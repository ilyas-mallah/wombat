from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position
from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
from wombat.interpretability.explain import explain_move
from wombat.losses import POLICY_ONLY, complete_targets, unrolled_loss
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.network import phase_buckets

BASE = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=1,
    latent_tokens=2,
    policy_readout="square",
    move_readout="from_to",
    normalize_state=True,
    max_residual_update=None,
    square_attention_bias=True,
    local_attention=True,
    lookahead_heads=True,
    move_consequences=True,
    value_bins=8,
)
BUCKETS = 4
WITH = dataclasses.replace(BASE, phase_heads=BUCKETS)


def _positions(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Kings, some pawns and 0 to 16 other pieces, so every bucket appears."""

    generator = torch.Generator().manual_seed(seed)
    pieces = torch.zeros(count, 64, dtype=torch.long)
    for row in range(count):
        squares = torch.randperm(64, generator=generator)
        others = row % 17
        pawns = int(torch.randint(0, 9, (), generator=generator))
        pieces[row, squares[0]], pieces[row, squares[1]] = 6, 12
        kinds = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])
        choice = torch.randint(0, 8, (others,), generator=generator)
        pieces[row, squares[2 : 2 + others]] = kinds[choice]
        pawn_ids = 1 + 6 * torch.randint(0, 2, (pawns,), generator=generator)
        pieces[row, squares[2 + others : 2 + others + pawns]] = pawn_ids
    return pieces, torch.rand((count, 26), generator=generator)


def _single_to_phase(single: WombatNetwork, model: WombatNetwork) -> None:
    """Load the single head's weights into every copy of the phase model's output layers."""

    state = single.state_dict()
    for name, tensor in model.state_dict().items():
        source = state[name]
        if source.shape != tensor.shape:
            source = source.repeat(BUCKETS, *[1] * (source.ndim - 1))
        tensor.copy_(source)


def test_the_option_is_off_by_default_and_needs_the_square_readout() -> None:
    assert NetworkConfig().phase_heads == 0
    for command in ("train-games", "train-puzzles"):
        assert build_parser().parse_args([command, "--output", "run"]).phase_heads == 0
        parsed = build_parser().parse_args([command, "--output", "run", "--phase-heads", "4"])
        assert parsed.phase_heads == 4
    with pytest.raises(ValueError, match="square policy readout"):
        WombatNetwork(NetworkConfig(phase_heads=4))
    with pytest.raises(ValueError, match="negative"):
        WombatNetwork(dataclasses.replace(WITH, phase_heads=-1))


def test_buckets_split_non_pawn_pieces_at_the_documented_edges() -> None:
    counts = [0, 1, 3, 4, 7, 8, 11, 12, 14, 16, 18]
    pieces = torch.zeros(len(counts), 64, dtype=torch.long)
    pieces[:, 0], pieces[:, 1] = 6, 12
    # Pawns and kings never count; both sides' knights, bishops, rooks and queens do.
    pieces[:, 2:10] = 1
    pieces[:, 10:18] = 7
    for row, count in enumerate(counts):
        kinds = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11]).repeat(3)[:count]
        pieces[row, 18 : 18 + count] = kinds
    assert phase_buckets(pieces, 4).tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
    assert phase_buckets(pieces, 2).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert phase_buckets(pieces, 1).tolist() == [0] * len(counts)
    boards = [
        chess.Board(),
        chess.Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1"),
        chess.Board("r3k3/8/2n5/8/8/2B5/8/R2QK3 w - - 0 1"),
    ]
    encoded = torch.from_numpy(np.stack([encode_position(board).pieces for board in boards]))
    assert phase_buckets(encoded.long(), 4).tolist() == [3, 0, 1]
    model = WombatNetwork(WITH)
    phase = model.phase(encoded.long())
    assert phase is not None and phase.tolist() == [3, 0, 1]
    assert WombatNetwork(BASE).phase(encoded.long()) is None


@pytest.mark.parametrize("value_bins", [8, 0])
def test_equal_copies_give_the_single_head_outputs(value_bins: int) -> None:
    config = dataclasses.replace(BASE, value_bins=value_bins)
    phased = dataclasses.replace(config, phase_heads=BUCKETS)
    fresh = WombatNetwork(phased)
    # A fresh model's copies start equal.
    for layer in fresh.phase_layers():
        for tensor in (layer.weight, layer.bias):
            if tensor is not None:
                copies = tensor.detach().unflatten(0, (BUCKETS, -1))
                assert torch.equal(copies, copies[:1].expand_as(copies))
    torch.manual_seed(1)
    single = WombatNetwork(config).eval()
    with torch.no_grad():
        for parameter in single.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    model = WombatNetwork(phased).eval()
    with torch.no_grad():
        _single_to_phase(single, model)
    pieces, state = _positions(34, seed=2)
    assert set(phase_buckets(pieces, BUCKETS).tolist()) == set(range(BUCKETS))
    steps = frozenset({0, 2})
    with torch.no_grad():
        expected = single(pieces, state, 2, intermediate_steps=steps, include_move_values=True)
        outputs = model(pieces, state, 2, intermediate_steps=steps, include_move_values=True)
    assert isinstance(expected, dict) and isinstance(outputs, dict)
    for step, tensors in expected.items():
        for reference, tensor in zip(tensors, outputs[step], strict=True):
            if value_bins:
                assert torch.equal(reference, tensor)
            else:
                # A one-wide product rounds apart from a four-wide one, by about 4e-8.
                assert torch.allclose(reference, tensor, rtol=0, atol=1e-6)
    # Each output layer adds its parameters again for every further copy.
    added = count_parameters(model) - count_parameters(single)
    per_copy = sum(
        parameter.numel() // BUCKETS
        for layer in model.phase_layers()
        for parameter in layer.parameters()
    )
    assert added == (BUCKETS - 1) * per_copy
    assert len(model.phase_layers()) == (13 if value_bins else 11)


def test_each_position_reads_the_copy_of_its_bucket() -> None:
    torch.manual_seed(3)
    model = WombatNetwork(WITH).eval()
    pieces, state = _positions(34, seed=4)
    buckets = phase_buckets(pieces, BUCKETS)
    with torch.no_grad():
        before = model(pieces, state, 1, include_move_values=True)
        for layer in model.phase_layers():
            layer.weight.unflatten(0, (BUCKETS, -1))[2].add_(0.5)
        after = model(pieces, state, 1, include_move_values=True)
    for reference, tensor in zip(before, after, strict=True):
        moved = (reference != tensor).reshape(len(pieces), -1).any(1)
        assert torch.equal(moved, buckets == 2)


def test_training_reaches_only_the_copies_of_the_batch_buckets() -> None:
    torch.manual_seed(5)
    model = WombatNetwork(WITH).train()
    pieces, state = _positions(34, seed=6)
    keep = phase_buckets(pieces, BUCKETS) != 1
    pieces, state = pieces[keep], state[keep]
    batch = complete_targets(
        {
            "pieces": pieces,
            "state": state,
            "action": torch.zeros(len(pieces), dtype=torch.long),
            "legal_mask": torch.ones(len(pieces), 4672, dtype=torch.bool),
            "value": torch.zeros(len(pieces)),
        }
    )
    memory = model.encode_board(pieces, state)
    loss, _ = unrolled_loss(
        model, memory, model.initial_hidden(memory), 1, batch, POLICY_ONLY, every_step=False
    )
    loss.backward()
    policy = model.square_policy_head.weight.grad
    assert policy is not None
    by_copy = policy.unflatten(0, (BUCKETS, -1)).abs().sum(dim=(1, 2))
    assert by_copy[1] == 0 and bool((by_copy[[0, 2, 3]] > 0).all())


def test_readouts_without_the_buckets_are_refused() -> None:
    model = WombatNetwork(WITH).eval()
    pieces, state = _positions(2, seed=7)
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory)
    with pytest.raises(ValueError, match="phase buckets"):
        model.heads(hidden, consequences=model.consequences(pieces, state))


def test_phase_head_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "phase-heads"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "2"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--relative-attention", "--state-attention-bias"]
    arguments += ["--board-features", "--move-consequences", "--value-bins", "8"]
    arguments += ["--phase-heads", str(BUCKETS)]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["phase_heads"] == BUCKETS
    assert saved["model"]["square_policy_head.weight"].shape == (73 * BUCKETS, 24)
    model = load_checkpoint(checkpoint, torch.device("cpu"))
    assert model.config == NetworkConfig(**saved["model_config"])
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    report = explain_move(model, board, chess.Move.from_uci("f1b5"), steps=2, depths=(0, 2))
    assert report["move"]["san"] == "Bb5"
    split = GameSplit(game_corpus / "validation")
    indices = np.arange(min(6, len(split)))
    cpu = torch.device("cpu")
    for select in ("policy", "mix"):
        predicted = predict_depths(
            model, split, indices, (0, 1, 3), batch_size=4, device=cpu, select=select
        )
        assert all(len(actions) == len(indices) for actions in predicted.values())
    capsys.readouterr()
