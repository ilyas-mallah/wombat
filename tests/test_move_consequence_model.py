from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import random

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import legal_action_mask, legal_actions
from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.eval.interventions import evaluate_hidden_interventions
from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
from wombat.interpretability.explain import explain_move
from wombat.losses import LossWeights, complete_targets, unrolled_loss
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.move_consequences import CONSEQUENCES
from wombat.model.network import board_consequences

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
    board_features=True,
)
WITH = dataclasses.replace(BASE, move_consequences=True)
MAP_LAYERS = (
    "policy_consequence_origin",
    "policy_consequence_destination",
    "value_consequence_origin",
    "value_consequence_destination",
)


def _boards(count: int, seed: int) -> list[chess.Board]:
    randomizer = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(randomizer.randrange(6, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        if any(board.legal_moves):
            boards.append(board)
    return boards


def _batch(boards: list[chess.Board]) -> dict[str, torch.Tensor]:
    encoded = [encode_position(board) for board in boards]
    return {
        "pieces": torch.from_numpy(np.stack([item.pieces for item in encoded])).long(),
        "state": torch.from_numpy(np.stack([state_vector(item) for item in encoded])),
        "legal_mask": torch.from_numpy(np.stack([legal_action_mask(board) for board in boards])),
        "action": torch.tensor([legal_actions(board)[0] for board in boards]),
    }


def test_the_option_is_off_by_default_and_needs_the_square_readout() -> None:
    assert not NetworkConfig().move_consequences
    for command in ("train-games", "train-puzzles"):
        assert not build_parser().parse_args([command, "--output", "run"]).move_consequences
        parsed = build_parser().parse_args([command, "--output", "run", "--move-consequences"])
        assert parsed.move_consequences
    with pytest.raises(ValueError, match="square policy readout"):
        WombatNetwork(NetworkConfig(move_consequences=True))


@pytest.mark.parametrize("readout", ["square", "from_to"])
def test_zero_initialised_map_leaves_every_readout_bit_identical(readout: str) -> None:
    torch.manual_seed(0)
    plain = WombatNetwork(dataclasses.replace(BASE, move_readout=readout)).eval()
    model = WombatNetwork(dataclasses.replace(WITH, move_readout=readout)).eval()
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert {key.split(".")[0] for key in missing} == set(MAP_LAYERS)
    # Four maps of CONSEQUENCES weights per model width, plus two origin biases.
    added = count_parameters(model) - count_parameters(plain)
    assert added == 4 * CONSEQUENCES * BASE.d_model + 2 * CONSEQUENCES
    batch = _batch(_boards(6, seed=1))
    steps = frozenset({0, 2})
    with torch.no_grad():
        inputs = (batch["pieces"], batch["state"], 3)
        expected = plain(*inputs, intermediate_steps=steps, include_move_values=True)
        outputs = model(*inputs, intermediate_steps=steps, include_move_values=True)
    assert isinstance(expected, dict) and isinstance(outputs, dict)
    assert expected.keys() == outputs.keys() == {0, 2, 3}
    for step, readout_tensors in expected.items():
        for reference, tensor in zip(readout_tensors, outputs[step], strict=True):
            assert torch.equal(reference, tensor)
    features = board_consequences(model.encode_board(batch["pieces"], batch["state"]))
    assert features is not None and features.abs().sum() > 0


def test_readouts_need_the_consequences() -> None:
    model = WombatNetwork(WITH)
    batch = _batch(_boards(2, seed=2))
    memory = model.encode_board(batch["pieces"], batch["state"], consequences=False)
    assert board_consequences(memory) is None
    with pytest.raises(ValueError, match="move consequences"):
        model.heads(model.initial_hidden(memory))


def test_training_moves_the_map_and_its_weights_follow_the_tokens() -> None:
    torch.manual_seed(3)
    model = WombatNetwork(WITH).train()
    boards = _boards(8, seed=3)
    batch = _batch(boards)
    generator = torch.Generator().manual_seed(0)
    legal = batch["legal_mask"]
    targets = complete_targets(batch)
    targets["move_value_target"] = torch.rand(legal.shape, generator=generator) * 2 - 1
    targets["move_value_mask"] = legal.clone()
    weights = LossWeights(move_value=1.0, value=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    memory = model.encode_board(batch["pieces"], batch["state"])
    loss, _ = unrolled_loss(
        model, memory, model.initial_hidden(memory), 2, targets, weights, every_step=True
    )
    loss.backward()
    for name in MAP_LAYERS:
        layer = getattr(model, name)
        assert float(layer.weight.grad.abs().sum()) > 0, name
        if layer.bias is not None:
            assert float(layer.bias.grad.abs().sum()) > 0, name
    optimizer.step()
    model.eval()
    with torch.no_grad():
        memory = model.encode_board(batch["pieces"], batch["state"])
        features = board_consequences(memory)
        assert features is not None
        hidden = model.initial_hidden(memory)
        for _ in range(2):
            hidden = model.recurrent_step(memory, hidden)
        policy = model.heads(hidden, consequences=features)[0]
        bare = model.heads(hidden, consequences=torch.zeros_like(features))[0]
    terms = (policy - bare).view(len(boards), 64, 73)
    # Legal moves with the same nonzero features get different terms from different tokens.
    groups: dict[tuple[float, ...], list[float]] = {}
    for row, origin, plane in legal.view(len(boards), 64, 73).nonzero().tolist():
        key = tuple(features[row, origin, plane].tolist())
        if any(key):
            groups.setdefault(key, []).append(float(terms[row, origin, plane]))
    spreads = [max(values) - min(values) for values in groups.values() if len(values) > 1]
    assert spreads and max(spreads) > 1e-4


def test_consequence_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "consequences"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "1"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--board-features", "--board-features-long-range"]
    arguments += ["--move-consequences"]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["move_consequences"]
    # The feature tables stay out of the state dict; the maps learned something.
    assert not any(key.startswith("move_consequences.") for key in saved["model"])
    assert float(saved["model"]["policy_consequence_origin.weight"].abs().sum()) > 0
    model = load_checkpoint(checkpoint, torch.device("cpu"))
    assert model.config == NetworkConfig(**saved["model_config"])
    snapshot = load_checkpoint(output / "snapshot-step-000004.pt", torch.device("cpu"))
    assert snapshot.config == model.config
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
    batch = split.batch(indices[:3])
    outputs = evaluate_hidden_interventions(
        model, batch["pieces"], batch["state"], total_steps=3, midpoint=1
    )
    assert outputs.swapped is not None and outputs.normal[0].shape == (3, 4672)
    capsys.readouterr()
    explain = ["explain-move", "--checkpoint", str(checkpoint), "--fen", board.fen()]
    assert main([*explain, "--move", "f1b5", "--steps", "1", "--depths", "0,1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["move"]["san"] == "Bb5"
