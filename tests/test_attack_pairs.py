from __future__ import annotations

import dataclasses
import json
import math
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
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.board_features import ATTACK_CODES
from wombat.model.network import RecurrentBlock

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
)
WITH = dataclasses.replace(BASE, attack_pairs=True)
# Sparse positions whose pieces attack across the board: rooks on open files, a long diagonal.
FENS = (
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3q4/8/8/6B1/4K3 b - - 0 1",
)


def _blocks(model: WombatNetwork) -> list[RecurrentBlock]:
    blocks = list(model.recurrent_core)
    assert all(isinstance(block, RecurrentBlock) for block in blocks)
    return blocks  # type: ignore[return-value]


def _off_identity(model: WombatNetwork, tables: bool = False) -> None:
    """Move the recurrent core off its identity start, and the attack tables off zero."""

    with torch.no_grad():
        for block in _blocks(model):
            torch.nn.init.normal_(block.attention.out_proj.weight, std=0.3)
            torch.nn.init.normal_(block.mlp.output.weight, std=0.1)
            block.square_bias.normal_()
            if model.config.relative_attention:
                block.relative_keys.normal_()
                block.relative_values.normal_()
            if tables:
                block.attack_bias.normal_()


def _positions() -> tuple[torch.Tensor, torch.Tensor]:
    pieces = np.stack([encode_position(chess.Board(fen)).pieces for fen in FENS])
    generator = torch.Generator().manual_seed(0)
    return torch.from_numpy(pieces).long(), torch.rand((len(FENS), 26), generator=generator)


def _key_square(column: int) -> int | None:
    """The square of a memory column: the board memory's squares, then H's."""

    if column < 64:
        return column
    if 65 <= column < 65 + 64:
        return column - 65
    return None


def test_the_option_is_off_by_default_and_needs_the_square_bias() -> None:
    assert not NetworkConfig().attack_pairs
    for command in ("train-games", "train-puzzles"):
        assert not build_parser().parse_args([command, "--output", "run"]).attack_pairs
        parsed = build_parser().parse_args([command, "--output", "run", "--attack-pairs"])
        assert parsed.attack_pairs
    plain = dataclasses.replace(WITH, local_attention=False, square_attention_bias=False)
    with pytest.raises(ValueError, match="square attention bias"):
        WombatNetwork(plain)
    with pytest.raises(ValueError, match="king neighbourhood"):
        WombatNetwork(dataclasses.replace(WITH, local_neighbourhood="move_graph"))
    # Per block, a table per direction of a logit per head and attack code.
    added = count_parameters(WombatNetwork(WITH)) - count_parameters(WombatNetwork(BASE))
    assert added == BASE.recurrent_blocks * 2 * BASE.attention_heads * ATTACK_CODES


def test_pairs_open_the_local_mask_and_a_table_moves_only_their_logits() -> None:
    torch.manual_seed(1)
    model = WombatNetwork(WITH).eval()
    block = _blocks(model)[0]
    pieces, _ = _positions()
    codes = model.local_edges(pieces)
    assert codes is not None
    paired = (codes > 0) | (codes.transpose(1, 2) > 0)
    hidden_tokens = model.hidden_tokens
    with torch.no_grad():
        plain = WombatNetwork(BASE).eval()
        king = _blocks(plain)[0].attention_bias(65, hidden_tokens)
        zero = block.attention_bias(65, hidden_tokens, None, codes)
        block.attack_bias.normal_()
        moved = block.attention_bias(65, hidden_tokens, None, codes)
    # Each rook sees the one across its file in both copies of the board, and back.
    a1, a8, h1 = chess.A1, chess.A8, chess.H1
    for query, key in ((a1, a8), (a8, a1), (h1, chess.H8), (chess.H8, h1)):
        for column in (key, 65 + key):
            assert torch.isfinite(zero[0, :, query, column]).all()
        assert not torch.isfinite(king[:, query, key]).any()
    # Every square row opens exactly its king steps and its pairs; other rows are unchanged.
    near = torch.isfinite(king[0, :64, :64])
    for index in range(len(FENS)):
        opened = near | paired[index]
        for columns in (slice(0, 64), slice(65, 129)):
            assert torch.equal(torch.isfinite(zero[index, 0, :64, columns]), opened)
        assert torch.equal(zero[index, :, 64:], king.expand_as(zero[index])[:, 64:])
    assert bool((paired & ~near).any(dim=(1, 2)).all())
    # A nonzero table moves the logits of the pairs only, by one entry per direction.
    difference = (moved - zero)[:, :, :64, :64]
    changed = (difference != 0) & torch.isfinite(zero[:, :, :64, :64])
    assert torch.equal(changed.any(1), paired)
    table = torch.nn.functional.pad(block.attack_bias.detach(), (1, 0))
    for index, (query, key) in enumerate(((a1, a8), (a1, chess.E1))):
        expected = table[0][:, codes[0, query, key]] + table[1][:, codes[0, key, query]]
        assert torch.allclose(difference[0, :, query, key], expected, atol=1e-6), index
    assert torch.equal(moved[:, :, 64:], zero[:, :, 64:])


@pytest.mark.parametrize("relative", [False, True])
def test_logits_and_outputs_match_a_slow_reference(relative: bool) -> None:
    config = dataclasses.replace(WITH, relative_attention=relative, latent_tokens=1)
    torch.manual_seed(2)
    model = WombatNetwork(config).eval()
    _off_identity(model, tables=True)
    block = _blocks(model)[0]
    hidden_tokens = model.hidden_tokens
    heads, width = config.attention_heads, config.d_model
    head_width = width // heads
    pieces, _ = _positions()
    codes = model.local_edges(pieces)
    assert codes is not None
    batch = len(pieces)
    generator = torch.Generator().manual_seed(3)
    queries = torch.randn(batch, hidden_tokens, width, generator=generator)
    memory = torch.cat((torch.randn(batch, 65, width, generator=generator), queries), dim=1)
    attention = block.attention
    q, k, v = (
        torch.nn.functional.linear(source, weight, bias)
        .unflatten(-1, (heads, head_width))
        .transpose(1, 2)
        for source, weight, bias in zip(
            (queries, memory, memory),
            attention.in_proj_weight.chunk(3),
            attention.in_proj_bias.chunk(3),
            strict=True,
        )
    )
    square_bias = block.square_bias.detach()
    table = torch.nn.functional.pad(block.attack_bias.detach(), (1, 0))
    logits = torch.full((batch, heads, hidden_tokens, 65 + hidden_tokens), -math.inf)
    offsets: list[list[int | None]] = []
    for row in range(hidden_tokens):
        offsets.append([])
        for column in range(65 + hidden_tokens):
            key = _key_square(column)
            offset = None
            if row < 64 and key is None:
                offset = 9
            elif row < 64 and key is not None:
                file_step, rank_step = key % 8 - row % 8, key // 8 - row // 8
                if abs(file_step) <= 1 and abs(rank_step) <= 1:
                    offset = 3 * (rank_step + 1) + file_step + 1
            offsets[-1].append(offset if relative else None)
    with torch.no_grad():
        for index in range(batch):
            for row in range(hidden_tokens):
                for column in range(65 + hidden_tokens):
                    key = _key_square(column)
                    square_row = row < 64
                    bias = torch.zeros(heads)
                    if square_row and key is not None:
                        forward, backward = codes[index, row, key], codes[index, key, row]
                        near = max(abs(key % 8 - row % 8), abs(key // 8 - row // 8)) <= 1
                        if not (near or forward or backward):
                            continue
                        bias = square_bias[:, row, key] + table[0][:, forward]
                        bias = bias + table[1][:, backward]
                    elif square_row and 128 < column:
                        # Square queries never read H's latent tokens.
                        continue
                    score = (q[index, :, row] * k[index, :, column]).sum(-1)
                    offset = offsets[row][column]
                    if offset is not None:
                        score = score + (q[index, :, row] * block.relative_keys[:, offset]).sum(-1)
                    logits[index, :, row, column] = score / math.sqrt(head_width) + bias
        probabilities = logits.softmax(-1)
        mixed = probabilities @ v
        if relative:
            for row in range(64):
                for column in range(65 + hidden_tokens):
                    offset = offsets[row][column]
                    if offset is not None:
                        weight = probabilities[:, :, row, column, None]
                        mixed[:, :, row] += weight * block.relative_values[:, offset]
        expected = attention.out_proj(mixed.transpose(1, 2).flatten(-2))
        traced, traced_probabilities = block._biased_attention(queries, memory, True, codes)
        fused, _ = block._biased_attention(queries, memory, False, codes)
    assert traced_probabilities is not None
    finite = torch.isfinite(logits)
    assert torch.equal(traced_probabilities == 0, ~finite)
    assert torch.allclose(traced_probabilities, probabilities, atol=1e-6)
    assert torch.allclose(traced, expected, atol=1e-5)
    assert torch.allclose(fused, expected, atol=1e-5)


def test_zero_tables_without_local_attention_leave_the_outputs_unchanged() -> None:
    config = dataclasses.replace(BASE, local_attention=False)
    torch.manual_seed(4)
    plain = WombatNetwork(config).eval()
    _off_identity(plain)
    model = WombatNetwork(dataclasses.replace(config, attack_pairs=True)).eval()
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {f"recurrent_core.{i}.attack_bias" for i in range(2)}
    pieces, state = _positions()
    with torch.no_grad():
        expected = plain(pieces, state, 2, include_move_values=True)
        outputs = model(pieces, state, 2, include_move_values=True)
    for reference, tensor in zip(expected, outputs, strict=True):
        assert torch.allclose(reference, tensor, atol=1e-5)


def test_gradients_reach_the_tables_of_every_block() -> None:
    torch.manual_seed(5)
    model = WombatNetwork(dataclasses.replace(WITH, relative_attention=True)).train()
    _off_identity(model, tables=True)
    pieces, state = _positions()
    policy, value = model(pieces, state, 2)
    (policy.logsumexp(-1).mean() + value.square().mean()).backward()
    for block in _blocks(model):
        grad = block.attack_bias.grad
        assert grad is not None and float(grad.abs().sum()) > 0
        # Codes no position holds get no gradient.
        codes = model.local_edges(pieces)
        assert codes is not None
        unused = torch.ones(ATTACK_CODES + 1, dtype=torch.bool)
        unused[codes.unique()] = False
        assert float(grad[..., unused[1:]].abs().sum()) == 0


def test_attack_pair_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "attack-pairs"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "2"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--relative-attention", "--state-attention-bias"]
    arguments += ["--board-features", "--move-consequences", "--value-bins", "8", "--attack-pairs"]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["attack_pairs"]
    assert not any("square_offset" in key for key in saved["model"])
    assert float(saved["model"]["recurrent_core.1.attack_bias"].abs().sum()) > 0
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


def test_checkpoints_without_the_option_load_unchanged(tmp_path: Path) -> None:
    config = dataclasses.replace(BASE, relative_attention=True, state_attention_bias=True)
    torch.manual_seed(6)
    model = WombatNetwork(config).eval()
    _off_identity(model)
    recorded = dataclasses.asdict(config)
    for name in ("attack_pairs", "phase_heads", "king_relative"):
        recorded.pop(name, None)
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "model_config": recorded}, path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded.config == config
    pieces, state = _positions()
    with torch.no_grad():
        for mine, reference in zip(
            loaded(pieces, state, 2, include_move_values=True),
            model(pieces, state, 2, include_move_values=True),
            strict=True,
        ):
            assert torch.equal(mine, reference)
