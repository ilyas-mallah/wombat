from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
from wombat.interpretability.explain import explain_move
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.network import RELATIVE_OFFSETS, RecurrentBlock

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
WITH = dataclasses.replace(BASE, relative_attention=True)
RELATIVE = ("relative_keys", "relative_values")


def _blocks(model: WombatNetwork) -> list[RecurrentBlock]:
    blocks = list(model.recurrent_core)
    assert all(isinstance(block, RecurrentBlock) for block in blocks)
    return blocks  # type: ignore[return-value]


def _off_identity(model: WombatNetwork, relative: bool = False) -> None:
    """Move the recurrent core off its identity start, and the relative vectors off zero."""

    with torch.no_grad():
        for block in _blocks(model):
            torch.nn.init.normal_(block.attention.out_proj.weight, std=0.3)
            torch.nn.init.normal_(block.mlp.output.weight, std=0.1)
            block.square_bias.normal_()
            if relative:
                block.relative_keys.normal_()
                block.relative_values.normal_()


def _positions(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pieces = torch.randint(0, 13, (count, 64), generator=generator)
    return pieces, torch.rand((count, 26), generator=generator)


def test_the_option_is_off_by_default_and_needs_king_local_attention() -> None:
    assert not NetworkConfig().relative_attention
    for command in ("train-games", "train-puzzles"):
        assert not build_parser().parse_args([command, "--output", "run"]).relative_attention
        parsed = build_parser().parse_args([command, "--output", "run", "--relative-attention"])
        assert parsed.relative_attention
    for config in (
        dataclasses.replace(WITH, local_attention=False),
        dataclasses.replace(WITH, local_neighbourhood="king_knight"),
        dataclasses.replace(WITH, local_neighbourhood="move_graph"),
    ):
        with pytest.raises(ValueError, match="king steps"):
            WombatNetwork(config)
    with pytest.raises(ValueError, match="scratch boards"):
        WombatNetwork(dataclasses.replace(WITH, scratch_boards=1))


@pytest.mark.parametrize(("width", "heads"), [(24, 6), (32, 4)])
def test_zero_initialised_vectors_leave_every_output_bit_identical(width: int, heads: int) -> None:
    config = dataclasses.replace(BASE, d_model=width, attention_heads=heads)
    torch.manual_seed(0)
    plain = WombatNetwork(config).eval()
    _off_identity(plain)
    model = WombatNetwork(dataclasses.replace(config, relative_attention=True)).eval()
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {f"recurrent_core.{i}.{name}" for i in range(2) for name in RELATIVE}
    # Per block, a key and a value vector per offset and head: 2 * 10 * width.
    added = count_parameters(model) - count_parameters(plain)
    assert added == config.recurrent_blocks * 2 * RELATIVE_OFFSETS * width
    pieces, state = _positions(5, seed=1)
    steps = frozenset({0, 2})
    with torch.no_grad():
        expected = plain(pieces, state, 3, intermediate_steps=steps, include_move_values=True)
        outputs = model(pieces, state, 3, intermediate_steps=steps, include_move_values=True)
    assert isinstance(expected, dict) and isinstance(outputs, dict)
    assert expected.keys() == outputs.keys() == {0, 2, 3}
    for step, tensors in expected.items():
        for reference, tensor in zip(tensors, outputs[step], strict=True):
            assert torch.equal(reference, tensor)
    traced = model.trace_reasoning(pieces, state, 2)
    plain_traced = plain.trace_reasoning(pieces, state, 2)
    for ours, theirs in zip(traced, plain_traced, strict=True):
        assert torch.equal(ours.hidden, theirs.hidden)
        for attention, reference in zip(ours.attention_maps, theirs.attention_maps, strict=True):
            assert torch.equal(attention, reference)


def _reference_offsets(hidden_tokens: int) -> list[list[int | None]]:
    """Per H row and memory column, the king offset index, 9 for non-square keys, None when the
    row is not a square, the pair is not within a king step, or the row gets no relative term."""

    def square(column: int) -> int | None:
        if column < 64:
            return column
        if 65 <= column < 65 + 64:
            return column - 65
        return None

    table: list[list[int | None]] = []
    for row in range(hidden_tokens):
        entries: list[int | None] = []
        for column in range(65 + hidden_tokens):
            key = square(column)
            if row >= 64:
                entries.append(None)
            elif key is None:
                entries.append(9)
            else:
                file_step, rank_step = key % 8 - row % 8, key // 8 - row // 8
                near = abs(file_step) <= 1 and abs(rank_step) <= 1
                entries.append(3 * (rank_step + 1) + file_step + 1 if near else None)
        table.append(entries)
    return table


@pytest.mark.parametrize(("width", "heads"), [(24, 6), (16, 2)])
def test_logits_and_outputs_match_a_slow_reference(width: int, heads: int) -> None:
    config = dataclasses.replace(WITH, d_model=width, attention_heads=heads, latent_tokens=1)
    torch.manual_seed(2)
    model = WombatNetwork(config).eval()
    _off_identity(model, relative=True)
    block = _blocks(model)[0]
    hidden_tokens = model.hidden_tokens
    batch, head_width = 2, width // heads
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
    bias = block.attention_bias(65, hidden_tokens).detach()
    offsets = _reference_offsets(hidden_tokens)
    keys, values = block.relative_keys.detach(), block.relative_values.detach()
    logits = torch.empty(batch, heads, hidden_tokens, 65 + hidden_tokens)
    mixed = torch.zeros(batch, heads, hidden_tokens, head_width)
    with torch.no_grad():
        for row in range(hidden_tokens):
            for column in range(65 + hidden_tokens):
                score = (q[:, :, row] * k[:, :, column]).sum(-1)
                offset = offsets[row][column]
                if offset is not None:
                    score = score + (q[:, :, row] * keys[:, offset]).sum(-1)
                logits[:, :, row, column] = score / math.sqrt(head_width) + bias[:, row, column]
        probabilities = logits.softmax(-1)
        for row in range(hidden_tokens):
            for column in range(65 + hidden_tokens):
                weight = probabilities[:, :, row, column, None]
                offset = offsets[row][column]
                read = v[:, :, column] + (0 if offset is None else values[:, offset])
                mixed[:, :, row] += weight * read
        expected = attention.out_proj(mixed.transpose(1, 2).flatten(-2))
        traced, traced_probabilities = block._biased_attention(queries, memory, True, None)
        fused, _ = block._biased_attention(queries, memory, False, None)
    assert traced_probabilities is not None
    finite = torch.isfinite(logits)
    # Square rows read their nine king neighbours in each half and the board's state token.
    assert int(finite[0, 0, 27].sum()) == 19 and int(finite[0, 0, 0].sum()) == 9
    assert torch.equal(traced_probabilities == 0, ~finite)
    log_probabilities = traced_probabilities.log()
    reference = logits.log_softmax(-1)
    assert torch.allclose(log_probabilities[finite], reference[finite], atol=1e-5)
    assert torch.allclose(traced, expected, atol=1e-5)
    assert torch.allclose(fused, expected, atol=1e-5)
    # The relative terms change the square rows and nothing else.
    with torch.no_grad():
        block.relative_keys.zero_()
        block.relative_values.zero_()
        bare, _ = block._biased_attention(queries, memory, False, None)
    assert not torch.allclose(bare[:, :64], expected[:, :64], atol=1e-3)
    assert torch.allclose(bare[:, 64:], expected[:, 64:], atol=1e-5)


def test_gradients_reach_both_vector_sets_and_training_moves_them_from_zero() -> None:
    torch.manual_seed(4)
    model = WombatNetwork(WITH).train()
    pieces, state = _positions(4, seed=5)
    targets = torch.randint(0, 4672, (4,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        policy, value = model(pieces, state, 2)
        loss = torch.nn.functional.cross_entropy(policy, targets) + value.square().mean()
        loss.backward()
        optimizer.step()
    for block in _blocks(model):
        for name in RELATIVE:
            parameter = getattr(block, name)
            assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0, name
            assert float(parameter.detach().abs().sum()) > 0, name


def test_relative_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "relative"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "2"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--relative-attention"]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["relative_attention"]
    # The index tables stay out of the state dict; the vectors learned something.
    assert not any("class" in key or "relative_codes" in key for key in saved["model"])
    for name in RELATIVE:
        assert float(saved["model"][f"recurrent_core.1.{name}"].abs().sum()) > 0
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
    capsys.readouterr()
    explain = ["explain-move", "--checkpoint", str(checkpoint), "--fen", board.fen()]
    assert main([*explain, "--move", "f1b5", "--steps", "1", "--depths", "0,1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["move"]["san"] == "Bb5"
