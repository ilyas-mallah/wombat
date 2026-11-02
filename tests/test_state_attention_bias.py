from __future__ import annotations

import copy
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
from wombat.model.network import STATE_BIAS_SQUARE_WIDTH, RecurrentBlock

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
WIDTH = 4
WITH = dataclasses.replace(BASE, state_attention_bias=True, state_attention_bias_width=WIDTH)


def _blocks(model: WombatNetwork) -> list[RecurrentBlock]:
    blocks = list(model.recurrent_core)
    assert all(isinstance(block, RecurrentBlock) for block in blocks)
    return blocks  # type: ignore[return-value]


def _off_identity(model: WombatNetwork, templates: bool = False) -> None:
    """Move the recurrent core off its identity start, and the shared templates off zero."""

    with torch.no_grad():
        for block in _blocks(model):
            torch.nn.init.normal_(block.attention.out_proj.weight, std=0.3)
            torch.nn.init.normal_(block.mlp.output.weight, std=0.1)
            block.square_bias.normal_()
        if templates:
            model.state_bias_templates.weight.normal_()


def _new_parameters(model: WombatNetwork) -> dict[str, torch.nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if "state_bias" in name}


def _positions(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pieces = torch.randint(0, 13, (count, 64), generator=generator)
    return pieces, torch.rand((count, 26), generator=generator)


def test_the_option_is_off_by_default_and_needs_the_square_bias() -> None:
    assert not NetworkConfig().state_attention_bias
    for command in ("train-games", "train-puzzles"):
        parsed = build_parser().parse_args([command, "--output", "run"])
        assert not parsed.state_attention_bias
        assert parsed.state_attention_bias_width == NetworkConfig().state_attention_bias_width
        arguments = [command, "--output", "run", "--state-attention-bias"]
        parsed = build_parser().parse_args([*arguments, "--state-attention-bias-width", "8"])
        assert parsed.state_attention_bias and parsed.state_attention_bias_width == 8
    plain = dataclasses.replace(WITH, local_attention=False, square_attention_bias=False)
    with pytest.raises(ValueError, match="square attention bias"):
        WombatNetwork(plain)
    with pytest.raises(ValueError, match="must be positive"):
        WombatNetwork(dataclasses.replace(WITH, state_attention_bias_width=0))


@pytest.mark.parametrize("relative", [False, True])
def test_zero_initialised_templates_leave_every_output_bit_identical(relative: bool) -> None:
    config = dataclasses.replace(BASE, relative_attention=relative)
    torch.manual_seed(0)
    plain = WombatNetwork(config).eval()
    _off_identity(plain)
    if relative:
        with torch.no_grad():
            for block in _blocks(plain):
                block.relative_keys.normal_()
                block.relative_values.normal_()
    biased = dataclasses.replace(
        config, state_attention_bias=True, state_attention_bias_width=WIDTH
    )
    model = WombatNetwork(biased).eval()
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == set(_new_parameters(model))
    assert "state_bias_templates.weight" in missing
    # Per block the compression, the board vector with its norm, the head weights with theirs;
    # then the shared templates.
    heads, width = config.attention_heads, config.d_model
    per_block = width * STATE_BIAS_SQUARE_WIDTH + (64 * STATE_BIAS_SQUARE_WIDTH + 3) * WIDTH
    per_block += (WIDTH + 3) * heads * WIDTH
    added = count_parameters(model) - count_parameters(plain)
    assert added == config.recurrent_blocks * per_block + 64 * 64 * WIDTH
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


def _key_square(column: int) -> int | None:
    """The square of a memory column: the board memory's squares, then H's."""

    if column < 64:
        return column
    if 65 <= column < 65 + 64:
        return column - 65
    return None


@pytest.mark.parametrize("relative", [False, True])
def test_logits_and_outputs_match_a_slow_reference(relative: bool) -> None:
    config = dataclasses.replace(WITH, relative_attention=relative)
    torch.manual_seed(2)
    model = WombatNetwork(config).eval()
    _off_identity(model, templates=True)
    block = _blocks(model)[0]
    hidden_tokens = model.hidden_tokens
    heads, width = config.attention_heads, config.d_model
    batch, head_width = 2, width // heads
    generator = torch.Generator().manual_seed(3)
    queries = torch.randn(batch, hidden_tokens, width, generator=generator)
    memory = torch.cat((torch.randn(batch, 65, width, generator=generator), queries), dim=1)
    templates = model.state_bias_templates.weight
    with torch.no_grad():
        # Smolgen step by step: compress, flatten, board vector, head weights, templates.
        smolgen = block.state_bias
        compressed = smolgen.compress(queries[:, :64]).reshape(batch, -1)
        board = smolgen.board(compressed)
        weights = smolgen.weights(board).reshape(batch, heads, WIDTH)
        state = torch.einsum("bhw,pw->bhp", weights, templates).reshape(batch, heads, 64, 64)
        base = block.attention_bias(65, hidden_tokens)
        logits = torch.empty(batch, heads, hidden_tokens, 65 + hidden_tokens)
        for row in range(hidden_tokens):
            for column in range(65 + hidden_tokens):
                logits[:, :, row, column] = base[:, row, column]
                key = _key_square(column)
                if row < 64 and key is not None:
                    logits[:, :, row, column] += state[:, :, row, key]
        with_state = logits.clone()
        without, _ = block._biased_attention(queries, memory, True, None, templates * 0)
        traced, probabilities = block._biased_attention(queries, memory, True, None, templates)
        fused, _ = block._biased_attention(queries, memory, False, None, templates)
    assert probabilities is not None
    finite = torch.isfinite(with_state)
    # The king mask still decides which pairs exist; the state bias moves only those.
    assert torch.equal(finite, torch.isfinite(base).expand_as(finite))
    assert torch.equal(probabilities == 0, ~finite)
    if not relative:
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
        with torch.no_grad():
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_width) + with_state
            reference = scores.softmax(-1)
            expected = attention.out_proj((reference @ v).transpose(1, 2).flatten(-2))
        assert torch.allclose(probabilities, reference, atol=1e-6)
        assert torch.allclose(traced, expected, atol=1e-5)
    assert torch.allclose(fused, traced, atol=1e-5)
    # The bias changes the square rows and nothing else.
    assert not torch.allclose(without[:, :64], traced[:, :64], atol=1e-3)
    assert torch.allclose(without[:, 64:], traced[:, 64:], atol=1e-5)


def test_the_bias_follows_the_thinking_state_at_every_step() -> None:
    torch.manual_seed(6)
    model = WombatNetwork(WITH).eval()
    _off_identity(model, templates=True)
    recorded: list[torch.Tensor] = []
    first = _blocks(model)[0]
    handle = first.state_bias.register_forward_hook(
        lambda module, inputs, output: recorded.append(output.detach().clone())
    )
    pieces, state = _positions(3, seed=7)
    with torch.no_grad():
        model(pieces, state, 3)
    handle.remove()
    assert len(recorded) == 3
    # Recomputed from H at every step, so it moves as H moves, and differs between positions.
    assert not torch.allclose(recorded[0], recorded[1], atol=1e-4)
    assert not torch.allclose(recorded[1], recorded[2], atol=1e-4)
    assert not torch.allclose(recorded[0][0], recorded[0][1], atol=1e-4)


def test_a_training_step_moves_the_templates_and_changes_the_attention() -> None:
    torch.manual_seed(4)
    model = WombatNetwork(WITH).train()
    _off_identity(model)
    pieces, state = _positions(4, seed=5)
    targets = torch.randint(0, 4672, (4,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        policy, value = model(pieces, state, 2)
        loss = torch.nn.functional.cross_entropy(policy, targets) + value.square().mean()
        loss.backward()
        optimizer.step()
    # From the second iteration the templates are nonzero and every new parameter is reached.
    for name, parameter in _new_parameters(model).items():
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0, name
    model.eval()
    zeroed = copy.deepcopy(model)
    with torch.no_grad():
        zeroed.state_bias_templates.weight.zero_()
    assert float(model.state_bias_templates.weight.detach().abs().sum()) > 0
    ours = model.trace_reasoning(pieces, state, 2)
    theirs = zeroed.trace_reasoning(pieces, state, 2)
    for step in (1, 2):
        for attention, reference in zip(
            ours[step].attention_maps, theirs[step].attention_maps, strict=True
        ):
            assert not torch.allclose(attention[:, :, :64], reference[:, :, :64], atol=1e-5)
    # The first block of the first step sees the same H, so only its square rows differ.
    first, reference = ours[1].attention_maps[0], theirs[1].attention_maps[0]
    assert torch.equal(first[:, :, 64:], reference[:, :, 64:])


def test_gradients_reach_every_new_parameter_with_nonzero_templates() -> None:
    torch.manual_seed(8)
    model = WombatNetwork(WITH).train()
    _off_identity(model, templates=True)
    pieces, state = _positions(4, seed=9)
    policy, value = model(pieces, state, 2)
    (policy.logsumexp(-1).mean() + value.square().mean()).backward()
    parameters = _new_parameters(model)
    # Per block the compression, the board and head layers and their norms; then the templates.
    assert len(parameters) == 2 * 9 + 1
    for name, parameter in parameters.items():
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0, name


def test_state_bias_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "state-bias"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "2"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--state-attention-bias"]
    arguments += ["--state-attention-bias-width", str(WIDTH)]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["state_attention_bias"]
    assert saved["model_config"]["state_attention_bias_width"] == WIDTH
    # The templates learned something from zero.
    assert float(saved["model"]["state_bias_templates.weight"].abs().sum()) > 0
    assert "recurrent_core.1.state_bias.compress.weight" in saved["model"]
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
