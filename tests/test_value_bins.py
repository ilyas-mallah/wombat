from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from statistics import NormalDist

import chess
import numpy as np
import pytest
import torch

from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
from wombat.interpretability.explain import explain_move
from wombat.losses import (
    POLICY_ONLY,
    LossWeights,
    complete_targets,
    graded_loss,
    hl_gauss_targets,
    step_loss,
)
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.network import board_consequences, expected_value, value_bin_values

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
)
BINS = 8
WITH = dataclasses.replace(BASE, value_bins=BINS)
NEW = {
    "move_value_bin_origin.weight",
    "move_value_bin_origin.bias",
    "move_value_bin_destination.weight",
    "move_value_bin_tilt",
}


def _positions(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pieces = torch.randint(0, 13, (count, 64), generator=generator)
    return pieces, torch.rand((count, 26), generator=generator)


def _live_consequences(model: WombatNetwork) -> None:
    """Move the zero-initialised consequence weights off zero, so they reach every readout."""

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "consequence" in name:
                parameter.normal_(std=0.1)


def _state(
    model: WombatNetwork, pieces: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Board memory, H after one step, and the board's move consequences."""

    memory = model.encode_board(pieces, state)
    hidden = model.recurrent_step(memory, model.initial_hidden(memory))
    assert isinstance(hidden, torch.Tensor)
    return memory, hidden, board_consequences(memory)


def test_the_option_is_off_by_default_and_needs_the_square_readout() -> None:
    assert NetworkConfig().value_bins == 0
    for command in ("train-games", "train-puzzles"):
        parsed = build_parser().parse_args([command, "--output", "run"])
        assert parsed.value_bins == 0
        parsed = build_parser().parse_args([command, "--output", "run", "--value-bins", "64"])
        assert parsed.value_bins == 64
    for bins in (-1, 1):
        with pytest.raises(ValueError, match="at least 2"):
            WombatNetwork(dataclasses.replace(WITH, value_bins=bins))
    with pytest.raises(ValueError, match="square policy readout"):
        WombatNetwork(NetworkConfig(value_bins=BINS))
    # The scalar model has none of the new parameters and keeps its one-output value head.
    names = set(WombatNetwork(BASE).state_dict())
    assert not NEW & names
    assert WombatNetwork(BASE).value_head[2].out_features == 1


@pytest.mark.parametrize("bins", [4, 64])
def test_hl_gauss_targets_match_a_direct_computation(bins: int) -> None:
    values = torch.tensor([-1.0, -0.37, 0.0, 0.5, 0.999, 1.0])
    targets = hl_gauss_targets(values, bins)
    assert targets.shape == (len(values), bins)
    sigma = 0.75 / bins
    for row, value in enumerate(values.tolist()):
        # A Gaussian on the win chance, cut to [0, 1] and renormalised (truncated, not folded).
        normal = NormalDist((value + 1) / 2, sigma)
        inside = normal.cdf(1) - normal.cdf(0)
        edges = [k / bins for k in range(bins + 1)]
        expected = [
            (normal.cdf(high) - normal.cdf(low)) / inside
            for low, high in zip(edges, edges[1:], strict=False)
        ]
        assert targets[row].tolist() == pytest.approx(expected, abs=1e-6)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(len(values)))
    assert (targets >= 0).all()


def test_hl_gauss_targets_are_centred_with_sigma_three_quarters_of_a_bin() -> None:
    bins = 8
    # Win chance 4.5 / 8 is the centre of bin 4.
    targets = hl_gauss_targets(torch.tensor([2 * 4.5 / 8 - 1]), bins)[0]
    assert int(targets.argmax()) == 4
    for offset in (1, 2, 3):
        assert targets[4 - offset].item() == pytest.approx(targets[4 + offset].item(), abs=1e-6)
    # The centre bin holds P(|z| < 0.5 / 0.75) of a standard normal.
    assert targets[4].item() == pytest.approx(2 * NormalDist().cdf(0.5 / 0.75) - 1, abs=1e-6)
    # Its mean is the value, less the tail cut above bin 7 (3.5 bins, 4.7 sigma, away).
    centres = value_bin_values(bins)
    assert float((targets * centres).sum()) == pytest.approx(2 * 4.5 / 8 - 1, abs=1e-5)
    # A certain win keeps the upper half of its Gaussian, renormalised onto [0, 1].
    top = hl_gauss_targets(torch.tensor([1.0]), bins)[0]
    assert top[-1].item() == pytest.approx(2 * (0.5 - NormalDist().cdf(-1 / 0.75)), abs=1e-6)


def test_expected_values_read_a_hand_set_distribution_on_the_scalar_scale() -> None:
    torch.manual_seed(1)
    scalar = WombatNetwork(BASE).eval()
    binned = WombatNetwork(dataclasses.replace(BASE, value_bins=4)).eval()
    shapes = {name: tensor.shape for name, tensor in binned.state_dict().items()}
    shared = {k: v for k, v in scalar.state_dict().items() if shapes[k] == v.shape}
    missing, unexpected = binned.load_state_dict(shared, strict=False)
    assert not unexpected
    assert set(missing) == NEW | {"value_head.2.weight", "value_head.2.bias"}
    # Win chance 0.1 / 8 + 0.2 * 3 / 8 + 0.3 * 5 / 8 + 0.4 * 7 / 8 = 0.625, value 0.25.
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert value_bin_values(4).tolist() == [-0.75, -0.25, 0.25, 0.75]
    assert float(expected_value(probabilities.log(), value_bin_values(4))) == pytest.approx(0.25)
    with torch.no_grad():
        for layer in (binned.value_head[2], binned.move_value_bin_origin):
            layer.weight.zero_()
            layer.bias.copy_(probabilities.log())
        binned.move_value_bin_destination.weight.zero_()
        binned.move_value_bin_tilt.zero_()
        # The scalar heads made constant 0.25 the same way.
        for layer in (scalar.value_head[2], scalar.square_move_value_head[0]):
            layer.weight.zero_()
            layer.bias.fill_(math.atanh(0.25))
        scalar.move_value_query.weight.zero_()
        scalar.move_value_query.bias.zero_()
    pieces, state = _positions(3, seed=2)
    with torch.no_grad():
        ours = binned(pieces, state, 2, include_move_values=True)
        theirs = scalar(pieces, state, 2, include_move_values=True)
    assert torch.equal(ours[0], theirs[0])
    for mine, reference in zip(ours[1:], theirs[1:], strict=True):
        assert mine.dtype == torch.float32
        assert torch.allclose(mine, torch.full_like(mine, 0.25), atol=1e-6)
        assert torch.allclose(mine, reference, atol=1e-6)
    # All mass in the top bin reads as that bin's value.
    with torch.no_grad():
        binned.value_head[2].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 60.0]))
        value = binned(pieces, state, 2)[1]
    assert torch.allclose(value, torch.full_like(value, 0.75))


def test_move_logits_are_origin_plus_destination_plus_tilted_score() -> None:
    torch.manual_seed(3)
    model = WombatNetwork(WITH).eval()
    _live_consequences(model)
    pieces, state = _positions(2, seed=4)
    with torch.no_grad():
        _, hidden, consequences = _state(model, pieces, state)
        policy, value_logits, bins = model.value_bin_heads(hidden, consequences=consequences)
        readouts = model.heads(hidden, include_move_values=True, consequences=consequences)
    assert value_logits.shape == (2, BINS)
    dense = bins.logits()
    assert dense.shape == (2, 4672, BINS)
    destinations = model.move_destinations
    for action in (0, 73 * 12 + 5, 73 * 63 + 72, 2345):
        origin, plane = divmod(action, 73)
        expected = (
            bins.origin[:, origin]
            + bins.destination[:, int(destinations[origin, plane])]
            + bins.scores.flatten(1)[:, action, None] * model.move_value_bin_tilt
        )
        assert torch.allclose(dense[:, action], expected, atol=1e-6)
    actions = torch.randint(0, 4672, (2, 30), generator=torch.Generator().manual_seed(5))
    rows = torch.arange(2).unsqueeze(1)
    assert torch.allclose(bins.logits(actions), dense[rows, actions], atol=1e-6)
    # The scores are the scalar move value readout before its tanh.
    tokens, _ = model._readout_tokens(hidden, 65, consequences)
    assert torch.equal(bins.scores, model._move_value_scores(tokens[:, :64], consequences))
    assert torch.equal(readouts[0], policy)
    assert torch.allclose(readouts[1], expected_value(value_logits, value_bin_values(BINS)))
    assert torch.allclose(readouts[2], expected_value(dense, value_bin_values(BINS)))
    assert readouts[2].abs().max() < 1


def _old_heads(
    model: WombatNetwork, hidden: torch.Tensor, consequences: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The scalar square readout written out term by term."""

    hidden = model.final_norm(hidden[:, : 65 + model.config.latent_tokens])
    value = model.value_head(hidden[:, 64]).squeeze(-1)
    squares = hidden[:, :64]
    policy = model.square_policy_head(squares)
    policy = policy + model._destination_scores(squares, model.policy_query, model.policy_key)
    assert consequences is not None
    policy = policy + model._consequence_scores(
        squares,
        consequences,
        model.policy_consequence_origin,
        model.policy_consequence_destination,
    )
    linear, tanh = model.square_move_value_head
    scores = linear(squares)
    scores = scores + model._destination_scores(
        squares, model.move_value_query, model.move_value_key
    )
    scores = scores + model._consequence_scores(
        squares, consequences, model.value_consequence_origin, model.value_consequence_destination
    )
    return policy.flatten(1), value, tanh(scores).flatten(1)


def test_scalar_heads_are_bit_identical_to_their_terms() -> None:
    torch.manual_seed(6)
    model = WombatNetwork(BASE).eval()
    _live_consequences(model)
    pieces, state = _positions(3, seed=7)
    with torch.no_grad():
        _, hidden, consequences = _state(model, pieces, state)
        outputs = model.heads(hidden, include_move_values=True, consequences=consequences)
        short = model.heads(hidden, consequences=consequences)
        expected = _old_heads(model, hidden, consequences)
    assert len(outputs) == 3 and len(short) == 2
    for tensor, reference in zip(outputs, expected, strict=True):
        assert torch.equal(tensor, reference)
    assert torch.equal(short[0], expected[0]) and torch.equal(short[1], expected[1])
    with pytest.raises(RuntimeError, match="scalar value heads"):
        model.value_bin_heads(hidden, consequences=consequences)


def test_the_binned_loss_is_hl_gauss_cross_entropy_on_the_same_masks(game_corpus: Path) -> None:
    batch = complete_targets(GameSplit(game_corpus / "train").batch(np.arange(4)))
    assert batch["move_value_mask"].any() and batch["value_mask"].all()
    torch.manual_seed(8)
    model = WombatNetwork(WITH)
    _live_consequences(model)
    memory, hidden, consequences = _state(model, batch["pieces"], batch["state"])
    policy, value_logits, bins = model.value_bin_heads(hidden, consequences=consequences)
    weights = LossWeights(move_value=0.5, value=0.25)
    loss, correct = graded_loss(policy, value_logits, bins, batch, weights)
    policy_only, same = graded_loss(policy, value_logits, bins, batch, POLICY_ONLY)
    assert torch.equal(correct, same)
    # Dense references: every action's logits, then the labelled ones' cross-entropy.
    log_probs = torch.log_softmax(bins.logits().float(), dim=-1)
    move_targets = hl_gauss_targets(batch["move_value_target"], BINS)
    cross = -(move_targets * log_probs).sum(dim=-1)
    mask = batch["move_value_mask"].float()
    moves = ((cross * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()
    value_targets = hl_gauss_targets(batch["value"], BINS)
    values = -(value_targets * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
    expected = policy_only + 0.5 * moves + 0.25 * values
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    stepped, _ = step_loss(model, hidden, memory.shape[1], batch, weights, consequences)
    assert stepped.item() == pytest.approx(loss.item(), rel=1e-6)
    # Unlabelled positions and moves add nothing, as with the squared errors.
    unlabelled = dict(batch)
    unlabelled["move_value_mask"] = torch.zeros_like(batch["move_value_mask"])
    unlabelled["value_mask"] = torch.zeros_like(batch["value_mask"])
    empty, _ = graded_loss(policy, value_logits, bins, unlabelled, weights)
    assert empty.item() == policy_only.item()


def test_gradients_reach_the_bin_heads(game_corpus: Path) -> None:
    batch = complete_targets(GameSplit(game_corpus / "train").batch(np.arange(4)))
    torch.manual_seed(9)
    model = WombatNetwork(WITH).train()
    _live_consequences(model)
    memory, hidden, consequences = _state(model, batch["pieces"], batch["state"])
    weights = LossWeights(move_value=0.5, value=0.25)
    loss, _ = step_loss(model, hidden, memory.shape[1], batch, weights, consequences)
    loss.backward()
    reached = [
        "value_head.0.weight",
        "value_head.2.weight",
        "value_head.2.bias",
        *NEW,
        # The scalar score still feeds each move's bins through the tilt.
        "square_move_value_head.0.weight",
        "move_value_query.weight",
        "value_consequence_origin.weight",
    ]
    parameters = dict(model.named_parameters())
    for name in reached:
        gradient = parameters[name].grad
        assert gradient is not None and float(gradient.abs().sum()) > 0, name


def test_bin_counts_add_the_documented_parameters() -> None:
    plain = count_parameters(WombatNetwork(BASE))
    width = BASE.d_model
    for bins in (2, BINS, 64):
        model = WombatNetwork(dataclasses.replace(BASE, value_bins=bins))
        # The value head's last layer widens; origin (with bias), destination and the tilt.
        added = (width + 1) * (bins - 1) + (2 * width + 2) * bins
        assert count_parameters(model) - plain == added


def test_value_bin_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "value-bins"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "2"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--move-consequences", "--relative-attention"]
    arguments += ["--value-bins", str(BINS)]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["value_bins"] == BINS
    assert saved["model"]["value_head.2.weight"].shape == (BINS, 24)
    assert saved["model"]["move_value_bin_origin.weight"].shape == (BINS, 24)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["model_config"]["value_bins"] == BINS
    model = load_checkpoint(checkpoint, torch.device("cpu"), select="value")
    assert model.config == NetworkConfig(**saved["model_config"])
    snapshot = load_checkpoint(output / "snapshot-step-000004.pt", torch.device("cpu"))
    assert snapshot.config == model.config
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    report = explain_move(model, board, chess.Move.from_uci("f1b5"), steps=2, depths=(0, 2))
    assert report["move"]["san"] == "Bb5"
    split = GameSplit(game_corpus / "validation")
    indices = np.arange(min(6, len(split)))
    cpu = torch.device("cpu")
    for select in ("policy", "value", "mix"):
        predicted = predict_depths(
            model, split, indices, (0, 1, 3), batch_size=4, device=cpu, select=select
        )
        assert all(len(actions) == len(indices) for actions in predicted.values())
    pieces, state = _positions(3, seed=10)
    with torch.no_grad():
        _, value, move_values = model(pieces, state, 2, include_move_values=True)
    assert value.abs().max() < 1 and move_values.abs().max() < 1
    capsys.readouterr()
    explain = ["explain-move", "--checkpoint", str(checkpoint), "--fen", board.fen()]
    assert main([*explain, "--move", "f1b5", "--steps", "1", "--depths", "0,1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["move"]["san"] == "Bb5"


def test_checkpoints_without_the_option_load_unchanged(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = WombatNetwork(BASE).eval()
    recorded = dataclasses.asdict(BASE)
    del recorded["value_bins"]
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "model_config": recorded}, path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded.config == BASE
    pieces, state = _positions(2, seed=12)
    with torch.no_grad():
        for mine, reference in zip(
            loaded(pieces, state, 2, include_move_values=True),
            model(pieces, state, 2, include_move_values=True),
            strict=True,
        ):
            assert torch.equal(mine, reference)
