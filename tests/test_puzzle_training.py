from __future__ import annotations

import dataclasses
from io import StringIO
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.cli import build_parser, main
from wombat.chess.moves import legal_action_mask, move_to_action
from wombat.data.game_corpus import IGNORE, GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit
from wombat.eval.puzzle_corpus_eval import (
    candidate_quality,
    compare_correctness,
    evaluate_corpus_curve,
    evaluate_puzzle_depth,
    full_line_solved,
    load_checkpoint,
    paired_difference,
    predict_depths,
    remaining_groups,
    score_predictions,
    wilson_interval,
)
from wombat.interpretability.explain import explain_move
from wombat.losses import (
    LOOKAHEAD_SHARES,
    POLICY_ONLY,
    LossWeights,
    complete_targets,
    graded_loss,
    lookahead_loss,
    reply_pair_loss,
    step_loss,
    unrolled_loss,
)
from wombat.model import NetworkConfig, WombatNetwork
from wombat.model.network import RecurrentBlock, noisy_squares
from wombat import puzzle_training
from wombat.puzzle_training import PuzzleTrainingConfig, train_puzzles
from wombat.visualizer.models import load_network

TINY = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=1,
    recurrent_blocks=1,
    latent_tokens=4,
    policy_readout="square",
    normalize_state=True,
    max_residual_update=None,
)


@pytest.fixture(autouse=True)
def _normal_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)


def _config(corpus: Path, output: Path, **overrides: object) -> PuzzleTrainingConfig:
    settings: dict[str, object] = {
        "corpus": str(corpus),
        "output": str(output),
        "steps": 4,
        "batch_size": 2,
        "warmup_steps": 1,
        "max_depth": 3,
        "max_gradient_steps": 2,
        "eval_every": 2,
        "eval_positions": 4,
        "eval_depths": (0, 1, 3),
        "device": "cpu",
    }
    settings.update(overrides)
    return PuzzleTrainingConfig(**settings)  # type: ignore[arg-type]


def _metrics(output: Path) -> list[dict[str, Any]]:
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_progressive_segments_stay_inside_the_trained_horizon() -> None:
    config = _config(Path("."), Path("."), max_depth=16, max_gradient_steps=8)
    randomizer = random.Random(0)
    for _ in range(500):
        prefix, tail = puzzle_training._segment(config, randomizer)
        assert 0 <= prefix <= 15
        assert 1 <= tail <= min(8, 16 - prefix)
    fixed = _config(Path("."), Path("."), fixed_depth=1)
    assert puzzle_training._segment(fixed, randomizer) == (0, 1)


def test_learning_rate_warms_up_then_decays_to_a_tenth() -> None:
    config = _config(Path("."), Path("."), steps=100, warmup_steps=10, learning_rate=1e-3)
    assert puzzle_training._learning_rate(config, 0) == pytest.approx(1e-4)
    assert puzzle_training._learning_rate(config, 9) == pytest.approx(1e-3)
    assert puzzle_training._learning_rate(config, 100) == pytest.approx(1e-4)


def test_training_logs_depth_curves_and_resumes(puzzle_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    train_puzzles(_config(puzzle_corpus, output), TINY)
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4]
    for row in rows:
        assert np.isfinite(row["train_loss"])
        assert set(row["validation"]) == {"t0", "t1", "t3"}
    train_puzzles(_config(puzzle_corpus, output, steps=6), TINY, resume=True)
    assert [row["step"] for row in _metrics(output)] == [2, 4, 6]


def test_progressive_share_mixes_full_horizons_with_continued_segments() -> None:
    config = _config(
        Path("."),
        Path("."),
        fixed_depth=8,
        progressive_share=0.5,
        min_gradient_steps=4,
        max_gradient_steps=8,
        max_depth=12,
    )
    randomizer = random.Random(0)
    segments = [puzzle_training._segment(config, randomizer) for _ in range(2000)]
    full = sum(segment == (0, 8) for segment in segments)
    assert 0.4 < full / len(segments) < 0.7
    for prefix, tail in segments:
        assert 0 <= prefix <= 8
        assert 4 <= tail <= 8
        assert prefix + tail <= 12
    assert {prefix for prefix, _ in segments} == set(range(9))
    assert puzzle_training._segment(_config(Path("."), Path("."), fixed_depth=8), randomizer) == (
        0,
        8,
    )


def test_puzzle_max_tail_runs_the_rest_of_the_horizon_without_gradients() -> None:
    config = _config(
        Path("."),
        Path("."),
        task="games",
        fixed_depth=8,
        puzzle_corpus="p",
        puzzle_share=1.0,
        puzzle_remaining=(1, 2, 3, 4),
        puzzle_max_tail=8,
    )
    randomizer = random.Random(0)
    draws = {puzzle_training._draw(config, randomizer) for _ in range(400)}
    assert draws == {(1, 0, 4), (2, 0, 8), (3, 4, 8), (4, 8, 8)}
    puzzle_training._check_puzzle_mix(config)
    with pytest.raises(ValueError):
        puzzle_training._check_puzzle_mix(dataclasses.replace(config, puzzle_max_tail=None))


def test_random_prefix_moves_the_graded_step_but_not_the_tail() -> None:
    config = _config(
        Path("."),
        Path("."),
        task="games",
        fixed_depth=8,
        puzzle_corpus="p",
        puzzle_share=0.25,
        random_prefix=4,
    )
    randomizer = random.Random(0)
    draws = [puzzle_training._draw(config, randomizer) for _ in range(4000)]
    assert {prefix for _, prefix, _ in draws} == set(range(5))
    for remaining, prefix, tail in draws:
        assert tail == (8 if remaining is None else 4 * remaining)
    games = [prefix for remaining, prefix, _ in draws if remaining is None]
    assert 1.8 < sum(games) / len(games) < 2.2


def test_random_prefix_starts_at_its_training_step() -> None:
    config = _config(
        Path("."), Path("."), task="games", fixed_depth=8, random_prefix=4, random_prefix_from=100
    )
    randomizer = random.Random(0)
    assert {puzzle_training._draw(config, randomizer, 99) for _ in range(200)} == {(None, 0, 8)}
    later = {puzzle_training._draw(config, randomizer, 100)[1] for _ in range(400)}
    assert later == set(range(5))
    parsed = build_parser().parse_args(["train-games", "--output", "run"])
    assert parsed.random_prefix_from == 0


def test_random_prefix_trains_with_puzzles(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "random-prefix"
    config = _config(
        game_corpus,
        output,
        task="games",
        fixed_depth=2,
        random_prefix=2,
        puzzle_corpus=str(puzzle_corpus),
        puzzle_share=0.5,
        puzzle_steps_per_move=1,
        steps=6,
    )
    train_puzzles(config, TINY)
    rows = _metrics(output)
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    assert any(row["mean_prefix"] > 0 for row in rows)


def test_puzzle_batches_run_steps_per_move_times_remaining() -> None:
    config = _config(
        Path("."), Path("."), task="games", fixed_depth=8, puzzle_corpus="p", puzzle_share=0.25
    )
    randomizer = random.Random(0)
    draws = [puzzle_training._draw(config, randomizer) for _ in range(4000)]
    puzzles = [draw for draw in draws if draw[0] is not None]
    assert 0.22 < len(puzzles) / len(draws) < 0.28
    assert all(draw == (None, 0, 8) for draw in draws if draw[0] is None)
    assert set(puzzles) == {(1, 0, 4), (2, 0, 8)}
    ones = sum(draw[0] == 1 for draw in puzzles)
    assert 0.45 < ones / len(puzzles) < 0.55
    # Without puzzles the draw consumes the randomizer exactly as the game segments did.
    plain = _config(
        Path("."),
        Path("."),
        task="games",
        fixed_depth=8,
        progressive_share=0.5,
        max_depth=12,
        max_gradient_steps=8,
    )
    first, second = random.Random(3), random.Random(3)
    for _ in range(50):
        segment = puzzle_training._segment(plain, second)
        assert puzzle_training._draw(plain, first) == (None, *segment)


def _lognormal(**overrides: object) -> PuzzleTrainingConfig:
    settings: dict[str, object] = {
        "task": "games",
        "depth_distribution": "lognormal",
        "max_gradient_steps": 8,
        **overrides,
    }
    return _config(Path("."), Path("."), **settings)


def test_lognormal_depths_have_a_heavy_tail_and_an_eight_step_gradient_cap() -> None:
    config = _lognormal()
    randomizer = random.Random(0)
    draws = [puzzle_training._draw(config, randomizer) for _ in range(20_000)]
    depths = np.array([prefix + tail for _, prefix, tail in draws])
    for remaining, prefix, tail in draws:
        assert remaining is None
        assert tail == min(prefix + tail, 8)
        assert 1 <= prefix + tail <= 48
    # Poisson(exp(tau)) + 1 has mean depth_mean + 1 before clipping at depth_max.
    assert 16.5 < depths.mean() < 17.3
    assert 14 <= np.median(depths) <= 16
    assert 0.12 < np.mean(depths <= 8) < 0.19
    assert 0.005 < np.mean(depths == 48) < 0.02
    capped = [puzzle_training._draw(_lognormal(depth_max=20), randomizer) for _ in range(2_000)]
    assert max(prefix + tail for _, prefix, tail in capped) == 20
    short = [puzzle_training._draw(_lognormal(depth_mean=2.0), randomizer) for _ in range(2_000)]
    assert {tail for _, _, tail in short} >= {1, 2, 3, 4}
    assert all(prefix == 0 for _, prefix, tail in short if tail < 8)


def test_lognormal_puzzles_keep_their_tails_and_add_the_drawn_prefix() -> None:
    config = _lognormal(
        puzzle_corpus="p", puzzle_share=0.5, puzzle_remaining=(1, 2, 3, 4), puzzle_max_tail=8
    )
    puzzle_training._check_puzzle_mix(config)
    randomizer = random.Random(1)
    draws = [puzzle_training._draw(config, randomizer) for _ in range(8_000)]
    extras: dict[str, list[int]] = {"games": [], "puzzles": []}
    for remaining, prefix, tail in draws:
        if remaining is None:
            extras["games"].append(prefix)
            continue
        horizon = 4 * remaining
        assert tail == min(horizon, 8)
        assert prefix >= horizon - tail
        extras["puzzles"].append(prefix - (horizon - tail))
    # Both kinds of batch take the same no-gradient steps from the same draw.
    means = {kind: float(np.mean(values)) for kind, values in extras.items()}
    assert 8.5 < means["games"] < 9.9
    assert abs(means["games"] - means["puzzles"]) < 0.8


def test_fixed_depth_draws_ignore_the_lognormal_settings() -> None:
    config = _config(
        Path("."),
        Path("."),
        task="games",
        fixed_depth=8,
        random_prefix=8,
        puzzle_corpus="p",
        puzzle_share=0.25,
        puzzle_remaining=(1, 2, 3, 4),
        puzzle_max_tail=8,
    )
    assert (config.depth_distribution, config.initial_noise) == ("fixed", 0.0)
    other = dataclasses.replace(config, depth_mean=3.0, depth_max=5)
    first, second = random.Random(4), random.Random(4)
    for _ in range(500):
        assert puzzle_training._draw(config, first) == puzzle_training._draw(other, second)
    assert first.getstate() == second.getstate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"fixed_depth": 8},
        {"random_prefix": 4},
        {"progressive_share": 0.5},
        {"supervise_every_step": True},
        {"depth_mean": 0.0},
        {"depth_max": 0},
        {"depth_distribution": "uniform"},
        {"initial_noise": -0.1},
    ],
)
def test_lognormal_depth_is_checked_before_training(overrides: dict[str, object]) -> None:
    puzzle_training._check_depth_distribution(_lognormal())
    with pytest.raises(ValueError):
        puzzle_training._check_depth_distribution(_lognormal(**overrides))


def test_initial_noise_perturbs_only_training_square_tokens(game_corpus: Path) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(TINY)
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    device = torch.device("cpu")
    clean, _ = puzzle_training._loss(model, batch, 1, 2, device)
    state = torch.get_rng_state()
    unperturbed, _ = puzzle_training._loss(model, batch, 1, 2, device, initial_noise=0.0)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(clean, unperturbed)
    noisy, _ = puzzle_training._loss(model, batch, 1, 2, device, initial_noise=0.5)
    assert not torch.equal(clean, noisy)
    model.eval()
    evaluated, _ = puzzle_training._loss(model, batch, 1, 2, device, initial_noise=0.5)
    assert torch.equal(clean, evaluated)
    hidden = torch.zeros(3, 70, 8)
    perturbed = noisy_squares(hidden, 1.0)
    assert (perturbed[:, :64] != 0).all()
    assert torch.equal(perturbed[:, 64:], hidden[:, 64:])


def test_train_games_cli_runs_lognormal_depths_with_initial_noise_and_resumes(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "lognormal"
    arguments = [
        "train-games",
        "--corpus",
        str(game_corpus),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--steps",
        "4",
        "--batch-size",
        "2",
        "--warmup-steps",
        "1",
        "--max-gradient-steps",
        "2",
        "--depth-distribution",
        "lognormal",
        "--depth-mean",
        "3",
        "--depth-max",
        "6",
        "--initial-noise",
        "0.1",
        "--no-supervise-every-step",
        "--eval-every",
        "2",
        "--eval-depths",
        "0,1,3",
        "--d-model",
        "24",
        "--heads",
        "6",
        "--mlp-width",
        "48",
        "--latent-tokens",
        "4",
        "--recurrent-blocks",
        "1",
        "--puzzle-corpus",
        str(puzzle_corpus),
        "--puzzle-share",
        "0.5",
        "--puzzle-steps-per-move",
        "1",
    ]
    assert main(arguments) == 0
    training = json.loads((output / "config.json").read_text(encoding="utf-8"))["training_config"]
    assert (training["depth_distribution"], training["depth_mean"]) == ("lognormal", 3.0)
    assert (training["depth_max"], training["initial_noise"]) == (6, 0.1)
    assert training["fixed_depth"] is None
    arguments[arguments.index("--steps") + 1] = "8"
    assert main([*arguments, "--resume"]) == 0
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4, 6, 8]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    assert all(1 <= row["mean_tail"] <= 2 for row in rows)
    assert any(row["mean_prefix"] > 0 for row in rows)
    args = build_parser().parse_args(["train-games", "--output", "run"])
    assert (args.depth_distribution, args.depth_mean, args.depth_max) == ("fixed", 16.0, 48)
    assert args.initial_noise == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"puzzle_share": 0.5},
        {"puzzle_corpus": "p", "puzzle_share": 0.5, "task": "puzzles"},
        {"puzzle_corpus": "p", "puzzle_share": 1.5},
        {"puzzle_corpus": "p", "puzzle_share": 0.5, "supervise_every_step": True},
        {"puzzle_corpus": "p", "puzzle_share": 0.5, "puzzle_remaining": (0, 1)},
        {"puzzle_corpus": "p", "puzzle_share": 0.5, "puzzle_steps_per_move": 2},
    ],
)
def test_puzzle_mix_is_checked_before_training(overrides: dict[str, object]) -> None:
    settings: dict[str, object] = {"task": "games", "fixed_depth": 2, **overrides}
    with pytest.raises(ValueError):
        puzzle_training._check_puzzle_mix(_config(Path("."), Path("."), **settings))
    fits = _config(Path("."), Path("."), task="games", fixed_depth=8, puzzle_corpus="p")
    puzzle_training._check_puzzle_mix(dataclasses.replace(fits, puzzle_share=0.25))


def test_mixed_training_grades_puzzles_at_their_own_horizon(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bool, int, int, int]] = []
    loss = puzzle_training._loss

    def recorded(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch, prefix, tail = args[1], args[2], args[3]
        remaining = int(batch["remaining"][0]) if "remaining" in batch else 0
        if remaining:
            assert (batch["remaining"] == remaining).all()
        calls.append(("remaining" in batch, remaining, prefix, tail))
        return loss(*args, **kwargs)

    monkeypatch.setattr(puzzle_training, "_loss", recorded)
    config = _config(
        game_corpus,
        tmp_path / "mixed",
        task="games",
        fixed_depth=2,
        puzzle_corpus=str(puzzle_corpus),
        puzzle_share=0.5,
        puzzle_steps_per_move=1,
        value_weight=0.25,
        move_value_weight=0.5,
        steps=12,
        eval_every=6,
        eval_positions=3,
    )
    train_puzzles(config, TINY)
    assert len(calls) == 12
    assert {remaining for _, remaining, _, _ in calls} == {0, 1, 2}
    for puzzle, remaining, prefix, tail in calls:
        assert prefix == 0
        assert tail == (remaining if puzzle else 2)
    rows = _metrics(tmp_path / "mixed")
    shares = [row["puzzle_batch_share"] for row in rows]
    assert sum(shares) * 6 == pytest.approx(sum(puzzle for puzzle, _, _, _ in calls))


def test_train_games_cli_mixes_puzzles_and_resumes(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "v10"
    arguments = [
        "train-games",
        "--corpus",
        str(game_corpus),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--steps",
        "4",
        "--batch-size",
        "2",
        "--warmup-steps",
        "1",
        "--fixed-depth",
        "2",
        "--no-supervise-every-step",
        "--eval-every",
        "2",
        "--eval-depths",
        "0,1,3",
        "--snapshot-every",
        "2",
        "--d-model",
        "24",
        "--heads",
        "6",
        "--mlp-width",
        "48",
        "--latent-tokens",
        "4",
        "--recurrent-blocks",
        "1",
        "--ema-decay",
        "0.9",
        "--puzzle-corpus",
        str(puzzle_corpus),
        "--puzzle-share",
        "0.5",
        "--puzzle-steps-per-move",
        "1",
    ]
    assert main(arguments) == 0
    run = json.loads((output / "config.json").read_text(encoding="utf-8"))
    training = run["training_config"]
    assert (training["puzzle_share"], training["puzzle_remaining"]) == (0.5, [1, 2])
    assert training["puzzle_steps_per_move"] == 1
    assert training["supervise_every_step"] is False
    validation = PuzzleSplit(puzzle_corpus / "validation")
    groups = {
        f"r{name}"
        for name, members in remaining_groups(validation.remaining).items()
        if len(members)
    }
    sizes = run["puzzle_validation_positions"]
    assert set(sizes) == {"r1", "r2", "r3", "r4", "r5+"}
    assert {name for name, size in sizes.items() if size} == groups
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4]
    for row in rows:
        assert np.isfinite(row["train_loss"])
        assert 0 <= row["puzzle_batch_share"] <= 1
        assert set(row["puzzle_validation"]) == groups
        for curve in row["puzzle_validation"].values():
            assert set(curve) == {"t0", "t1", "t3"}
            assert all(0 <= value <= 1 for value in curve.values())
    assert (output / "snapshot-step-000004.pt").exists()
    arguments[arguments.index("--steps") + 1] = "6"
    assert main([*arguments, "--resume"]) == 0
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4, 6]
    assert set(rows[-1]["puzzle_validation"]) == groups


def test_train_games_cli_leaves_puzzles_off_unless_asked() -> None:
    args = build_parser().parse_args(["train-games", "--output", "run"])
    assert (args.puzzle_corpus, args.puzzle_share) == (None, 0.0)
    assert (args.puzzle_remaining, args.puzzle_steps_per_move) == ("1,2", 4)
    puzzles = build_parser().parse_args(["train-puzzles", "--output", "run"])
    assert (puzzles.puzzle_corpus, puzzles.puzzle_share) == (None, 0.0)


def test_readout_and_neighbourhood_default_to_the_square_readout_and_king_steps() -> None:
    for command in ("train-games", "train-puzzles"):
        args = build_parser().parse_args([command, "--output", "run"])
        assert (args.move_readout, args.local_neighbourhood) == ("square", "king")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-games", "--output", "run", "--move-readout", "to"])


def test_from_to_king_knight_checkpoint_loads_everywhere(
    game_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "from-to"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "1"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--local-neighbourhood", "king_knight", "--move-readout", "from_to"]
    assert main(arguments) == 0
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_config"]
    assert (saved["move_readout"], saved["local_neighbourhood"]) == ("from_to", "king_knight")
    # The UCI engine, explain-move and the evaluations load through load_checkpoint.
    strict = load_checkpoint(checkpoint, torch.device("cpu"))
    served = load_network(checkpoint)
    assert strict.config == served.config == NetworkConfig(**saved)
    block = strict.recurrent_core[0]
    assert isinstance(block, RecurrentBlock)
    assert int((block.king_mask[0] == 0).sum()) == 6
    snapshot = load_checkpoint(output / "snapshot-step-000002.pt", torch.device("cpu"))
    assert snapshot.config == strict.config
    board = chess.Board()
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces).long()[None]
    state = torch.from_numpy(state_vector(encoded).astype(np.float32))[None]
    with torch.no_grad():
        expected = strict(pieces, state, 3, include_move_values=True)
        outputs = served(pieces, state, 3, include_move_values=True)
    assert isinstance(outputs, tuple) and isinstance(expected, tuple)
    assert len(outputs) == len(expected) == 3
    for tensor, reference in zip(outputs, expected):
        assert torch.equal(tensor, reference)
    monkeypatch.setattr("sys.stdin", StringIO("uci\nposition startpos\ngo nodes 1\nquit\n"))
    capsys.readouterr()
    assert main(["uci", "--checkpoint", str(checkpoint)]) == 0
    best = [line.split()[1] for line in capsys.readouterr().out.splitlines() if "bestmove" in line]
    assert len(best) == 1 and chess.Move.from_uci(best[0]) in board.legal_moves
    explain = ["explain-move", "--checkpoint", str(checkpoint), "--fen", board.fen()]
    assert main([*explain, "--move", "e2e4", "--steps", "1", "--depths", "0,1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["move"]["san"] == "e4"


@pytest.mark.parametrize("neighbourhood", ["move_graph", "move_lines"])
def test_move_graph_trains_resumes_and_loads_everywhere(
    neighbourhood: str, game_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / neighbourhood
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "1"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--local-neighbourhood", neighbourhood, "--move-readout", "from_to"]
    arguments += ["--board-features", "--board-features-long-range"]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["local_neighbourhood"] == neighbourhood
    # The move graph's tables stay out of the state dict; move_lines adds one blocker bias.
    assert not any("move_graph" in key for key in saved["model"])
    blocker = saved["model"].get("recurrent_core.0.blocker_bias")
    assert (blocker is not None) == (neighbourhood == "move_lines")
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
    predicted = predict_depths(model, split, indices, (0, 1, 3), batch_size=4, device=cpu)
    assert all(len(actions) == len(indices) for actions in predicted.values())


def test_progressive_share_needs_room_for_the_shortest_tail(
    game_corpus: Path, tmp_path: Path
) -> None:
    config = _config(
        game_corpus,
        tmp_path / "run",
        task="games",
        fixed_depth=3,
        progressive_share=0.5,
        min_gradient_steps=2,
        max_depth=4,
    )
    with pytest.raises(ValueError):
        train_puzzles(config, TINY)


def test_progressive_share_trains_with_final_only_grading(
    game_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "progressive"
    config = _config(
        game_corpus,
        output,
        task="games",
        fixed_depth=2,
        progressive_share=0.5,
        min_gradient_steps=1,
        max_gradient_steps=2,
        max_depth=4,
        steps=6,
    )
    train_puzzles(config, TINY)
    rows = _metrics(output)
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    assert any(row["mean_prefix"] > 0 for row in rows)


def test_stop_after_keeps_the_full_learning_rate_schedule(
    puzzle_corpus: Path, tmp_path: Path
) -> None:
    config = _config(puzzle_corpus, tmp_path / "run", steps=100, stop_after=4, warmup_steps=1)
    train_puzzles(config, TINY)
    rows = _metrics(tmp_path / "run")
    assert [row["step"] for row in rows] == [2, 4]
    assert rows[-1]["learning_rate"] == pytest.approx(puzzle_training._learning_rate(config, 3))
    assert rows[-1]["learning_rate"] > 0.9 * config.learning_rate


def test_learning_rate_without_cooldown_is_warmup_then_cosine() -> None:
    config = _config(Path("."), Path("."), steps=100, warmup_steps=10, stop_after=60)
    for step in range(120):
        if step < 10:
            expected = config.learning_rate * (step + 1) / 10
        else:
            progress = min(1.0, (step - 10) / 90)
            expected = config.learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        assert puzzle_training._learning_rate(config, step) == expected


def test_cooldown_decays_from_the_scheduled_rate_to_zero_at_the_last_step() -> None:
    plain = _config(Path("."), Path("."), steps=100, warmup_steps=10, stop_after=60)
    config = dataclasses.replace(plain, cooldown_steps=20)
    rates = [puzzle_training._learning_rate(config, step) for step in range(70)]
    assert rates[:41] == [puzzle_training._learning_rate(plain, step) for step in range(41)]
    peak = puzzle_training._learning_rate(plain, 40)
    for step in range(40, 61):
        assert rates[step] == pytest.approx(peak * (1 - math.sqrt((step - 40) / 20)))
    assert rates[50] == pytest.approx(peak * (1 - math.sqrt(0.5)))
    assert rates[59] > 0
    assert rates[60:] == [0.0] * 10
    assert all(later < earlier for earlier, later in zip(rates[40:60], rates[41:61]))
    # Without stop_after the cooldown ends at ``steps``.
    full = dataclasses.replace(plain, stop_after=None, cooldown_steps=10)
    assert puzzle_training._learning_rate(full, 89) == puzzle_training._learning_rate(plain, 89)
    assert puzzle_training._learning_rate(full, 100) == 0.0


def test_cooldown_must_fit_before_the_last_step(puzzle_corpus: Path, tmp_path: Path) -> None:
    config = _config(puzzle_corpus, tmp_path / "run", stop_after=2, cooldown_steps=3)
    with pytest.raises(ValueError):
        train_puzzles(config, TINY)


def test_resume_with_a_cooldown_decays_the_logged_learning_rate(
    puzzle_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    first = _config(puzzle_corpus, output, steps=100, stop_after=4, warmup_steps=1)
    train_puzzles(first, TINY)
    resumed = dataclasses.replace(first, stop_after=8, cooldown_steps=4)
    train_puzzles(resumed, TINY, resume=True)
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4, 6, 8]
    rates = [row["learning_rate"] for row in rows]
    assert rates[1] == pytest.approx(puzzle_training._learning_rate(first, 3))
    # The cooldown starts at step 4 of the resumed run's own schedule.
    peak = puzzle_training._learning_rate(first, 4)
    assert rates[2] == pytest.approx(peak * (1 - math.sqrt(1 / 4)))
    assert rates[3] == pytest.approx(peak * (1 - math.sqrt(3 / 4)))
    assert rates[1] > rates[2] > rates[3] > 0
    assert all(np.isfinite(row["train_loss"]) for row in rows)


def test_cooldown_is_off_unless_asked() -> None:
    for command in ("train-games", "train-puzzles"):
        args = build_parser().parse_args([command, "--output", "run"])
        assert args.cooldown_steps == 0
    args = build_parser().parse_args(
        ["train-games", "--output", "run", "--stop-after", "34000", "--cooldown-steps", "6000"]
    )
    assert (args.stop_after, args.cooldown_steps) == (34000, 6000)


def test_fixed_depth_control_trains(puzzle_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "control"
    train_puzzles(_config(puzzle_corpus, output, fixed_depth=1, steps=2), TINY)
    rows = _metrics(output)
    assert rows[-1]["mean_prefix"] == 0
    assert rows[-1]["mean_tail"] == 1


def test_curve_report_and_paired_comparison(puzzle_corpus: Path, tmp_path: Path) -> None:
    train_puzzles(_config(puzzle_corpus, tmp_path / "run", steps=2), TINY)
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    report = evaluate_corpus_curve(
        checkpoint,
        puzzle_corpus / "test",
        (0, 2, 4),
        output=tmp_path / "eval",
        device_name="cpu",
    )
    curve = report["curve"]
    assert isinstance(curve, list)
    assert [row["reasoning_steps"] for row in curve] == [0, 1, 2, 4]
    assert all("full_line_solved" in row for row in curve)
    assert curve[1]["versus_t1"]["difference"] == 0
    correctness = tmp_path / "eval" / "correctness.npz"
    assert compare_correctness(correctness, 2, correctness, 2)["difference"] == 0


def test_puzzle_depth_report_groups_positions_and_pairs_every_depth_with_t8(
    puzzle_corpus: Path, tmp_path: Path
) -> None:
    train_puzzles(_config(puzzle_corpus, tmp_path / "run", steps=2), TINY)
    output = tmp_path / "depth"
    arguments = [
        "evaluate-puzzle-depth",
        "--checkpoint",
        str(tmp_path / "run" / "checkpoint.pt"),
        "--output",
        str(output),
        "--split",
        str(puzzle_corpus / "test"),
        "--depths",
        "1,2,4,12",
        "--device",
        "cpu",
    ]
    assert main(arguments) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["recurrent"] is True
    assert report["depths"] == [1, 2, 4, 8, 12]
    split = PuzzleSplit(puzzle_corpus / "test")
    available = {name: len(members) for name, members in remaining_groups(split.remaining).items()}
    untrained = available["3"] + available["4"] + available["5+"]
    assert report["available"] == {**available, "3+": untrained}
    # Groups 3, 4 and 5+ are whole; larger groups are sampled down to the largest of them.
    assert report["per_group"] == max(available["3"], available["4"], available["5+"])
    groups = report["groups"]
    assert set(groups) == {name for name, size in report["available"].items() if size}
    assert groups["3"]["positions"] == groups["3+"]["positions"] == available["3"]
    for group in groups.values():
        assert group["positions"] <= report["per_group"]
        assert set(group["depths"]) == {"t1", "t2", "t4", "t8", "t12"}
        assert group["depths"]["t8"]["versus_t8"]["difference"] == 0
        for row in group["depths"].values():
            assert 0 <= row["accuracy"] <= 1
            assert row["versus_t8"]["positions"] == group["positions"]
        assert group["best_above_t8"]["reasoning_steps"] == 12
    with np.load(output / "correctness.npz") as stored:
        assert len(stored["indices"]) == report["positions"]
        assert np.array_equal(stored["remaining"], split.remaining[stored["indices"]])
        assert {key for key in stored.files if key.startswith("t")} == set(groups["1"]["depths"])
        pooled = stored["remaining"] >= 3
        expected = paired_difference(stored["t8"][pooled], stored["t12"][pooled])
    assert groups["3+"]["depths"]["t12"]["versus_t8"] == expected
    sampled = [
        evaluate_puzzle_depth(
            tmp_path / "run" / "checkpoint.pt",
            puzzle_corpus / "test",
            (4,),
            output=tmp_path / f"sampled{attempt}",
            per_group=2,
            device_name="cpu",
        )
        for attempt in range(2)
    ]
    assert sampled[0]["groups"] == sampled[1]["groups"]
    sampled_groups = cast(dict[str, Any], sampled[0]["groups"])
    sizes = {name: group["positions"] for name, group in sampled_groups.items()}
    assert sizes == {name: min(2, size) for name, size in report["available"].items() if size}


def test_puzzle_depth_reads_a_feedforward_model_once(puzzle_corpus: Path, tmp_path: Path) -> None:
    baseline = dataclasses.replace(
        TINY, prelude_blocks=2, recurrent_blocks=0, square_attention_bias=True, lookahead_heads=True
    )
    torch.manual_seed(0)
    snapshot = tmp_path / "snapshot-step-000001.pt"
    torch.save(
        {
            "model": WombatNetwork(baseline).state_dict(),
            "model_config": dataclasses.asdict(baseline),
            "step": 1,
        },
        snapshot,
    )
    report = evaluate_puzzle_depth(
        snapshot, puzzle_corpus / "test", (1, 32), output=tmp_path / "depth", device_name="cpu"
    )
    assert report["recurrent"] is False
    for group in cast(dict[str, Any], report["groups"]).values():
        for row in group["depths"].values():
            assert row["versus_t8"]["gained"] == row["versus_t8"]["lost"] == 0
    with np.load(tmp_path / "depth" / "correctness.npz") as stored:
        assert np.array_equal(stored["t1"], stored["t32"])


def test_any_checkmating_move_counts_as_solved() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3R1K1 w - - 0 1")
    target = move_to_action(board, chess.Move.from_uci("a1a8"))
    split = cast(
        PuzzleSplit,
        SimpleNamespace(
            action=np.array([target] * 3), fens=np.array([board.fen()] * 3), mate_cache={}
        ),
    )
    predicted = np.array(
        [
            target,
            move_to_action(board, chess.Move.from_uci("e1e8")),
            move_to_action(board, chess.Move.from_uci("g1f1")),
        ]
    )
    assert score_predictions(split, np.arange(3), predicted).tolist() == [True, True, False]


def test_full_line_needs_every_solver_move() -> None:
    split = cast(PuzzleSplit, SimpleNamespace(puzzle=np.array([0, 0, 1, 1, 2])))
    correct = np.array([True, True, True, False, True])
    assert full_line_solved(split, np.arange(5), correct) == pytest.approx(2 / 3)


def test_paired_difference_counts_discordant_positions() -> None:
    first = np.array([True, False, True, False])
    second = np.array([True, True, False, True])
    result = paired_difference(first, second)
    assert result["gained"] == 2
    assert result["lost"] == 1
    assert result["difference"] == pytest.approx(0.25)
    assert result["low"] < 0.25 < result["high"]
    low, high = wilson_interval(30, 100)
    assert low < 0.3 < high


def test_games_task_trains_on_graded_targets_and_writes_snapshots(
    game_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "games"
    config = _config(
        game_corpus,
        output,
        task="games",
        move_value_weight=0.5,
        value_weight=0.25,
        snapshot_every=2,
        eval_positions=3,
    )
    train_puzzles(config, TINY)
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4]
    for row in rows:
        assert np.isfinite(row["train_loss"])
        assert all(0 <= value <= 1 for value in row["validation"].values())
    snapshots = sorted(path.name for path in output.glob("snapshot-step-*.pt"))
    assert snapshots == ["snapshot-step-000002.pt", "snapshot-step-000004.pt"]
    snapshot = torch.load(output / snapshots[-1], map_location="cpu", weights_only=False)
    assert snapshot["step"] == 4
    WombatNetwork(NetworkConfig(**snapshot["model_config"])).load_state_dict(snapshot["model"])


def test_one_hot_graded_loss_is_cross_entropy_plus_weighted_value_errors() -> None:
    torch.manual_seed(0)
    policy = torch.randn(3, 4672)
    value = torch.tensor([0.2, -0.1, 0.0])
    move_values = torch.zeros(3, 4672)
    legal = torch.zeros(3, 4672, dtype=torch.bool)
    legal[:, :10] = True
    batch = complete_targets({"legal_mask": legal, "action": torch.tensor([1, 2, 3])})
    loss, correct = graded_loss(policy, value, move_values, batch, POLICY_ONLY)
    masked = policy.masked_fill(~legal, float("-inf"))
    assert loss.item() == pytest.approx(
        torch.nn.functional.cross_entropy(masked, batch["action"]).item(), rel=1e-5
    )
    assert 0 <= correct.item() <= 1
    batch["value"] = torch.tensor([1.0, 1.0, 1.0])
    batch["value_mask"] = torch.ones(3, dtype=torch.bool)
    weighted, _ = graded_loss(policy, value, move_values, batch, LossWeights(value=0.5))
    expected = loss + 0.5 * (value - 1).square().mean()
    assert weighted.item() == pytest.approx(expected.item(), rel=1e-5)


def test_value_errors_average_over_valued_positions_only() -> None:
    policy = torch.zeros(3, 5)
    value = torch.tensor([0.0, 0.5, -0.5])
    targets = complete_targets(
        {"legal_mask": torch.ones(3, 5, dtype=torch.bool), "action": torch.tensor([0, 1, 2])}
    )
    targets["value"] = torch.tensor([1.0, 1.0, 1.0])
    targets["value_mask"] = torch.tensor([True, False, True])
    plain, _ = graded_loss(policy, value, policy, targets, POLICY_ONLY)
    weighted, _ = graded_loss(policy, value, policy, targets, LossWeights(value=0.5))
    assert (weighted - plain).item() == pytest.approx(0.5 * (1.0 + 2.25) / 2)


def test_puzzle_batches_carry_no_value_move_value_or_lookahead_targets(
    puzzle_corpus: Path,
) -> None:
    batch = PuzzleSplit(puzzle_corpus / "train").batch(np.arange(4))
    targets = complete_targets(batch)
    assert not targets["value_mask"].any()
    assert not targets["move_value_mask"].any()
    for key in ("reply_from", "reply_to", "board2", "board4"):
        assert (targets[key] == IGNORE).all()
    torch.manual_seed(0)
    model = WombatNetwork(dataclasses.replace(TINY, lookahead_heads=True))
    cpu = torch.device("cpu")
    graded, _ = puzzle_training._loss(model, batch, 0, 2, cpu, LossWeights(0.5, 0.25, 0.25))
    policy_only, _ = puzzle_training._loss(model, batch, 0, 2, cpu, POLICY_ONLY)
    assert graded.item() == policy_only.item()
    graded.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("value_head", "square_move_value_head", "reply_", "ply")):
            assert parameter.grad is None or not parameter.grad.any(), name


def test_value_mask_leaves_game_batch_losses_unchanged(game_corpus: Path) -> None:
    batch = complete_targets(GameSplit(game_corpus / "train").batch(np.arange(4)))
    assert batch["value_mask"].all()
    torch.manual_seed(0)
    policy = torch.randn(4, 4672)
    value = torch.rand(4) * 2 - 1
    move_values = torch.rand(4, 4672) * 2 - 1
    weights = LossWeights(move_value=0.5, value=0.25)
    loss, correct = graded_loss(policy, value, move_values, batch, weights)
    # The loss as written before value masks existed.
    logits = policy.masked_fill(~batch["legal_mask"], float("-inf"))
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    expected = -(batch["policy_target"] * log_probs.clamp_min(-1e4)).sum(dim=1).mean()
    mask = batch["move_value_mask"].float()
    errors = (move_values - batch["move_value_target"]).square() * mask
    expected = expected + 0.5 * (errors.sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()
    expected = expected + 0.25 * (value - batch["value"]).square().mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6, abs=0)
    assert correct.item() == (logits.argmax(dim=1) == batch["action"]).float().mean().item()


def test_candidate_quality_scores_moves_by_win_chance_lost() -> None:
    split = cast(
        GameSplit,
        SimpleNamespace(
            candidate_offsets=np.array([0, 3, 6]),
            candidate_actions=np.array([10, 11, 12, 20, 21, 22]),
            candidate_win=np.array([0.60, 0.58, 0.30, 0.90, 0.50, 0.40]),
        ),
    )
    quality = candidate_quality(split, np.arange(2), np.array([11, 99]))
    assert quality["good"] == pytest.approx(0.5)
    assert quality["in_candidates"] == pytest.approx(0.5)
    # Position two's move is unscored, so its loss is bounded below by best minus worst.
    assert quality["mean_lost_lower_bound"] == pytest.approx((2 + 50) / 2)


def _opening_batch(size: int) -> dict[str, torch.Tensor]:
    board = chess.Board()
    encoded = encode_position(board)
    targets = complete_targets(
        {
            "legal_mask": torch.from_numpy(legal_action_mask(board)).repeat(size, 1),
            "action": torch.full((size,), move_to_action(board, chess.Move.from_uci("e2e4"))),
        }
    )
    targets["reply_from"] = torch.tensor([52, IGNORE])[:size]
    targets["board2"][0] = torch.from_numpy(encoded.pieces.astype(np.int64))
    return {
        "pieces": torch.from_numpy(encoded.pieces.astype(np.int64)).repeat(size, 1),
        "state": torch.from_numpy(state_vector(encoded).astype(np.float32)).repeat(size, 1),
        **targets,
    }


def test_every_step_supervision_weights_later_steps_more() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(dataclasses.replace(TINY, lookahead_heads=True))
    batch = _opening_batch(2)
    weights = LossWeights(move_value=0.5, lookahead=0.25)
    memory = model.encode_board(batch["pieces"], batch["state"])
    start = model.initial_hidden(memory)
    states = [start]
    for _ in range(3):
        states.append(model.recurrent_step(memory, states[-1]))
    graded = [step_loss(model, state, memory.shape[1], batch, weights)[0] for state in states]
    every, _ = unrolled_loss(model, memory, start, 3, batch, weights, every_step=True)
    expected = (graded[1] + 2 * graded[2] + 3 * graded[3]) / 6
    assert every.item() == pytest.approx(expected.item(), rel=1e-5)
    final, _ = unrolled_loss(model, memory, start, 3, batch, weights, every_step=False)
    assert final.item() == pytest.approx(graded[3].item(), rel=1e-5)
    initial, _ = unrolled_loss(model, memory, start, 0, batch, weights, every_step=True)
    assert initial.item() == pytest.approx(graded[0].item(), rel=1e-5)


V8 = dataclasses.replace(TINY, square_attention_bias=True, lookahead_heads=True, scratch_boards=2)


def _v8_gradients(
    batch: dict[str, torch.Tensor], prefix: int, *, recompute: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Loss, gradients and recurrent block runs of one every-step iteration of three steps."""

    torch.manual_seed(0)
    model = WombatNetwork(V8)
    with torch.no_grad():
        for block in model.recurrent_core:
            assert isinstance(block, RecurrentBlock)
            block.attention.out_proj.weight.normal_(std=0.3)
            block.mlp.output.weight.normal_(std=0.3)
            block.square_bias.normal_()
    runs: list[int] = []
    model.recurrent_core[0].register_forward_hook(lambda *_: runs.append(1))
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    loss, _ = puzzle_training._loss(
        model,
        batch,
        prefix,
        3,
        torch.device("cpu"),
        weights,
        every_step=True,
        recompute=recompute,
    )
    loss.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), gradients, len(runs)


@pytest.mark.parametrize("prefix", [0, 1])
def test_recomputed_steps_give_the_same_loss_and_gradients(prefix: int) -> None:
    batch = _opening_batch(2)
    loss, gradients, runs = _v8_gradients(batch, prefix, recompute=False)
    recomputed_loss, recomputed_gradients, recomputed_runs = _v8_gradients(
        batch, prefix, recompute=True
    )
    # The three tail steps run again in the backward pass; the no-gradient prefix does not.
    assert (runs, recomputed_runs) == (prefix + 3, prefix + 6)
    assert torch.allclose(recomputed_loss, loss, rtol=1e-6, atol=0)
    assert gradients.keys() == recomputed_gradients.keys()
    # H0, scratch boards included, is trained only when the gradient tail starts from it.
    assert ("scratch_board_embedding" in gradients) == (prefix == 0)
    if prefix == 0:
        assert gradients["scratch_board_embedding"].abs().sum() > 0
    for name, gradient in gradients.items():
        assert torch.allclose(recomputed_gradients[name], gradient, rtol=1e-5, atol=1e-8), name


def test_lookahead_loss_averages_only_labelled_entries() -> None:
    torch.manual_seed(0)
    outputs = {
        "reply_from": torch.randn(2, 64),
        "reply_to": torch.randn(2, 64),
        "board2": torch.randn(2, 64, 13),
        "board4": torch.randn(2, 64, 13),
    }
    unlabelled = complete_targets(
        {"legal_mask": torch.ones(2, 5, dtype=torch.bool), "action": torch.tensor([0, 1])}
    )
    assert lookahead_loss(outputs, unlabelled).item() == 0
    targets = dict(unlabelled, reply_from=torch.tensor([7, IGNORE]))
    targets["board2"] = unlabelled["board2"].clone()
    targets["board2"][1, :3] = torch.tensor([0, 5, 12])
    reply = torch.nn.functional.cross_entropy(outputs["reply_from"][:1], torch.tensor([7]))
    board = torch.nn.functional.cross_entropy(outputs["board2"][1, :3], torch.tensor([0, 5, 12]))
    expected = LOOKAHEAD_SHARES["reply_from"] * reply + LOOKAHEAD_SHARES["board2"] * board
    assert lookahead_loss(outputs, targets).item() == pytest.approx(expected.item(), rel=1e-5)


def test_reply_pair_loss_is_cross_entropy_over_labelled_square_pairs() -> None:
    torch.manual_seed(0)
    pair = torch.randn(4, 64, 64)
    targets = complete_targets(
        {"legal_mask": torch.ones(4, 5, dtype=torch.bool), "action": torch.tensor([0, 1, 2, 3])}
    )
    assert reply_pair_loss(pair, targets).item() == 0
    targets["reply_from"] = torch.tensor([12, IGNORE, 52, 3])
    targets["reply_to"] = torch.tensor([28, 5, IGNORE, 63])
    manual = torch.nn.functional.cross_entropy(
        pair[[0, 3]].flatten(1), torch.tensor([12 * 64 + 28, 3 * 64 + 63])
    )
    assert reply_pair_loss(pair, targets).item() == pytest.approx(manual.item(), rel=1e-6)
    # The pair readout has no square heads, so the lookahead loss keeps only its board shares.
    boards = {"board2": torch.randn(4, 64, 13), "board4": torch.randn(4, 64, 13)}
    targets["board2"] = torch.randint(0, 13, (4, 64))
    targets["board4"][1] = torch.randint(0, 13, (64,))
    expected = LOOKAHEAD_SHARES["board2"] * torch.nn.functional.cross_entropy(
        boards["board2"].flatten(0, 1), targets["board2"].flatten()
    ) + LOOKAHEAD_SHARES["board4"] * torch.nn.functional.cross_entropy(
        boards["board4"][1], targets["board4"][1]
    )
    both = lookahead_loss({"reply_pair": pair, **boards}, targets)
    assert both.item() == pytest.approx(expected.item(), rel=1e-5)


PAIR = dataclasses.replace(TINY, lookahead_heads=True, reply_readout="pair")


def test_reply_weight_adds_the_pair_loss_and_zero_leaves_it_out() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(PAIR)
    batch = _opening_batch(2)
    batch["reply_to"] = torch.tensor([36, 36])
    memory = model.encode_board(batch["pieces"], batch["state"])
    hidden = cast(torch.Tensor, model.recurrent_step(memory, model.initial_hidden(memory)))
    board_tokens = memory.shape[1]
    lookahead = LossWeights(move_value=0.5, lookahead=0.25)
    base, _ = step_loss(model, hidden, board_tokens, batch, lookahead)
    outputs = model.lookahead(hidden)
    graded, _ = step_loss(model, hidden, board_tokens, batch, LossWeights(move_value=0.5))
    boards = 0.25 * lookahead_loss(outputs, batch)
    assert base.item() == pytest.approx((graded + boards).item(), rel=1e-6)
    with_reply = dataclasses.replace(lookahead, reply=0.5)
    total, _ = step_loss(model, hidden, board_tokens, batch, with_reply)
    pair = reply_pair_loss(outputs["reply_pair"], batch)
    assert pair.item() > 0
    assert total.item() == pytest.approx((base + 0.5 * pair).item(), rel=1e-6)
    # At weight 0 the pair projections get no gradient.
    base.backward(retain_graph=True)
    assert model.reply_query.weight.grad is None
    model.zero_grad()
    total.backward()
    for layer in (model.reply_query, model.reply_key, model.reply_origin_bias):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_feedforward_baseline_trains_at_depth_zero(game_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "feedforward"
    baseline = dataclasses.replace(TINY, prelude_blocks=2, recurrent_blocks=0)
    config = _config(
        game_corpus, output, task="games", fixed_depth=0, eval_depths=(0,), eval_positions=3
    )
    train_puzzles(config, baseline)
    rows = _metrics(output)
    assert rows[-1]["mean_tail"] == 0
    assert np.isfinite(rows[-1]["train_loss"])
    assert set(rows[-1]["validation"]) == {"t0"}


def test_train_games_cli_runs_scratch_boards_and_resumes(game_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "v8"
    arguments = [
        "train-games",
        "--corpus",
        str(game_corpus),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--steps",
        "4",
        "--batch-size",
        "2",
        "--warmup-steps",
        "1",
        "--max-depth",
        "3",
        "--max-gradient-steps",
        "2",
        "--eval-every",
        "2",
        "--eval-depths",
        "0,1,3",
        "--snapshot-every",
        "2",
        "--d-model",
        "24",
        "--heads",
        "6",
        "--mlp-width",
        "48",
        "--latent-tokens",
        "4",
        "--recurrent-blocks",
        "1",
        "--ema-decay",
        "0.9",
        "--scratch-boards",
        "2",
        "--recompute-steps",
    ]
    assert main(arguments) == 0
    run = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert run["model_config"]["scratch_boards"] == 2
    assert run["model_config"]["flat_scratch"] is False
    assert run["model_config"]["lookahead_heads"] is True
    assert run["model_config"]["square_attention_bias"] is True
    assert run["training_config"]["supervise_every_step"] is True
    assert run["training_config"]["recompute_steps"] is True
    assert [row["step"] for row in _metrics(output)] == [2, 4]
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = WombatNetwork(NetworkConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    assert model.hidden_tokens == 65 + 4 + 128
    key = "scratch_board_embedding"
    assert not torch.equal(checkpoint["model"][key], checkpoint["raw_model"][key])
    snapshot_path = output / "snapshot-step-000004.pt"
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    model.load_state_dict(snapshot["model"])
    arguments[arguments.index("--steps") + 1] = "6"
    assert main([*arguments, "--resume"]) == 0
    rows = _metrics(output)
    assert [row["step"] for row in rows] == [2, 4, 6]
    assert all(np.isfinite(row["train_loss"]) for row in rows)


def test_refreshed_board_trains_with_a_scratch_board(game_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "refresh"
    model_config = dataclasses.replace(
        TINY, square_attention_bias=True, lookahead_heads=True, scratch_boards=1, refresh_board=True
    )
    config = _config(
        game_corpus,
        output,
        task="games",
        lookahead_weight=0.25,
        supervise_every_step=True,
        ema_decay=0.9,
        eval_positions=3,
    )
    train_puzzles(config, model_config)
    assert all(np.isfinite(row["train_loss"]) for row in _metrics(output))
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["refresh_board"] is True


def test_refresh_board_flag_is_off_unless_asked() -> None:
    assert build_parser().parse_args(["train-games", "--output", "run"]).refresh_board is False
    asked = build_parser().parse_args(["train-games", "--output", "run", "--refresh-board"])
    assert asked.refresh_board is True
    assert build_parser().parse_args(["train-puzzles", "--output", "run"]).refresh_board is False


def test_train_puzzles_cli_keeps_scratch_boards_off() -> None:
    args = build_parser().parse_args(["train-puzzles", "--output", "run"])
    assert (args.scratch_boards, args.flat_scratch, args.recompute_steps) == (0, False, False)
    games = build_parser().parse_args(["train-games", "--output", "run", "--flat-scratch"])
    assert (games.scratch_boards, games.flat_scratch, games.recompute_steps) == (0, True, False)


def test_board_features_are_off_unless_asked_and_never_on_puzzles() -> None:
    assert build_parser().parse_args(["train-games", "--output", "run"]).board_features is False
    asked = build_parser().parse_args(["train-games", "--output", "run", "--board-features"])
    assert asked.board_features is True
    assert build_parser().parse_args(["train-puzzles", "--output", "run"]).board_features is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-puzzles", "--output", "run", "--board-features"])


def test_board_features_train_through_the_cli_and_load(game_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "features"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--board-features"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "1"]
    arguments += ["--prelude-blocks", "1", "--latent-tokens", "4", "--local-attention"]
    assert main(arguments) == 0
    assert all(np.isfinite(row["train_loss"]) for row in _metrics(output))
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["board_features"] is True
    assert "board_feature_projection.weight" in saved["model"]
    model = load_checkpoint(checkpoint, torch.device("cpu"))
    assert model.config.board_features
    trained = model.board_feature_projection.weight
    assert torch.equal(trained, saved["model"]["board_feature_projection.weight"])
    snapshot = load_checkpoint(output / "snapshot-step-000002.pt", torch.device("cpu"))
    assert snapshot.config == model.config
    board = chess.Board("4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1")
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces).long()[None]
    state = torch.from_numpy(state_vector(encoded).astype(np.float32))[None]
    with torch.no_grad():
        policy, value = model(pieces, state, 2)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()


def test_long_range_board_features_are_off_unless_asked_and_never_on_puzzles() -> None:
    games = build_parser().parse_args(["train-games", "--output", "run"])
    assert games.board_features_long_range is False
    arguments = ["train-games", "--output", "run", "--board-features"]
    asked = build_parser().parse_args([*arguments, "--board-features-long-range"])
    assert asked.board_features and asked.board_features_long_range
    puzzles = build_parser().parse_args(["train-puzzles", "--output", "run"])
    assert puzzles.board_features_long_range is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["train-puzzles", "--output", "run", "--board-features-long-range"]
        )


def test_long_range_board_features_train_through_the_cli_and_load(
    game_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "long-range"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--board-features", "--board-features-long-range"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "1"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to"]
    assert main(arguments) == 0
    assert all(np.isfinite(row["train_loss"]) for row in _metrics(output))
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["board_features_long_range"] is True
    assert saved["model"]["board_feature_projection.weight"].shape == (24, 61)
    model = load_checkpoint(checkpoint, torch.device("cpu"))
    assert model.config.board_features and model.config.board_features_long_range
    assert model.board_features.width == 61
    trained = model.board_feature_projection.weight
    assert torch.equal(trained, saved["model"]["board_feature_projection.weight"])
    snapshot = load_checkpoint(output / "snapshot-step-000002.pt", torch.device("cpu"))
    assert snapshot.config == model.config
    board = chess.Board("4k3/8/3q3n/8/1n6/8/8/1R2K2Q b - - 0 1")
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces).long()[None]
    state = torch.from_numpy(state_vector(encoded).astype(np.float32))[None]
    with torch.no_grad():
        policy, value = model(pieces, state, 2)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()
    alone = [item for item in arguments if item != "--board-features"]
    with pytest.raises(ValueError, match="needs board_features"):
        main([*alone, "--output", str(tmp_path / "alone")])
