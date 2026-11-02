from __future__ import annotations

import dataclasses
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from wombat import puzzle_training, surprise
from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit
from wombat.model import NetworkConfig, WombatNetwork
from wombat.puzzle_training import PuzzleTrainingConfig, train_puzzles
from wombat.surprise import SurpriseSampler, sampling_weights, score_surprise

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


class _Interrupted(Exception):
    pass


@pytest.fixture(autouse=True)
def _normal_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surprise, "lower_priority", lambda: None)
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)


@pytest.fixture
def labels(game_corpus: Path, tmp_path: Path) -> Path:
    """Random every-move labels for every training position but the second."""

    split = game_corpus / "train"
    offsets = np.load(split / "legal_offsets.npy").astype(np.int64)
    actions = np.load(split / "legal_actions.npy")
    chosen = np.delete(np.arange(len(offsets) - 1), 1)
    counts = offsets[chosen + 1] - offsets[chosen]
    output = tmp_path / "labels"
    output.mkdir()
    np.save(output / "label_indices.npy", chosen)
    np.save(output / "label_offsets.npy", np.concatenate(([0], np.cumsum(counts))))
    np.save(
        output / "label_actions.npy",
        np.concatenate([actions[offsets[index] : offsets[index + 1]] for index in chosen]),
    )
    wins = np.random.default_rng(0).uniform(0, 1, int(counts.sum()))
    np.save(output / "label_win.npy", wins.astype(np.float16))
    return output


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = WombatNetwork(TINY)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(TINY),
            "training_config": {"temperature": 2.0},
        },
        path,
    )
    return path


def _score(
    checkpoint: Path,
    labels: Path,
    game_corpus: Path,
    puzzle_corpus: Path,
    output: Path,
    **extra: Any,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "corpus": game_corpus,
        "puzzle_corpus": puzzle_corpus,
        "depth": 2,
        "batch_size": 3,
        "chunk_size": 5,
        "device_name": "cpu",
        **extra,
    }
    return score_surprise(checkpoint, labels, output, **settings)


def _arrays(output: Path) -> dict[str, np.ndarray]:
    names = ("games_indices", "games_policy_kl", "games_value_error", "puzzles_cross_entropy")
    return {name: np.load(output / f"{name}.npy") for name in names}


def test_scores_match_a_direct_computation(
    checkpoint: Path, labels: Path, game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    manifest = _score(checkpoint, labels, game_corpus, puzzle_corpus, tmp_path / "scores")
    arrays = _arrays(tmp_path / "scores")
    games = GameSplit(game_corpus / "train", temperature=2.0, labels=labels)
    puzzles = PuzzleSplit(puzzle_corpus / "train")
    np.testing.assert_array_equal(arrays["games_indices"], games.trainable())
    assert manifest["complete"] and manifest["temperature"] == 2.0
    assert (manifest["game_positions"], manifest["puzzle_positions"]) == (
        len(games.trainable()),
        len(puzzles),
    )
    model = WombatNetwork(TINY)
    model.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
    model.eval()

    def log_policy(batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            policy, value = model(batch["pieces"], batch["state"], 2)
        logits = policy.masked_fill(~batch["legal_mask"], float("-inf"))
        return torch.log_softmax(logits, dim=1).numpy(), value.numpy()

    batch = games.batch(games.trainable())
    log_probs, value = log_policy(batch)
    target = batch["policy_target"].numpy()
    scored = target > 0
    gap = np.log(np.where(scored, target, 1)) - np.where(scored, log_probs, 0)
    kl = (target * gap).sum(1)
    np.testing.assert_allclose(arrays["games_policy_kl"], kl, rtol=1e-4, atol=1e-5)
    errors = (value - batch["value"].numpy()) ** 2
    np.testing.assert_allclose(arrays["games_value_error"], errors, rtol=1e-4, atol=1e-6)
    everything = np.arange(len(puzzles))
    log_probs, _ = log_policy(puzzles.batch(everything))
    cross_entropy = -log_probs[everything, puzzles.action]
    np.testing.assert_allclose(arrays["puzzles_cross_entropy"], cross_entropy, rtol=1e-4)
    assert manifest["games_surprise"]["mean"] == pytest.approx(float((kl + errors).mean()))


def test_scoring_resumes_from_the_last_finished_chunk(
    checkpoint: Path,
    labels: Path,
    game_corpus: Path,
    puzzle_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _score(checkpoint, labels, game_corpus, puzzle_corpus, tmp_path / "whole")
    output = tmp_path / "resumed"
    readout = surprise._readout
    calls = {"count": 0}

    def interrupted(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls["count"] += 1
        if calls["count"] > 4:
            raise _Interrupted
        return readout(*args)

    monkeypatch.setattr(surprise, "_readout", interrupted)
    with pytest.raises(_Interrupted):
        _score(checkpoint, labels, game_corpus, puzzle_corpus, output)
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    # The seven games take chunks of 5 and 2 in three batches of at most 3; the fifth
    # batch stops inside the first puzzle chunk.
    assert progress["done"] == {"games": 7, "puzzles": 0}
    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError):
        _score(checkpoint, labels, game_corpus, puzzle_corpus, output, depth=3)
    calls["count"] = -1000
    _score(checkpoint, labels, game_corpus, puzzle_corpus, output)
    puzzles = len(PuzzleSplit(puzzle_corpus / "train"))
    batches = sum(-(-min(5, puzzles - start) // 3) for start in range(0, puzzles, 5))
    assert calls["count"] == -1000 + batches
    whole, resumed = _arrays(tmp_path / "whole"), _arrays(output)
    for name, values in whole.items():
        np.testing.assert_array_equal(resumed[name], values)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["complete"]


def test_weights_are_capped_at_the_99th_percentile_and_floored_above_zero() -> None:
    values = np.concatenate((np.zeros(50), np.linspace(0.1, 1.0, 949), [1000.0]))
    weights, summary = sampling_weights(values)
    assert weights.max() == pytest.approx(np.quantile(values, 0.99))
    assert weights.min() == pytest.approx(summary["floor"]) and summary["floor"] > 0
    assert summary["cap"] == pytest.approx(weights.max())
    assert 0.01 < summary["top_1_percent_weight"] < summary["top_10_percent_weight"] < 1
    with pytest.raises(ValueError):
        sampling_weights(np.array([1.0, np.nan]))


def test_draws_follow_the_weights_with_the_uniform_share_kept() -> None:
    pool = np.arange(100, 110)
    values = np.array([1.0, 1, 1, 1, 1, 2, 2, 4, 8, 8])
    weights, _ = sampling_weights(values)
    sampler = SurpriseSampler(pool, values, 0.25, 8, np.random.default_rng(0))
    assert (sampler.uniform, sampler.weighted) == (6, 2)
    counts = np.zeros(len(pool))
    for _ in range(30_000):
        chosen = next(sampler)
        assert len(chosen) == 8
        counts += np.bincount(chosen[6:] - 100, minlength=len(pool))
    frequency = counts / counts.sum()
    np.testing.assert_allclose(frequency, weights / weights.sum(), atol=0.01)
    assert sampler.drain_mean() == pytest.approx(
        0.75 * values.mean() + 0.25 * float((values * weights).sum() / weights.sum()), rel=0.02
    )
    assert sampler.drain_mean() is None
    # The uniform part walks a permutation: one pass of 10 gives one batch of 6 distinct rows.
    fresh = SurpriseSampler(pool, values, 0.25, 8, np.random.default_rng(1))
    assert len(set(next(fresh)[:6].tolist())) == 6
    only = SurpriseSampler(pool, values, 1.0, 8, np.random.default_rng(0))
    assert len(next(only)) == 8


def test_no_surprise_share_draws_exactly_the_default_batches() -> None:
    pool = np.arange(3, 40)
    values = np.linspace(0.1, 2.0, len(pool))
    first, second = np.random.default_rng(5), np.random.default_rng(5)
    sampler = SurpriseSampler(pool, values, 0.0, 4, first)
    for _ in range(25):
        order = pool[second.permutation(len(pool))]
        for start in range(0, len(order) - 4 + 1, 4):
            np.testing.assert_array_equal(next(sampler), order[start : start + 4])


def test_score_surprise_cli_writes_scores_and_a_manifest(
    checkpoint: Path, labels: Path, game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    scores = tmp_path / "scores"
    command = ["score-surprise", "--checkpoint", str(checkpoint), "--labels", str(labels)]
    command += ["--output", str(scores), "--corpus", str(game_corpus)]
    command += ["--puzzle-corpus", str(puzzle_corpus), "--depth", "2", "--device", "cpu"]
    assert main(command) == 0
    manifest = json.loads((scores / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["depth"] == 2 and manifest["checkpoint"] == str(checkpoint)
    assert manifest["labels"] == str(labels) and manifest["complete"]
    assert manifest["games_weights"]["positions"] == manifest["game_positions"]
    for name, values in _arrays(scores).items():
        assert np.isfinite(values).all(), name


def _training(
    game_corpus: Path, puzzle_corpus: Path, labels: Path, output: Path, **overrides: object
) -> PuzzleTrainingConfig:
    settings: dict[str, Any] = {
        "corpus": str(game_corpus),
        "output": str(output),
        "task": "games",
        "labels": str(labels),
        "steps": 6,
        "batch_size": 2,
        "warmup_steps": 1,
        "max_depth": 3,
        "max_gradient_steps": 2,
        "fixed_depth": 2,
        "eval_every": 3,
        "eval_positions": 3,
        "eval_depths": (0, 2),
        "puzzle_corpus": str(puzzle_corpus),
        "puzzle_share": 0.5,
        "puzzle_steps_per_move": 1,
        "device": "cpu",
        **overrides,
    }
    return PuzzleTrainingConfig(**settings)


def test_training_with_no_surprise_share_matches_the_default_bit_for_bit(
    checkpoint: Path, labels: Path, game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    scores = tmp_path / "scores"
    _score(checkpoint, labels, game_corpus, puzzle_corpus, scores)
    runs = {
        "default": _training(game_corpus, puzzle_corpus, labels, tmp_path / "default"),
        "zero": _training(
            game_corpus,
            puzzle_corpus,
            labels,
            tmp_path / "zero",
            surprise=str(scores),
            surprise_share=0.0,
        ),
    }
    states = {}
    for name, config in runs.items():
        train_puzzles(config, TINY)
        states[name] = torch.load(Path(config.output) / "checkpoint.pt", weights_only=False)
    for key, tensor in states["default"]["raw_model"].items():
        assert torch.equal(tensor, states["zero"]["raw_model"][key]), key
    default = json.loads((tmp_path / "default" / "config.json").read_text(encoding="utf-8"))
    assert "surprise" not in default and default["training_config"]["surprise"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "puzzles", "surprise": "s", "puzzle_corpus": None, "puzzle_share": 0.0},
        {"surprise": "s", "surprise_share": 1.5},
    ],
)
def test_surprise_settings_are_checked_before_training(overrides: dict[str, object]) -> None:
    base = _training(Path("."), Path("."), Path("."), Path("."))
    config = dataclasses.replace(base, **overrides)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        puzzle_training._check_surprise(config)


def test_surprise_scores_must_cover_the_training_pool(
    checkpoint: Path,
    labels: Path,
    game_corpus: Path,
    puzzle_corpus: Path,
    tmp_path: Path,
) -> None:
    scores = tmp_path / "scores"
    _score(checkpoint, labels, game_corpus, puzzle_corpus, scores)
    labelled = _training(game_corpus, puzzle_corpus, labels, tmp_path / "run", surprise=str(scores))
    unlabelled = dataclasses.replace(labelled, labels=None)
    with pytest.raises(ValueError, match="other game positions"):
        train_puzzles(unlabelled, TINY)


def test_train_games_cli_draws_by_surprise_and_resumes(
    checkpoint: Path, labels: Path, game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    scores = tmp_path / "scores"
    manifest = _score(checkpoint, labels, game_corpus, puzzle_corpus, scores)
    output = tmp_path / "run"
    arguments = [
        "train-games",
        "--corpus",
        str(game_corpus),
        "--labels",
        str(labels),
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
        "0,2",
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
        "--surprise-weights",
        str(scores),
        "--surprise-share",
        "0.75",
    ]
    assert main(arguments) == 0
    run = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert run["training_config"]["surprise"] == str(scores)
    assert run["training_config"]["surprise_share"] == 0.75
    assert run["surprise"]["source"] == str(scores) and run["surprise"]["share"] == 0.75
    assert run["surprise"]["games"]["positions"] == manifest["game_positions"]
    assert set(run["surprise"]["puzzles"]) == {"1", "2"}
    arguments[arguments.index("--steps") + 1] = "6"
    assert main([*arguments, "--resume"]) == 0
    rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [2, 4, 6]
    for row in rows:
        assert np.isfinite(row["train_loss"])
        drawn = row["surprise_drawn"]
        assert drawn["source"] == str(scores) and set(drawn) == {"source", "games", "r1", "r2"}
        assert any(value is not None for key, value in drawn.items() if key != "source")


def test_surprise_sampling_is_off_unless_asked() -> None:
    games = build_parser().parse_args(["train-games", "--output", "run"])
    assert (games.surprise_weights, games.surprise_share) == (None, 0.5)
    puzzles = build_parser().parse_args(["train-puzzles", "--output", "run"])
    assert (puzzles.surprise_weights, puzzles.surprise_share) == (None, 0.5)
    config = PuzzleTrainingConfig(corpus="c", output="o")
    assert (config.surprise, config.surprise_share) == (None, 0.5)
