"""Train a recurrent core from scratch with a progressive depth objective.

The ``puzzles`` task uses single-answer puzzle lines; the ``games`` task uses
ordinary positions with graded win-chance targets over scored candidates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import cast

import numpy as np
import torch
from torch import nn

from wombat.cuda_graphs import GraphedPredictor, GraphedTrainer
from wombat.data.game_corpus import GameSplit, JoinedGameSplit, densify, with_children
from wombat.data.puzzle_corpus import PuzzleSplit, lower_priority
from wombat.eval.puzzle_corpus_eval import (
    candidate_quality,
    predict_depths,
    remaining_groups,
    sample_groups,
    score_predictions,
)
from wombat.gpu_share import gpu_priority_class, share_gpu
from wombat.losses import (
    POLICY_ONLY,
    LineWeights,
    LossWeights,
    complete_targets,
    line_loss,
    unrolled_loss,
)
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.network import noisy_squares
from wombat.surprise import SurpriseSampler, load_surprise
from wombat.training import _sample_progressive_segment

TASKS = ("puzzles", "games")
DEPTH_DISTRIBUTIONS = ("fixed", "lognormal")
# Spread of the log-normal depth draw, as in Huginn (Geiping et al. 2025).
DEPTH_SIGMA = 0.5
TARGET_KEYS = (
    "action",
    "legal_mask",
    "policy_target",
    "move_value_target",
    "move_value_mask",
    "value",
    "value_mask",
    "reply_from",
    "reply_to",
    "board2",
    "board4",
)


@dataclass(frozen=True)
class PuzzleTrainingConfig:
    corpus: str
    output: str
    task: str = "puzzles"
    # Every-move Stockfish labels for the games task; training then uses the labelled subset.
    labels: str | None = None
    # Games task: (corpus, labels) pairs whose labelled train positions join the pool;
    # validation stays the first corpus's.
    extra_train: tuple[tuple[str, str], ...] = ()
    # Win-chance points per unit of target log-probability (games task).
    temperature: float = 4.0
    move_value_weight: float = 0.0
    value_weight: float = 0.0
    # Reply-square and future-board predictions; needs a model with lookahead heads.
    lookahead_weight: float = 0.0
    # Joint reply (origin, destination) prediction; needs the pair reply readout.
    reply_weight: float = 0.0
    # Grade every step of the gradient tail, later steps weighted more, not just the last.
    supervise_every_step: bool = False
    # Exponential moving average of the weights, used for checkpoints and evaluation.
    ema_decay: float | None = None
    # Recompute each gradient-tail step's activations in the backward pass to save memory.
    recompute_steps: bool = False
    # Model-only snapshots every this many steps; 0 disables them.
    snapshot_every: int = 0
    steps: int = 30_000
    # Stop at this step while keeping the learning-rate schedule of ``steps``.
    stop_after: int | None = None
    # The last this many steps before the final step decay from the scheduled rate to 0.
    cooldown_steps: int = 0
    # 256 exceeds a 4 GiB VRAM ceiling with an eight-step gradient tail.
    batch_size: int = 128
    learning_rate: float = 5e-4
    warmup_steps: int = 1_000
    weight_decay: float = 0.05
    max_depth: int = 16
    max_gradient_steps: int = 8
    # When set, every batch runs exactly this depth with full backpropagation.
    fixed_depth: int | None = None
    # With a fixed depth, this share of batches instead continues a random no-gradient
    # prefix of up to fixed_depth steps with a tail of min_gradient_steps to
    # max_gradient_steps, never past max_depth in total.
    progressive_share: float = 0.0
    min_gradient_steps: int = 1
    # Up to this many extra no-gradient steps before every batch's tail, drawn per batch,
    # so the graded step varies and the model cannot count its way to the answer.
    random_prefix: int = 0
    # The random prefix starts at this training step, so the exact-depth algorithm forms first.
    random_prefix_from: int = 0
    # "lognormal" replaces fixed_depth and random_prefix: each batch draws its total steps as
    # Poisson(exp(tau)) + 1 up to depth_max, the last max_gradient_steps of them with gradients.
    # A puzzle batch keeps its own horizon and runs the draw's no-gradient steps before it.
    depth_distribution: str = "fixed"
    depth_mean: float = 16.0
    depth_max: int = 48
    # Std of Gaussian noise added to the square tokens of H0 in training, never in evaluation.
    initial_noise: float = 0.0
    # Games task: a puzzle corpus whose train split replaces this share of batches, each
    # drawn at one remaining count and run for that many moves times steps_per_move.
    puzzle_corpus: str | None = None
    puzzle_share: float = 0.0
    puzzle_remaining: tuple[int, ...] = (1, 2)
    puzzle_steps_per_move: int = 4
    # Longest gradient tail of a puzzle batch; a longer horizon runs its first steps without
    # gradients, so later steps must improve the state the model already reached.
    puzzle_max_tail: int | None = None
    # Validation puzzles per solver-moves-remaining group, when a puzzle corpus is given.
    puzzle_eval_per_group: int = 512
    # Scores from score-surprise; game and puzzle batches then draw this share of their rows
    # in proportion to capped surprise and the rest uniformly. Evaluation stays uniform.
    surprise: str | None = None
    surprise_share: float = 0.5
    # Games task: this share of each game batch's rows become child positions (a labelled
    # position after one of its scored moves) whose only target is the value.
    child_value_share: float = 0.0
    # A checkpoint whose raw and averaged weights start a fresh run: a new optimizer, step count
    # and schedule. Resuming this run's own checkpoint takes precedence.
    init_from: str | None = None
    # Workspace core: each batch imagines a number of lines of a number of plies, both drawn
    # uniformly from these ranges and run exactly; random_prefix adds surplus steps from
    # random_prefix_from on. Given final maxima, the maxima ramp linearly to them between
    # budget_ramp_from and budget_ramp_until.
    lines_min: int = 1
    lines_max: int = 3
    line_plies_min: int = 1
    line_plies_max: int = 3
    lines_max_final: int | None = None
    line_plies_max_final: int | None = None
    budget_ramp_from: int = 0
    budget_ramp_until: int = 0
    canonical_weight: float = 1.0
    line_policy_weight: float = 0.25
    # Share of batches whose lines follow the reference line after a matching first move, a
    # warm start falling linearly to 0 at reference_line_until (None: 30% of the last step).
    reference_line_share: float = 1.0
    reference_line_until: int | None = None
    eval_every: int = 1_000
    eval_positions: int = 8_192
    eval_depths: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    # VRAM ceiling in GiB.
    max_vram_gb: float | None = 3.5
    # Replay captured CUDA graphs instead of launching every kernel from Python.
    cuda_graphs: bool = True
    seed: int = 0
    device: str = "cuda"


def _last_step(config: PuzzleTrainingConfig) -> int:
    return min(config.steps, config.stop_after or config.steps)


def _scheduled_rate(config: PuzzleTrainingConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, progress))))


def _learning_rate(config: PuzzleTrainingConfig, step: int) -> float:
    """Warmup and cosine over ``steps``, then a 1 - sqrt cooldown to 0 at the last step."""

    start = _last_step(config) - config.cooldown_steps
    if not config.cooldown_steps or step < start:
        return _scheduled_rate(config, step)
    fraction = min(1.0, (step - start) / config.cooldown_steps)
    return _scheduled_rate(config, start) * (1 - math.sqrt(fraction))


def _segment(config: PuzzleTrainingConfig, randomizer: random.Random) -> tuple[int, int]:
    if config.fixed_depth is not None:
        if config.progressive_share and randomizer.random() < config.progressive_share:
            prefix = randomizer.randint(0, config.fixed_depth)
            longest = min(config.max_gradient_steps, config.max_depth - prefix)
            return prefix, randomizer.randint(config.min_gradient_steps, longest)
        return 0, config.fixed_depth
    return _sample_progressive_segment(randomizer, config.max_depth, config.max_gradient_steps)


def _sample_depth(config: PuzzleTrainingConfig, randomizer: random.Random) -> int:
    """Total steps Poisson(exp(tau)) + 1 with tau ~ N(log(depth_mean) - sigma^2 / 2, sigma)."""

    tau = randomizer.normalvariate(math.log(config.depth_mean) - DEPTH_SIGMA**2 / 2, DEPTH_SIGMA)
    rate = math.exp(tau)
    # Inverse CDF from one uniform; counts past depth_max - 1 would be clipped anyway.
    uniform = randomizer.random()
    count, probability = 0, math.exp(-rate)
    cumulative = probability
    while uniform > cumulative and count < config.depth_max - 1:
        count += 1
        probability *= rate / count
        cumulative += probability
    return count + 1


def _draw(
    config: PuzzleTrainingConfig, randomizer: random.Random, step: int = 0
) -> tuple[int | None, int, int]:
    """Solver moves remaining for a puzzle batch (None for a game batch), prefix and tail."""

    drawn: tuple[int | None, int, int]
    lognormal = config.depth_distribution == "lognormal"
    if config.puzzle_share and randomizer.random() < config.puzzle_share:
        remaining = randomizer.choice(config.puzzle_remaining)
        horizon = config.puzzle_steps_per_move * remaining
        tail = min(horizon, config.puzzle_max_tail or horizon)
        drawn = remaining, horizon - tail, tail
    elif lognormal:
        drawn = None, 0, 0
    else:
        drawn = None, *_segment(config, randomizer)
    remaining_moves, prefix, tail = drawn
    if lognormal:
        # A game batch runs the drawn depth; a puzzle batch runs its no-gradient part first.
        depth = _sample_depth(config, randomizer)
        graded = min(depth, config.max_gradient_steps)
        tail = graded if remaining_moves is None else tail
        drawn = remaining_moves, prefix + depth - graded, tail
    elif config.random_prefix and step >= config.random_prefix_from:
        drawn = remaining_moves, prefix + randomizer.randint(0, config.random_prefix), tail
    return drawn


def _ramped(config: PuzzleTrainingConfig, maximum: int, final: int | None, step: int) -> int:
    """A budget maximum at ``step``, moving linearly to ``final`` over the ramp's steps."""

    if final is None:
        return maximum
    span = max(1, config.budget_ramp_until - config.budget_ramp_from)
    progress = min(1.0, max(0.0, (step - config.budget_ramp_from) / span))
    return round(maximum + (final - maximum) * progress)


def _reference_share(config: PuzzleTrainingConfig, step: int) -> float:
    until = config.reference_line_until
    if until is None:
        until = int(0.3 * _last_step(config))
    if until <= 0:
        return 0.0
    return config.reference_line_share * max(0.0, 1 - step / until)


def _draw_line(
    config: PuzzleTrainingConfig, randomizer: random.Random, step: int = 0
) -> tuple[int | None, int, int, int, bool]:
    """Workspace core: solver moves remaining for a puzzle batch (None for games), lines,
    plies per line, surplus steps, and whether lines follow the reference."""

    remaining = None
    if config.puzzle_share and randomizer.random() < config.puzzle_share:
        remaining = randomizer.choice(config.puzzle_remaining)
    lines_max = _ramped(config, config.lines_max, config.lines_max_final, step)
    plies_max = _ramped(config, config.line_plies_max, config.line_plies_max_final, step)
    lines = randomizer.randint(config.lines_min, lines_max)
    plies = randomizer.randint(config.line_plies_min, plies_max)
    surplus = 0
    if config.random_prefix and step >= config.random_prefix_from:
        surplus = randomizer.randint(0, config.random_prefix)
    follow = randomizer.random() < _reference_share(config, step)
    return remaining, lines, plies, surplus, follow


def _check_workspace(config: PuzzleTrainingConfig, model_config: NetworkConfig) -> None:
    if model_config.core != "workspace":
        return
    if config.task != "games" or config.labels is None:
        raise ValueError("the workspace core trains on labelled game lines (and puzzles)")
    if config.depth_distribution != "fixed" or config.progressive_share:
        raise ValueError("the workspace core draws its budget of lines and plies per batch")
    if config.fixed_depth is not None:
        raise ValueError("the workspace core draws its budget per batch; leave fixed_depth unset")
    if config.supervise_every_step:
        raise ValueError("the workspace core grades the root once, after its lines")
    if config.initial_noise:
        raise ValueError("initial_noise perturbs the recurrent core's H0")
    for low, high, final in (
        (config.lines_min, config.lines_max, config.lines_max_final),
        (config.line_plies_min, config.line_plies_max, config.line_plies_max_final),
    ):
        if not 1 <= low <= high or (final is not None and final < high):
            raise ValueError("budget ranges need 1 <= minimum <= maximum <= final maximum")
    if not 0 <= config.reference_line_share <= 1:
        raise ValueError("reference_line_share must be in [0, 1]")
    if config.canonical_weight < 0 or config.line_policy_weight < 0:
        raise ValueError("line loss weights cannot be negative")


def _check_depth_distribution(config: PuzzleTrainingConfig) -> None:
    if config.depth_distribution not in DEPTH_DISTRIBUTIONS:
        raise ValueError(f"depth_distribution must be one of {DEPTH_DISTRIBUTIONS}")
    if config.initial_noise < 0:
        raise ValueError("initial_noise must be at least 0")
    if config.depth_distribution == "fixed":
        return
    if config.fixed_depth is not None or config.progressive_share or config.random_prefix:
        raise ValueError("lognormal depth replaces fixed_depth and random_prefix")
    if config.supervise_every_step:
        raise ValueError("lognormal depth is graded only at the last step")
    if config.depth_mean <= 0 or config.depth_max < 1 or config.max_gradient_steps < 1:
        raise ValueError("lognormal depth needs a positive mean, maximum and gradient tail")


def _max_tail(config: PuzzleTrainingConfig) -> int:
    """Longest gradient tail a game batch can draw, which the graphed trainer captures."""

    return max(
        config.fixed_depth or config.max_gradient_steps,
        config.max_gradient_steps if config.progressive_share else 0,
    )


def _check_puzzle_mix(config: PuzzleTrainingConfig, workspace: bool = False) -> None:
    """``workspace`` puzzle batches run the line's cadence, so their horizon settings are unused."""

    if config.puzzle_corpus is None:
        if config.puzzle_share:
            raise ValueError("puzzle batches need a puzzle corpus")
        return
    if config.task != "games":
        raise ValueError("puzzle batches mix into the games task")
    if not 0 <= config.puzzle_share <= 1:
        raise ValueError("puzzle_share must be in [0, 1]")
    if not config.puzzle_share:
        return
    if config.supervise_every_step:
        raise ValueError("puzzle batches are graded only at the end of their horizon")
    if not config.puzzle_remaining or min(config.puzzle_remaining) < 1:
        raise ValueError("puzzle batches need remaining counts of 1 or more")
    if workspace:
        return
    if config.puzzle_steps_per_move < 1:
        raise ValueError("puzzle_steps_per_move must be at least 1")
    longest = config.puzzle_steps_per_move * max(config.puzzle_remaining)
    if config.puzzle_max_tail is not None:
        if config.puzzle_max_tail < 1:
            raise ValueError("puzzle_max_tail must be at least 1")
        longest = min(longest, config.puzzle_max_tail)
    allowed = _max_tail(config)
    if longest > allowed:
        raise ValueError(f"puzzle tails reach {longest} steps; the longest tail is {allowed}")


def _check_surprise(config: PuzzleTrainingConfig) -> None:
    if config.surprise is None:
        return
    if config.task != "games":
        raise ValueError("surprise-weighted draws need the games task")
    if not 0 <= config.surprise_share <= 1:
        raise ValueError("surprise_share must be in [0, 1]")


def _check_children(config: PuzzleTrainingConfig) -> None:
    if not 0 <= config.child_value_share <= 1:
        raise ValueError("child_value_share must be in [0, 1]")
    if config.child_value_share and (config.task != "games" or config.labels is None):
        raise ValueError("child positions come from labelled game positions")


def _start_from(
    path: Path, model: WombatNetwork, average: WombatNetwork | None, model_config: NetworkConfig
) -> None:
    """Load a checkpoint's raw and averaged weights; its model config must be this run's."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    # Settings added after the checkpoint was written take their defaults, as loading does.
    saved = asdict(NetworkConfig(**state["model_config"]))
    differing = [key for key, value in asdict(model_config).items() if saved[key] != value]
    if differing:
        raise ValueError(f"{path} has another model config: {', '.join(differing)}")
    model.load_state_dict(state.get("raw_model", state["model"]))
    if average is not None:
        average.load_state_dict(state["model"])


def _surprise_samplers(
    config: PuzzleTrainingConfig,
    train: GameSplit | JoinedGameSplit,
    puzzle_pools: dict[int, np.ndarray],
    puzzle_positions: int,
    generator: np.random.Generator,
) -> tuple[SurpriseSampler, dict[int, SurpriseSampler], dict[str, object]]:
    """Surprise-weighted draws for the game pool and each puzzle pool, and their summary."""

    source = Path(cast(str, config.surprise))
    indices, games, puzzles, manifest = load_surprise(source)
    if not np.array_equal(indices, train.trainable()):
        raise ValueError(f"{source} scores other game positions than the training pool")
    if puzzle_pools and len(puzzles) != puzzle_positions:
        raise ValueError(f"{source} scores another puzzle training split")
    share, size = config.surprise_share, config.batch_size
    game_sampler = SurpriseSampler(indices, games, share, size, generator)
    puzzle_samplers = {
        moves: SurpriseSampler(pool, puzzles[pool], share, size, generator)
        for moves, pool in puzzle_pools.items()
    }
    summary: dict[str, object] = {
        "source": str(source),
        "checkpoint": manifest["checkpoint"],
        "depth": manifest["depth"],
        "share": share,
        "games": game_sampler.summary,
        "puzzles": {str(moves): sampler.summary for moves, sampler in puzzle_samplers.items()},
    }
    return game_sampler, puzzle_samplers, summary


def _loss(
    model: WombatNetwork,
    batch: dict[str, torch.Tensor],
    prefix: int,
    tail: int,
    device: torch.device,
    weights: LossWeights = POLICY_ONLY,
    *,
    every_step: bool = False,
    recompute: bool = False,
    initial_noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graded loss on the trained tail after an unobserved prefix."""

    batch = complete_targets(batch)
    pieces = batch["pieces"].to(device, non_blocking=True)
    state = batch["state"].to(device, non_blocking=True)
    targets = {key: batch[key].to(device, non_blocking=True) for key in TARGET_KEYS}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        memory = model.encode_board(pieces, state)
        hidden = model.initial_hidden(memory)
        if initial_noise and model.training:
            hidden = noisy_squares(hidden, initial_noise)
        if prefix:
            with torch.no_grad():
                for _ in range(prefix):
                    hidden = cast(torch.Tensor, model.recurrent_step(memory, hidden))
            hidden = hidden.detach()
        return unrolled_loss(
            model,
            memory,
            hidden,
            tail,
            targets,
            weights,
            every_step=every_step,
            recompute=recompute,
        )


def _line_loss(
    model: WombatNetwork,
    batch: dict[str, torch.Tensor],
    budget: tuple[int, int, int],
    follow: bool,
    device: torch.device,
    weights: LossWeights,
    line: LineWeights,
    *,
    recompute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """The workspace core's graded loss after its imagined lines (see ``line_loss``).

    ``budget`` holds the lines, the plies per line and the surplus steps.
    """

    batch = complete_targets(batch)
    pieces = batch["pieces"].to(device, non_blocking=True)
    state = batch["state"].to(device, non_blocking=True)
    keys = (*TARGET_KEYS, "line_actions")
    targets = {key: batch[key].to(device, non_blocking=True) for key in keys}
    lines, plies, surplus = budget
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return line_loss(
            model,
            pieces,
            state,
            targets,
            weights,
            line,
            lines=lines,
            plies=plies,
            surplus=surplus,
            follow_reference=follow,
            recompute=recompute,
        )


def _evaluate(
    model: WombatNetwork,
    split: PuzzleSplit | GameSplit,
    indices: np.ndarray,
    depths: tuple[int, ...],
    device: torch.device,
    predictor: GraphedPredictor | None,
) -> dict[str, float]:
    predictions = predict_depths(
        model,
        split,
        indices,
        depths,
        batch_size=512 if predictor is None else predictor.batch_size,
        device=device,
        predictor=predictor,
    )
    model.train()
    if isinstance(split, GameSplit):
        # Share of moves within 5 win-chance points of the best scored candidate.
        return {
            f"t{depth}": candidate_quality(split, indices, predicted)["good"]
            for depth, predicted in predictions.items()
        }
    return {
        f"t{depth}": float(score_predictions(split, indices, predicted).mean())
        for depth, predicted in predictions.items()
    }


def _evaluate_puzzle_groups(
    model: WombatNetwork,
    split: PuzzleSplit,
    groups: dict[str, np.ndarray],
    depths: tuple[int, ...],
    device: torch.device,
    predictor: GraphedPredictor | None,
) -> dict[str, dict[str, float]]:
    """First-move accuracy at each depth per solver-moves-remaining group, keyed like "r3"."""

    groups = {name: members for name, members in groups.items() if len(members)}
    if not groups:
        return {}
    indices = np.concatenate(list(groups.values()))
    predictions = predict_depths(
        model,
        split,
        indices,
        depths,
        batch_size=512 if predictor is None else predictor.batch_size,
        device=device,
        predictor=predictor,
    )
    model.train()
    correctness = {
        depth: score_predictions(split, indices, predicted)
        for depth, predicted in predictions.items()
    }
    results: dict[str, dict[str, float]] = {}
    start = 0
    for name, members in groups.items():
        stop = start + len(members)
        results[f"r{name}"] = {
            f"t{depth}": float(correct[start:stop].mean()) for depth, correct in correctness.items()
        }
        start = stop
    return results


def _on_device(
    batches: Iterator[dict[str, torch.Tensor]], device: torch.device
) -> Iterator[dict[str, torch.Tensor]]:
    for batch in batches:
        yield {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _update_average(average: WombatNetwork, model: WombatNetwork, decay: float) -> None:
    """Move the averaged weights a (1 - decay) step toward the trained ones."""

    with torch.no_grad():
        torch._foreach_lerp_(list(average.parameters()), list(model.parameters()), 1 - decay)


def _save(
    path: Path,
    model: WombatNetwork,
    average: WombatNetwork | None,
    optimizer: torch.optim.Optimizer,
    config: PuzzleTrainingConfig,
    model_config: NetworkConfig,
    step: int,
    randomizer: random.Random,
    generator: np.random.Generator,
) -> None:
    """``model`` holds the weights to play with; ``raw_model`` the ones training resumes from."""

    temporary = path.with_suffix(".partial")
    torch.save(
        {
            "model": (average or model).state_dict(),
            "raw_model": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "python_random": randomizer.getstate(),
            "numpy_random": generator.bit_generator.state,
            "torch_random": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def train_puzzles(
    config: PuzzleTrainingConfig, model_config: NetworkConfig, *, resume: bool = False
) -> dict[str, object]:
    lower_priority()
    device = torch.device(config.device)
    share_gpu(device, config.max_vram_gb)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    metrics_path = output / "metrics.jsonl"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if config.progressive_share and (
        config.fixed_depth is None
        or config.min_gradient_steps < 1
        or config.fixed_depth + config.min_gradient_steps > config.max_depth
        or config.min_gradient_steps > config.max_gradient_steps
    ):
        raise ValueError("progressive batches need a fixed depth and room for their shortest tail")
    if not 0 <= config.cooldown_steps <= _last_step(config):
        raise ValueError("cooldown_steps must be between 0 and the last training step")
    workspace = model_config.core == "workspace"
    _check_depth_distribution(config)
    _check_puzzle_mix(config, workspace)
    _check_surprise(config)
    _check_children(config)
    _check_workspace(config, model_config)
    line = LineWeights(config.canonical_weight, config.line_policy_weight)
    if config.reply_weight and model_config.reply_readout != "pair":
        raise ValueError("reply_weight trains the pair reply readout")
    weights = LossWeights(
        config.move_value_weight, config.value_weight, config.lookahead_weight, config.reply_weight
    )
    if config.extra_train and config.task != "games":
        raise ValueError("extra training corpora join the games task")
    train: PuzzleSplit | GameSplit | JoinedGameSplit
    validation: PuzzleSplit | GameSplit
    if config.task == "games":
        train = GameSplit(
            Path(config.corpus) / "train",
            temperature=config.temperature,
            labels=None if config.labels is None else Path(config.labels),
        )
        if config.extra_train:
            extra = [
                GameSplit(
                    Path(corpus) / "train", temperature=config.temperature, labels=Path(labels)
                )
                for corpus, labels in config.extra_train
            ]
            train = JoinedGameSplit([train, *extra])
        validation = GameSplit(Path(config.corpus) / "validation", temperature=config.temperature)
    else:
        train = PuzzleSplit(Path(config.corpus) / "train")
        validation = PuzzleSplit(Path(config.corpus) / "validation")
    eval_indices = np.random.default_rng(config.seed).permutation(len(validation))[
        : config.eval_positions
    ]
    puzzle_train = puzzle_validation = None
    puzzle_pools: dict[int, np.ndarray] = {}
    puzzle_eval_groups: dict[str, np.ndarray] = {}
    if config.puzzle_corpus is not None:
        puzzle_validation = PuzzleSplit(Path(config.puzzle_corpus) / "validation")
        puzzle_eval_groups = sample_groups(
            remaining_groups(puzzle_validation.remaining), config.puzzle_eval_per_group, config.seed
        )
        if config.puzzle_share:
            puzzle_train = PuzzleSplit(Path(config.puzzle_corpus) / "train")
            solver_moves = np.asarray(puzzle_train.remaining)
            for moves in sorted(set(config.puzzle_remaining)):
                puzzle_pools[moves] = np.flatnonzero(solver_moves == moves)
                if len(puzzle_pools[moves]) < config.batch_size:
                    raise ValueError(f"fewer than a batch of puzzles with {moves} moves remaining")
    model = WombatNetwork(model_config).to(device)
    # The feedforward baseline (fixed depth 0) and the workspace core train eagerly; graphs cover
    # the recurrent core's gradient tails of 1+.
    graphed = (
        config.cuda_graphs and device.type == "cuda" and config.fixed_depth != 0 and not workspace
    )
    # A device-resident learning rate lets captured optimizer steps follow the schedule.
    learning_rate: float | torch.Tensor = (
        torch.tensor(config.learning_rate, device=device) if graphed else config.learning_rate
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
        fused=graphed or None,
        capturable=graphed,
    )
    randomizer = random.Random(config.seed)
    generator = np.random.default_rng(config.seed)
    step = 0
    average = None
    if config.ema_decay is not None:
        average = WombatNetwork(model_config).to(device)
        average.load_state_dict(model.state_dict())
        average.requires_grad_(False)
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("raw_model", state["model"]))
        if average is not None:
            average.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        step = int(state["step"])
        randomizer.setstate(state["python_random"])
        generator.bit_generator.state = state["numpy_random"]
        torch.set_rng_state(state["torch_random"])
    else:
        if metrics_path.exists():
            metrics_path.unlink()
        if config.init_from is not None:
            _start_from(Path(config.init_from), model, average, model_config)
    game_sampler: SurpriseSampler | None = None
    puzzle_samplers: dict[int, SurpriseSampler] = {}
    surprise_summary: dict[str, object] | None = None
    if config.surprise is not None:
        game_sampler, puzzle_samplers, surprise_summary = _surprise_samplers(
            config,
            cast(GameSplit | JoinedGameSplit, train),
            puzzle_pools,
            0 if puzzle_train is None else len(puzzle_train),
            generator,
        )

    run_info: dict[str, object] = {
        "schema": "wombat.puzzle-training.v1",
        "parameters": count_parameters(model),
        "train_positions": len(train.trainable())
        if isinstance(train, GameSplit | JoinedGameSplit)
        else len(train),
        "validation_positions": len(validation),
        "puzzle_train_positions": {str(moves): len(pool) for moves, pool in puzzle_pools.items()},
        "puzzle_validation_positions": {
            f"r{name}": len(members) for name, members in puzzle_eval_groups.items()
        },
        "gpu_priority_class": gpu_priority_class() if device.type == "cuda" else None,
        "training_config": asdict(config),
        "model_config": asdict(model_config),
    }
    if surprise_summary is not None:
        run_info["surprise"] = surprise_summary
    (output / "config.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info), flush=True)

    if isinstance(train, GameSplit | JoinedGameSplit):
        # Game targets are densified on the device from the few scored moves per position.
        sparse: Iterator[dict[str, np.ndarray]] = (
            train.shuffled_batches(config.batch_size, generator, sparse=True)
            if game_sampler is None
            else (train.sparse_batch(chosen) for chosen in game_sampler)
        )
        if config.child_value_share:
            share = config.child_value_share
            sparse = (with_children(item, share, generator) for item in sparse)
        batches: Iterator[dict[str, torch.Tensor]] = (
            densify(item, config.temperature, device) for item in sparse
        )
    else:
        batches = train.shuffled_batches(config.batch_size, generator)

    def puzzle_stream(moves: int, pool: np.ndarray) -> Iterator[dict[str, torch.Tensor]]:
        split = cast(PuzzleSplit, puzzle_train)
        if moves in puzzle_samplers:
            return (split.batch(chosen, lines=workspace) for chosen in puzzle_samplers[moves])
        return split.shuffled_batches(config.batch_size, generator, pool, lines=workspace)

    puzzle_batches = {
        moves: _on_device(puzzle_stream(moves, pool), device)
        for moves, pool in puzzle_pools.items()
    }
    model.train()
    # Evaluation, snapshots and checkpoints use the averaged weights when there are any.
    served = average or model
    trainer = predictor = None
    if graphed:
        # One pool for training and evaluation graphs keeps VRAM at the larger of the two.
        pool = torch.cuda.graph_pool_handle()
        trainer = GraphedTrainer(
            model,
            optimizer,
            batch_size=config.batch_size,
            max_tail=_max_tail(config),
            device=device,
            weights=weights,
            every_step=config.supervise_every_step,
            recompute=config.recompute_steps,
            initial_noise=config.initial_noise,
            pool=pool,
        )
        # Training-sized batches let evaluation graphs reuse blocks freed by training graphs.
        predictor = GraphedPredictor(served, config.batch_size, device, pool=pool)
    window = {"loss": 0.0, "correct": 0.0, "prefix": 0, "tail": 0, "count": 0, "puzzle": 0}
    # Workspace core: the budget drawn and the line losses of ``line_loss``, summed.
    line_window: dict[str, float] = {}
    started = time.perf_counter()
    last = _last_step(config)
    while step < last:
        rate = _learning_rate(config, step)
        lines = plies = 0
        follow = False
        if workspace:
            # The prefix column logs the surplus steps and the tail the exact budget.
            remaining, lines, plies, prefix, follow = _draw_line(config, randomizer, step)
            tail = lines * plies * model_config.steps_per_ply
        else:
            remaining, prefix, tail = _draw(config, randomizer, step)
        batch = next(batches) if remaining is None else next(puzzle_batches[remaining])
        if trainer is not None:
            cast(torch.Tensor, learning_rate).fill_(rate)
            trainer.load(batch)
            trainer.run(prefix, tail)
        else:
            for group in optimizer.param_groups:
                group["lr"] = rate
            if workspace:
                loss, correct, details = _line_loss(
                    model,
                    batch,
                    (lines, plies, prefix),
                    follow,
                    device,
                    weights,
                    line,
                    recompute=config.recompute_steps,
                )
                drawn = {"lines": lines, "plies": plies, "follow_reference": float(follow)}
                drawn.update({key: float(value) for key, value in details.items()})
                for key, value in drawn.items():
                    line_window[key] = line_window.get(key, 0.0) + value
            else:
                loss, correct = _loss(
                    model,
                    batch,
                    prefix,
                    tail,
                    device,
                    weights,
                    every_step=config.supervise_every_step,
                    recompute=config.recompute_steps,
                    initial_noise=config.initial_noise,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            window["loss"] += float(loss.detach())
            window["correct"] += float(correct)
        if average is not None and config.ema_decay is not None:
            _update_average(average, model, config.ema_decay)
        step += 1
        window["prefix"] += prefix
        window["tail"] += tail
        window["count"] += 1
        window["puzzle"] += remaining is not None
        if config.snapshot_every and step % config.snapshot_every == 0:
            torch.save(
                {"model": served.state_dict(), "model_config": asdict(model_config), "step": step},
                output / f"snapshot-step-{step:06d}.pt",
            )
        if step % config.eval_every == 0 or step == last:
            if trainer is not None:
                window["loss"], window["correct"] = trainer.drain_totals()
            count = window["count"]
            row: dict[str, object] = {
                "step": step,
                "learning_rate": _learning_rate(config, step - 1),
                "train_loss": window["loss"] / count,
                "train_accuracy": window["correct"] / count,
                "mean_prefix": window["prefix"] / count,
                "mean_tail": window["tail"] / count,
                "positions_per_second": count * config.batch_size / (time.perf_counter() - started),
                "validation": _evaluate(
                    served,
                    validation,
                    eval_indices,
                    config.eval_depths,
                    device,
                    predictor,
                ),
            }
            if config.puzzle_share:
                row["puzzle_batch_share"] = window["puzzle"] / count
            if line_window:
                row["line"] = {key: value / count for key, value in line_window.items()}
                line_window = {}
            if game_sampler is not None:
                # Mean raw surprise of the training positions drawn since the last row.
                row["surprise_drawn"] = {
                    "source": config.surprise,
                    "games": game_sampler.drain_mean(),
                    **{f"r{moves}": item.drain_mean() for moves, item in puzzle_samplers.items()},
                }
            if puzzle_validation is not None:
                row["puzzle_validation"] = _evaluate_puzzle_groups(
                    served,
                    puzzle_validation,
                    puzzle_eval_groups,
                    config.eval_depths,
                    device,
                    predictor,
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            _save(
                checkpoint_path,
                model,
                average,
                optimizer,
                config,
                model_config,
                step,
                randomizer,
                generator,
            )
            window = {"loss": 0.0, "correct": 0.0, "prefix": 0, "tail": 0, "count": 0, "puzzle": 0}
            started = time.perf_counter()
    return {"checkpoint": str(checkpoint_path), **run_info}
