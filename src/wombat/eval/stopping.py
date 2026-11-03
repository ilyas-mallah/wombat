"""Stopping rules for the recurrent steps, judged without training (backlog item 15 in docs/plan.md).

Every position is stepped once to ``MAX_STEPS``, keeping the legal argmax and the change of the
policy after each step. A rule picks, per position, the step whose argmax is played. Games are
scored by the evaluate-strength judge and cache, whole puzzles as ``evaluate-full-puzzles``
scores them. Rules with free settings are tuned on the even rows of each set (positions of the
strength sample, puzzles of the CSV) and reported on the odd rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from typing import cast

import chess
import numpy as np
import torch

from wombat.chess.moves import action_to_move
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import lower_priority, solver_positions
from wombat.eval.full_puzzles import EncodedPositions, read_puzzle_csv
from wombat.eval.puzzle_corpus_eval import (
    PositionBatches,
    load_checkpoint,
    paired_difference,
    wilson_interval,
)
from wombat.eval.strength import (
    BLUNDER_FROM,
    StockfishJudge,
    _paired_mean,
    _summary,
    move_accuracy,
    valid_sample,
)
from wombat.gpu_share import share_gpu
from wombat.model import WombatNetwork
from wombat.model.network import board_consequences, board_phase

MAX_STEPS = 32
FIXED_STEPS = (8, 12, 16)
AGREE_WINDOWS = (2, 3, 4, 6)
MIN_STEPS = (2, 4)
# Nats; the per-step KL of a settled policy is far below 0.01.
KL_THRESHOLDS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
KL_WINDOWS = (1, 2, 3)
GAME_BASELINE = 12
PUZZLE_BASELINE = 16


@dataclass(frozen=True)
class StoppingRule:
    """Where to stop, never past ``cap``; H0's readout counts as step 0.

    ``fixed`` stops at ``steps``. ``agree`` stops at the first step t >= ``minimum`` whose argmax
    equals the argmax after each of the ``window`` steps before it. ``kl`` stops at the first
    t >= ``minimum`` where KL(policy_t || policy_t-1) was below ``threshold`` for the last
    ``window`` steps. A position that never qualifies stops at ``cap``.
    """

    kind: str
    steps: int = 0
    window: int = 0
    minimum: int = 1
    threshold: float = 0.0
    cap: int = MAX_STEPS

    @property
    def name(self) -> str:
        if self.kind == "fixed":
            return f"t{self.steps}"
        if self.kind == "agree":
            return f"agree k{self.window} m{self.minimum}"
        if self.kind == "kl":
            return f"kl {self.threshold:g} k{self.window} m{self.minimum}"
        raise ValueError(f"unknown stopping rule {self.kind!r}")

    def stop(self, actions: np.ndarray, kl: np.ndarray) -> np.ndarray:
        """Chosen step per row of (positions, steps + 1) argmax actions and KL columns."""

        if self.kind == "fixed":
            return np.full(len(actions), self.steps, dtype=np.int64)
        if self.kind == "agree":
            settled = np.zeros(actions.shape, dtype=np.bool_)
            settled[:, 1:] = actions[:, 1:] == actions[:, :-1]
        elif self.kind == "kl":
            settled = np.zeros(kl.shape, dtype=np.bool_)
            settled[:, 1:] = kl[:, 1:] < self.threshold
        else:
            raise ValueError(f"unknown stopping rule {self.kind!r}")
        return first_step(run_lengths(settled) >= self.window, self.minimum, self.cap)


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """Consecutive True columns ending at each column, per row."""

    runs = np.zeros(flags.shape, dtype=np.int64)
    for column in range(flags.shape[1]):
        previous = runs[:, column - 1] if column else 0
        runs[:, column] = np.where(flags[:, column], previous + 1, 0)
    return runs


def first_step(condition: np.ndarray, minimum: int, cap: int) -> np.ndarray:
    """First column in ``minimum..cap`` where ``condition`` holds, else ``cap``."""

    allowed = condition.copy()
    allowed[:, :minimum] = False
    allowed[:, cap + 1 :] = False
    return np.where(allowed.any(axis=1), allowed.argmax(axis=1), cap)


def candidate_rules(max_steps: int = MAX_STEPS) -> dict[str, list[StoppingRule]]:
    """Every rule by family; ``agree`` and ``kl`` are tuned within their family."""

    return {
        "fixed": [StoppingRule("fixed", steps=steps) for steps in FIXED_STEPS],
        "agree": [
            StoppingRule("agree", window=window, minimum=minimum, cap=max_steps)
            for window in AGREE_WINDOWS
            for minimum in MIN_STEPS
        ],
        "kl": [
            StoppingRule("kl", window=window, minimum=minimum, threshold=threshold, cap=max_steps)
            for threshold in KL_THRESHOLDS
            for window in KL_WINDOWS
            for minimum in MIN_STEPS
        ],
    }


def halves(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Tuning rows (even) and test rows (odd)."""

    rows = np.arange(count)
    return rows[0::2], rows[1::2]


def game_oracle_steps(lost: np.ndarray, cap: int = MAX_STEPS) -> np.ndarray:
    """Earliest step in 1..cap whose move loses the least win chance."""

    return 1 + np.argmin(lost[:, 1 : cap + 1], axis=1)


def puzzle_oracle_steps(match: np.ndarray, mate: np.ndarray, cap: int = MAX_STEPS) -> np.ndarray:
    """Earliest step in 1..cap that mates, else earliest that plays the solution, else ``cap``."""

    return np.where(
        mate[:, 1 : cap + 1].any(axis=1),
        first_step(mate, 1, cap),
        first_step(match, 1, cap),
    )


@torch.inference_mode()
def step_trajectories(
    model: WombatNetwork,
    positions: PositionBatches,
    indices: np.ndarray,
    max_steps: int = MAX_STEPS,
    *,
    batch_size: int = 512,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Legal argmax action after steps 0..max_steps and KL(policy_t || policy_t-1) per step.

    The KL is over the legal moves; column 0 is 0 because H0 has no previous policy.
    """

    model.eval()
    actions = np.zeros((len(indices), max_steps + 1), dtype=np.int64)
    kl = np.zeros((len(indices), max_steps + 1), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        rows = slice(start, start + batch_size)
        batch = positions.batch(indices[rows])
        legal = batch["legal_mask"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            memory = model.encode_board(batch["pieces"].to(device), batch["state"].to(device))
            consequences, phase = board_consequences(memory), board_phase(memory)
            hidden = model.initial_hidden(memory)
            previous: torch.Tensor | None = None
            step_actions, step_kl = [], []
            for step in range(max_steps + 1):
                if step:
                    hidden = cast(torch.Tensor, model.recurrent_step(memory, hidden))
                policy = model.heads(
                    hidden, memory.shape[1], consequences=consequences, phase=phase
                )[0]
                log_policy = policy.float().masked_fill(~legal, float("-inf")).log_softmax(-1)
                step_actions.append(log_policy.argmax(dim=1))
                if previous is None:
                    step_kl.append(torch.zeros(len(legal), device=device))
                else:
                    change = log_policy.exp() * (log_policy - previous)
                    step_kl.append(torch.where(legal, change, 0.0).sum(dim=1))
                previous = log_policy
        actions[rows] = torch.stack(step_actions, dim=1).cpu().numpy()
        kl[rows] = torch.stack(step_kl, dim=1).cpu().numpy()
    return actions, kl


def game_costs(
    fens: Sequence[str], actions: np.ndarray, judge: StockfishJudge, *, workers: int
) -> np.ndarray:
    """Win chance lost by the argmax after each step 1..; NaN at step 0.

    Each distinct move of a position is judged once.
    """

    pairs = sorted({(row, int(action)) for row in range(len(fens)) for action in actions[row, 1:]})

    def cost(pair: tuple[int, int]) -> float:
        board = chess.Board(fens[pair[0]])
        return judge.lost(board, action_to_move(board, pair[1]))

    with ThreadPoolExecutor(workers) as executor:
        costs = dict(zip(pairs, executor.map(cost, pairs)))
    lost = np.full(actions.shape, np.nan)
    for row in range(len(fens)):
        for step in range(1, actions.shape[1]):
            lost[row, step] = costs[(row, int(actions[row, step]))]
    return lost


def puzzle_outcomes(
    boards: Sequence[chess.Board], solutions: Sequence[chess.Move], actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per position and step: the solution move was played, or another move that mates."""

    match = np.zeros(actions.shape, dtype=np.bool_)
    mate = np.zeros(actions.shape, dtype=np.bool_)
    for row, (board, solution) in enumerate(zip(boards, solutions)):
        verdicts: dict[int, tuple[bool, bool]] = {}
        for step, action in enumerate(actions[row]):
            action = int(action)
            if action not in verdicts:
                move = action_to_move(board, action)
                after = board.copy(stack=False)
                after.push(move)
                verdicts[action] = (move == solution, move != solution and after.is_checkmate())
            match[row, step], mate[row, step] = verdicts[action]
    return match, mate


def lines_solved(
    match: np.ndarray, mate: np.ndarray, starts: np.ndarray, steps: np.ndarray
) -> np.ndarray:
    """Whole-puzzle verdicts as ``line_verdict`` gives them, for one chosen step per position.

    ``starts`` holds each puzzle's first position; a puzzle's positions are contiguous.
    """

    rows = np.arange(len(steps))
    matched = match[rows, steps]
    missed = np.where(matched, len(steps), rows)
    first_miss = np.minimum.reduceat(missed, starts)
    ends = np.append(starts[1:], len(steps))
    mated = mate[rows, steps]
    return (first_miss >= ends) | mated[np.minimum(first_miss, len(steps) - 1)]


def _accuracy_values(lost: np.ndarray) -> np.ndarray:
    return np.array([move_accuracy(value) for value in lost])


def _game_row(lost: np.ndarray, steps: np.ndarray, rows: np.ndarray, baseline: int) -> dict:
    chosen = lost[rows, steps[rows]]
    base = lost[rows, baseline]
    return {
        **_summary(chosen),
        "mean_steps": float(steps[rows].mean()),
        f"versus_t{baseline}": {
            "lichess_accuracy": _paired_mean(_accuracy_values(base), _accuracy_values(chosen)),
            "blunder_rate": _paired_mean(base >= BLUNDER_FROM, chosen >= BLUNDER_FROM),
        },
        "positions": len(rows),
    }


@dataclass(frozen=True)
class _PuzzleSet:
    match: np.ndarray
    mate: np.ndarray
    starts: np.ndarray
    line: np.ndarray  # Puzzle row of each position.


def _puzzle_row(puzzles: _PuzzleSet, steps: np.ndarray, rows: np.ndarray, baseline: int) -> dict:
    solved = lines_solved(puzzles.match, puzzles.mate, puzzles.starts, steps)[rows]
    base = lines_solved(puzzles.match, puzzles.mate, puzzles.starts, np.full(len(steps), baseline))[
        rows
    ]
    positions = np.isin(puzzles.line, rows)
    count = int(solved.sum())
    low, high = wilson_interval(count, len(rows))
    return {
        "accuracy": count / max(1, len(rows)),
        "low": low,
        "high": high,
        "solved": count,
        "puzzles": len(rows),
        "mean_steps": float(steps[positions].mean()),
        f"versus_t{baseline}": paired_difference(base, solved),
    }


def _tune(
    family: Sequence[StoppingRule],
    score: Callable[[StoppingRule], tuple[float, float]],
    floor: float,
) -> dict[str, StoppingRule | None]:
    """The most accurate rule (fewer steps on ties), and the cheapest at least ``floor``."""

    scored = [(rule, *score(rule)) for rule in family]
    best = max(scored, key=lambda item: (item[1], -item[2]))[0]
    enough = [item for item in scored if item[1] >= floor]
    cheapest = min(enough, key=lambda item: (item[2], -item[1]))[0] if enough else None
    return {"best": best, "cheapest": cheapest}


def rule_report(
    rules: dict[str, list[StoppingRule]],
    stops: Callable[[StoppingRule], np.ndarray],
    oracle: np.ndarray,
    row: Callable[[np.ndarray, np.ndarray], dict],
    count: int,
    baseline: int,
    metric: str,
) -> dict[str, object]:
    """Every rule on the tuning half, the test half and all rows, and the tuned choices."""

    tune_rows, test_rows = halves(count)
    parts = {"tune": tune_rows, "test": test_rows, "all": np.arange(count)}
    chosen = {rule.name: stops(rule) for family in rules.values() for rule in family}
    chosen["oracle"] = oracle
    table = {
        name: {part: row(steps, rows) for part, rows in parts.items()}
        for name, steps in chosen.items()
    }

    def score(rule: StoppingRule) -> tuple[float, float]:
        tuned = table[rule.name]["tune"]
        return float(tuned[metric]), float(tuned["mean_steps"])

    floor = float(table[f"t{baseline}"]["tune"][metric])
    tuned: dict[str, object] = {}
    for family in ("agree", "kl"):
        for goal, rule in _tune(rules[family], score, floor).items():
            entry = None
            if rule is not None:
                entry = {"rule": rule.name, **table[rule.name]}
            tuned[f"{family}_{goal}"] = entry
    return {
        "tune_rows": "even",
        "test_rows": "odd",
        "tune_count": len(tune_rows),
        "test_count": len(test_rows),
        "baseline": f"t{baseline}",
        "rules": table,
        "tuned": tuned,
    }


def held_out_summary(report: dict[str, object]) -> dict[str, dict[str, object]]:
    """Test-half rows of the fixed rules, the tuned rules and the oracle, trimmed for printing."""

    keys = ("lichess_accuracy", "blunder_rate", "accuracy", "mean_steps")
    results = report["results"]
    assert isinstance(results, dict)
    summary: dict[str, dict[str, object]] = {}
    for part, section in results.items():
        plain = [*(f"t{steps}" for steps in FIXED_STEPS), "oracle"]
        chosen = {name: section["rules"][name] for name in plain}
        chosen.update({name: entry for name, entry in section["tuned"].items() if entry})
        summary[part] = {}
        for name, entry in chosen.items():
            row = entry["test"]
            versus = row[f"versus_{section['baseline']}"]
            difference = versus.get("lichess_accuracy", versus)
            summary[part][name] = {
                "rule": entry.get("rule", name),
                **{key: row[key] for key in keys if key in row},
                "versus": [difference["difference"], difference["low"], difference["high"]],
            }
    return summary


def evaluate_stopping(
    checkpoint: Path,
    *,
    puzzles: Path,
    games: Path,
    output: Path,
    stockfish: Path,
    positions: int = 2_000,
    nodes: int = 20_000,
    engines: int = 4,
    cache_dir: Path = Path("runs/stockfish-cache"),
    max_steps: int = MAX_STEPS,
    batch_size: int = 512,
    device_name: str = "cpu",
    max_vram_gb: float | None = 1.0,
) -> dict[str, object]:
    """Stopping rules on the strength positions (paired with T12) and whole puzzles (with T16)."""

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    model = load_checkpoint(checkpoint, device)
    rules = candidate_rules(max_steps)
    trajectories = partial(
        step_trajectories,
        max_steps=max_steps,
        batch_size=batch_size,
        device=device,
    )

    game_split = GameSplit(games)
    if game_split.fens is None:
        raise ValueError("stopping rules on games need a split with FENs")
    game_indices = valid_sample(game_split, positions)
    game_actions, game_kl = trajectories(model, game_split, game_indices)
    fens = [str(game_split.fens[index]) for index in game_indices]
    with StockfishJudge(stockfish, nodes, cache_dir, engines=engines) as judge:
        lost = game_costs(fens, game_actions, judge, workers=engines)
    games_report = rule_report(
        rules,
        lambda rule: rule.stop(game_actions, game_kl),
        game_oracle_steps(lost, max_steps),
        lambda steps, rows: _game_row(lost, steps, rows, GAME_BASELINE),
        len(game_indices),
        GAME_BASELINE,
        "lichess_accuracy",
    )

    records = read_puzzle_csv(puzzles)
    lines = [solver_positions(record) for record in records]
    flat = [position for line in lines for position in line]
    boards = [position.board for position in flat]
    puzzle_actions, puzzle_kl = trajectories(model, EncodedPositions(boards), np.arange(len(flat)))
    match, mate = puzzle_outcomes(boards, [position.move for position in flat], puzzle_actions)
    lengths = np.array([len(line) for line in lines])
    puzzle_set = _PuzzleSet(
        match, mate, np.cumsum(lengths) - lengths, np.repeat(np.arange(len(lines)), lengths)
    )
    puzzles_report = rule_report(
        rules,
        lambda rule: rule.stop(puzzle_actions, puzzle_kl),
        puzzle_oracle_steps(match, mate, max_steps),
        lambda steps, rows: _puzzle_row(puzzle_set, steps, rows, PUZZLE_BASELINE),
        len(lines),
        PUZZLE_BASELINE,
        "accuracy",
    )

    report: dict[str, object] = {
        "schema": "wombat.stopping.v1",
        "checkpoint": str(checkpoint),
        "games": str(games),
        "puzzles_csv": str(puzzles),
        "max_steps": max_steps,
        "nodes": nodes,
        "game_positions": len(game_indices),
        "puzzles": len(lines),
        "puzzle_positions": len(flat),
        "results": {"games": games_report, "puzzles": puzzles_report},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(
        output / "trajectories.npz",
        game_indices=game_indices,
        game_actions=game_actions,
        game_kl=game_kl,
        game_lost=lost,
        puzzle_actions=puzzle_actions,
        puzzle_kl=puzzle_kl,
        puzzle_match=match,
        puzzle_mate=mate,
        puzzle_lengths=lengths,
    )
    return report
