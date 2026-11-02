"""Accuracy-versus-depth reports on held-out puzzle corpus splits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import math
from pathlib import Path
from typing import Protocol

import chess
import numpy as np
import torch

from wombat.chess.encode import decode_position, encode_position, state_vector
from wombat.chess.moves import action_to_move
from wombat.cuda_graphs import GraphedPredictor
from wombat.data.columns import ragged_entries
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit, lower_priority
from wombat.gpu_share import share_gpu
from wombat.model import NetworkConfig, WombatNetwork
from wombat.selection import (
    CHILD_MODES,
    CHILD_TOP_K,
    MIX_BETA,
    child_choice,
    require_move_values,
    selection_scores,
)


class PositionBatches(Protocol):
    """Anything that serves pieces, state and legal_mask tensors for chosen positions."""

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]: ...


RATING_BUCKETS = ((0, 1500), (1500, 2000), (2000, 2500), (2500, 10_000))
# Solver moves remaining, inclusive bounds; v10 trains on 1 and 2 and tests on the rest.
REMAINING_GROUPS = {"1": (1, 1), "2": (2, 2), "3": (3, 3), "4": (4, 4), "5+": (5, 127)}
UNTRAINED_GROUPS = ("3", "4", "5+")
PUZZLE_DEPTHS = (1, 2, 4, 8, 12, 16, 24, 32)


def remaining_groups(remaining: np.ndarray) -> dict[str, np.ndarray]:
    """Positions in each solver-moves-remaining group, in corpus order."""

    remaining = np.asarray(remaining)
    return {
        name: np.flatnonzero((remaining >= low) & (remaining <= high))
        for name, (low, high) in REMAINING_GROUPS.items()
    }


def sample_groups(groups: dict[str, np.ndarray], size: int, seed: int) -> dict[str, np.ndarray]:
    """At most ``size`` positions of each group, drawn without replacement, in corpus order."""

    generator = np.random.default_rng(seed)
    return {
        name: members
        if len(members) <= size
        else np.sort(generator.choice(members, size, replace=False))
        for name, members in groups.items()
    }


@torch.inference_mode()
def predict_depths(
    model: WombatNetwork,
    split: PositionBatches,
    indices: np.ndarray,
    depths: Sequence[int],
    *,
    batch_size: int = 512,
    device: torch.device,
    predictor: GraphedPredictor | None = None,
    graphs: bool = True,
    select: str = "policy",
    beta: float = MIX_BETA,
    top_k: int = CHILD_TOP_K,
) -> dict[int, np.ndarray]:
    """Predicted action per position at every requested depth, from one recurrent pass.

    On CUDA the pass replays captured graphs, building a predictor when none is given.
    ``select`` and ``beta`` choose the move as ``wombat.selection.selection_scores`` does;
    the child modes read the ``top_k`` children eagerly, as ``_predict_by_children`` does.
    """

    model.eval()
    if select in CHILD_MODES:
        if predictor is not None:
            raise ValueError("the predictor was captured for a different move selection")
        return _predict_by_children(
            model,
            split,
            indices,
            depths,
            batch_size=batch_size,
            device=device,
            select=select,
            beta=beta,
            top_k=top_k,
        )
    # The workspace core plays its own line with data-dependent legal moves, so it runs eagerly.
    graphs = graphs and model.config.core == "recurrent"
    if predictor is None and graphs and device.type == "cuda":
        predictor = GraphedPredictor(model, batch_size, device, select=select, beta=beta)
    if predictor is not None and (
        predictor.select != select or (select == "mix" and predictor.beta != beta)
    ):
        raise ValueError("the predictor was captured for a different move selection")
    values = select != "policy"
    wanted = frozenset(depths)
    predictions: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    for start in range(0, len(indices), batch_size):
        batch = split.batch(indices[start : start + batch_size])
        if predictor is not None:
            for depth, actions in predictor.predict(batch, depths).items():
                predictions[depth].append(actions)
            continue
        mask = batch["legal_mask"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(
                batch["pieces"].to(device),
                batch["state"].to(device),
                max(depths),
                intermediate_steps=wanted,
                include_move_values=values,
            )
        assert isinstance(outputs, dict)
        for depth in depths:
            readout = outputs[depth]
            scores = selection_scores(
                readout[0], readout[2] if values else None, mask, select, beta
            )
            predictions[depth].append(scores.argmax(dim=1).cpu().numpy())
    return {depth: np.concatenate(items) for depth, items in predictions.items()}


def _readouts(
    model: WombatNetwork,
    pieces: torch.Tensor,
    state: torch.Tensor,
    depths: Sequence[int],
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, ...]]:
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(
            pieces.to(device),
            state.to(device),
            max(depths),
            intermediate_steps=frozenset(depths),
        )
    assert isinstance(outputs, dict)
    return outputs


def _root_boards(
    split: PositionBatches, indices: np.ndarray, batch: dict[str, torch.Tensor]
) -> list[chess.Board]:
    """The batch's boards, from the split's FENs when it keeps them."""

    fens = getattr(split, "fens", None)
    if fens is not None:
        return [chess.Board(str(fens[index])) for index in indices]
    pairs = zip(batch["pieces"].numpy(), batch["state"].numpy())
    return [decode_position(pieces, state) for pieces, state in pairs]


def _child_values(
    model: WombatNetwork,
    children: Sequence[chess.Board],
    depths: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Each child's value for its side to move at every depth; mated -1, other ends 0."""

    values = {depth: np.zeros(len(children), dtype=np.float32) for depth in depths}
    live: list[int] = []
    for row, board in enumerate(children):
        if board.is_checkmate():
            for depth in depths:
                values[depth][row] = -1.0
        elif not board.is_game_over():
            live.append(row)
    for start in range(0, len(live), batch_size):
        rows = live[start : start + batch_size]
        encoded = [encode_position(children[row]) for row in rows]
        pieces = torch.from_numpy(np.stack([item.pieces for item in encoded]).astype(np.int64))
        state = torch.from_numpy(np.stack([state_vector(item) for item in encoded]))
        outputs = _readouts(model, pieces, state, depths, device)
        for depth in depths:
            values[depth][rows] = outputs[depth][1].float().cpu().numpy()
    return values


def _predict_by_children(
    model: WombatNetwork,
    split: PositionBatches,
    indices: np.ndarray,
    depths: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    select: str,
    beta: float,
    top_k: int,
) -> dict[int, np.ndarray]:
    """Actions chosen among the policy's top ``top_k`` moves by their children's values.

    Each child is read at the depth whose policy proposed it; a batch's children share passes.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    predictions: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch = split.batch(chunk)
        boards = _root_boards(split, chunk, batch)
        mask = batch["legal_mask"].to(device)
        outputs = _readouts(model, batch["pieces"], batch["state"], depths, device)
        slots: dict[tuple[int, int], int] = {}
        children: list[chess.Board] = []
        candidates: dict[int, tuple[torch.Tensor, np.ndarray, np.ndarray]] = {}
        for depth in depths:
            logits = outputs[depth][0].float().masked_fill(~mask, float("-inf"))
            actions = logits.topk(min(top_k, logits.shape[1]), dim=1).indices
            log_probs = logits.log_softmax(dim=1).gather(1, actions).cpu()
            actions_np = actions.cpu().numpy()
            child = np.zeros_like(actions_np)
            for row, column in np.argwhere(np.isfinite(log_probs.numpy())):
                key = (int(row), int(actions_np[row, column]))
                if key not in slots:
                    slots[key] = len(children)
                    board = boards[key[0]].copy(stack=False)
                    board.push(action_to_move(board, key[1]))
                    children.append(board)
                child[row, column] = slots[key]
            candidates[depth] = (log_probs, actions_np, child)
        values = _child_values(model, children, depths, batch_size=batch_size, device=device)
        for depth, (log_probs, actions_np, child) in candidates.items():
            scores = torch.from_numpy(-values[depth][child])
            column = child_choice(log_probs, scores, select, beta).numpy()
            predictions[depth].append(actions_np[np.arange(len(chunk)), column])
    return {depth: np.concatenate(items) for depth, items in predictions.items()}


def score_predictions(split: PuzzleSplit, indices: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Exact solution moves, plus any other move that delivers checkmate when FENs exist."""

    correct = predicted == split.action[indices]
    if split.fens is None:
        return correct
    for row in np.flatnonzero(~correct):
        key = (int(indices[row]), int(predicted[row]))
        if key not in split.mate_cache:
            board = chess.Board(str(split.fens[key[0]]))
            board.push(action_to_move(board, key[1]))
            split.mate_cache[key] = board.is_checkmate()
        correct[row] = split.mate_cache[key]
    return correct


def candidate_quality(
    split: GameSplit, indices: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Stockfish-free move quality from each position's scored candidates.

    A move outside the candidates is only known to be no better than the worst
    one, so its win-chance loss is a lower bound.
    """

    rows, actions = ragged_entries(split.candidate_offsets, split.candidate_actions, indices)
    _, wins = ragged_entries(split.candidate_offsets, split.candidate_win, indices)
    wins = wins.astype(np.float64) * 100
    best = np.full(len(indices), -np.inf)
    worst = np.full(len(indices), np.inf)
    np.maximum.at(best, rows, wins)
    np.minimum.at(worst, rows, wins)
    chosen = np.full(len(indices), np.nan)
    hit = actions.astype(np.int64) == predicted[rows]
    chosen[rows[hit]] = wins[hit]
    found = ~np.isnan(chosen)
    lost = np.where(found, best - chosen, best - worst)
    return {
        "good": float((found & (lost < 5)).mean()),
        "in_candidates": float(found.mean()),
        "mean_lost_lower_bound": float(lost.mean()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    return (centre - spread / denominator, centre + spread / denominator)


def paired_difference(
    first: np.ndarray, second: np.ndarray, z: float = 1.96
) -> dict[str, float | int]:
    """Accuracy of ``second`` minus ``first`` on the same positions, with a normal CI."""

    if first.shape != second.shape:
        raise ValueError("paired comparisons need identical positions")
    total = len(first)
    gained = int((second & ~first).sum())
    lost = int((first & ~second).sum())
    difference = (gained - lost) / total
    variance = (gained + lost - (gained - lost) ** 2 / total) / total**2
    spread = z * math.sqrt(max(variance, 0.0))
    return {
        "difference": difference,
        "low": difference - spread,
        "high": difference + spread,
        "gained": gained,
        "lost": lost,
        "positions": total,
    }


def _accuracy(correct: np.ndarray) -> dict[str, float | int]:
    successes = int(correct.sum())
    low, high = wilson_interval(successes, len(correct))
    return {
        "accuracy": successes / max(1, len(correct)),
        "low": low,
        "high": high,
        "correct": successes,
        "positions": len(correct),
    }


def _groups(split: PuzzleSplit, indices: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    rating = split.rating[indices]
    remaining = split.remaining[indices]
    return {
        "rating": {
            f"{low}-{high - 1}" if high < 10_000 else f"{low}+": (rating >= low) & (rating < high)
            for low, high in RATING_BUCKETS
        },
        "remaining": {
            "1": remaining == 1,
            "2": remaining == 2,
            "3": remaining == 3,
            "4+": remaining >= 4,
        },
    }


def full_line_solved(split: PuzzleSplit, indices: np.ndarray, correct: np.ndarray) -> float:
    """Fraction of puzzles whose every solver move is correct, as Lichess scores them."""

    puzzles = split.puzzle[indices]
    failed = np.zeros(int(puzzles.max()) + 1, dtype=np.bool_)
    np.logical_or.at(failed, puzzles, ~correct)
    present = np.unique(puzzles)
    return float((~failed[present]).mean())


def curve_report(
    split: PuzzleSplit,
    indices: np.ndarray,
    correctness: dict[int, np.ndarray],
    *,
    baseline_depth: int,
    whole_puzzles: bool,
) -> list[dict[str, object]]:
    groups = _groups(split, indices)
    baseline = correctness[baseline_depth]
    rows: list[dict[str, object]] = []
    for depth, correct in sorted(correctness.items()):
        row: dict[str, object] = {
            "reasoning_steps": depth,
            **_accuracy(correct),
            f"versus_t{baseline_depth}": paired_difference(baseline, correct),
        }
        for family, members in groups.items():
            row[family] = {
                name: {
                    **_accuracy(correct[member]),
                    f"versus_t{baseline_depth}": paired_difference(
                        baseline[member], correct[member]
                    ),
                }
                for name, member in members.items()
                if member.any()
            }
        if whole_puzzles:
            row["full_line_solved"] = full_line_solved(split, indices, correct)
        rows.append(row)
    return rows


def load_checkpoint(
    path: Path, device: torch.device, select: str = "policy", *, line_plies: int | None = None
) -> WombatNetwork:
    """The checkpoint's network; value and mix ``select`` also need trained move values.

    ``line_plies`` replaces the plies per line a workspace core runs its steps in.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = dict(checkpoint["model_config"])
    if line_plies is not None:
        if settings.get("core") != "workspace":
            raise ValueError("line_plies applies to the workspace core")
        settings["line_plies"] = line_plies
    model = WombatNetwork(NetworkConfig(**settings))
    require_move_values(checkpoint, path, model.state_dict(), select)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def workspace_line_plies(model: WombatNetwork) -> int | None:
    """Plies per line a workspace core ran its steps in, None for the recurrent core."""

    return model.config.line_plies if model.config.core == "workspace" else None


def evaluate_corpus_curve(
    checkpoint: Path,
    split_path: Path,
    depths: Iterable[int],
    *,
    output: Path,
    limit: int | None = None,
    baseline_depth: int = 1,
    batch_size: int = 512,
    device_name: str = "cuda",
    max_vram_gb: float | None = 3.5,
) -> dict[str, object]:
    """Evaluate one checkpoint at every depth; also store per-position correctness."""

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    depths = tuple(sorted(set(depths) | {baseline_depth}))
    model = load_checkpoint(checkpoint, device)
    split = PuzzleSplit(split_path)
    indices = np.arange(len(split) if limit is None else min(limit, len(split)))
    predictions = predict_depths(
        model, split, indices, depths, batch_size=batch_size, device=device
    )
    correctness = {
        depth: score_predictions(split, indices, predicted)
        for depth, predicted in predictions.items()
    }
    report: dict[str, object] = {
        "schema": "wombat.puzzle-corpus-curve.v1",
        "checkpoint": str(checkpoint),
        "split": str(split_path),
        "positions": len(indices),
        "curve": curve_report(
            split,
            indices,
            correctness,
            baseline_depth=baseline_depth,
            whole_puzzles=limit is None,
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    arrays = {"indices": indices, **{f"t{depth}": row for depth, row in correctness.items()}}
    np.savez(output / "correctness.npz", **arrays)  # type: ignore[arg-type]
    return report


def _group_report(
    correctness: dict[int, np.ndarray], rows: np.ndarray, baseline_depth: int
) -> dict[str, object]:
    baseline = correctness[baseline_depth][rows]
    versus = f"versus_t{baseline_depth}"
    depths: dict[str, object] = {}
    best: tuple[float, int] | None = None
    for depth, correct in sorted(correctness.items()):
        accuracy = _accuracy(correct[rows])
        depths[f"t{depth}"] = {**accuracy, versus: paired_difference(baseline, correct[rows])}
        if depth > baseline_depth and (best is None or accuracy["accuracy"] > best[0]):
            best = (float(accuracy["accuracy"]), depth)
    report: dict[str, object] = {"positions": len(rows), "depths": depths}
    if best is not None:
        chosen = correctness[best[1]][rows]
        report[f"best_above_t{baseline_depth}"] = {
            "reasoning_steps": best[1],
            **paired_difference(baseline, chosen),
        }
    return report


def evaluate_puzzle_depth(
    checkpoint: Path,
    split_path: Path,
    depths: Iterable[int] = PUZZLE_DEPTHS,
    *,
    output: Path,
    per_group: int | None = None,
    baseline_depth: int = 8,
    seed: int = 0,
    batch_size: int = 512,
    device_name: str = "cuda",
    max_vram_gb: float | None = 3.5,
    select: str = "policy",
    beta: float = MIX_BETA,
) -> dict[str, object]:
    """First-move accuracy by depth per solver-moves-remaining group, each depth paired with T8.

    By default groups 3, 4 and 5+ are complete and every group is sampled (seed 0) down to
    the largest of them. "3+" pools those three. A model without recurrent blocks is read
    out once and that answer stands for every depth. ``select`` and ``beta`` choose moves.
    """

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    depths = tuple(sorted(set(depths) | {baseline_depth}))
    model = load_checkpoint(checkpoint, device, select)
    split = PuzzleSplit(split_path)
    available = remaining_groups(split.remaining)
    if per_group is None:
        per_group = max(len(available[name]) for name in UNTRAINED_GROUPS)
    chosen = sample_groups(available, per_group, seed)
    indices = np.concatenate(list(chosen.values()))
    if not len(indices):
        raise ValueError("no positions to evaluate")
    recurrent = model.config.recurrent_blocks > 0
    predictions = predict_depths(
        model,
        split,
        indices,
        depths if recurrent else (0,),
        batch_size=batch_size,
        device=device,
        select=select,
        beta=beta,
    )
    if recurrent:
        correctness = {
            depth: score_predictions(split, indices, predicted)
            for depth, predicted in predictions.items()
        }
    else:
        once = score_predictions(split, indices, predictions[0])
        correctness = {depth: once for depth in depths}
    ends = np.cumsum([len(members) for members in chosen.values()])
    rows = {name: np.arange(end - len(chosen[name]), end) for name, end in zip(chosen, ends)}
    rows["3+"] = np.concatenate([rows[name] for name in UNTRAINED_GROUPS])
    report: dict[str, object] = {
        "schema": "wombat.puzzle-depth.v1",
        "checkpoint": str(checkpoint),
        "split": str(split_path),
        "recurrent": recurrent,
        "depths": list(depths),
        "baseline_depth": baseline_depth,
        "per_group": per_group,
        "seed": seed,
        "select": select,
        "select_beta": beta if select == "mix" else None,
        "available": {
            **{name: len(members) for name, members in available.items()},
            "3+": sum(len(available[name]) for name in UNTRAINED_GROUPS),
        },
        "positions": len(indices),
        "groups": {
            name: _group_report(correctness, members, baseline_depth)
            for name, members in rows.items()
            if len(members)
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    arrays = {
        "indices": indices,
        "remaining": np.asarray(split.remaining[indices]),
        **{f"t{depth}": correct for depth, correct in correctness.items()},
    }
    np.savez(output / "correctness.npz", **arrays)  # type: ignore[arg-type]
    return report


def compare_correctness(
    first: Path, first_depth: int, second: Path, second_depth: int
) -> dict[str, float | int]:
    """Paired accuracy difference between two stored evaluations of the same positions."""

    with np.load(first) as left, np.load(second) as right:
        if not np.array_equal(left["indices"], right["indices"]):
            raise ValueError("evaluations cover different positions")
        return paired_difference(left[f"t{first_depth}"], right[f"t{second_depth}"])
