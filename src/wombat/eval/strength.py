"""Playing strength: Stockfish-judged move quality and whole-puzzle rating.

Moves are judged the way Lichess annotates games. Each Stockfish score becomes
a win chance for the side to move; a move's cost is the win chance it gives up
against Stockfish's best move.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import queue
import statistics
import threading
from typing import Self

import chess
import chess.engine
import numpy as np
import torch

from wombat.chess.moves import action_to_move
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit, lower_priority
from wombat.eval.puzzle_corpus_eval import (
    load_checkpoint,
    predict_depths,
    score_predictions,
    workspace_line_plies,
)
from wombat.selection import CHILD_MODES, CHILD_TOP_K, MIX_BETA, recorded_beta

GOOD_BELOW = 5.0
MISTAKE_FROM = 10.0
BLUNDER_FROM = 15.0


def win_percent(cp: float) -> float:
    """Lichess's centipawn-to-win-chance curve, 0..100 for the side to move."""

    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(lost: float) -> float:
    """Lichess per-move accuracy from win-chance points lost."""

    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * lost) - 3.1669))


def puzzle_rating(ratings: np.ndarray, solved: np.ndarray) -> tuple[float, float]:
    """Elo rating whose expected solves match, with its standard error."""

    target = min(max(float(solved.sum()), 0.5), len(solved) - 0.5)
    low, high = -500.0, 4000.0
    for _ in range(60):
        middle = (low + high) / 2
        expected = (1 / (1 + 10 ** ((ratings - middle) / 400))).sum()
        low, high = (middle, high) if expected < target else (low, middle)
    rating = (low + high) / 2
    probability = 1 / (1 + 10 ** ((ratings - rating) / 400))
    information = (probability * (1 - probability)).sum() * (math.log(10) / 400) ** 2
    return rating, 1 / math.sqrt(max(information, 1e-12))


class StockfishJudge:
    """Win chances from a pool of Stockfish processes, cached on disk.

    ``None`` as the move asks for the position's best move.
    """

    def __init__(self, executable: Path, nodes: int, cache_dir: Path, *, engines: int = 4) -> None:
        self.nodes = nodes
        self.cache_dir = cache_dir
        self.pool: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        for _ in range(engines):
            engine = chess.engine.SimpleEngine.popen_uci(str(executable))
            engine.configure({"Threads": 1})
            self.pool.put(engine)
        self.label = str(self._peek().id.get("name", "stockfish"))

    def _peek(self) -> chess.engine.SimpleEngine:
        engine = self.pool.get()
        self.pool.put(engine)
        return engine

    def close(self) -> None:
        while not self.pool.empty():
            self.pool.get().quit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _path(self, fen: str, move: str | None) -> Path:
        payload = json.dumps(
            {
                "schema": "wombat.win-judge.v1",
                "engine": self.label,
                "nodes": self.nodes,
                "fen": fen,
                "move": move,
            },
            sort_keys=True,
        )
        key = sha256(payload.encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def win(self, board: chess.Board, move: chess.Move | None) -> float:
        path = self._path(board.fen(), None if move is None else move.uci())
        if path.exists():
            try:
                return float(json.loads(path.read_text(encoding="utf-8"))["win"])
            except (ValueError, KeyError, OSError):
                pass  # A torn write from an older run; judge it again.
        engine = self.pool.get()
        try:
            info = engine.analyse(
                board,
                chess.engine.Limit(nodes=self.nodes),
                root_moves=None if move is None else [move],
            )
        finally:
            self.pool.put(engine)
        cp = info["score"].pov(board.turn).score(mate_score=100_000)
        win = win_percent(float(cp))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent reports share the cache, so readers must never see a partial file.
        pending = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        pending.write_text(json.dumps({"win": win}), encoding="utf-8")
        try:
            os.replace(pending, path)
        except PermissionError:
            pending.unlink(missing_ok=True)  # Another writer holds the same judgement.
        return win

    def lost(self, board: chess.Board, move: chess.Move) -> float:
        return max(0.0, self.win(board, None) - self.win(board, move))


def _paired_mean(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Mean of ``second - first`` with a normal 95% interval."""

    difference = second.astype(float) - first.astype(float)
    spread = 1.96 * float(difference.std(ddof=1)) / math.sqrt(len(difference))
    mean = float(difference.mean())
    return {"difference": mean, "low": mean - spread, "high": mean + spread}


def _summary(lost: np.ndarray) -> dict[str, float]:
    return {
        "lichess_accuracy": statistics.fmean(move_accuracy(value) for value in lost),
        "good_rate": float((lost < GOOD_BELOW).mean()),
        "mistake_rate": float(((lost >= MISTAKE_FROM) & (lost < BLUNDER_FROM)).mean()),
        "blunder_rate": float((lost >= BLUNDER_FROM).mean()),
        "mean_win_lost": float(lost.mean()),
    }


def judge_positions(
    split: GameSplit,
    indices: np.ndarray,
    predictions: dict[int, np.ndarray],
    judge: StockfishJudge,
    *,
    workers: int,
) -> dict[int, np.ndarray]:
    """Win chance lost by each predicted move, per depth."""

    if split.fens is None:
        raise ValueError("move quality needs a split with FENs")
    fens = split.fens

    def cost(depth: int, row: int) -> float:
        board = chess.Board(str(fens[indices[row]]))
        return judge.lost(board, action_to_move(board, int(predictions[depth][row])))

    lost: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(workers) as executor:
        for depth in predictions:
            costs = executor.map(partial(cost, depth), range(len(indices)))
            lost[depth] = np.array(list(costs))
    return lost


def valid_sample(split: GameSplit, count: int, seed: int = 0) -> np.ndarray:
    """First ``count`` legal-by-the-rules positions of a seeded shuffle, sorted.

    The archive holds a few impossible boards that Stockfish cannot analyse.
    """

    if split.fens is None:
        raise ValueError("sampling valid positions needs a split with FENs")
    chosen: list[int] = []
    for index in np.random.default_rng(seed).permutation(len(split)):
        if chess.Board(str(split.fens[index])).status() == chess.STATUS_VALID:
            chosen.append(int(index))
            if len(chosen) == count:
                break
    return np.sort(np.array(chosen, dtype=np.int64))


def whole_puzzles_solved(
    split: PuzzleSplit, indices: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solved flag and rating per puzzle among ``indices``, in puzzle order."""

    correct = score_predictions(split, indices, predicted)
    puzzles = np.asarray(split.puzzle)[indices]
    ids = np.unique(puzzles)
    solved = np.array([bool(correct[puzzles == puzzle].all()) for puzzle in ids])
    rating = np.asarray(split.rating)[indices]
    ratings = np.array([rating[puzzles == puzzle][0] for puzzle in ids], dtype=float)
    return solved, ratings


def evaluate_strength(
    checkpoint: Path,
    *,
    games: Path,
    puzzles: Path,
    output: Path,
    depths: Sequence[int],
    stockfish: Path,
    positions: int = 2_000,
    puzzle_count: int = 1_000,
    nodes: int = 20_000,
    engines: int = 4,
    cache_dir: Path = Path("runs/stockfish-cache"),
    device_name: str = "cpu",
    select: str = "policy",
    beta: float = MIX_BETA,
    top_k: int = CHILD_TOP_K,
    line_plies: int | None = None,
) -> dict[str, object]:
    """Registered v7 strength report for one checkpoint at every depth.

    ``select``, ``beta`` and ``top_k`` choose moves as ``predict_depths`` does; ``line_plies``
    as ``load_checkpoint`` takes it.
    """

    lower_priority()
    torch.set_num_threads(4)
    device = torch.device(device_name)
    depths = tuple(sorted(set(depths) | {1}))
    model = load_checkpoint(checkpoint, device, select, line_plies=line_plies)
    predict = partial(predict_depths, device=device, select=select, beta=beta, top_k=top_k)
    game_split = GameSplit(games)
    game_indices = valid_sample(game_split, positions)
    game_moves = predict(model, game_split, game_indices, depths)
    puzzle_split = PuzzleSplit(puzzles)
    chosen = np.random.default_rng(0).choice(
        int(np.asarray(puzzle_split.puzzle).max()) + 1, puzzle_count, replace=False
    )
    puzzle_indices = np.flatnonzero(np.isin(np.asarray(puzzle_split.puzzle), chosen))
    puzzle_moves = predict(model, puzzle_split, puzzle_indices, depths)
    with StockfishJudge(stockfish, nodes, cache_dir, engines=engines) as judge:
        lost = judge_positions(game_split, game_indices, game_moves, judge, workers=engines)
    curve: list[dict[str, object]] = []
    solved_by_depth: dict[int, np.ndarray] = {}
    for depth in depths:
        solved, ratings = whole_puzzles_solved(puzzle_split, puzzle_indices, puzzle_moves[depth])
        solved_by_depth[depth] = solved
        rating, error = puzzle_rating(ratings, solved)
        accuracy = np.array([move_accuracy(value) for value in lost[depth]])
        base = np.array([move_accuracy(value) for value in lost[1]])
        curve.append(
            {
                "reasoning_steps": depth,
                **_summary(lost[depth]),
                "accuracy_versus_t1": _paired_mean(base, accuracy),
                "good_versus_t1": _paired_mean(lost[1] < GOOD_BELOW, lost[depth] < GOOD_BELOW),
                "puzzle_rating": rating,
                "puzzle_rating_error": error,
                "puzzles_solved": float(solved.mean()),
            }
        )
    report: dict[str, object] = {
        "schema": "wombat.strength.v1",
        "checkpoint": str(checkpoint),
        "games": str(games),
        "puzzles": str(puzzles),
        "positions": len(game_indices),
        "puzzle_count": len(chosen),
        "nodes": nodes,
        "select": select,
        "select_beta": recorded_beta(select, beta),
        "select_top_k": top_k if select in CHILD_MODES else None,
        "line_plies": workspace_line_plies(model),
        "curve": curve,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "strength.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    arrays = {f"lost_t{depth}": values for depth, values in lost.items()}
    arrays.update({f"solved_t{depth}": values for depth, values in solved_by_depth.items()})
    np.savez(output / "strength.npz", game_indices=game_indices, **arrays)  # type: ignore[arg-type]
    return report


def compare_strength(first: Path, first_depth: int, second: Path, second_depth: int) -> dict:
    """Paired accuracy, good-move and blunder differences between two strength runs."""

    with np.load(first) as left, np.load(second) as right:
        if not np.array_equal(left["game_indices"], right["game_indices"]):
            raise ValueError("strength runs cover different positions")
        a, b = left[f"lost_t{first_depth}"], right[f"lost_t{second_depth}"]
    accuracy_a = np.array([move_accuracy(value) for value in a])
    accuracy_b = np.array([move_accuracy(value) for value in b])
    return {
        "lichess_accuracy": _paired_mean(accuracy_a, accuracy_b),
        "good_rate": _paired_mean(a < GOOD_BELOW, b < GOOD_BELOW),
        "blunder_rate": _paired_mean(a >= BLUNDER_FROM, b >= BLUNDER_FROM),
    }
