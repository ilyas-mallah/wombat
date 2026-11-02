"""Whole puzzle lines rebuilt from a memory-mapped held-out puzzle split."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import threading
from typing import Any

import chess
import numpy as np

from wombat.chess.moves import action_to_move
from wombat.data.puzzle_corpus import PuzzleSplit


@dataclass(frozen=True)
class PuzzleStep:
    """One solver-to-move position, its solution, and the opponent's answer to it."""

    fen: str
    solution: str
    solution_san: str
    reply: str | None
    reply_san: str | None


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    rating: int
    steps: tuple[PuzzleStep, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _position_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def find_reply(board: chess.Board, next_fen: str) -> chess.Move:
    """The legal move from ``board`` that reaches ``next_fen``."""

    target = _position_key(next_fen)
    for move in board.legal_moves:
        board.push(move)
        reached = _position_key(board.fen())
        board.pop()
        if reached == target:
            return move
    raise ValueError(f"no legal move reaches {next_fen}")


def reconstruct_line(fens: Sequence[str], actions: Sequence[int]) -> tuple[PuzzleStep, ...]:
    """Pair each solver position with its solution and the reply leading to the next one."""

    steps: list[PuzzleStep] = []
    for index, (fen, action) in enumerate(zip(fens, actions, strict=True)):
        board = chess.Board(fen)
        solution = action_to_move(board, int(action))
        solution_san = board.san(solution)
        reply: str | None = None
        reply_san: str | None = None
        if index + 1 < len(fens):
            board.push(solution)
            move = find_reply(board, fens[index + 1])
            reply, reply_san = move.uci(), board.san(move)
        steps.append(PuzzleStep(fen, solution.uci(), solution_san, reply, reply_san))
    return tuple(steps)


def is_solution(board: chess.Board, move: chess.Move, solution: chess.Move) -> bool:
    """The listed move, or any move that checkmates (the Lichess rule)."""

    if move == solution:
        return True
    if move not in board.legal_moves:
        return False
    board = board.copy(stack=False)
    board.push(move)
    return board.is_checkmate()


class PuzzleBook:
    """Random access to whole puzzles; the split is opened lazily and memory-mapped."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._split: PuzzleSplit | None = None
        self._starts: np.ndarray | None = None
        self._ratings: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.directory / "fens.npy").exists()

    def _open(self) -> tuple[PuzzleSplit, np.ndarray, np.ndarray]:
        with self._lock:
            if self._split is None:
                split = PuzzleSplit(self.directory)
                if split.fens is None:
                    raise FileNotFoundError(f"{self.directory} has no FENs")
                puzzle = np.asarray(split.puzzle)
                # Rows of one puzzle are contiguous, so a puzzle starts wherever the id changes.
                changes = np.flatnonzero(puzzle[1:] != puzzle[:-1]) + 1
                self._starts = np.concatenate(([0], changes, [len(puzzle)])).astype(np.int64)
                self._ratings = np.asarray(split.rating[self._starts[:-1]], dtype=np.int64)
                self._split = split
            assert self._starts is not None and self._ratings is not None
            return self._split, self._starts, self._ratings

    def __len__(self) -> int:
        return len(self._open()[1]) - 1

    def puzzle(self, number: int) -> Puzzle:
        split, starts, ratings = self._open()
        rows = np.arange(starts[number], starts[number + 1])
        rows = rows[np.argsort(np.asarray(split.line_index[rows]), kind="stable")]
        assert split.fens is not None
        steps = reconstruct_line(
            [str(split.fens[row]) for row in rows], [int(split.action[row]) for row in rows]
        )
        puzzle_id = str(split.puzzle_ids[int(split.puzzle[rows[0]])])
        return Puzzle(puzzle_id, int(ratings[number]), steps)

    def random(
        self,
        generator: random.Random,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> Puzzle:
        _, _, ratings = self._open()
        keep = np.ones(len(ratings), dtype=np.bool_)
        if min_rating is not None:
            keep &= ratings >= min_rating
        if max_rating is not None:
            keep &= ratings <= max_rating
        candidates = np.flatnonzero(keep)
        if len(candidates) == 0:
            raise LookupError("no puzzle in that rating range")
        return self.puzzle(int(candidates[generator.randrange(len(candidates))]))
