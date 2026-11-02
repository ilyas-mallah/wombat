"""Optional Stockfish judging with the Lichess win-percent and accuracy conventions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Any

import chess
import chess.engine

MATE_SCORE = 100_000
CP_CEILING = 1000
# The UCI_Elo range Stockfish accepts.
MIN_ELO, MAX_ELO = 1320, 3190


def win_percent(cp: float) -> float:
    """Lichess win chance, in percent, for the side whose centipawn score is ``cp``."""

    cp = max(-CP_CEILING, min(CP_CEILING, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(lost: float) -> float:
    """Lichess per-move accuracy from win-percent points lost."""

    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * lost) - 3.1669))


def classify(lost: float) -> str:
    if lost >= 15:
        return "blunder"
    if lost >= 10:
        return "mistake"
    if lost >= 5:
        return "inaccuracy"
    return "good"


def strength(
    *, nodes: int | None = None, elo: int | None = None, movetime: float = 0.1
) -> tuple[chess.engine.Limit, dict[str, bool | int]]:
    """Search limit and UCI options for a node budget or an Elo-limited timed move."""

    if nodes is not None:
        return chess.engine.Limit(nodes=max(1, nodes)), {}
    limit = chess.engine.Limit(time=max(0.01, movetime))
    if elo is None:
        return limit, {"UCI_LimitStrength": False}
    return limit, {"UCI_LimitStrength": True, "UCI_Elo": max(MIN_ELO, min(MAX_ELO, elo))}


@dataclass(frozen=True)
class Evaluation:
    """A score from the side to move, with Stockfish's best move when one exists."""

    cp: int
    mate: int | None
    best: str | None


def _terminal(board: chess.Board) -> Evaluation | None:
    if board.is_checkmate():
        return Evaluation(-MATE_SCORE, 0, None)
    if board.is_game_over(claim_draw=False):
        return Evaluation(0, None, None)
    return None


class Stockfish:
    """A lazily started single-thread engine shared by all requests."""

    def __init__(self, path: str | None, nodes: int = 20_000, cache_size: int = 4096) -> None:
        self.path = path
        self.nodes = nodes
        self.cache_size = cache_size
        self._engine: chess.engine.SimpleEngine | None = None
        self._cache: OrderedDict[tuple[str, int], Evaluation] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.path is not None and Path(self.path).exists()

    def _started(self) -> chess.engine.SimpleEngine:
        if not self.available:
            raise RuntimeError("Stockfish is not configured")
        if self._engine is None:
            assert self.path is not None
            engine = chess.engine.SimpleEngine.popen_uci(self.path)
            engine.configure({"Threads": 1, "Hash": 16})
            self._engine = engine
        return self._engine

    def evaluate(self, board: chess.Board, nodes: int | None = None) -> Evaluation:
        terminal = _terminal(board)
        if terminal is not None:
            return terminal
        nodes = nodes or self.nodes
        key = (board.fen(), nodes)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            info = self._started().analyse(board, chess.engine.Limit(nodes=nodes))
            score = info["score"].pov(board.turn)
            pv = info.get("pv") or []
            result = Evaluation(
                int(score.score(mate_score=MATE_SCORE) or 0),
                score.mate(),
                pv[0].uci() if pv else None,
            )
            self._cache[key] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return result

    def position(self, board: chess.Board) -> dict[str, Any]:
        """Eval-bar data from White's side."""

        evaluation = self.evaluate(board)
        sign = 1 if board.turn == chess.WHITE else -1
        return {
            "cp_white": sign * evaluation.cp,
            "mate_white": None if evaluation.mate is None else sign * evaluation.mate,
            "win_white": win_percent(sign * evaluation.cp),
            "best": evaluation.best,
        }

    def judge(self, board: chess.Board, move: chess.Move) -> dict[str, Any]:
        """Win-percent points ``move`` gives up against Stockfish's best move."""

        before = self.evaluate(board)
        after_board = board.copy(stack=False)
        after_board.push(move)
        after = self.evaluate(after_board)
        best_win = win_percent(before.cp)
        played_win = 100.0 - win_percent(after.cp)
        lost = 0.0 if move.uci() == before.best else max(0.0, best_win - played_win)
        best_san = board.san(chess.Move.from_uci(before.best)) if before.best else None
        return {
            "move": move.uci(),
            "best": before.best,
            "best_san": best_san,
            "win_before": best_win,
            "win_after": played_win,
            "lost": lost,
            "label": classify(lost),
            "accuracy": move_accuracy(lost),
            "eval_after": self.position(after_board),
        }

    def play(
        self,
        board: chess.Board,
        *,
        nodes: int | None = None,
        elo: int | None = None,
        movetime: float = 0.1,
    ) -> chess.Move:
        """A move at a node budget, or at ``elo`` (None is full strength) in ``movetime`` s."""

        limit, options = strength(nodes=nodes, elo=elo, movetime=movetime)
        with self._lock:
            # python-chess restores these options after the search, so judging is unaffected.
            result = self._started().play(board, limit, options=options)
        if result.move is None:
            raise RuntimeError("Stockfish returned no move")
        return result.move

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.quit()
                self._engine = None
