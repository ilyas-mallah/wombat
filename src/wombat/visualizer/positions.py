"""Random ordinary game positions from an evaluation corpus split."""

from __future__ import annotations

import io
import json
from pathlib import Path
import random
import threading

import chess
import zstandard


class PositionPool:
    """Reads at most ``limit`` FENs once, on first use, from a ``.jsonl.zst`` file."""

    def __init__(self, path: Path, limit: int = 5000) -> None:
        self.path = path
        self.limit = limit
        self._fens: list[str] | None = None
        self._lock = threading.Lock()

    def _read(self) -> list[str]:
        if not self.path.exists():
            return []
        fens: list[str] = []
        with self.path.open("rb") as source:
            reader = zstandard.ZstdDecompressor().stream_reader(source)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if not line.strip():
                    continue
                board = chess.Board(json.loads(line)["fen"])
                if board.is_valid() and not board.is_game_over():
                    fens.append(board.fen())
                if len(fens) >= self.limit:
                    break
        return fens

    @property
    def fens(self) -> list[str]:
        with self._lock:
            if self._fens is None:
                self._fens = self._read()
            return self._fens

    def random(self, generator: random.Random) -> str:
        """A random position, or the standard start when the corpus is missing."""

        fens = self.fens
        return generator.choice(fens) if fens else chess.STARTING_FEN
