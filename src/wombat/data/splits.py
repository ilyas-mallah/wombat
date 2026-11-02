"""Stable position identity and dataset partitioning."""

from __future__ import annotations

from hashlib import sha256

import chess


def normalized_position(fen: str) -> str:
    """Normalize to the four state fields recoverable from isolated positions."""

    board = chess.Board(fen)
    return " ".join(board.fen(en_passant="fen").split()[:4])


def position_hash(fen: str) -> str:
    return sha256(normalized_position(fen).encode()).hexdigest()


def split_for_fen(fen: str) -> str:
    bucket = int(position_hash(fen)[:16], 16) % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"

