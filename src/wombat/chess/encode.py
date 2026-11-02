"""Canonical, compact position encoding.

Black-to-move positions are rotated 180 degrees and piece colors are swapped,
so the player to move is always represented as White and its pawns travel north.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import chess
import numpy as np


EMPTY = 0
NO_EN_PASSANT_FILE = 8
HALFMOVE_BUCKET_SIZE = 10
MAX_HALFMOVE_BUCKET = 10


@dataclass(frozen=True)
class EncodedPosition:
    """Inputs needed to construct Wombat's 64 square and one state token."""

    pieces: np.ndarray
    side_to_move_was_black: int
    castling: np.ndarray
    en_passant_file: int
    halfmove_bucket: int

    def __post_init__(self) -> None:
        if self.pieces.shape != (64,):
            raise ValueError("pieces must have shape (64,)")
        if self.castling.shape != (4,):
            raise ValueError("castling must have shape (4,)")


def canonical_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Map an absolute square into side-to-move coordinates."""

    return square if turn == chess.WHITE else 63 - square


def original_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Invert :func:`canonical_square`. Rotation by 180 degrees is self-inverse."""

    return canonical_square(square, turn)


def canonical_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Map a move into side-to-move coordinates while preserving promotion type."""

    return chess.Move(
        canonical_square(move.from_square, turn),
        canonical_square(move.to_square, turn),
        promotion=move.promotion,
        drop=move.drop,
    )


def _piece_id(piece: chess.Piece, turn: chess.Color) -> int:
    color = piece.color if turn == chess.WHITE else not piece.color
    return piece.piece_type + (0 if color == chess.WHITE else 6)


def _castling_features(board: chess.Board) -> np.ndarray:
    """Return self-Q, self-K, opponent-Q, opponent-K in canonical coordinates."""

    us = board.turn
    them = not us

    def rights(color: chess.Color) -> tuple[bool, bool]:
        queenside = board.has_queenside_castling_rights(color)
        kingside = board.has_kingside_castling_rights(color)
        if board.turn == chess.BLACK:
            queenside, kingside = kingside, queenside
        return queenside, kingside

    return np.asarray((*rights(us), *rights(them)), dtype=np.uint8)


def encode_position(board: chess.Board) -> EncodedPosition:
    """Encode a board without mutating it."""

    turn = board.turn
    pieces = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        pieces[canonical_square(square, turn)] = _piece_id(piece, turn)

    if board.ep_square is None:
        ep_file = NO_EN_PASSANT_FILE
    else:
        ep_file = chess.square_file(canonical_square(board.ep_square, turn))

    return EncodedPosition(
        pieces=pieces,
        side_to_move_was_black=int(turn == chess.BLACK),
        castling=_castling_features(board),
        en_passant_file=ep_file,
        halfmove_bucket=min(board.halfmove_clock // HALFMOVE_BUCKET_SIZE, MAX_HALFMOVE_BUCKET),
    )


def play_encoding(board: chess.Board, *, zero_halfmove: bool = True) -> EncodedPosition:
    """Encode a board for play, by default with halfmove bucket 0.

    Training game positions come from FENs without move counters, so all of them are in bucket 0.
    """

    encoded = encode_position(board)
    return replace(encoded, halfmove_bucket=0) if zero_halfmove else encoded


def state_vector(encoded: EncodedPosition) -> np.ndarray:
    """Expand categorical state fields into the network's 26 float features."""

    state = np.zeros(26, dtype=np.float32)
    state[encoded.side_to_move_was_black] = 1.0
    state[2:6] = encoded.castling
    state[6 + encoded.en_passant_file] = 1.0
    state[15 + encoded.halfmove_bucket] = 1.0
    return state


def decode_position(pieces: np.ndarray, state: np.ndarray) -> chess.Board:
    """Invert ``encode_position`` and ``state_vector``.

    The halfmove clock is only known to its bucket, so it is set to the bucket start.
    """

    turn = chess.BLACK if state[1] else chess.WHITE
    board = chess.Board.empty()
    board.turn = turn
    for canonical, piece_id in enumerate(np.asarray(pieces)):
        if piece_id == EMPTY:
            continue
        own = piece_id <= 6
        color = turn if own else not turn
        piece_type = int(piece_id) if own else int(piece_id) - 6
        board.set_piece_at(original_square(canonical, turn), chess.Piece(piece_type, color))
    own_first, own_second, their_first, their_second = (bool(value) for value in state[2:6])
    # Features are (queenside, kingside) in canonical order, which swaps them for Black.
    if turn == chess.BLACK:
        own_first, own_second = own_second, own_first
        their_first, their_second = their_second, their_first
    rights = {
        turn: (own_first, own_second),
        (not turn): (their_first, their_second),
    }
    fen_rights = ""
    for color, (letter_king, letter_queen) in ((chess.WHITE, "KQ"), (chess.BLACK, "kq")):
        queenside, kingside = rights[color]
        fen_rights += (letter_king if kingside else "") + (letter_queen if queenside else "")
    board.set_castling_fen(fen_rights or "-")
    ep_file = int(np.argmax(np.asarray(state[6:15])))
    if ep_file != NO_EN_PASSANT_FILE:
        file = ep_file if turn == chess.WHITE else 7 - ep_file
        board.ep_square = chess.square(file, 5 if turn == chess.WHITE else 2)
    board.halfmove_clock = int(np.argmax(np.asarray(state[15:26]))) * HALFMOVE_BUCKET_SIZE
    return board
