"""AlphaZero-style 64 x 73 move encoding."""

from __future__ import annotations

import chess
import numpy as np

from wombat.chess.encode import canonical_move


PLANES_PER_SQUARE = 73
ACTION_SIZE = 64 * PLANES_PER_SQUARE

# Plane order is part of the persisted data schema.
RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _delta(move: chess.Move) -> tuple[int, int]:
    return (
        chess.square_file(move.to_square) - chess.square_file(move.from_square),
        chess.square_rank(move.to_square) - chess.square_rank(move.from_square),
    )


def _plane(move: chess.Move) -> int:
    df, dr = _delta(move)

    if move.promotion in UNDERPROMOTION_PIECES:
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError(f"invalid canonical underpromotion: {move.uci()}")
        piece_index = UNDERPROMOTION_PIECES.index(move.promotion)
        return 64 + piece_index * 3 + (df + 1)

    for direction_index, (unit_df, unit_dr) in enumerate(RAY_DIRECTIONS):
        for distance in range(1, 8):
            if (df, dr) == (unit_df * distance, unit_dr * distance):
                return direction_index * 7 + (distance - 1)

    try:
        return 56 + KNIGHT_DELTAS.index((df, dr))
    except ValueError as error:
        raise ValueError(f"move cannot be represented: {move.uci()}") from error


def move_to_action(board: chess.Board, move: chess.Move, *, require_legal: bool = True) -> int:
    """Encode an absolute move for ``board`` as an action index."""

    if require_legal and move not in board.legal_moves:
        raise ValueError(f"illegal move for position: {move.uci()}")
    # Lichess may spell standard castling with the Chess960 king-to-rook UCI
    # form (for example e8h8). python-chess accepts it as legal but yields e8g8
    # from legal_moves, so normalize before assigning the persisted action.
    move = board.parse_uci(move.uci())
    return _canonical_action(move, board.turn)


def _canonical_action(move: chess.Move, turn: chess.Color) -> int:
    move = canonical_move(move, turn)
    return move.from_square * PLANES_PER_SQUARE + _plane(move)


def uci_action(uci: str, turn: chess.Color) -> int:
    """Action of a standard-notation UCI move by ``turn``, unchecked against any board."""

    return _canonical_action(chess.Move.from_uci(uci), turn)


def square_pair_actions() -> np.ndarray:
    """(64, 64) action of the queen-like or knight move between two squares, -1 where none.

    Squares are in the side-to-move frame; a pawn reaching the last rank this way is a queen
    promotion.
    """

    table = np.full((64, 64), -1, dtype=np.int64)
    for square in range(64):
        file, rank = chess.square_file(square), chess.square_rank(square)
        for direction, (unit_df, unit_dr) in enumerate(RAY_DIRECTIONS):
            for distance in range(1, 8):
                target_file, target_rank = file + unit_df * distance, rank + unit_dr * distance
                if 0 <= target_file < 8 and 0 <= target_rank < 8:
                    plane = direction * 7 + distance - 1
                    target = target_rank * 8 + target_file
                    table[square, target] = square * PLANES_PER_SQUARE + plane
        for index, (df, dr) in enumerate(KNIGHT_DELTAS):
            if 0 <= file + df < 8 and 0 <= rank + dr < 8:
                table[square, square + df + 8 * dr] = square * PLANES_PER_SQUARE + 56 + index
    return table


def legal_actions(board: chess.Board) -> list[int]:
    """Encode every legal move; generated moves need no castling normalization."""

    return [_canonical_action(move, board.turn) for move in board.legal_moves]


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask containing exactly the board's legal moves."""

    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    for action in legal_actions(board):
        if mask[action]:
            raise AssertionError(f"move encoding collision at action {action}")
        mask[action] = True
    return mask


def action_to_move(board: chess.Board, action: int) -> chess.Move:
    """Decode an action by matching it against the legal move set."""

    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action must be in [0, {ACTION_SIZE}): {action}")
    matches = [move for move in board.legal_moves if move_to_action(board, move, require_legal=False) == action]
    if not matches:
        raise ValueError(f"action {action} is not legal in this position")
    if len(matches) != 1:
        raise AssertionError(f"action {action} maps to multiple legal moves")
    return matches[0]
