"""Operational chess-concept labels for diagnostic probes only.

These labels are never model inputs or training targets for the base network.
Each name encodes a deliberately narrow definition rather than claiming to
capture every human meaning of a chess concept.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from wombat.chess.encode import canonical_square


@dataclass(frozen=True)
class ConceptLabels:
    attacked_by_opponent: np.ndarray
    defended_by_self: np.ndarray
    own_attacked_undefended_piece: np.ndarray
    in_check: bool
    legal_destinations: np.ndarray


def label_position(board: chess.Board) -> ConceptLabels:
    """Label square-level facts in Wombat's side-to-move coordinates."""

    turn = board.turn
    opponent = not turn
    attacked = np.zeros(64, dtype=np.bool_)
    defended = np.zeros(64, dtype=np.bool_)
    hanging = np.zeros(64, dtype=np.bool_)
    legal_destinations = np.zeros(64, dtype=np.bool_)

    for square in chess.SQUARES:
        canonical = canonical_square(square, turn)
        attacked[canonical] = board.is_attacked_by(opponent, square)
        defended[canonical] = board.is_attacked_by(turn, square)
        piece = board.piece_at(square)
        hanging[canonical] = bool(
            piece
            and piece.color == turn
            and piece.piece_type != chess.KING
            and attacked[canonical]
            and not defended[canonical]
        )
    for move in board.legal_moves:
        legal_destinations[canonical_square(move.to_square, turn)] = True
    return ConceptLabels(
        attacked_by_opponent=attacked,
        defended_by_self=defended,
        own_attacked_undefended_piece=hanging,
        in_check=board.is_check(),
        legal_destinations=legal_destinations,
    )


# Pawn units for static exchange; a king is never captured, so its value only has to dominate.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def static_exchange(board: chess.Board, move: chess.Move) -> int:
    """Material the mover nets from ``move`` and the best run of captures on its square after it.

    Each side recaptures with its least valuable attacker or stops. X-rays count, pins are
    ignored, a king never recaptures onto a square the other side still attacks, and castling
    scores 0.
    """

    if board.is_castling(move):
        return 0
    target = move.to_square
    mover = board.piece_at(move.from_square)
    if mover is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")
    work = board.copy(stack=False)
    if board.is_en_passant(move):
        captured = PIECE_VALUES[chess.PAWN]
        work.remove_piece_at(target + (-8 if board.turn == chess.WHITE else 8))
    else:
        victim = board.piece_type_at(target)
        captured = PIECE_VALUES[victim] if victim is not None else 0
    on_target = chess.Piece(move.promotion, mover.color) if move.promotion else mover
    gains = [captured + PIECE_VALUES[on_target.piece_type] - PIECE_VALUES[mover.piece_type]]
    work.remove_piece_at(move.from_square)
    work.set_piece_at(target, on_target)
    side = not board.turn
    while True:
        attackers = list(work.attackers(side, target))
        if not attackers:
            break
        square = min(attackers, key=lambda item: PIECE_VALUES[work.piece_type_at(item) or 1])
        piece = work.remove_piece_at(square)
        assert piece is not None
        if piece.piece_type == chess.KING and work.attackers(not side, target):
            break
        gains.append(PIECE_VALUES[on_target.piece_type] - gains[-1])
        work.set_piece_at(target, piece)
        on_target = piece
        side = not side
    for depth in range(len(gains) - 1, 0, -1):
        gains[depth - 1] = -max(-gains[depth - 1], gains[depth])
    return gains[0]


@dataclass(frozen=True)
class StepLabels:
    """Square targets of the step probes in the side-to-move frame, each with its mask.

    ``destination_loses``: every legal move to the square loses material by static exchange
    (masked to legal destinations). ``free_capture``: a side-to-move piece other than the king
    that the opponent can capture winning material (masked to those pieces). ``own_mate`` and
    ``opponent_mate``: the destination of a mating move for the side to move, or for the
    opponent were it to move (masked to each side's legal destinations; the opponent's mask is
    empty while the side to move is in check).
    """

    destination_loses: np.ndarray
    destination_mask: np.ndarray
    free_capture: np.ndarray
    free_capture_mask: np.ndarray
    own_mate: np.ndarray
    own_mate_mask: np.ndarray
    opponent_mate: np.ndarray
    opponent_mate_mask: np.ndarray


def _mates(board: chess.Board, turn: chess.Color) -> tuple[np.ndarray, np.ndarray]:
    """Mating destinations and legal destinations of ``board``'s mover, in ``turn``'s frame."""

    mates = np.zeros(64, dtype=np.bool_)
    reached = np.zeros(64, dtype=np.bool_)
    for move in board.legal_moves:
        square = canonical_square(move.to_square, turn)
        reached[square] = True
        if not mates[square] and board.gives_check(move):
            board.push(move)
            mates[square] = board.is_checkmate()
            board.pop()
    return mates, reached


def step_labels(board: chess.Board) -> StepLabels:
    """Label material, capture and mate targets of one position for the step probes."""

    turn = board.turn
    work = board.copy(stack=False)
    exchange: dict[int, int] = {}
    for move in work.legal_moves:
        square = canonical_square(move.to_square, turn)
        exchange[square] = max(exchange.get(square, -1_000), static_exchange(work, move))
    destination_mask = np.zeros(64, dtype=np.bool_)
    destination_loses = np.zeros(64, dtype=np.bool_)
    for square, best in exchange.items():
        destination_mask[square] = True
        destination_loses[square] = best < 0
    free_capture_mask = np.zeros(64, dtype=np.bool_)
    for square in chess.SquareSet(work.occupied_co[turn] & ~work.kings):
        free_capture_mask[canonical_square(square, turn)] = True
    flipped = work.copy(stack=False)
    flipped.push(chess.Move.null())
    free_capture = np.zeros(64, dtype=np.bool_)
    for move in flipped.generate_legal_captures():
        square = canonical_square(move.to_square, turn)
        if free_capture_mask[square] and not free_capture[square]:
            free_capture[square] = static_exchange(flipped, move) > 0
    own_mate, own_mate_mask = _mates(work, turn)
    if work.is_check():
        opponent_mate = opponent_mate_mask = np.zeros(64, dtype=np.bool_)
    else:
        opponent_mate, opponent_mate_mask = _mates(flipped, turn)
    return StepLabels(
        destination_loses=destination_loses,
        destination_mask=destination_mask,
        free_capture=free_capture,
        free_capture_mask=free_capture_mask,
        own_mate=own_mate,
        own_mate_mask=own_mate_mask,
        opponent_mate=opponent_mate,
        opponent_mate_mask=opponent_mate_mask,
    )


def move_squares(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    """A move's origin and destination in ``board``'s side-to-move frame."""

    return canonical_square(move.from_square, board.turn), canonical_square(
        move.to_square, board.turn
    )
