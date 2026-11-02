from __future__ import annotations

import chess
import numpy as np
import pytest

from wombat.interpretability.labels import move_squares, static_exchange, step_labels


@pytest.mark.parametrize(
    ("fen", "uci", "expected"),
    [
        # A queen takes a pawn defended by a pawn.
        ("4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1", "e2e5", -8),
        ("4k3/8/8/4p3/8/8/4Q3/4K3 w - - 0 1", "e2e5", 1),
        # Rook takes, rook retakes, and the rook behind recaptures through the x-ray.
        ("4k3/4r3/8/4p3/8/8/4R3/4R2K w - - 0 1", "e2e5", 1),
        ("4k3/4r3/8/4p3/8/8/4R3/R6K w - - 0 1", "e2e5", -4),
        # The king retakes only where no white piece still attacks the square.
        ("8/8/3k4/4p3/8/8/4Q3/4K3 w - - 0 1", "e2e5", -8),
        ("8/8/3k4/4p3/3P4/8/4Q3/4K3 w - - 0 1", "e2e5", 1),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 1),
        ("8/4P3/8/8/8/8/8/k6K w - - 0 1", "e7e8q", 8),
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", 0),
    ],
)
def test_static_exchange(fen: str, uci: str, expected: int) -> None:
    board = chess.Board(fen)
    assert static_exchange(board, chess.Move.from_uci(uci)) == expected
    assert board.fen() == fen


def test_destinations_that_lose_material_in_both_frames() -> None:
    # The knight may go to e2 or h3 safely but to f3 only into the e4 pawn's capture.
    board = chess.Board("4k3/8/8/8/4p3/8/8/4K1N1 w - - 0 1")
    labels = step_labels(board)
    reached = {move.to_square for move in board.legal_moves}
    assert set(np.flatnonzero(labels.destination_mask)) == reached
    assert list(np.flatnonzero(labels.destination_loses)) == [chess.F3]

    mirrored = step_labels(board.mirror())
    assert list(np.flatnonzero(mirrored.destination_loses)) == [63 - chess.F6]


def test_the_best_move_to_a_square_decides_its_label() -> None:
    # d2-d3 is met by exd3 Qxd3 and loses nothing; the knight's f3 still loses.
    labels = step_labels(chess.Board("4k3/8/8/8/4p3/8/3P4/3QK1N1 w - - 0 1"))
    assert labels.destination_mask[chess.D3] and not labels.destination_loses[chess.D3]
    assert labels.destination_loses[chess.F3]


def test_free_captures_need_a_winning_exchange() -> None:
    # exd4 wins a knight for a pawn; exf4 gxf4 only trades pawns.
    board = chess.Board("4k3/8/8/4p3/3N1P2/2P3P1/8/4K3 w - - 0 1")
    labels = step_labels(board)
    assert set(np.flatnonzero(labels.free_capture_mask)) == {
        chess.D4,
        chess.F4,
        chess.C3,
        chess.G3,
    }
    assert list(np.flatnonzero(labels.free_capture)) == [chess.D4]
    mirrored = step_labels(board.mirror())
    assert list(np.flatnonzero(mirrored.free_capture)) == [63 - chess.D5]


def test_mate_in_one_for_either_side_and_either_colour() -> None:
    back_rank = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    labels = step_labels(back_rank)
    assert list(np.flatnonzero(labels.own_mate)) == [chess.A8]
    assert not labels.opponent_mate.any() and labels.opponent_mate_mask.any()
    # In the mirror Black mates on a1, which Black's frame puts at h8.
    mirrored = step_labels(back_rank.mirror())
    assert list(np.flatnonzero(mirrored.own_mate)) == [63 - chess.A1]
    assert not mirrored.opponent_mate.any()

    board = chess.Board("r5k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    threatened = step_labels(board)
    assert not threatened.own_mate.any()
    assert list(np.flatnonzero(threatened.opponent_mate)) == [chess.A1]
    assert threatened.opponent_mate_mask[chess.A1] and threatened.own_mate_mask.any()
    mirrored = step_labels(board.mirror())
    assert list(np.flatnonzero(mirrored.opponent_mate)) == [63 - chess.A8]


def test_no_opponent_mate_labels_while_in_check() -> None:
    labels = step_labels(chess.Board("r5k1/5ppp/8/8/8/8/5PPP/r5K1 w - - 0 1"))
    assert not labels.opponent_mate_mask.any()


def test_move_squares_use_the_side_to_move_frame() -> None:
    white = chess.Board()
    assert move_squares(white, chess.Move.from_uci("g1f3")) == (chess.G1, chess.F3)
    white.push_uci("e2e4")
    assert move_squares(white, chess.Move.from_uci("g8f6")) == (63 - chess.G8, 63 - chess.F6)
