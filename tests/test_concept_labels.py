import chess
import numpy as np

from wombat.interpretability import label_position


def test_attacked_undefended_piece_label_has_explicit_semantics() -> None:
    board = chess.Board("7k/8/8/6b1/8/8/3N4/K7 w - - 0 1")
    labels = label_position(board)
    assert labels.own_attacked_undefended_piece[chess.D2]
    assert labels.attacked_by_opponent[chess.D2]
    assert not labels.defended_by_self[chess.D2]


def test_labels_use_side_to_move_coordinates() -> None:
    white = chess.Board("7k/8/8/6b1/8/8/3N4/K7 w - - 0 1")
    black = chess.Board("7k/4n3/8/8/1B6/8/8/K7 b - - 0 1")
    assert np.array_equal(
        label_position(white).own_attacked_undefended_piece,
        label_position(black).own_attacked_undefended_piece,
    )
