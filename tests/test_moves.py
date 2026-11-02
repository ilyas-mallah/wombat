import random

import chess
import numpy as np
import pytest

from wombat.chess.moves import ACTION_SIZE, action_to_move, legal_action_mask, move_to_action


SPECIAL_FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/8/8/3pP3/8/8/8/K6k w - d6 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
    "7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
]


@pytest.mark.parametrize("fen", SPECIAL_FENS)
def test_legal_moves_round_trip_without_collisions(fen: str) -> None:
    board = chess.Board(fen)
    actions = {move_to_action(board, move) for move in board.legal_moves}
    mask = legal_action_mask(board)
    assert mask.shape == (ACTION_SIZE,)
    assert mask.dtype == np.bool_
    assert set(np.flatnonzero(mask)) == actions
    assert len(actions) == board.legal_moves.count()
    for move in board.legal_moves:
        assert action_to_move(board, move_to_action(board, move)) == move


def test_random_legal_positions_round_trip() -> None:
    randomizer = random.Random(0)
    board = chess.Board()
    positions_checked = 0
    while positions_checked < 500:
        legal = list(board.legal_moves)
        actions = [move_to_action(board, move) for move in legal]
        assert len(actions) == len(set(actions))
        assert int(legal_action_mask(board).sum()) == len(legal)
        for move, action in zip(legal, actions, strict=True):
            assert action_to_move(board, action) == move
        positions_checked += 1
        if board.is_game_over() or board.ply() >= 160:
            board.reset()
        else:
            board.push(randomizer.choice(legal))


def test_illegal_move_is_rejected() -> None:
    with pytest.raises(ValueError, match="illegal move"):
        move_to_action(chess.Board(), chess.Move.from_uci("e2e5"))


def test_lichess_king_to_rook_castling_alias_is_normalized() -> None:
    board = chess.Board("rn1qk2r/pbppppbp/1p3np1/8/2PP4/5NP1/PP2PPBP/RNBQ1RK1 b kq -")
    assert move_to_action(board, chess.Move.from_uci("e8h8")) == move_to_action(
        board, chess.Move.from_uci("e8g8")
    )
