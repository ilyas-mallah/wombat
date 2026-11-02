from __future__ import annotations

import random

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import legal_action_mask, move_to_action, square_pair_actions
from wombat.model.lines import LineRules, own_moves

RULES = LineRules()
SPECIAL_FENS = (
    # Both sides may castle either way; a black pawn can take en passant on e3.
    "r3k2r/pppq1ppp/2n2n2/3p4/3Pp3/2N2N2/PPPQPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppq1ppp/2n2n2/3p4/3PpP2/2N2N2/PPPQP1PP/R3K2R b KQkq f3 0 1",
    # Promotions and underpromotions, with captures onto rook corners.
    "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R w KQkq - 0 1",
    "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R b KQkq - 0 1",
    # White may take en passant on d6.
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    # In check, with a pinned piece.
    "4k3/8/8/1b6/8/8/3N4/4K2r w - - 0 1",
)


def _tensors(boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_position(board) for board in boards]
    pieces = torch.from_numpy(np.stack([item.pieces for item in encoded]).astype(np.int64))
    state = torch.from_numpy(np.stack([state_vector(item) for item in encoded]))
    return pieces, state


def _positions(count: int, seed: int) -> list[chess.Board]:
    randomizer = random.Random(seed)
    boards = [chess.Board(fen) for fen in SPECIAL_FENS]
    while len(boards) < count:
        board = chess.Board()
        for _ in range(randomizer.randrange(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        if not board.is_game_over():
            boards.append(board)
    for board in boards:
        # A cleared clock keeps the halfmove bucket exact after any move.
        board.halfmove_clock = 0
    return boards


def test_legal_actions_match_python_chess() -> None:
    boards = _positions(120, seed=1)
    pieces, state = _tensors(boards)
    legal = RULES.legal(pieces, state).numpy()
    expected = np.stack([legal_action_mask(board) for board in boards])
    assert np.array_equal(legal, expected)


def test_every_child_matches_python_chess_after_the_move() -> None:
    boards = _positions(60, seed=2)
    parents, moves = [], []
    for board in boards:
        for move in board.legal_moves:
            parents.append(board)
            moves.append(move)
    pieces, state = _tensors(parents)
    actions = torch.tensor([move_to_action(b, m) for b, m in zip(parents, moves)])
    child_pieces, child_state = RULES.child(pieces, state, actions)
    children = []
    for board, move in zip(parents, moves):
        child = board.copy(stack=False)
        child.push(move)
        children.append(child)
    expected_pieces, expected_state = _tensors(children)
    assert torch.equal(child_pieces, expected_pieces)
    assert torch.equal(child_state, expected_state)
    kinds = {
        "castling": any(board.is_castling(move) for board, move in zip(parents, moves)),
        "en passant": any(board.is_en_passant(move) for board, move in zip(parents, moves)),
        "underpromotion": any(move.promotion == chess.KNIGHT for move in moves),
        "promotion": any(move.promotion == chess.QUEEN for move in moves),
    }
    assert all(kinds.values()), kinds


def test_null_move_passes_the_turn_and_clears_en_passant() -> None:
    board = chess.Board(SPECIAL_FENS[1])
    pieces, state = _tensors([board])
    child_pieces, child_state = RULES.child(pieces, state, torch.tensor([-1]))
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    expected_pieces, expected_state = _tensors([passed])
    assert torch.equal(child_pieces, expected_pieces)
    assert torch.equal(child_state, expected_state)


def test_replies_are_recovered_from_the_position_two_plies_later() -> None:
    randomizer = random.Random(3)
    boards = _positions(80, seed=3)
    parents, firsts, followings, expected = [], [], [], []
    for board in boards:
        first = randomizer.choice(list(board.legal_moves))
        middle = board.copy(stack=False)
        middle.push(first)
        replies = list(middle.legal_moves)
        if not replies:
            continue
        reply = randomizer.choice(replies)
        later = middle.copy(stack=False)
        later.push(reply)
        parents.append(board)
        firsts.append(move_to_action(board, first))
        followings.append(encode_position(later).pieces)
        expected.append(move_to_action(middle, reply))
    special: tuple[tuple[str, str, str], ...] = (
        # Recapture with the same piece type on the same square.
        ("4k3/8/3n4/1n6/4N3/8/8/4K3 w - - 0 1", "e4d6", "b5d6"),
        # A castling reply and an underpromotion reply.
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1a2", "e8c8"),
        ("4k3/8/8/8/8/8/1p6/R3K3 w Q - 0 1", "e1d1", "b2a1n"),
    )
    for fen, first_uci, reply_uci in special:
        board = chess.Board(fen)
        first = chess.Move.from_uci(first_uci)
        middle = board.copy(stack=False)
        middle.push(first)
        reply = chess.Move.from_uci(reply_uci)
        later = middle.copy(stack=False)
        later.push(reply)
        parents.append(board)
        firsts.append(move_to_action(board, first))
        followings.append(encode_position(later).pieces)
        expected.append(move_to_action(middle, reply))
    pieces, state = _tensors(parents)
    following = torch.from_numpy(np.stack(followings).astype(np.int64))
    found = RULES.reply(pieces, state, torch.tensor(firsts), following)
    assert found.tolist() == expected


def test_a_position_that_cannot_follow_has_no_reply() -> None:
    board = chess.Board()
    pieces, state = _tensors([board])
    first = torch.tensor([move_to_action(board, chess.Move.from_uci("e2e4"))])
    assert RULES.reply(pieces, state, first, pieces).tolist() == [-1]


def test_square_pairs_name_queen_and_knight_moves() -> None:
    table = square_pair_actions()
    board = chess.Board("4k3/8/8/8/3Q4/8/8/4K1N1 w - - 0 1")
    for move in board.legal_moves:
        assert table[move.from_square, move.to_square] == move_to_action(board, move)
    assert table[chess.A1, chess.B4] == -1


def test_own_moves_pick_the_best_legal_action_or_none() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    legal = torch.tensor([[False, True, True], [False, False, False]])
    assert own_moves(logits, legal).tolist() == [2, -1]


@pytest.mark.parametrize("fen", SPECIAL_FENS)
def test_special_positions_have_legal_actions(fen: str) -> None:
    board = chess.Board(fen)
    pieces, state = _tensors([board])
    assert int(RULES.legal(pieces, state).sum()) == board.legal_moves.count()
