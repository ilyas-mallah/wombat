from __future__ import annotations

import random

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import ACTION_SIZE, move_to_action
from wombat.interpretability.labels import static_exchange
from wombat.model.move_consequences import CONSEQUENCES, MATERIAL_SCALE, MoveConsequences

# Castling, en passant, promotions and pins for either side to move, and replies that castle
# into check or mate, mate by underpromotion or en passant, take en passant only in pseudo-legal
# terms, or check when only en passant answers.
HAND_MADE = (
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
    "r3k3/1P6/8/8/8/8/8/4K3 w q - 0 1",
    "4k3/8/8/8/8/8/1p6/R3K3 b Q - 0 1",
    "8/8/8/8/k2p3R/8/4P3/4K3 w - - 0 1",
    "5k2/2Q5/8/3p4/1p4P1/B3P3/2P5/R3K2R w KQ - 0 1",
    "8/5k2/2pPP3/8/8/P7/3P4/R3K2R b KQ - 0 1",
    "r3k2r/6P1/3BP3/p7/8/5p2/2pN1r2/R3K2R b KQkq - 0 1",
    "r3k2r/8/4R3/1P6/1P6/6R1/1P2P3/R3K2R b KQkq - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
    "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1",
    "8/8/6K1/8/1BQkp3/8/3P4/2Q5 b - - 0 1",
    "8/3p1K2/8/Q3Pk2/7R/3PB3/4q3/8 b - - 0 1",
)


def _played(count: int, seed: int) -> list[chess.Board]:
    randomizer = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(randomizer.randrange(0, 160)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        if any(board.legal_moves):
            boards.append(board)
    return boards


def _scattered(count: int, seed: int) -> list[chess.Board]:
    """Kings and up to 24 other pieces on random squares, often pinned or checking."""

    randomizer = random.Random(seed)
    others = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board.empty()
        squares = randomizer.sample(range(64), 2 + randomizer.randrange(0, 25))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for square in squares[2:]:
            kind = randomizer.choice(others)
            if kind == chess.PAWN and chess.square_rank(square) in (0, 7):
                continue
            board.set_piece_at(square, chess.Piece(kind, randomizer.random() < 0.5))
        board.turn = randomizer.random() < 0.5
        # The side not to move may not stand in check, and the side to move needs a move.
        if not board.was_into_check() and any(board.legal_moves):
            boards.append(board)
    return boards


def _reference(board: chess.Board) -> dict[int, tuple[int, int, int, int]]:
    """Per legal action: its exchange, the reply's best capture, a mating reply, check."""

    rows = {}
    for move in board.legal_moves:
        own = static_exchange(board, move)
        board.push(move)
        captures = [static_exchange(board, reply) for reply in board.generate_legal_captures()]
        mate = False
        for reply in board.legal_moves:
            if not mate and board.gives_check(reply):
                board.push(reply)
                mate = board.is_checkmate()
                board.pop()
        check = board.is_check()
        board.pop()
        rows[move_to_action(board, move)] = (own, max([0, *captures]), int(mate), int(check))
    return rows


def _inputs(boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_position(board) for board in boards]
    pieces = torch.from_numpy(np.stack([item.pieces for item in encoded])).long()
    state = torch.from_numpy(np.stack([state_vector(item) for item in encoded]))
    return pieces, state


def _raw(boards: list[chess.Board], static: bool = False) -> torch.Tensor:
    module = MoveConsequences()
    module.static = static
    with torch.no_grad():
        return module.raw(*_inputs(boards))


def _check(boards: list[chess.Board]) -> np.ndarray:
    """Compare every legal move with python-chess; count nonzero features per colour."""

    raw = _raw(boards)
    assert raw.shape == (len(boards), ACTION_SIZE, CONSEQUENCES)
    counts = np.zeros((2, CONSEQUENCES), dtype=np.int64)
    for board, row in zip(boards, raw):
        reference = _reference(board)
        for action, expected in reference.items():
            assert tuple(row[action].tolist()) == expected, (board.fen(), action)
            counts[int(board.turn == chess.BLACK)] += np.array(expected) != 0
        # Illegal actions, pseudo-legal ones included, get zeros.
        assert set(row.abs().sum(1).nonzero().flatten().tolist()) <= set(reference)
    return counts


def test_played_positions_match_python_chess() -> None:
    counts = _check(_played(200, seed=1))
    # Both colours see every kind of consequence, mates included.
    assert (counts > 0).all()


def test_scattered_positions_match_python_chess() -> None:
    counts = _check(_scattered(200, seed=2))
    assert (counts[:, 2] >= 20).all() and (counts > 0).all()


def test_castling_en_passant_and_promotions_match_python_chess() -> None:
    boards = [chess.Board(fen) for fen in HAND_MADE]
    _check(boards)
    raw = _raw(boards)
    # Castling exchanges nothing; replies mate by underpromotion and by castling.
    assert raw[0, move_to_action(boards[0], chess.Move.from_uci("e1g1")), 0] == 0
    assert raw[9, move_to_action(boards[9], chess.Move.from_uci("h8f8")), 2] == 1
    assert raw[10, move_to_action(boards[10], chess.Move.from_uci("e8f8")), 2] == 1
    # The back rank: Kh8 allows Rd8 mate, Kf8 leaves e7 free.
    for uci, mate in (("g8h8", 1), ("g8f8", 0)):
        assert raw[11, move_to_action(boards[11], chess.Move.from_uci(uci)), 2] == mate
    # En passant that would bare the king along the rank is no capture for the reply.
    assert raw[6, move_to_action(boards[6], chess.Move.from_uci("e2e4")), 1] == 0
    # Nor one pinned along the diagonal; the reply's best capture is then bxa3.
    board = boards[7].copy()
    board.push_uci("c2c4")
    assert not any(board.generate_legal_ep()) and any(board.generate_pseudo_legal_ep())
    assert raw[7, move_to_action(boards[7], chess.Move.from_uci("c2c4")), 1] == 2
    # After Ke5 no reply mates: d4 checks, but only exd3 en passant answers it.
    board = boards[13].copy()
    board.push_uci("d4e5")
    board.push_uci("d2d4")
    assert board.is_check() and all(board.is_en_passant(move) for move in board.legal_moves)
    assert raw[13, move_to_action(boards[13], chess.Move.from_uci("d4e5")), 2] == 0
    # After d5 the reply exd6 en passant mates.
    assert raw[14, move_to_action(boards[14], chess.Move.from_uci("d7d5")), 2] == 1


def test_fixed_caps_give_the_exact_features() -> None:
    boards = _played(24, seed=3) + [chess.Board(fen) for fen in HAND_MADE]
    assert torch.equal(_raw(boards, static=True), _raw(boards))
    # Positions in check have many pseudo-legal moves that would hand over the king; only the
    # legal ones use the reply caps, so small batches of them still fit.
    checked = [board for board in _scattered(200, seed=5) if board.is_check()][:4]
    assert len(checked) == 4
    assert torch.equal(_raw(checked, static=True), _raw(checked))


def test_features_are_scaled_material_and_flags() -> None:
    boards = _played(8, seed=4)
    pieces, state = _inputs(boards)
    module = MoveConsequences()
    with torch.no_grad():
        raw = module.raw(pieces, state).view(len(boards), 64, 73, CONSEQUENCES)
        features = module(pieces, state)
    assert features.shape == (len(boards), 64, 73, CONSEQUENCES)
    expected = (raw[..., :2].float() / MATERIAL_SCALE).clamp(-1, 1)
    assert torch.equal(features[..., :2], expected)
    assert torch.equal(features[..., 2:], raw[..., 2:].float())


@pytest.mark.parametrize("fen", ["8/8/8/8/8/8/8/K6k w - - 0 1", "k7/8/8/8/8/8/8/7K b - - 0 1"])
def test_bare_kings_have_no_captures_or_mates(fen: str) -> None:
    raw = _raw([chess.Board(fen)])
    assert int(raw[..., 1:3].abs().sum()) == 0
