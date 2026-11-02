import random

import chess
import numpy as np

from wombat.chess.encode import decode_position, encode_position, state_vector


def test_starting_position_encoding() -> None:
    encoded = encode_position(chess.Board())
    assert encoded.pieces.shape == (64,)
    assert np.count_nonzero(encoded.pieces) == 32
    assert encoded.side_to_move_was_black == 0
    assert encoded.castling.tolist() == [1, 1, 1, 1]
    assert encoded.en_passant_file == 8


def test_black_position_is_rotated_and_colors_swapped() -> None:
    white = chess.Board("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
    black = chess.Board("K2k4/3p4/8/8/8/8/8/8 b - - 0 1")
    assert np.array_equal(encode_position(white).pieces, encode_position(black).pieces)


def test_en_passant_file_is_canonicalized() -> None:
    board = chess.Board("8/8/8/3pP3/8/8/8/K6k w - d6 0 1")
    assert encode_position(board).en_passant_file == chess.FILE_NAMES.index("d")
    board.turn = chess.BLACK
    assert encode_position(board).en_passant_file == chess.FILE_NAMES.index("e")


def test_decoding_inverts_encoding_for_random_positions() -> None:
    randomizer = random.Random(11)
    seen_black = seen_en_passant = seen_partial_castling = False
    for _ in range(3000):
        board = chess.Board()
        for _ in range(randomizer.randrange(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        encoded = encode_position(board)
        decoded = decode_position(encoded.pieces, state_vector(encoded))
        assert decoded.fen().split()[:4] == board.fen().split()[:4]
        assert decoded.halfmove_clock == min(board.halfmove_clock // 10, 10) * 10
        assert set(decoded.legal_moves) == set(board.legal_moves)
        seen_black |= board.turn == chess.BLACK
        seen_en_passant |= board.ep_square is not None
        seen_partial_castling |= board.castling_xfen() not in ("KQkq", "-")
    assert seen_black and seen_en_passant and seen_partial_castling
