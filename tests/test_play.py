from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
from typing import cast

import chess
import chess.engine
import numpy as np
import torch

from wombat.chess.encode import encode_position, play_encoding, state_vector
from wombat.chess.moves import ACTION_SIZE, move_to_action
from wombat.cli import build_parser
from wombat.engine.uci import run_uci
from wombat.eval.agreement import DRAW_GUARD_VALUE, draws_immediately, guard_draws, select_move
from wombat.eval.games import adjudicate, play_game
from wombat.model import NetworkConfig, WombatNetwork
from wombat.model.adapter import TorchPolicyProvider
from wombat.policy import FixturePolicy, ValuedPolicyProvider

TINY = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=1,
    recurrent_blocks=1,
    latent_tokens=4,
    policy_readout="square",
    normalize_state=True,
    max_residual_update=None,
)
# Halfmove clock 57 is bucket 5; Black to move exercises the rotated frame.
LATE = "8/5k2/8/3r4/8/2K5/8/4R3 b - - 57 120"
HALFMOVE = slice(15, 26)


def _bucket(state: np.ndarray) -> int:
    assert state[HALFMOVE].sum() == 1
    return int(np.argmax(state[HALFMOVE]))


def test_training_encodings_keep_the_real_halfmove_bucket() -> None:
    board = chess.Board(LATE)
    assert encode_position(board).halfmove_bucket == 5
    assert _bucket(state_vector(encode_position(board))) == 5
    expected = encode_position(board)
    unchanged = play_encoding(board, zero_halfmove=False)
    assert np.array_equal(state_vector(unchanged), state_vector(expected))
    assert np.array_equal(unchanged.pieces, expected.pieces)
    played = play_encoding(board)
    assert played.halfmove_bucket == 0
    assert np.array_equal(played.pieces, expected.pieces)
    assert np.array_equal(played.castling, expected.castling)
    assert played.side_to_move_was_black == expected.side_to_move_was_black == 1
    assert played.en_passant_file == expected.en_passant_file


def test_provider_encodes_play_in_halfmove_bucket_zero() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(TINY)
    board = chess.Board(LATE)
    pieces, state = TorchPolicyProvider(model).inputs(board)
    assert _bucket(state[0].numpy()) == 0
    reference = np.delete(state_vector(encode_position(board)), HALFMOVE)
    assert np.array_equal(np.delete(state[0].numpy(), HALFMOVE), reference)
    assert np.array_equal(pieces[0].numpy(), encode_position(board).pieces)
    _, real = TorchPolicyProvider(model, zero_halfmove=False).inputs(board)
    assert _bucket(real[0].numpy()) == 5
    # The logits are the network's output on the bucket-zero inputs.
    with torch.no_grad():
        policy, _ = model(pieces, state, 2)
    logits = TorchPolicyProvider(model).policy_logits(board, 2)
    assert np.array_equal(logits, policy[0].numpy())


class RankedPolicy:
    """Stub provider: ``ranked`` UCI moves get descending logits, with a fixed ``value``."""

    def __init__(self, ranked: Sequence[str], value: float) -> None:
        self.ranked = ranked
        self.value = value

    def policy_logits(self, board: chess.Board, reasoning_steps: int) -> np.ndarray:
        del reasoning_steps
        logits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for rank, uci in enumerate(self.ranked):
            logits[move_to_action(board, chess.Move.from_uci(uci))] = len(self.ranked) - rank
        return logits

    def policy_and_value(
        self, board: chess.Board, reasoning_steps: int
    ) -> tuple[np.ndarray, float]:
        return self.policy_logits(board, reasoning_steps), self.value


# Qg6 stalemates; Qh6 mates.
QUEEN_ENDING = "7k/5K2/4Q3/8/8/8/8/8 w - - 0 1"
# Knights out and back twice; Black's Ng8 would make the start position occur a third time.
SHUFFLE = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1")
# Every one of White's four legal moves stalemates.
ALL_STALEMATE = "7k/7B/6Q1/2p5/2Bp4/2KN4/8/8 w - - 0 1"


def _shuffled() -> chess.Board:
    board = chess.Board()
    for uci in SHUFFLE:
        board.push_uci(uci)
    return board


def test_winning_side_avoids_a_stalemating_top_move() -> None:
    board = chess.Board(QUEEN_ENDING)
    provider = RankedPolicy(["e6g6", "e6h6"], value=0.9)
    assert draws_immediately(board, chess.Move.from_uci("e6g6"))
    assert select_move(provider, board, 1, draw_guard=0.3).uci() == "e6h6"
    assert board.fen() == QUEEN_ENDING and not board.move_stack
    # Off by default, as for evaluate-strength and puzzle evaluation.
    assert select_move(provider, board, 1).uci() == "e6g6"
    assert select_move(provider, board, 1, draw_guard=None).uci() == "e6g6"


def test_winning_side_avoids_a_third_repetition() -> None:
    board = _shuffled()
    provider = RankedPolicy(["f6g8", "e7e5"], value=0.5)
    assert select_move(provider, board, 1, draw_guard=0.3).uci() == "e7e5"
    assert len(board.move_stack) == len(SHUFFLE)
    # The same move is kept where it repeats only for the second time.
    fresh = chess.Board(board.fen())
    assert select_move(provider, fresh, 1, draw_guard=0.3).uci() == "f6g8"


def test_winning_side_avoids_giving_the_opponent_a_repetition() -> None:
    board = chess.Board()
    for uci in SHUFFLE[:6]:
        board.push_uci(uci)
    retreat = chess.Move.from_uci("f3g1")
    # After Ng1 the position occurs only twice, but Black's Ng8 would make it three.
    board.push(retreat)
    assert not board.is_repetition(3) and board.can_claim_threefold_repetition()
    board.pop()
    assert draws_immediately(board, retreat)
    provider = RankedPolicy(["f3g1", "e2e4"], value=0.6)
    assert select_move(provider, board, 1, draw_guard=0.3).uci() == "e2e4"


def test_games_are_drawn_only_after_a_real_threefold_repetition() -> None:
    board = chess.Board()
    for uci in SHUFFLE:
        board.push_uci(uci)
    # Black could claim by playing Ng8, but has not repeated yet, so play goes on.
    assert board.can_claim_threefold_repetition()
    assert adjudicate(board) is None
    board.push_uci("f6g8")
    assert adjudicate(board) == ("1/2-1/2", "threefold_repetition")
    fifty = chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 100 80")
    assert adjudicate(fifty) == ("1/2-1/2", "fifty_moves")
    mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    mate.push_uci("f7g7")
    assert adjudicate(mate) == ("1-0", "checkmate")


def test_positions_not_won_are_untouched() -> None:
    for value in (-0.9, 0.0, 0.3):
        stalemate = RankedPolicy(["e6g6", "e6h6"], value=value)
        board = chess.Board(QUEEN_ENDING)
        assert select_move(stalemate, board, 1, draw_guard=0.3).uci() == "e6g6"
        repeat = RankedPolicy(["f6g8", "e7e5"], value=value)
        assert select_move(repeat, _shuffled(), 1, draw_guard=0.3).uci() == "f6g8"
    # A provider without a value is never guarded.
    board = chess.Board(QUEEN_ENDING)
    fixture = FixturePolicy({board.fen(): "e6g6"})
    assert select_move(fixture, board, 1, draw_guard=0.3).uci() == "e6g6"


def test_every_move_drawing_falls_back_to_the_top_move() -> None:
    board = chess.Board(ALL_STALEMATE)
    moves = list(board.legal_moves)
    assert len(moves) == 4 and all(draws_immediately(board, move) for move in moves)
    provider = RankedPolicy(["c3d2", "c3b2", "c3b3"], value=0.95)
    assert select_move(provider, board, 1, draw_guard=0.3).uci() == "c3d2"
    ranked = [chess.Move.from_uci(uci) for uci in ("c3b2", "c3d2")]
    assert guard_draws(board, ranked, 0.95, 0.3).uci() == "c3b2"


def test_guard_is_on_for_uci_and_games() -> None:
    commands = "position startpos moves " + " ".join(SHUFFLE) + "\ngo\nquit\n"
    provider = RankedPolicy(["f6g8", "e7e5"], value=0.8)
    output = StringIO()
    run_uci(StringIO(commands), output, provider)
    assert "bestmove e7e5" in output.getvalue().splitlines()
    output = StringIO()
    run_uci(StringIO(commands), output, provider, draw_guard=None)
    assert "bestmove f6g8" in output.getvalue().splitlines()
    # Wombat's first move ends the game, so the engine is never asked.
    engine = cast(chess.engine.SimpleEngine, None)
    provider = RankedPolicy(["e6g6", "e6h6"], value=0.9)
    limit = chess.engine.Limit(nodes=1)
    guarded, _ = play_game(
        provider, engine, QUEEN_ENDING, pair=0, wombat_white=True, steps=1, limit=limit, max_plies=4
    )
    assert (guarded.result, guarded.termination) == ("1-0", "checkmate")
    unguarded, _ = play_game(
        provider,
        engine,
        QUEEN_ENDING,
        pair=0,
        wombat_white=True,
        steps=1,
        limit=limit,
        max_plies=4,
        draw_guard=None,
    )
    assert (unguarded.result, unguarded.termination) == ("1/2-1/2", "stalemate")
    arguments = build_parser().parse_args(["play-stockfish", "--checkpoint", "x.pt"])
    assert arguments.draw_guard == DRAW_GUARD_VALUE
    off = ["play-stockfish", "--checkpoint", "x.pt", "--no-draw-guard"]
    assert build_parser().parse_args(off).draw_guard is None


def test_torch_provider_reports_its_value() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(TINY)
    board = chess.Board(QUEEN_ENDING)
    provider = TorchPolicyProvider(model)
    assert isinstance(provider, ValuedPolicyProvider)
    logits, value = provider.policy_and_value(board, 2)
    with torch.no_grad():
        policy, expected = model(*provider.inputs(board), 2)
    assert np.array_equal(logits, policy[0].numpy()) and value == float(expected[0])
    assert np.array_equal(provider.policy_logits(board, 2), logits)
