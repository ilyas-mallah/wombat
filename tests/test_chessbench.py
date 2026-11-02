from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from wombat.cli import build_parser
from wombat.eval.games import MovePlayer, play_stockfish
from wombat.external.chessbench import ActionValue9M, ChessBenchPlayer, tokenize

FAKE_UCI = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci.py")]
WEIGHTS = os.environ.get("CHESSBENCH_9M_WEIGHTS")
# Moves the verified PyTorch port of the released 9M chose from these positions.
PORT_MOVES = {
    "rnbqk1nr/pp1pppbp/2p3p1/8/3PP3/5N1P/PPP2PP1/RNBQKB1R b KQkq - 0 1": "d7d5",
    "2r2rk1/ppN2ppp/2n1b3/1Bb5/8/4PN2/Pq3PPP/R2Q1RK1 w - - 0 1": "c7e6",
    "8/8/4K2k/8/8/6Q1/8/8 w - - 0 1": "g3g4",
    "r1bq2kr/1p1n1p2/p2p2pP/2pPp3/2P1P3/3B1NP1/PP1Q4/R3K2R w KQ - 0 1": "e1c1",
    "r1bqkb1r/ppp2ppp/2n5/3p4/3Pn3/3B1N2/PPP2PPP/RNBQ1RK1 b kq - 0 1": "f8e7",
}


def _actions() -> list[str]:
    """Every from-to pair and pawn promotion, a superset of their action list."""

    actions = []
    for start in chess.SQUARES:
        for end in chess.SQUARES:
            if start == end:
                continue
            actions.append(chess.Move(start, end).uci())
            ranks = (chess.square_rank(start), chess.square_rank(end))
            near = abs(chess.square_file(start) - chess.square_file(end)) <= 1
            if near and ranks in ((6, 7), (1, 0)):
                actions.extend(chess.Move(start, end, piece).uci() for piece in (2, 3, 4, 5))
    return actions


@pytest.fixture(scope="module")
def random_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    actions = _actions()
    path = tmp_path_factory.mktemp("chessbench") / "random.pt"
    torch.save(
        {
            "model": ActionValue9M(vocab=len(actions)).state_dict(),
            "actions": actions,
            "bucket_values": torch.linspace(0.5 / 128, 1 - 0.5 / 128, 128),
        },
        path,
    )
    return path


def test_tokenize_matches_their_layout() -> None:
    tokens = tokenize(chess.STARTING_FEN)
    assert tokens.shape == (77,)
    # Side to move, then 64 squares from a8, then castling, en passant and the two clocks.
    assert "0123456789abcdefghpnrkqPBNRQKw."[tokens[0]] == "w"
    assert "0123456789abcdefghpnrkqPBNRQKw."[tokens[1]] == "r"


def test_random_player_returns_legal_moves(random_weights: Path) -> None:
    player = ChessBenchPlayer(random_weights)
    assert isinstance(player, MovePlayer)
    board = chess.Board()
    for _ in range(12):
        fen = board.fen()
        move = player.choose_move(board)
        assert board.fen() == fen
        assert move in board.legal_moves
        board.push(move)


def test_moves_allowing_a_repetition_claim_score_one_half(
    random_weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player = ChessBenchPlayer(random_weights)
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    moves = player.ordered_moves(board)
    repeat = moves.index(chess.Move.from_uci("f6g8"))
    scores = np.full(len(moves), 0.4, dtype=np.float32)
    scores[repeat] = 0.45
    monkeypatch.setattr(player, "win_probabilities", lambda board: (moves, scores.copy()))
    # A losing side takes the draw; a winning side does not.
    assert player.choose_move(board).uci() == "f6g8"
    scores[:] = 0.6
    scores[repeat] = 0.9
    assert player.choose_move(board).uci() != "f6g8"


@pytest.mark.skipif(
    WEIGHTS is None or not Path(WEIGHTS).exists(), reason="set CHESSBENCH_9M_WEIGHTS to 9m.pt"
)
def test_released_weights_choose_the_port_moves() -> None:
    assert WEIGHTS is not None
    player = ChessBenchPlayer(Path(WEIGHTS))
    for fen, expected in PORT_MOVES.items():
        assert player.choose_move(chess.Board(fen)).uci() == expected


def test_play_stockfish_takes_an_external_player(
    game_corpus: Path, random_weights: Path, tmp_path: Path
) -> None:
    command = ["play-stockfish", "--external", "chessbench9m", "--external-weights", "w.pt"]
    args = build_parser().parse_args(command)
    assert (args.external, args.external_weights, args.checkpoint) == (
        "chessbench9m",
        Path("w.pt"),
        None,
    )
    for wrong in (["play-stockfish"], [*command, "--checkpoint", "x.pt"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(wrong)
    reports: list[dict[str, Any]] = play_stockfish(
        None,
        FAKE_UCI,
        pairs=1,
        max_plies=6,
        output=tmp_path / "match",
        split_dir=game_corpus / "test",
        win_band=(0.0, 1.0),
        low_priority=False,
        external="chessbench9m",
        external_weights=random_weights,
    )
    report = reports[0]
    assert report["games"] == 2
    assert report["config"]["player"] == "chessbench9m"
    assert report["config"]["draw_guard"] is None
    assert "checkpoint" not in report["config"]
    with (tmp_path / "match" / "elo-1320" / "games.pgn").open() as handle:
        game = chess.pgn.read_game(handle)
    assert game is not None and game.headers["White"] == "ChessBench 9M"
    with pytest.raises(ValueError, match="either a checkpoint"):
        play_stockfish(None, FAKE_UCI, output=tmp_path)
