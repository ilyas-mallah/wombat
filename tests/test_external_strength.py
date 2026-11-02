from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import pytest

from wombat.data.game_corpus import GameSplit
from wombat.eval.external_strength import evaluate_external_strength, export_strength_fens
from wombat.eval.strength import valid_sample, win_percent


class FakeEngine:
    """Scores the best move +300 cp and any restricted move by its target square."""

    def __init__(self) -> None:
        self.id = {"name": "Fake"}

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(self, board: chess.Board, limit: Any, root_moves: Any = None) -> dict:
        cp = 300 if root_moves is None else root_moves[0].to_square * 5
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}

    def quit(self) -> None:
        pass


def test_external_moves_are_judged_on_the_strength_positions(
    game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    games = game_corpus / "test"
    fens_path = tmp_path / "positions.txt"
    count = export_strength_fens(games, fens_path)
    fens = fens_path.read_text(encoding="utf-8").splitlines()
    split = GameSplit(games)
    assert split.fens is not None
    assert fens == [str(split.fens[index]) for index in valid_sample(split, 2_000)]
    assert count == len(fens) > 0
    chosen = {
        fen: min(chess.Board(fen).legal_moves, key=lambda move: move.uci()).uci() for fen in fens
    }
    moves = tmp_path / "moves.json"
    moves.write_text(json.dumps(chosen), encoding="utf-8")
    report = evaluate_external_strength(
        moves,
        games=games,
        output=tmp_path / "out",
        stockfish=Path("stockfish"),
        engines=1,
        cache_dir=tmp_path / "cache",
    )
    expected = [
        win_percent(300) - win_percent(chess.Move.from_uci(chosen[fen]).to_square * 5)
        for fen in fens
    ]
    assert report["positions"] == len(fens)
    assert report["mean_win_lost"] == pytest.approx(float(np.mean(np.maximum(expected, 0))))
    with np.load(tmp_path / "out" / "strength.npz") as saved:
        assert saved["lost"] == pytest.approx(np.maximum(expected, 0))
    del chosen[fens[0]]
    moves.write_text(json.dumps(chosen), encoding="utf-8")
    with pytest.raises(ValueError, match="no move"):
        evaluate_external_strength(
            moves,
            games=games,
            output=tmp_path / "out",
            stockfish=Path("stockfish"),
            cache_dir=tmp_path / "cache",
        )
