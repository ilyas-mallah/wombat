from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import chess
import chess.engine
import numpy as np
import pytest

from wombat.chess.moves import action_to_move, move_to_action
from wombat.eval import strength
from wombat.eval.strength import (
    StockfishJudge,
    compare_strength,
    move_accuracy,
    puzzle_rating,
    whole_puzzles_solved,
    win_percent,
)
from wombat.data.puzzle_corpus import PuzzleSplit


class FakeEngine:
    """Scores the best move +300 cp and any restricted move by its target square."""

    def __init__(self) -> None:
        self.id = {"name": "Fake"}
        self.calls = 0

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(self, board: chess.Board, limit: Any, root_moves: Any = None) -> dict:
        self.calls += 1
        cp = 300 if root_moves is None else root_moves[0].to_square * 5
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}

    def quit(self) -> None:
        pass


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    engine = FakeEngine()
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: engine)
    return engine


def test_lichess_curves_match_their_reference_points() -> None:
    assert win_percent(0) == pytest.approx(50)
    assert win_percent(300) == pytest.approx(75.1, abs=0.1)
    assert win_percent(-300) == pytest.approx(100 - win_percent(300))
    assert move_accuracy(0) == pytest.approx(100, abs=0.01)
    assert move_accuracy(15) == pytest.approx(50.52, abs=0.01)
    assert move_accuracy(100) == 0


def test_puzzle_rating_recovers_the_rating_that_explains_the_solves() -> None:
    ratings = np.full(400, 1500.0)
    solved = np.zeros(400, dtype=bool)
    solved[:200] = True
    rating, error = puzzle_rating(ratings, solved)
    assert rating == pytest.approx(1500, abs=1)
    # 400 / ln 10 / sqrt(400 * 0.25) for 400 even-odds puzzles.
    assert error == pytest.approx(17.37, abs=0.05)
    stronger, _ = puzzle_rating(ratings, np.arange(400) < 300)
    assert stronger > 1600


def test_judge_scores_win_chance_lost_and_caches_on_disk(
    fake_engine: FakeEngine, tmp_path: Path
) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    with StockfishJudge(Path("stockfish"), 1000, tmp_path, engines=2) as judge:
        lost = judge.lost(board, move)
        assert lost == pytest.approx(win_percent(300) - win_percent(28 * 5))
        calls = fake_engine.calls
        assert judge.lost(board, move) == pytest.approx(lost)
        assert fake_engine.calls == calls
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_judge_replaces_an_empty_cache_file_and_leaves_no_partial_files(
    fake_engine: FakeEngine, tmp_path: Path
) -> None:
    board = chess.Board()
    with StockfishJudge(Path("stockfish"), 1000, tmp_path, engines=1) as judge:
        path = judge._path(board.fen(), None)
        path.parent.mkdir(parents=True)
        path.write_text("", encoding="utf-8")
        assert judge.win(board, None) == pytest.approx(win_percent(300))
    assert json.loads(path.read_text(encoding="utf-8")) == {"win": win_percent(300)}
    assert not list(tmp_path.rglob("*.tmp"))


def _mates(board: chess.Board, action: int) -> bool:
    after = board.copy()
    after.push(action_to_move(board, action))
    return after.is_checkmate()


def test_whole_puzzles_need_every_solver_move(puzzle_corpus: Path) -> None:
    split = PuzzleSplit(puzzle_corpus / "test")
    indices = np.arange(len(split))
    predicted = np.asarray(split.action).copy()
    solved, ratings = whole_puzzles_solved(split, indices, predicted)
    assert solved.all()
    assert len(ratings) == len(np.unique(np.asarray(split.puzzle)))
    legal = np.flatnonzero(split.legal_mask(np.arange(1))[0])
    board = chess.Board(str(split.fens[0])) if split.fens is not None else chess.Board()
    predicted[0] = next(
        action for action in legal if action != predicted[0] and not _mates(board, int(action))
    )
    solved_after, _ = whole_puzzles_solved(split, indices, predicted)
    assert solved_after.sum() == len(solved) - 1


def test_compare_strength_pairs_the_same_positions(tmp_path: Path) -> None:
    indices = np.arange(5)
    np.savez(tmp_path / "a.npz", game_indices=indices, lost_t1=np.array([0, 20, 20, 3, 1.0]))
    np.savez(tmp_path / "b.npz", game_indices=indices, lost_t8=np.array([0, 0, 20, 3, 1.0]))
    result = compare_strength(tmp_path / "a.npz", 1, tmp_path / "b.npz", 8)
    assert result["good_rate"]["difference"] == pytest.approx(0.2)
    assert result["blunder_rate"]["difference"] == pytest.approx(-0.2)
    assert result["lichess_accuracy"]["difference"] > 0
    np.savez(tmp_path / "c.npz", game_indices=indices + 1, lost_t8=np.zeros(5))
    with pytest.raises(ValueError):
        compare_strength(tmp_path / "a.npz", 1, tmp_path / "c.npz", 8)


def test_judge_positions_costs_each_depth(fake_engine: FakeEngine, tmp_path: Path) -> None:
    board = chess.Board()
    moves = [chess.Move.from_uci(uci) for uci in ("e2e4", "a2a3")]
    split = cast(Any, SimpleNamespace(fens=np.array([board.fen()])))
    predictions = {
        1: np.array([move_to_action(board, moves[0])]),
        2: np.array([move_to_action(board, moves[1])]),
    }
    with StockfishJudge(Path("stockfish"), 1000, tmp_path, engines=1) as judge:
        lost = strength.judge_positions(split, np.arange(1), predictions, judge, workers=2)
    assert lost[1][0] < lost[2][0]


def test_valid_sample_skips_impossible_boards() -> None:
    fens = np.array(
        [chess.Board().fen(), "4k3/8/PPPPPPPP/BBBQQNNN/RR1RR1RR/3QQ2K/8/8 w - - 0 1"] * 3
    )
    chosen = strength.valid_sample(cast(Any, _FenSplit(fens)), 3)
    assert sorted(chosen.tolist()) == [0, 2, 4]


class _FenSplit:
    def __init__(self, fens: np.ndarray) -> None:
        self.fens = fens

    def __len__(self) -> int:
        return len(self.fens)
