from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from wombat.data.game_corpus import GameSplit
from wombat.eval.agreement import DRAW_GUARD_VALUE
from wombat.eval.games import (
    elo_difference,
    pair_statistics,
    pentanomial,
    play_match,
    play_stockfish,
    select_openings,
)
from wombat.model import NetworkConfig, WombatNetwork
from wombat.policy import DeterministicRandomPolicy

FAKE_UCI = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake_uci.py")]
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
WIDE_BAND = (0.0, 1.0)


def _tiny_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    model = WombatNetwork(TINY)
    torch.save({"model": model.state_dict(), "model_config": asdict(TINY)}, path)
    return path


def test_elo_difference_is_zero_at_even_and_antisymmetric() -> None:
    assert elo_difference(0.5) == 0
    assert elo_difference(0.75) == pytest.approx(400 * np.log10(3))
    assert elo_difference(0.3) == pytest.approx(-elo_difference(0.7))
    assert elo_difference(0.0) == pytest.approx(-elo_difference(1.0))
    assert np.isfinite(elo_difference(0.0))


def test_pentanomial_counts_pairs() -> None:
    assert pentanomial([0, 0.5, 1, 1, 1.5, 2, 2, 2]) == [1, 1, 2, 1, 3]
    with pytest.raises(ValueError):
        pentanomial([0.25])


def test_pair_statistics_interval_contains_estimate() -> None:
    statistics = pair_statistics([2, 1.5, 1, 1, 0.5, 2, 1.5])
    assert statistics.pairs == 7
    assert statistics.pair_score == pytest.approx(9.5 / 14)
    assert statistics.elo_low < statistics.elo_difference < statistics.elo_high
    assert statistics.score_low < statistics.pair_score < statistics.score_high
    even = pair_statistics([1, 1, 2, 0])
    assert even.elo_difference == 0
    assert even.elo_low == pytest.approx(-even.elo_high)
    # Identical pairs have no spread, so the interval collapses to the estimate.
    same = pair_statistics([1, 1, 1])
    assert same.elo_low == same.elo_difference == same.elo_high == 0
    with pytest.raises(ValueError):
        pair_statistics([])


def test_select_openings_filters_and_is_seeded(game_corpus: Path) -> None:
    split = GameSplit(game_corpus / "test")
    assert split.fens is not None
    offsets = np.asarray(split.candidate_offsets)
    best = np.asarray(split.candidate_win, dtype=np.float32)[offsets[:-1]]
    boards = [chess.Board(str(fen)) for fen in split.fens]
    wanted = {
        board.fen()
        for board, win in zip(boards, best, strict=True)
        if 0.45 <= win <= 0.6 and len(board.piece_map()) >= 20
    }
    assert wanted
    chosen = select_openings(
        game_corpus / "test", len(wanted), seed=1, win_band=(0.45, 0.6), min_pieces=20
    )
    assert set(chosen) == wanted
    assert chosen == select_openings(
        game_corpus / "test", len(wanted), seed=1, win_band=(0.45, 0.6), min_pieces=20
    )
    with pytest.raises(ValueError, match="openings pass"):
        select_openings(game_corpus / "test", 1, win_band=(0.99, 1.0))
    with pytest.raises(ValueError, match="openings pass"):
        select_openings(game_corpus / "test", 1, win_band=WIDE_BAND, min_pieces=33)


def test_tiny_match_against_fake_engine(game_corpus: Path, tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path / "tiny.pt")
    output = tmp_path / "match"
    reports: list[dict[str, Any]] = play_stockfish(
        checkpoint,
        FAKE_UCI,
        levels=(1320, 1500),
        steps=2,
        pairs=2,
        max_plies=8,
        output=output,
        split_dir=game_corpus / "test",
        win_band=WIDE_BAND,
        low_priority=False,
    )
    assert [report["opponent"]["elo"] for report in reports] == [1320, 1500]
    report = reports[0]
    assert report["status"] == "complete"
    assert report["games"] == 4
    assert report["wins"] + report["draws"] + report["losses"] == 4
    assert report["pairs"] == 2
    assert sum(report["pentanomial"]) == 2
    assert report["wombat_elo"] == pytest.approx(1320 + report["elo_difference"])
    assert json.loads((output / "elo-1320" / "match.json").read_text())["games"] == 4
    assert report["config"]["draw_guard"] == DRAW_GUARD_VALUE

    with (output / "elo-1320" / "games.pgn").open() as handle:
        games = list(iter(lambda: chess.pgn.read_game(handle), None))
    assert len(games) == 4
    openings = report["config"]["openings"]
    for index, game in enumerate(games):
        headers = game.headers
        wombat_white = index % 2 == 0
        assert headers["Round"] == f"{index // 2 + 1}.{1 if wombat_white else 2}"
        assert headers["White" if wombat_white else "Black"] == "Wombat T2"
        assert headers["Black" if wombat_white else "White"] == "FakeFish Elo 1320"
        assert headers["SetUp"] == "1"
        assert headers["FEN"] == openings[index // 2]
        assert headers["Result"] in ("1-0", "0-1", "1/2-1/2")
        assert len(list(game.mainline_moves())) <= 8


def test_match_resumes_and_rejects_other_configs(game_corpus: Path, tmp_path: Path) -> None:
    openings = select_openings(game_corpus / "test", 2, win_band=WIDE_BAND, min_pieces=2)
    output = tmp_path / "resume"
    policy = DeterministicRandomPolicy()
    first = play_match(policy, FAKE_UCI, openings[:1], output, steps=1, max_plies=6)
    assert first["games"] == 2
    # A half-written trailing game is dropped and replayed.
    with (output / "games.pgn").open("a", encoding="utf-8") as handle:
        handle.write('[Event "partial"]\n[Round "2.1"]\n')
    with pytest.raises(ValueError, match="different match"):
        play_match(policy, FAKE_UCI, openings, output, steps=1, max_plies=6)
    (output / "match.json").unlink()
    resumed = play_match(policy, FAKE_UCI, openings, output, steps=1, max_plies=6)
    assert resumed["games"] == 4
    assert resumed["pairs"] == 2
    text = (output / "games.pgn").read_text(encoding="utf-8")
    assert text.count("[Round ") == 4
    assert "partial" not in text


def test_engine_without_elo_option_plays_full_strength(
    game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_UCI_NO_ELO", "1")
    openings = select_openings(game_corpus / "test", 1, win_band=WIDE_BAND, min_pieces=2)
    report: dict[str, Any] = play_match(
        DeterministicRandomPolicy(), FAKE_UCI, openings, tmp_path / "full", steps=1, max_plies=4
    )
    assert report["opponent"]["limit_strength"] is False
    assert report["opponent"]["label"] == "FakeFish full strength"
    assert "elo_difference" in report
    assert "wombat_elo" not in report


def test_level_zero_is_full_strength_and_nodes_replace_the_movetime(
    game_corpus: Path, tmp_path: Path
) -> None:
    checkpoint = _tiny_checkpoint(tmp_path / "tiny.pt")
    reports: list[dict[str, Any]] = play_stockfish(
        checkpoint,
        FAKE_UCI,
        levels=(0,),
        steps=1,
        pairs=1,
        nodes=100,
        max_plies=4,
        output=tmp_path / "match",
        split_dir=game_corpus / "test",
        win_band=WIDE_BAND,
        low_priority=False,
    )
    report = reports[0]
    assert report["opponent"]["limit_strength"] is False
    assert "wombat_elo" not in report
    assert report["config"]["nodes"] == 100
    assert report["config"]["movetime"] is None
    assert (tmp_path / "match" / "full-strength" / "match.json").exists()


def test_out_of_range_elo_is_rejected(game_corpus: Path, tmp_path: Path) -> None:
    openings = select_openings(game_corpus / "test", 1, win_band=WIDE_BAND, min_pieces=2)
    with pytest.raises(ValueError, match="outside 1320..3190"):
        play_match(DeterministicRandomPolicy(), FAKE_UCI, openings, tmp_path / "low", elo=1000)
