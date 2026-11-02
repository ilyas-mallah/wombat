from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat.chess.moves import action_to_move
from wombat.data.lichess import PuzzleRecord
from wombat.data.puzzle_corpus import solver_positions
from wombat.eval import stopping
from wombat.eval.full_puzzles import (
    EncodedPositions,
    export_puzzle_csv,
    line_verdict,
    read_puzzle_csv,
)
from wombat.eval.puzzle_corpus_eval import predict_depths
from wombat.eval.stopping import (
    StoppingRule,
    evaluate_stopping,
    first_step,
    game_oracle_steps,
    halves,
    lines_solved,
    puzzle_oracle_steps,
    step_trajectories,
)
from wombat.model import NetworkConfig, WombatNetwork

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


def test_agree_stops_once_the_argmax_held_for_the_window() -> None:
    actions = np.array(
        [
            [5, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 2, 3, 3, 3, 4, 4, 4],
        ]
    )
    kl = np.zeros(actions.shape)

    def agree(window: int, minimum: int, cap: int) -> list[int]:
        rule = StoppingRule("agree", window=window, minimum=minimum, cap=cap)
        return list(rule.stop(actions, kl).tolist())

    assert agree(2, 2, 8) == [3, 8, 2, 5]
    assert agree(2, 4, 8) == [4, 8, 4, 5]
    assert agree(3, 2, 6) == [4, 6, 3, 6]
    assert StoppingRule("fixed", steps=12).stop(actions, kl).tolist() == [12] * 4


def test_kl_stops_after_the_window_of_small_changes() -> None:
    kl = np.array(
        [
            [0.0, 0.5, 0.2, 0.001, 0.001, 0.5, 0.001, 0.001, 0.001],
            [0.0, 0.001, 0.001, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    actions = np.zeros(kl.shape, dtype=np.int64)
    rule = StoppingRule("kl", window=2, minimum=2, threshold=0.01, cap=8)
    assert rule.stop(actions, kl).tolist() == [4, 2]
    rule = StoppingRule("kl", window=3, minimum=2, threshold=0.01, cap=8)
    assert rule.stop(actions, kl).tolist() == [8, 8]
    # Column 0 never counts as a small change.
    rule = StoppingRule("kl", window=1, minimum=0, threshold=0.01, cap=8)
    assert rule.stop(actions, kl).tolist() == [3, 1]


def test_first_step_respects_the_minimum_and_the_cap() -> None:
    condition = np.array([[True, True, False, False], [False, False, False, True]])
    assert first_step(condition, 1, 3).tolist() == [1, 3]
    assert first_step(condition, 2, 2).tolist() == [2, 2]


def test_halves_split_even_rows_for_tuning_and_odd_rows_for_testing() -> None:
    tune, test = halves(5)
    assert tune.tolist() == [0, 2, 4] and test.tolist() == [1, 3]


def test_oracles_take_the_earliest_best_step() -> None:
    lost = np.array([[np.nan, 9.0, 1.0, 1.0, 5.0], [np.nan, 0.0, 3.0, 0.0, 0.0]])
    assert game_oracle_steps(lost, 4).tolist() == [2, 1]
    match = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]], dtype=bool)
    mate = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
    assert puzzle_oracle_steps(match, mate, 3).tolist() == [2, 3, 3]


def test_lines_solved_matches_line_verdict() -> None:
    # Three puzzles of 2, 1 and 3 positions, two steps each: 1 is the solution, 2 another mate.
    outcome = np.array([[1, 0], [0, 1], [0, 2], [1, 1], [2, 0], [0, 0]])
    match, mate = outcome == 1, outcome == 2
    starts = np.array([0, 2, 3])

    def solved(*steps: int) -> list[bool]:
        return list(lines_solved(match, mate, starts, np.array(steps)).tolist())

    assert solved(0, 0, 0, 0, 0, 0) == [False, False, True]
    assert solved(0, 1, 0, 0, 1, 1) == [True, False, False]
    assert solved(1, 1, 1, 0, 0, 0) == [False, True, True]


def test_trajectories_follow_the_model_readout(
    puzzle_archive: tuple[Path, list[PuzzleRecord]],
) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(TINY).eval()
    boards = [item.board for record in puzzle_archive[1] for item in solver_positions(record)]
    encoded = EncodedPositions(boards)
    indices = np.arange(len(boards))
    device = torch.device("cpu")
    actions, kl = step_trajectories(model, encoded, indices, 4, batch_size=3, device=device)
    expected = predict_depths(model, encoded, indices, (0, 1, 2, 4), batch_size=5, device=device)
    for depth, chosen in expected.items():
        assert actions[:, depth].tolist() == chosen.tolist()
    assert actions.shape == kl.shape == (len(boards), 5)
    assert (kl[:, 0] == 0).all() and (kl[:, 1:] >= -1e-5).all() and (kl[:, 1:] > 0).any()


class _FakeEngine:
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


def test_report_scores_every_rule_on_both_halves(
    monkeypatch: pytest.MonkeyPatch,
    puzzle_archive: tuple[Path, list[PuzzleRecord]],
    puzzle_corpus: Path,
    game_corpus: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: _FakeEngine())
    csv_path = tmp_path / "puzzles.csv"
    export_puzzle_csv(puzzle_corpus / "test", puzzle_archive[0], csv_path)
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {"model": WombatNetwork(TINY).state_dict(), "model_config": asdict(TINY)}, checkpoint
    )
    report = evaluate_stopping(
        checkpoint,
        puzzles=csv_path,
        games=game_corpus / "test",
        output=tmp_path / "out",
        stockfish=Path("stockfish"),
        engines=1,
        cache_dir=tmp_path / "cache",
        max_steps=16,
        batch_size=2,
    )
    results = report["results"]
    assert isinstance(results, dict)
    names = {rule.name for family in stopping.candidate_rules(16).values() for rule in family}
    for part in ("games", "puzzles"):
        section = results[part]
        assert set(section["rules"]) == names | {"oracle"}
        assert section["tune_count"] + section["test_count"] == (
            report["game_positions"] if part == "games" else report["puzzles"]
        )
        for name in ("agree_best", "kl_best"):
            chosen = section["tuned"][name]
            assert chosen["test"] == section["rules"][chosen["rule"]]["test"]
    games = results["games"]["rules"]
    assert games["oracle"]["all"]["lichess_accuracy"] >= games["t12"]["all"]["lichess_accuracy"]
    assert games["t12"]["all"]["versus_t12"]["lichess_accuracy"]["difference"] == 0
    puzzles = results["puzzles"]["rules"]
    assert puzzles["oracle"]["all"]["accuracy"] >= puzzles["t16"]["all"]["accuracy"]
    assert puzzles["t8"]["all"]["mean_steps"] == 8
    with np.load(tmp_path / "out" / "trajectories.npz") as saved:
        lengths = saved["puzzle_lengths"]
        actions = saved["puzzle_actions"]
    # The t16 verdicts equal line_verdict on the step-16 moves.
    lines = [solver_positions(record) for record in read_puzzle_csv(csv_path)]
    moves = [
        [
            action_to_move(item.board, int(action))
            for item, action in zip(line, actions[end - len(line) : end, 16])
        ]
        for line, end in zip(lines, np.cumsum(lengths))
    ]
    solved = [line_verdict(line, chosen)[0] for line, chosen in zip(lines, moves)]
    assert puzzles["t16"]["all"]["solved"] == sum(solved)
    summary = stopping.held_out_summary(report)
    assert {"t8", "t12", "t16", "oracle", "agree_best", "kl_best"} <= set(summary["games"])
    assert summary["puzzles"]["t16"]["versus"][0] == 0
    json.dumps(summary)


def test_tuning_prefers_accuracy_and_the_cheapest_rule_meets_the_floor() -> None:
    rules = [StoppingRule("agree", window=window, minimum=2) for window in (2, 3, 4)]
    scores = {2: (80.0, 5.0), 3: (82.0, 7.0), 4: (82.0, 9.0)}
    chosen = stopping._tune(rules, lambda rule: scores[rule.window], 81.0)
    assert chosen["best"] == rules[1] and chosen["cheapest"] == rules[1]
    assert stopping._tune(rules, lambda rule: scores[rule.window], 90.0)["cheapest"] is None
