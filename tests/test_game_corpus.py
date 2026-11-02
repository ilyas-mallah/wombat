from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zstandard

from wombat.chess.moves import legal_action_mask, move_to_action
from wombat.cli import main
from wombat.data.game_corpus import (
    SPLITS,
    GameSplit,
    _encode_lines,
    build_game_corpus,
    graded_candidates,
    select_evaluation,
    win_chance,
)
from wombat.eval.manifest import PositionManifest


def test_selection_prefers_most_variations_among_deep_evaluations() -> None:
    evaluations = [
        {"depth": 40, "pvs": [{}]},
        {"depth": 22, "pvs": [{}, {}, {}]},
        {"depth": 30, "pvs": [{}, {}, {}]},
        {"depth": 12, "pvs": [{}, {}, {}, {}, {}]},
    ]
    assert select_evaluation(evaluations) == evaluations[2]
    assert select_evaluation([{"depth": 5, "pvs": [{}]}]) is None


def test_candidates_are_graded_from_the_side_to_move() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    payload = {
        "evals": [
            {
                "depth": 25,
                "pvs": [
                    {"cp": 50, "line": "e7e5 g1f3"},
                    {"cp": -30, "line": "c7c5"},
                    {"mate": -3, "line": "d7d5"},
                ],
            }
        ]
    }
    candidates = graded_candidates(payload, board)
    assert not isinstance(candidates, str)
    # Lichess scores are White-relative; Black to move prefers negative scores.
    assert [action for action, _ in candidates] == [
        move_to_action(board, chess.Move.from_uci(uci)) for uci in ("d7d5", "c7c5", "e7e5")
    ]
    assert candidates[0][1] == 1.0
    assert candidates[1][1] == pytest.approx(win_chance(30))
    assert candidates[2][1] == pytest.approx(win_chance(-50))


def test_corpus_keeps_deep_multi_candidate_positions_with_exact_legal_moves(
    eval_archive: tuple[Path, list[dict]], tmp_path: Path
) -> None:
    source, _ = eval_archive
    manifest = build_game_corpus(
        source, tmp_path / "games", limit=100, low_priority=False, minimum_free_gb=0
    )
    assert manifest["rejected"] == {"shallow": 1, "few_candidates": 1}
    splits = manifest["splits"]
    assert isinstance(splits, dict)
    assert {name: splits[name]["positions"] for name in splits} == {
        "train": 8,
        "validation": 3,
        "test": 3,
    }
    for name in ("validation", "test"):
        data = GameSplit(tmp_path / "games" / name)
        assert data.fens is not None
        masks = data.legal_mask(np.arange(len(data)))
        for row, fen in enumerate(data.fens):
            assert np.array_equal(masks[row], legal_action_mask(chess.Board(str(fen))))


def test_graded_targets_favor_better_candidates(
    eval_archive: tuple[Path, list[dict]], tmp_path: Path
) -> None:
    source, _ = eval_archive
    build_game_corpus(source, tmp_path / "games", limit=100, low_priority=False, minimum_free_gb=0)
    sharp = GameSplit(tmp_path / "games" / "train", temperature=4.0)
    flat = GameSplit(tmp_path / "games" / "train", temperature=40.0)
    indices = np.arange(len(sharp))
    targets = sharp.targets(indices)
    assert np.allclose(targets["policy_target"].sum(axis=1), 1.0)
    assert np.array_equal(targets["policy_target"].argmax(axis=1), sharp.action[indices])
    assert np.array_equal(targets["move_value_mask"], targets["policy_target"] > 0)
    assert np.all(np.abs(targets["move_value_target"]) <= 1)
    best = targets["move_value_target"].max(axis=1, where=targets["move_value_mask"], initial=-1)
    assert np.allclose(targets["value"], best)
    softer = flat.targets(indices)["policy_target"]
    assert np.all(softer.max(axis=1) < targets["policy_target"].max(axis=1))
    batch = sharp.batch(indices[:4])
    assert batch["legal_mask"][range(4), batch["action"]].all()
    assert batch["policy_target"].shape == (4, 4672)


def test_limit_trims_and_hashes_reproduce_across_workers(
    eval_archive: tuple[Path, list[dict]], tmp_path: Path
) -> None:
    source, _ = eval_archive
    serial = build_game_corpus(
        source, tmp_path / "serial", limit=100, low_priority=False, minimum_free_gb=0
    )
    parallel = build_game_corpus(
        source,
        tmp_path / "parallel",
        limit=100,
        workers=2,
        chunk_size=3,
        low_priority=False,
        minimum_free_gb=0,
    )
    assert json.dumps(serial["splits"], sort_keys=True) == json.dumps(
        parallel["splits"], sort_keys=True
    )
    trimmed = build_game_corpus(
        source, tmp_path / "trimmed", limit=5, low_priority=False, minimum_free_gb=0
    )
    splits = trimmed["splits"]
    assert isinstance(splits, dict)
    assert sum(split["positions"] for split in splits.values()) == 5


@pytest.mark.parametrize("workers", [1, 2])
def test_skipped_records_and_their_positions_are_left_out(
    eval_archive: tuple[Path, list[dict]], tmp_path: Path, workers: int
) -> None:
    _, records = eval_archive
    # The first record's position again at the end, as a later record.
    text = "".join(json.dumps(record) + "\n" for record in [*records, records[0]])
    source = tmp_path / "evals.jsonl.zst"
    source.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))

    def build(name: str, skip: int) -> dict[str, Any]:
        return build_game_corpus(
            source,
            tmp_path / name,
            limit=100,
            skip_records=skip,
            workers=workers,
            chunk_size=2,
            low_priority=False,
            minimum_free_gb=0,
        )

    def fens(name: str) -> set[str]:
        return {
            str(fen)
            for split in ("validation", "test")
            for fen in np.load(tmp_path / name / split / "fens.npy")
        }

    full = build("full", 0)
    later = build("later", 3)
    # Records 0 and 1 are kept in the full build, record 2 is shallow.
    assert later["records_read"] == len(records) + 1
    assert later["earlier_positions"] == 3
    assert later["rejected"] == {"few_candidates": 1, "earlier_record": 1}
    assert _positions(later) == _positions(full) - 3
    skipped = {chess.Board(record["fen"]).fen() for record in records[:3]}
    assert fens("later") == fens("full") - skipped
    rest = build("rest", len(records) + 1)
    assert _positions(rest) == 0


def _positions(manifest: dict[str, Any]) -> int:
    return sum(split["positions"] for split in manifest["splits"].values())


def test_cli_leaves_out_the_positions_of_given_splits(
    eval_archive: tuple[Path, list[dict]],
    game_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wombat.data.game_corpus.lower_priority", lambda: None)
    frozen = tmp_path / "frozen.json"
    PositionManifest(name="none", positions=()).write(frozen)
    output = tmp_path / "games"
    arguments = ["build-game-corpus", "--source", str(eval_archive[0]), "--output", str(output)]
    arguments += ["--puzzle-manifest", str(frozen), "--limit", "100", "--workers", "1"]
    arguments += ["--exclude-splits", str(game_corpus / "validation"), str(game_corpus / "test")]
    assert main(arguments) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["excluded_fens"] == 6
    assert manifest["rejected"]["frozen_fen"] == 6
    assert [manifest["splits"][name]["positions"] for name in SPLITS] == [8, 0, 0]


def test_impossible_positions_are_rejected() -> None:
    impossible = "4k3/8/PPPPPPPP/BBBQQNNN/RR1RR1RR/3QQ2K/8/8 w - -"
    line = json.dumps({"fen": impossible, "evals": []})
    _, rejected = _encode_lines([line])
    assert rejected == {"invalid_position": 1}
