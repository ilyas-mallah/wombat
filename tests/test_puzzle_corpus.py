from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from wombat.chess.moves import legal_action_mask, move_to_action
from wombat.data.lichess import PuzzleRecord
from wombat.data.puzzle_corpus import (
    PuzzleSplit,
    build_puzzle_corpus,
    puzzle_split,
    solver_positions,
)


def test_solver_positions_follow_the_line_and_count_moves_remaining() -> None:
    board = chess.Board()
    record = PuzzleRecord("x", board.fen(), ("e2e4", "e7e5", "g1f3", "b8c6"), 1500, ())
    positions = solver_positions(record)
    assert [item.remaining for item in positions] == [2, 1]
    assert [item.line_index for item in positions] == [0, 1]
    assert [item.move.uci() for item in positions] == ["e7e5", "b8c6"]
    assert (
        positions[1].board.fen()
        == chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2").fen()
    )


def test_solver_positions_reject_lines_that_end_on_the_opponent() -> None:
    record = PuzzleRecord("x", chess.Board().fen(), ("e2e4", "e7e5", "g1f3"), 1500, ())
    with pytest.raises(ValueError):
        solver_positions(record)


def test_corpus_stores_every_solver_position_with_exact_legal_moves(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], tmp_path: Path
) -> None:
    source, records = puzzle_archive
    manifest = build_puzzle_corpus(
        source, tmp_path / "corpus", low_priority=False, minimum_free_gb=0
    )
    expected = {split: 0 for split in ("train", "validation", "test")}
    for record in records:
        expected[puzzle_split(record.puzzle_id)] += len(record.moves) // 2
    files = manifest["splits"]
    assert isinstance(files, dict)
    assert {split: files[split]["positions"] for split in expected} == expected
    for split in ("validation", "test"):
        data = PuzzleSplit(tmp_path / "corpus" / split)
        assert data.fens is not None
        masks = data.legal_mask(np.arange(len(data)))
        for row, fen in enumerate(data.fens):
            board = chess.Board(str(fen))
            assert np.array_equal(masks[row], legal_action_mask(board))
            assert masks[row, data.action[row]]
            assert data.remaining[row] >= 1
        record_by_id = {record.puzzle_id: record for record in records}
        first = record_by_id[str(data.puzzle_ids[data.puzzle[0]])]
        position = solver_positions(first)[0]
        assert data.action[0] == move_to_action(position.board, position.move)
        assert data.rating[0] == first.rating


def test_corpus_excludes_frozen_puzzles_and_positions(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], tmp_path: Path
) -> None:
    source, records = puzzle_archive
    frozen_board = solver_positions(records[1])[0].board
    manifest = build_puzzle_corpus(
        source,
        tmp_path / "corpus",
        excluded_puzzle_ids=[records[0].puzzle_id],
        excluded_fens=[frozen_board.fen(en_passant="fen")],
        low_priority=False,
        minimum_free_gb=0,
    )
    assert manifest["rejected"] == {"frozen_puzzle_id": 1, "frozen_fen_overlap": 1}
    kept: set[str] = set()
    for split in ("train", "validation", "test"):
        kept.update(str(item) for item in PuzzleSplit(tmp_path / "corpus" / split).puzzle_ids)
    assert records[0].puzzle_id not in kept
    assert records[1].puzzle_id not in kept
    assert len(kept) == len(records) - 2


def test_corpus_hashes_are_reproducible_across_worker_counts(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], tmp_path: Path
) -> None:
    source, _ = puzzle_archive
    serial = build_puzzle_corpus(source, tmp_path / "serial", low_priority=False, minimum_free_gb=0)
    parallel = build_puzzle_corpus(
        source,
        tmp_path / "parallel",
        workers=2,
        chunk_size=3,
        low_priority=False,
        minimum_free_gb=0,
    )
    assert json.dumps(serial["splits"], sort_keys=True) == json.dumps(
        parallel["splits"], sort_keys=True
    )


def test_shuffled_batches_cover_the_split_and_carry_legal_targets(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], tmp_path: Path
) -> None:
    source, _ = puzzle_archive
    build_puzzle_corpus(source, tmp_path / "corpus", low_priority=False, minimum_free_gb=0)
    data = PuzzleSplit(tmp_path / "corpus" / "train")
    batches = data.shuffled_batches(4, np.random.default_rng(0))
    batch = next(batches)
    assert batch["pieces"].shape == (4, 64)
    assert batch["state"].shape == (4, 26)
    assert batch["legal_mask"].shape == (4, 4672)
    assert bool(batch["legal_mask"][range(4), batch["action"]].all())
