from __future__ import annotations

import csv
from dataclasses import asdict
import io
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
import torch
import zstandard

from wombat.cli import main
from wombat.data.lichess import PuzzleRecord
from wombat.data.puzzle_corpus import PuzzleSplit, SolverPosition, solver_positions
from wombat.eval.full_puzzles import (
    EncodedPositions,
    evaluate_full_puzzles,
    export_puzzle_csv,
    line_verdict,
    rating_band,
    read_puzzle_csv,
    score_external_puzzles,
)
from wombat.model import NetworkConfig, WombatNetwork
from wombat.selection import CHILD_MIX_BETA, CHILD_TOP_K

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


@pytest.fixture
def exported(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], puzzle_corpus: Path, tmp_path: Path
) -> Path:
    path = tmp_path / "puzzles.csv"
    summary = export_puzzle_csv(puzzle_corpus / "test", puzzle_archive[0], path, count=50)
    split = PuzzleSplit(puzzle_corpus / "test")
    assert summary["puzzles"] == len(split.puzzle_ids)
    assert summary["positions"] == len(split)
    return path


def test_export_writes_whole_test_puzzles_that_chessbench_replays(
    exported: Path, puzzle_archive: tuple[Path, list[PuzzleRecord]], puzzle_corpus: Path
) -> None:
    records = {record.puzzle_id: record for record in puzzle_archive[1]}
    with exported.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["PuzzleId", "Rating", "PGN", "Solution", "FEN", "Moves"]
    split = PuzzleSplit(puzzle_corpus / "test")
    assert sorted(row["PuzzleId"] for row in rows) == sorted(map(str, split.puzzle_ids))
    for row in rows:
        record = records[row["PuzzleId"]]
        assert (row["FEN"], row["Moves"].split(), int(row["Rating"])) == (
            record.fen,
            list(record.moves),
            record.rating,
        )
        game = chess.pgn.read_game(io.StringIO(row["PGN"]))
        assert game is not None and game.end().board().fen() == record.fen
        assert len(row["Solution"].split()) == len(record.moves)


def test_export_refuses_an_archive_that_differs_from_the_split(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], puzzle_corpus: Path, tmp_path: Path
) -> None:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=("PuzzleId", "FEN", "Moves", "Rating", "Themes"))
    writer.writeheader()
    for record in puzzle_archive[1]:
        writer.writerow(
            {
                "PuzzleId": record.puzzle_id,
                "FEN": record.fen,
                "Moves": " ".join(record.moves),
                "Rating": record.rating + 1,
                "Themes": "",
            }
        )
    changed = tmp_path / "changed.csv.zst"
    changed.write_bytes(zstandard.ZstdCompressor().compress(text.getvalue().encode()))
    with pytest.raises(ValueError, match="differs"):
        export_puzzle_csv(puzzle_corpus / "test", changed, tmp_path / "x.csv")


def _mate_in_one() -> list[SolverPosition]:
    # Setup move h7h6, then Qxf7 mates; Qh5-f7 is the listed solution, Bc4xf7 is not mate.
    record = PuzzleRecord(
        "mate",
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        ("g8f6", "h5f7"),
        900,
        (),
    )
    return solver_positions(record)


def test_line_verdict_accepts_another_mate_but_not_as_exact() -> None:
    line = _mate_in_one()
    board = line[0].board
    assert line_verdict(line, [line[0].move]) == (True, True)
    assert line_verdict(line, [board.parse_uci("c4f7")]) == (False, False)
    other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1")
    rook = [SolverPosition(other, other.parse_uci("a1a8"), 1, 0)]
    assert line_verdict(rook, [other.parse_uci("e1e8")]) == (True, False)


def test_line_verdict_ignores_moves_after_the_first_miss(exported: Path) -> None:
    line = next(
        lines for lines in map(solver_positions, read_puzzle_csv(exported)) if len(lines) > 1
    )
    board = line[0].board
    wrong = next(
        move for move in board.legal_moves if move != line[0].move and not _mates(board, move)
    )
    chosen: list[chess.Move | None] = [wrong] + [None] * (len(line) - 1)
    assert line_verdict(line, chosen) == (False, False)
    with pytest.raises(ValueError):
        line_verdict(line, [line[0].move] + [None] * (len(line) - 1))


def test_rating_bands_are_chessbench_bands_of_200() -> None:
    assert rating_band(1200) == "1200-1399"
    assert rating_band(1399) == "1200-1399"
    assert rating_band(399) == "200-399"


def test_external_moves_score_whole_lines(exported: Path, tmp_path: Path) -> None:
    lines = [solver_positions(record) for record in read_puzzle_csv(exported)]
    solution = {item.board.fen(): item.move.uci() for line in lines for item in line}
    moves = tmp_path / "moves.json"
    moves.write_text(json.dumps(solution), encoding="utf-8")
    report = score_external_puzzles(moves, exported, output=tmp_path / "report")
    result = report["results"]["external"]  # type: ignore[index]
    assert result["accuracy"] == 1.0 and result["exact_accuracy"] == 1.0
    assert sum(band["puzzles"] for band in result["by_rating"].values()) == len(lines)
    first = lines[0][0]
    wrong = next(
        move
        for move in first.board.legal_moves
        if move != first.move and not _mates(first.board, move)
    )
    solution[first.board.fen()] = wrong.uci()
    moves.write_text(json.dumps(solution), encoding="utf-8")
    report = score_external_puzzles(moves, exported, output=tmp_path / "report")
    assert report["results"]["external"]["solved"] == len(lines) - 1  # type: ignore[index]
    with np.load(tmp_path / "report" / "solved.npz") as saved:
        assert not saved["external"][0] and saved["external"][1:].all()


def _mates(board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


def test_encoded_positions_match_the_corpus_batches(exported: Path, puzzle_corpus: Path) -> None:
    split = PuzzleSplit(puzzle_corpus / "test")
    boards = {
        item.board.fen(): item.board
        for record in read_puzzle_csv(exported)
        for item in solver_positions(record)
    }
    assert split.fens is not None
    encoded = EncodedPositions([boards[str(fen)] for fen in split.fens])
    indices = np.arange(len(split))
    ours, theirs = encoded.batch(indices), split.batch(indices)
    for key in ("pieces", "state", "legal_mask"):
        assert torch.equal(ours[key], theirs[key].to(ours[key].dtype))


def test_checkpoint_report_covers_each_depth(exported: Path, tmp_path: Path) -> None:
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    model = WombatNetwork(TINY)
    torch.save({"model": model.state_dict(), "model_config": asdict(TINY)}, checkpoint)
    report = evaluate_full_puzzles(
        checkpoint, exported, (2, 1), output=tmp_path / "out", batch_size=4
    )
    results = report["results"]
    assert isinstance(results, dict) and list(results) == ["t1", "t2"]
    for result in results.values():
        assert 0 <= result["accuracy"] <= 1 and result["puzzles"] == report["puzzles"]
    with np.load(tmp_path / "out" / "solved.npz") as saved:
        assert len(saved["t1"]) == report["puzzles"]
    assert (report["select"], report["select_beta"], report["select_top_k"]) == (
        "policy",
        None,
        None,
    )


def test_child_selection_report_records_the_mode(exported: Path, tmp_path: Path) -> None:
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    model = WombatNetwork(TINY)
    torch.save({"model": model.state_dict(), "model_config": asdict(TINY)}, checkpoint)
    command = [
        "evaluate-full-puzzles",
        "--puzzles",
        str(exported),
        "--checkpoint",
        str(checkpoint),
        "--depths",
        "1,2",
        "--batch-size",
        "4",
    ]
    output = tmp_path / "child-mix"
    assert main([*command, "--output", str(output), "--select", "child-mix"]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert (report["select"], report["select_beta"], report["select_top_k"]) == (
        "child-mix",
        CHILD_MIX_BETA,
        CHILD_TOP_K,
    )
    assert list(report["results"]) == ["t1", "t2"]
    output = tmp_path / "child"
    flags = ["--select", "child", "--select-top-k", "2"]
    assert main([*command, "--output", str(output), *flags]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert (report["select"], report["select_beta"], report["select_top_k"]) == ("child", None, 2)
