import json
from pathlib import Path

import chess
import zstandard

from wombat.chess.moves import move_to_action
from wombat.data.build import build_evaluation_corpus
from wombat.data.lichess import iter_evaluations, parse_evaluation
from wombat.data.splits import normalized_position, split_for_fen


def test_lichess_white_relative_score_is_flipped_for_black() -> None:
    payload = {
        "fen": "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
        "evals": [
            {
                "pvs": [{"cp": 69, "line": "f7g7 e6e2"}],
                "knodes": 123,
                "depth": 46,
            }
        ],
    }
    record = parse_evaluation(payload)
    assert record.score_cp == -69
    assert record.nodes == 123_000
    assert chess.Move.from_uci(record.best_move) in chess.Board(record.fen).legal_moves


def test_position_normalization_ignores_move_clocks() -> None:
    assert normalized_position(chess.STARTING_FEN) == normalized_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 42 99"
    )


def test_lichess_castling_pv_is_stored_in_standard_uci_form() -> None:
    payload = {
        "fen": "rn1qk2r/pbppppbp/1p3np1/8/2PP4/5NP1/PP2PPBP/RNBQ1RK1 b kq -",
        "evals": [{"pvs": [{"cp": 42, "line": "e8h8"}], "depth": 20}],
    }
    record = parse_evaluation(payload)
    assert record.best_move == "e8g8"
    assert record.action == move_to_action(chess.Board(record.fen), chess.Move.from_uci("e8g8"))
    assert split_for_fen(chess.STARTING_FEN) == split_for_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 42 99"
    )


def test_debug_corpus_is_deterministic_and_excludes_fens(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl.zst"
    rows = [
        {
            "fen": chess.STARTING_FEN,
            "evals": [{"pvs": [{"cp": 20, "line": "e2e4"}], "depth": 10}],
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "evals": [{"pvs": [{"cp": 15, "line": "c7c5"}], "depth": 11}],
        },
    ]
    with zstandard.open(source, mode="wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    first = build_evaluation_corpus(
        source, tmp_path / "first", limit=2, excluded_fens=(chess.STARTING_FEN,), minimum_free_gb=0
    )
    second = build_evaluation_corpus(
        source, tmp_path / "second", limit=2, excluded_fens=(chess.STARTING_FEN,), minimum_free_gb=0
    )
    assert first["written_records"] == 1
    assert first["rejected"] == {"puzzle_overlap": 1}
    assert first["files"]["train"]["sha256"] == second["files"]["train"]["sha256"]


def test_streaming_parser_can_count_and_skip_unsupported_records(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl.zst"
    rows = [
        {
            "fen": chess.STARTING_FEN,
            "evals": [{"pvs": [{"cp": 20, "line": "e2e4"}], "depth": 10}],
        },
        {
            "fen": chess.STARTING_FEN,
            "evals": [{"pvs": [{"cp": 20, "line": "e2e5"}], "depth": 10}],
        },
    ]
    with zstandard.open(source, mode="wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    rejected: list[str] = []
    records = list(iter_evaluations(source, rejected.append))
    assert len(records) == 1
    assert rejected == ["IllegalMoveError"]
