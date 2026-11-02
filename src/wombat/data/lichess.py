"""Parsers for official Lichess evaluation and puzzle archives."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from typing import Callable, Iterator

import chess
import zstandard

from wombat.chess.moves import move_to_action
from wombat.data.splits import normalized_position


@dataclass(frozen=True)
class PrincipalVariation:
    move: str
    score_cp: int | None
    score_mate: int | None


@dataclass(frozen=True)
class EvaluationRecord:
    fen: str
    best_move: str
    action: int
    score_cp: int | None
    score_mate: int | None
    depth: int
    nodes: int
    alternatives: tuple[PrincipalVariation, ...]

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["alternatives"] = [asdict(pv) for pv in self.alternatives]
        return payload


@dataclass(frozen=True)
class PuzzleRecord:
    puzzle_id: str
    fen: str
    moves: tuple[str, ...]
    rating: int
    themes: tuple[str, ...]

    def critical_position(self) -> tuple[str, str]:
        if len(self.moves) < 2:
            raise ValueError(f"puzzle {self.puzzle_id} has no solution move")
        board = chess.Board(self.fen)
        board.push_uci(self.moves[0])
        best_move = chess.Move.from_uci(self.moves[1])
        if best_move not in board.legal_moves:
            raise ValueError(f"puzzle {self.puzzle_id} has illegal critical move {best_move}")
        return normalized_position(board.fen(en_passant="fen")), best_move.uci()


def _side_to_move_score(value: int, turn: chess.Color) -> int:
    """Convert Lichess's White-relative score into root side-to-move perspective."""

    return value if turn == chess.WHITE else -value


def _parse_pv(item: dict[str, object], board: chess.Board) -> PrincipalVariation:
    line = str(item["line"]).split()
    if not line:
        raise ValueError("empty principal variation")
    move = board.parse_uci(line[0])
    if move not in board.legal_moves:
        raise ValueError(f"illegal first PV move: {move.uci()}")
    cp = item.get("cp")
    mate = item.get("mate")
    return PrincipalVariation(
        move=move.uci(),
        score_cp=_side_to_move_score(int(cp), board.turn) if cp is not None else None,
        score_mate=_side_to_move_score(int(mate), board.turn) if mate is not None else None,
    )


def parse_evaluation(payload: dict[str, object]) -> EvaluationRecord:
    board = chess.Board(str(payload["fen"]))
    evaluations = payload.get("evals")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("evaluation record has no evals")
    evaluation = max(evaluations, key=lambda item: int(item.get("depth", -1)))
    pvs = evaluation.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        raise ValueError("evaluation has no PVs")
    alternatives = tuple(_parse_pv(item, board) for item in pvs)
    best = alternatives[0]
    best_move = board.parse_uci(best.move)
    return EvaluationRecord(
        fen=normalized_position(board.fen(en_passant="fen")),
        best_move=best.move,
        action=move_to_action(board, best_move),
        score_cp=best.score_cp,
        score_mate=best.score_mate,
        depth=int(evaluation["depth"]),
        nodes=int(evaluation.get("knodes", 0)) * 1000,
        alternatives=alternatives,
    )


def iter_evaluations(
    path: Path, on_reject: Callable[[str], None] | None = None
) -> Iterator[EvaluationRecord]:
    with path.open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        with io.TextIOWrapper(reader, encoding="utf-8") as text:
            for line in text:
                if line.strip():
                    try:
                        yield parse_evaluation(json.loads(line))
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                        if on_reject is None:
                            raise
                        on_reject(type(error).__name__)


def iter_puzzles(path: Path) -> Iterator[PuzzleRecord]:
    with path.open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        with io.TextIOWrapper(reader, encoding="utf-8", newline="") as text:
            for row in csv.DictReader(text):
                yield PuzzleRecord(
                    puzzle_id=row["PuzzleId"],
                    fen=row["FEN"],
                    moves=tuple(row["Moves"].split()),
                    rating=int(row["Rating"]),
                    themes=tuple(row["Themes"].split()),
                )
