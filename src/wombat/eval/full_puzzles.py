"""Whole-puzzle solving on a Lichess-format puzzle CSV, scored the way ChessBench scores it.

The CSV follows ChessBench's ``puzzles.csv``: ``FEN`` is the position before the opponent's
setup move, ``Moves`` starts with that move, and ``PGN`` replays to ``FEN``, so their
``puzzles.py`` reads an exported file unchanged. A puzzle is solved when every solver move
matches the line, except that a different move which checkmates ends the puzzle solved.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import json
from pathlib import Path

import chess
import numpy as np
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import ACTION_SIZE, action_to_move, legal_actions
from wombat.data.lichess import PuzzleRecord, iter_puzzles
from wombat.data.puzzle_corpus import PuzzleSplit, SolverPosition, lower_priority, solver_positions
from wombat.eval.puzzle_corpus_eval import (
    load_checkpoint,
    predict_depths,
    wilson_interval,
    workspace_line_plies,
)
from wombat.eval.strength import puzzle_rating
from wombat.gpu_share import share_gpu
from wombat.selection import CHILD_MODES, CHILD_TOP_K, MIX_BETA, recorded_beta

CSV_FIELDS = ("PuzzleId", "Rating", "PGN", "Solution", "FEN", "Moves")
RATING_BAND = 200


def _setup_pgn(fen: str) -> str:
    """A game with no moves that starts at ``fen``, for readers that replay the PGN."""

    return f'[FEN "{fen}"]\n[SetUp "1"]\n\n*'


def _solution_san(record: PuzzleRecord) -> str:
    board = chess.Board(record.fen)
    sans = []
    for uci in record.moves:
        move = board.parse_uci(uci)
        sans.append(board.san(move))
        board.push(move)
    return " ".join(sans)


def export_puzzle_csv(
    split_path: Path, source: Path, output: Path, *, count: int = 10_000, seed: int = 0
) -> dict[str, object]:
    """Write ``count`` seeded whole puzzles of a corpus split as a ChessBench-format CSV.

    FENs and setup moves come from the Lichess archive the split was built from, and every
    replayed solver position must match the split's stored FEN.
    """

    split = PuzzleSplit(split_path)
    if split.fens is None:
        raise ValueError("exporting puzzles needs a split with FENs")
    puzzle = np.asarray(split.puzzle)
    if np.any(np.diff(puzzle) < 0):
        raise ValueError("split positions are not grouped by puzzle")
    total = len(split.puzzle_ids)
    numbers = np.sort(np.random.default_rng(seed).choice(total, min(count, total), replace=False))
    wanted = {str(split.puzzle_ids[number]): int(number) for number in numbers}
    found: dict[str, PuzzleRecord] = {}
    for record in iter_puzzles(source):
        if record.puzzle_id in wanted:
            found[record.puzzle_id] = record
            if len(found) == len(wanted):
                break
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"{len(missing)} puzzles are not in {source}, first {missing[0]}")
    starts = np.searchsorted(puzzle, numbers, side="left")
    ends = np.searchsorted(puzzle, numbers, side="right")
    output.parent.mkdir(parents=True, exist_ok=True)
    positions = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for number, start, end in zip(numbers, starts, ends):
            record = found[str(split.puzzle_ids[number])]
            replayed = [item.board.fen() for item in solver_positions(record)]
            stored = [str(fen) for fen in split.fens[start:end]]
            if replayed != stored or int(split.rating[start]) != record.rating:
                raise ValueError(f"puzzle {record.puzzle_id} differs from the split")
            positions += len(stored)
            writer.writerow(
                {
                    "PuzzleId": record.puzzle_id,
                    "Rating": record.rating,
                    "PGN": _setup_pgn(record.fen),
                    "Solution": _solution_san(record),
                    "FEN": record.fen,
                    "Moves": " ".join(record.moves),
                }
            )
    return {"puzzles": len(numbers), "positions": positions, "seed": seed, "output": str(output)}


def read_puzzle_csv(path: Path) -> list[PuzzleRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        PuzzleRecord(
            row["PuzzleId"], row["FEN"], tuple(row["Moves"].split()), int(row["Rating"]), ()
        )
        for row in rows
    ]


def line_verdict(
    positions: Sequence[SolverPosition], chosen: Sequence[chess.Move | None]
) -> tuple[bool, bool]:
    """ChessBench's verdict and the strict every-move-matches verdict for one line.

    Moves after the first miss are never read, so an engine may stop there as theirs does.
    """

    for position, move in zip(positions, chosen):
        if move == position.move:
            continue
        if move is None:
            raise ValueError(f"no move for {position.board.fen()}")
        board = position.board.copy(stack=False)
        board.push(move)
        return board.is_checkmate(), False
    return True, True


def rating_band(rating: int) -> str:
    low = rating // RATING_BAND * RATING_BAND
    return f"{low}-{low + RATING_BAND - 1}"


def solving_report(ratings: np.ndarray, solved: np.ndarray, exact: np.ndarray) -> dict:
    """Solve rate with its interval, strict rate, implied puzzle rating, and rate per band."""

    count = int(solved.sum())
    low, high = wilson_interval(count, len(solved))
    rating, error = puzzle_rating(ratings.astype(float), solved)
    lows = ratings // RATING_BAND * RATING_BAND
    bands = {
        rating_band(int(band)): {
            "accuracy": float(solved[lows == band].mean()),
            "solved": int(solved[lows == band].sum()),
            "puzzles": int((lows == band).sum()),
        }
        for band in np.unique(lows)
    }
    return {
        "accuracy": count / max(1, len(solved)),
        "low": low,
        "high": high,
        "solved": count,
        "puzzles": len(solved),
        "exact_accuracy": float(exact.mean()),
        "puzzle_rating": rating,
        "puzzle_rating_error": error,
        "by_rating": bands,
    }


class EncodedPositions:
    """Boards encoded as the puzzle corpus encodes them, served as model batches."""

    def __init__(self, boards: Sequence[chess.Board]) -> None:
        self.fens = [board.fen() for board in boards]
        encoded = [encode_position(board) for board in boards]
        self.pieces = np.stack([item.pieces for item in encoded]).astype(np.int64)
        self.state = np.stack([state_vector(item) for item in encoded]).astype(np.float32)
        self.legal = [np.asarray(legal_actions(board), dtype=np.int64) for board in boards]

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        mask = np.zeros((len(indices), ACTION_SIZE), dtype=np.bool_)
        for row, index in enumerate(indices):
            mask[row, self.legal[index]] = True
        return {
            "pieces": torch.from_numpy(self.pieces[indices]),
            "state": torch.from_numpy(self.state[indices]),
            "legal_mask": torch.from_numpy(mask),
        }


def _write(output: Path, report: dict, arrays: Mapping[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "solved.npz", **arrays)  # type: ignore[arg-type]


def _verdicts(
    lines: Sequence[list[SolverPosition]], chosen: Sequence[Sequence[chess.Move | None]]
) -> tuple[np.ndarray, np.ndarray]:
    verdicts = np.array([line_verdict(line, moves) for line, moves in zip(lines, chosen)])
    return verdicts[:, 0], verdicts[:, 1]


def evaluate_full_puzzles(
    checkpoint: Path,
    puzzles: Path,
    depths: Iterable[int],
    *,
    output: Path,
    batch_size: int = 512,
    device_name: str = "cpu",
    max_vram_gb: float | None = 1.0,
    select: str = "policy",
    beta: float = MIX_BETA,
    top_k: int = CHILD_TOP_K,
    line_plies: int | None = None,
) -> dict[str, object]:
    """Whole-puzzle accuracy of one checkpoint at each depth on a ChessBench-format CSV.

    ``select``, ``beta`` and ``top_k`` choose moves as ``predict_depths`` does; ``line_plies``
    as ``load_checkpoint`` takes it.
    """

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    depths = tuple(sorted(set(depths)))
    model = load_checkpoint(checkpoint, device, select, line_plies=line_plies)
    records = read_puzzle_csv(puzzles)
    lines = [solver_positions(record) for record in records]
    flat = [position for line in lines for position in line]
    predictions = predict_depths(
        model,
        EncodedPositions([position.board for position in flat]),
        np.arange(len(flat)),
        depths,
        batch_size=batch_size,
        device=device,
        select=select,
        beta=beta,
        top_k=top_k,
    )
    ratings = np.array([record.rating for record in records])
    results: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {"rating": ratings}
    ends = np.cumsum([len(line) for line in lines])
    for depth, actions in predictions.items():
        moves = [action_to_move(item.board, int(action)) for item, action in zip(flat, actions)]
        chosen = [moves[end - len(line) : end] for line, end in zip(lines, ends)]
        solved, exact = _verdicts(lines, chosen)
        results[f"t{depth}"] = solving_report(ratings, solved, exact)
        arrays[f"t{depth}"] = solved
    report: dict[str, object] = {
        "schema": "wombat.full-puzzles.v1",
        "checkpoint": str(checkpoint),
        "puzzles_csv": str(puzzles),
        "puzzles": len(records),
        "positions": len(flat),
        "select": select,
        "select_beta": recorded_beta(select, beta),
        "select_top_k": top_k if select in CHILD_MODES else None,
        "line_plies": workspace_line_plies(model),
        "results": results,
    }
    _write(output, report, arrays)
    return report


def score_external_puzzles(moves: Path, puzzles: Path, *, output: Path) -> dict[str, object]:
    """Whole-puzzle accuracy of moves chosen elsewhere, given as a JSON ``{fen: uci}`` map.

    Keys are ``chess.Board.fen()`` of each solver position replayed from the CSV.
    """

    chosen_by_fen: dict[str, str] = json.loads(moves.read_text(encoding="utf-8"))
    records = read_puzzle_csv(puzzles)
    lines = [solver_positions(record) for record in records]
    chosen = []
    for line in lines:
        picks = []
        for position in line:
            uci = chosen_by_fen.get(position.board.fen())
            picks.append(None if uci is None else chess.Move.from_uci(uci))
        chosen.append(picks)
    solved, exact = _verdicts(lines, chosen)
    ratings = np.array([record.rating for record in records])
    report: dict[str, object] = {
        "schema": "wombat.full-puzzles.v1",
        "moves": str(moves),
        "puzzles_csv": str(puzzles),
        "puzzles": len(records),
        "positions": sum(len(line) for line in lines),
        "results": {"external": solving_report(ratings, solved, exact)},
    }
    _write(output, report, {"rating": ratings, "external": solved})
    return report
