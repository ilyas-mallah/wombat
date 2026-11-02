from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import random

import chess
import pytest
import zstandard

from wombat.data.game_corpus import _split_and_hash, build_game_corpus
from wombat.data.lichess import PuzzleRecord
from wombat.data.puzzle_corpus import build_puzzle_corpus, puzzle_split

PUZZLE_FIELDS = ("PuzzleId", "FEN", "Moves", "Rating", "Themes")


def _random_line(randomizer: random.Random, plies: int) -> tuple[str, tuple[str, ...]]:
    while True:
        board = chess.Board()
        for _ in range(randomizer.randrange(4, 30)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        start = board.fen()
        line: list[str] = []
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = randomizer.choice(moves)
            line.append(move.uci())
            board.push(move)
        if len(line) == plies:
            return start, tuple(line)


def _puzzle_ids() -> list[str]:
    """Ids covering every split so held-out code paths are exercised."""

    wanted = {"train": 6, "validation": 3, "test": 3}
    ids: list[str] = []
    index = 0
    while any(wanted.values()):
        candidate = f"fx{index}"
        split = puzzle_split(candidate)
        if wanted[split]:
            wanted[split] -= 1
            ids.append(candidate)
        index += 1
    return ids


@pytest.fixture(scope="session")
def puzzle_archive(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[PuzzleRecord]]:
    """A small zstd Lichess-format puzzle CSV built from random legal lines."""

    randomizer = random.Random(7)
    records = []
    for number, puzzle_id in enumerate(_puzzle_ids()):
        fen, moves = _random_line(randomizer, 2 + 2 * (number % 3))
        records.append(PuzzleRecord(puzzle_id, fen, moves, 1000 + 100 * number, ("short",)))
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=PUZZLE_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "PuzzleId": record.puzzle_id,
                "FEN": record.fen,
                "Moves": " ".join(record.moves),
                "Rating": record.rating,
                "Themes": " ".join(record.themes),
            }
        )
    path = tmp_path_factory.mktemp("raw") / "puzzles.csv.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(text.getvalue().encode()))
    return path, records


def _random_board(randomizer: random.Random) -> chess.Board:
    while True:
        board = chess.Board()
        for _ in range(randomizer.randrange(4, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        if len(list(board.legal_moves)) >= 4:
            return board


def _record(board: chess.Board, randomizer: random.Random, pvs: int, depth: int) -> dict:
    moves = randomizer.sample(list(board.legal_moves), pvs)
    return {
        "fen": " ".join(board.fen().split()[:4]),
        "evals": [
            {
                "depth": depth,
                "knodes": 1000,
                "pvs": [
                    {"cp": 80 - 60 * index, "line": f"{move.uci()} {move.uci()}"}
                    for index, move in enumerate(moves)
                ],
            },
            # A shallower evaluation with more variations must never be chosen.
            {
                "depth": 10,
                "pvs": [{"cp": 0, "line": move.uci()} for move in list(board.legal_moves)[:4]],
            },
        ],
    }


@pytest.fixture(scope="session")
def eval_archive(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    """Lichess-format evaluations covering every split, plus shallow and thin rejects."""

    randomizer = random.Random(3)
    wanted = {"train": 8, "validation": 3, "test": 3}
    records: list[dict] = []
    while any(wanted.values()):
        board = _random_board(randomizer)
        split, _ = _split_and_hash(board)
        if wanted[split]:
            wanted[split] -= 1
            records.append(_record(board, randomizer, 3 + len(records) % 2, 24))
    records.insert(2, _record(_random_board(randomizer), randomizer, 4, 12))
    records.insert(5, _record(_random_board(randomizer), randomizer, 2, 30))
    path = tmp_path_factory.mktemp("raw") / "evals.jsonl.zst"
    text = "".join(json.dumps(record) + "\n" for record in records)
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    return path, records


@pytest.fixture(scope="session")
def game_corpus(
    eval_archive: tuple[Path, list[dict]], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    destination = tmp_path_factory.mktemp("games")
    build_game_corpus(
        eval_archive[0], destination, limit=100, low_priority=False, minimum_free_gb=0
    )
    return destination


@pytest.fixture(scope="session")
def puzzle_corpus(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    destination = tmp_path_factory.mktemp("corpus")
    build_puzzle_corpus(puzzle_archive[0], destination, low_priority=False, minimum_free_gb=0)
    return destination
