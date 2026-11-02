"""Every solver-to-move position of every Lichess puzzle line, as compact arrays.

A split is a directory of one ``.npy`` file per column. Builders stream columns
to disk and readers memory-map them, so neither side holds the corpus in RAM.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
import json
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Iterable, Iterator

import chess
import numpy as np
import psutil

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import ACTION_SIZE, legal_actions, move_to_action
from wombat.data.lichess import PuzzleRecord, iter_puzzles

if TYPE_CHECKING:
    import torch

PUZZLE_CORPUS_SCHEMA = "wombat.puzzle-corpus.v2"
SPLITS = ("train", "validation", "test")
# Only held-out splits keep FENs; they are needed for mate checks and replays.
FEN_SPLITS = ("validation", "test")
# Per-position columns: dtype and trailing shape.
COLUMNS: dict[str, tuple[type[np.generic], tuple[int, ...]]] = {
    "pieces": (np.uint8, (64,)),
    "state": (np.uint8, (26,)),
    "action": (np.int16, ()),
    "rating": (np.int16, ()),
    "remaining": (np.int8, ()),
    "line_index": (np.int8, ()),
    "puzzle": (np.int32, ()),
    "legal_counts": (np.int16, ()),
}


def puzzle_split(puzzle_id: str) -> str:
    """Assign whole puzzles, never single positions, to a split."""

    bucket = int(sha256(puzzle_id.encode()).hexdigest()[:16], 16) % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"


def _normalized_hash(board: chess.Board) -> str:
    fen = " ".join(board.fen(en_passant="fen").split()[:4])
    return sha256(fen.encode()).hexdigest()


@dataclass(frozen=True)
class SolverPosition:
    board: chess.Board
    move: chess.Move
    remaining: int
    line_index: int


def solver_positions(record: PuzzleRecord) -> list[SolverPosition]:
    """Yield each position where the solver must move, with moves left in the line.

    The first listed move is the opponent's setup move; solver and opponent
    moves then alternate and the line ends on a solver move.
    """

    if len(record.moves) < 2 or len(record.moves) % 2:
        raise ValueError(f"puzzle {record.puzzle_id} has malformed move count")
    board = chess.Board(record.fen)
    board.push(board.parse_uci(record.moves[0]))
    positions: list[SolverPosition] = []
    total = len(record.moves) // 2
    for line_index, index in enumerate(range(1, len(record.moves), 2)):
        move = board.parse_uci(record.moves[index])
        positions.append(
            SolverPosition(board.copy(stack=False), move, total - line_index, line_index)
        )
        board.push(move)
        if index + 1 < len(record.moves):
            board.push(board.parse_uci(record.moves[index + 1]))
    return positions


def lower_priority() -> None:
    """Yield the CPU to everything else on the machine."""

    process = psutil.Process()
    if psutil.WINDOWS:
        process.nice(psutil.IDLE_PRIORITY_CLASS)
    else:
        process.nice(19)


_EXCLUDED_HASHES: frozenset[str] = frozenset()


def _initialize_worker(hashes: frozenset[str], low_priority: bool) -> None:
    global _EXCLUDED_HASHES
    _EXCLUDED_HASHES = hashes
    if low_priority:
        lower_priority()


@dataclass
class _ChunkSplit:
    """Positions of one split from one chunk, with puzzle indices local to the chunk."""

    puzzle_ids: list[str]
    columns: dict[str, list[object]]
    legal_actions: list[int]
    fens: list[str]


def _encode_puzzles(records: list[PuzzleRecord]) -> tuple[dict[str, _ChunkSplit], Counter[str]]:
    """Encode one chunk into per-split columns plus rejection counts."""

    chunk = {split: _ChunkSplit([], {name: [] for name in COLUMNS}, [], []) for split in SPLITS}
    rejected: Counter[str] = Counter()
    for record in records:
        try:
            positions = solver_positions(record)
        except (ValueError, AssertionError):
            rejected["malformed_line"] += 1
            continue
        if any(_normalized_hash(item.board) in _EXCLUDED_HASHES for item in positions):
            rejected["frozen_fen_overlap"] += 1
            continue
        split = puzzle_split(record.puzzle_id)
        target = chunk[split]
        puzzle = len(target.puzzle_ids)
        target.puzzle_ids.append(record.puzzle_id)
        for item in positions:
            encoded = encode_position(item.board)
            actions = legal_actions(item.board)
            values: dict[str, object] = {
                "pieces": encoded.pieces,
                "state": state_vector(encoded),
                "action": move_to_action(item.board, item.move),
                "rating": record.rating,
                "remaining": item.remaining,
                "line_index": item.line_index,
                "puzzle": puzzle,
                "legal_counts": len(actions),
            }
            for name, value in values.items():
                target.columns[name].append(value)
            target.legal_actions.extend(actions)
            if split in FEN_SPLITS:
                target.fens.append(item.board.fen())
    return chunk, rejected


def _chunks(records: Iterable[PuzzleRecord], size: int) -> Iterator[list[PuzzleRecord]]:
    chunk: list[PuzzleRecord] = []
    for record in records:
        chunk.append(record)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _write_npy(path: Path, array: np.ndarray) -> str:
    np.save(path, array, allow_pickle=False)
    return _file_sha256(path)


class _SplitWriter:
    """Append chunks to raw column files, then convert them to ``.npy`` once."""

    def __init__(self, directory: Path, keep_fens: bool) -> None:
        self.directory = directory
        self.scratch = directory / ".partial"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.handles = {
            name: (self.scratch / f"{name}.bin").open("wb") for name in (*COLUMNS, "legal_actions")
        }
        self.ids = (self.scratch / "puzzle_ids.txt").open("w", encoding="utf-8")
        self.keep_fens = keep_fens
        self.fens: list[str] = []
        self.puzzles = 0
        self.positions = 0
        self.legal_total = 0

    def add(self, chunk: _ChunkSplit) -> None:
        for name, (dtype, _) in COLUMNS.items():
            values = np.asarray(chunk.columns[name], dtype=dtype)
            if name == "puzzle":
                values = values + np.int32(self.puzzles)
            self.handles[name].write(values.tobytes())
        self.handles["legal_actions"].write(np.asarray(chunk.legal_actions, np.int16).tobytes())
        self.ids.writelines(f"{item}\n" for item in chunk.puzzle_ids)
        if self.keep_fens:
            self.fens.extend(chunk.fens)
        self.puzzles += len(chunk.puzzle_ids)
        self.positions += len(chunk.columns["action"])
        self.legal_total += len(chunk.legal_actions)

    def _raw(self, name: str, dtype: type[np.generic], shape: tuple[int, ...]) -> np.ndarray:
        path = self.scratch / f"{name}.bin"
        count = self.legal_total if name == "legal_actions" else self.positions
        if count == 0:
            return np.zeros((0, *shape), dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r", shape=(count, *shape))

    def finish(self) -> dict[str, object]:
        for handle in self.handles.values():
            handle.close()
        self.ids.close()
        hashes: dict[str, str] = {}
        for name, (dtype, shape) in COLUMNS.items():
            if name == "legal_counts":
                counts = self._raw(name, dtype, shape).astype(np.int64)
                offsets = np.concatenate(([0], np.cumsum(counts)))
                hashes["legal_offsets"] = _write_npy(self.directory / "legal_offsets.npy", offsets)
                del counts
                continue
            hashes[name] = _write_npy(self.directory / f"{name}.npy", self._raw(name, dtype, shape))
        hashes["legal_actions"] = _write_npy(
            self.directory / "legal_actions.npy", self._raw("legal_actions", np.int16, ())
        )
        ids = (self.scratch / "puzzle_ids.txt").read_text(encoding="utf-8").split()
        hashes["puzzle_ids"] = _write_npy(self.directory / "puzzle_ids.npy", np.array(ids, str))
        if self.keep_fens:
            hashes["fens"] = _write_npy(self.directory / "fens.npy", np.array(self.fens, str))
        shutil.rmtree(self.scratch)
        return {"puzzles": self.puzzles, "positions": self.positions, "sha256": hashes}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_puzzle_corpus(
    source: Path,
    destination: Path,
    *,
    excluded_puzzle_ids: Iterable[str] = (),
    excluded_fens: Iterable[str] = (),
    limit: int | None = None,
    workers: int = 1,
    chunk_size: int = 2000,
    low_priority: bool = True,
    minimum_free_gb: float = 10.0,
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    free_gb = psutil.disk_usage(destination.resolve().anchor).free / 2**30
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"only {free_gb:.2f} GB free; refusing to build corpus")
    if low_priority:
        lower_priority()
    excluded_ids = set(excluded_puzzle_ids)
    hashes = frozenset(
        sha256(" ".join(fen.split()[:4]).encode()).hexdigest() for fen in excluded_fens
    )
    rejected: Counter[str] = Counter()

    def records() -> Iterator[PuzzleRecord]:
        taken = 0
        for record in iter_puzzles(source):
            if record.puzzle_id in excluded_ids:
                rejected["frozen_puzzle_id"] += 1
                continue
            yield record
            taken += 1
            if limit is not None and taken >= limit:
                return

    writers = {split: _SplitWriter(destination / split, split in FEN_SPLITS) for split in SPLITS}

    def consume(result: tuple[dict[str, _ChunkSplit], Counter[str]]) -> None:
        chunk, chunk_rejected = result
        rejected.update(chunk_rejected)
        for split, part in chunk.items():
            writers[split].add(part)

    chunks = _chunks(records(), chunk_size)
    if workers > 1:
        with Pool(workers, initializer=_initialize_worker, initargs=(hashes, low_priority)) as pool:
            # A bounded window keeps output ordered without queueing the whole archive.
            pending: deque[AsyncResult[tuple[dict[str, _ChunkSplit], Counter[str]]]] = deque()
            for chunk in chunks:
                pending.append(pool.apply_async(_encode_puzzles, (chunk,)))
                if len(pending) >= workers * 2:
                    consume(pending.popleft().get())
            while pending:
                consume(pending.popleft().get())
    else:
        _initialize_worker(hashes, low_priority=False)
        for chunk in chunks:
            consume(_encode_puzzles(chunk))

    manifest: dict[str, object] = {
        "schema": PUZZLE_CORPUS_SCHEMA,
        "source": str(source),
        "source_sha256": _file_sha256(source),
        "limit": limit,
        "excluded_puzzle_ids": len(excluded_ids),
        "excluded_fens": len(hashes),
        "rejected": dict(rejected),
        "splits": {split: writer.finish() for split, writer in writers.items()},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class PuzzleSplit:
    """One memory-mapped corpus split, served as ready-made tensor batches."""

    def __init__(self, directory: Path) -> None:
        def column(name: str) -> np.ndarray:
            return np.load(directory / f"{name}.npy", mmap_mode="r")

        self.pieces = column("pieces")
        self.state = column("state")
        self.action = np.asarray(column("action"), dtype=np.int64)
        self.rating = column("rating")
        self.remaining = column("remaining")
        self.line_index = column("line_index")
        self.puzzle = column("puzzle")
        self.legal_offsets = column("legal_offsets")
        self.legal_actions = column("legal_actions")
        self.puzzle_ids = column("puzzle_ids")
        fens = directory / "fens.npy"
        self.fens = np.load(fens) if fens.exists() else None
        # (position, action) -> whether that move checkmates; filled by evaluation.
        self.mate_cache: dict[tuple[int, int], bool] = {}

    def __len__(self) -> int:
        return len(self.action)

    def legal_mask(self, indices: np.ndarray) -> np.ndarray:
        starts = self.legal_offsets[indices]
        counts = self.legal_offsets[indices + 1] - starts
        rows = np.repeat(np.arange(len(indices)), counts)
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        mask = np.zeros((len(indices), ACTION_SIZE), dtype=np.bool_)
        mask[rows, self.legal_actions[np.repeat(starts, counts) + within]] = True
        return mask

    def batch(self, indices: np.ndarray, *, lines: bool = False) -> dict[str, torch.Tensor]:
        """Boards and answers; ``lines`` adds each position's ``line_actions``."""

        # Imported here so corpus-building worker processes never load torch.
        import torch

        def tensor(values: np.ndarray, dtype: type[np.generic]) -> torch.Tensor:
            return torch.from_numpy(np.asarray(values[indices], dtype=dtype))

        batch = {
            "pieces": tensor(self.pieces, np.int64),
            "state": tensor(self.state, np.float32),
            "action": tensor(self.action, np.int64),
            "legal_mask": torch.from_numpy(self.legal_mask(indices)),
            "rating": tensor(self.rating, np.int64),
            "remaining": tensor(self.remaining, np.int64),
        }
        if lines:
            batch["line_actions"] = torch.from_numpy(self.line_actions(indices))
        return batch

    def line_actions(self, indices: np.ndarray) -> np.ndarray:
        """(rows, LINE_PLIES) solution line from each position, -1 past its end.

        Solver moves are the stored actions; each opponent reply is read from the next solver
        position of the same puzzle. Every action is in its own mover's frame.
        """

        from wombat.data.move_labels import LINE_PLIES

        indices = np.asarray(indices, dtype=np.int64)
        lines = np.full((len(indices), LINE_PLIES), -1, dtype=np.int64)
        lines[:, 0] = self.action[indices]
        solver = indices
        for ply in range(1, LINE_PLIES):
            alive = lines[:, ply - 1] >= 0
            if ply % 2 == 0:
                lines[:, ply] = np.where(alive, self.action[solver], -1)
                continue
            later = solver + 1
            nearest = np.minimum(later, len(self) - 1)
            alive &= (later < len(self)) & (self.puzzle[nearest] == self.puzzle[solver])
            alive &= self.line_index[nearest] == self.line_index[solver] + 1
            rows = np.flatnonzero(alive)
            if len(rows):
                lines[rows, ply] = self._replies(solver[rows], later[rows])
            solver = np.where(alive, later, solver)
        return lines

    def _replies(self, solver: np.ndarray, later: np.ndarray) -> np.ndarray:
        import torch

        from wombat.model.lines import cpu_rules

        def tensor(values: np.ndarray, rows: np.ndarray, dtype: type[np.generic]) -> torch.Tensor:
            return torch.from_numpy(np.asarray(values[rows], dtype=dtype))

        found = cpu_rules().reply(
            tensor(self.pieces, solver, np.int64),
            tensor(self.state, solver, np.float32),
            torch.from_numpy(self.action[solver]),
            tensor(self.pieces, later, np.int64),
        )
        return found.numpy()

    def shuffled_batches(
        self,
        batch_size: int,
        generator: np.random.Generator,
        indices: np.ndarray | None = None,
        *,
        lines: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield full batches forever, reshuffling after each pass; ``indices`` limits the pool."""

        while True:
            if indices is None:
                order = generator.permutation(len(self))
            else:
                order = indices[generator.permutation(len(indices))]
            for start in range(0, len(order) - batch_size + 1, batch_size):
                yield self.batch(order[start : start + batch_size], lines=lines)
