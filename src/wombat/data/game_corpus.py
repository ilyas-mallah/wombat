"""Ordinary game positions with Stockfish-scored candidate moves, graded by win chance.

Each position keeps the Lichess evaluation with the most principal variations
among those deep enough, so several candidate moves carry a score. Candidates
are stored best first as win chances for the side to move.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Iterator, Sequence
from hashlib import sha256
import io
from itertools import islice
import json
import math
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
import numpy as np
import psutil
import zstandard

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import ACTION_SIZE, legal_actions, move_to_action, square_pair_actions
from wombat.data.columns import ColumnChunk, ColumnWriter, FixedSpec, RaggedSpec
from wombat.data.columns import ragged_entries
from wombat.data.puzzle_corpus import lower_priority

if TYPE_CHECKING:
    import torch

GAME_CORPUS_SCHEMA = "wombat.game-corpus.v1"
# Lookahead labels use 255 for "no square"; training targets use the loss's ignore index.
NO_LINE = 255
IGNORE = -100
SPLITS = ("train", "validation", "test")
FEN_SPLITS = ("validation", "test")
MIN_DEPTH = 20
MIN_CANDIDATES = 3
FIXED: FixedSpec = {
    "pieces": (np.uint8, (64,)),
    "state": (np.uint8, (26,)),
    "legal_counts": (np.int16, ()),
    "candidate_counts": (np.int8, ()),
}
RAGGED: RaggedSpec = {
    "legal_actions": (np.int16, "legal"),
    "candidate_actions": (np.int16, "candidate"),
    "candidate_win": (np.float16, "candidate"),
}


def win_chance(cp: float) -> float:
    """Lichess's win chance in [0, 1] for the side to move at ``cp`` centipawns."""

    return 1 / (1 + math.exp(-0.00368208 * cp))


def _candidate_win(pv: dict[str, Any], turn: chess.Color) -> float | None:
    sign = 1 if turn == chess.WHITE else -1
    if pv.get("mate") is not None:
        return 1.0 if sign * int(pv["mate"]) > 0 else 0.0
    if pv.get("cp") is not None:
        return win_chance(sign * int(pv["cp"]))
    return None


def select_evaluation(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The deep-enough evaluation with the most variations, deeper on ties."""

    deep = [item for item in evaluations if int(item.get("depth", 0)) >= MIN_DEPTH]
    if not deep:
        return None
    return max(deep, key=lambda item: (len(item.get("pvs", ())), int(item.get("depth", 0))))


def _split_and_hash(board: chess.Board) -> tuple[str, str]:
    normalized = " ".join(board.fen(en_passant="fen").split()[:4])
    digest = sha256(normalized.encode()).hexdigest()
    bucket = int(digest[:16], 16) % 100
    split = "train" if bucket < 98 else "validation" if bucket == 98 else "test"
    return split, digest


def graded_candidates(payload: dict[str, Any], board: chess.Board) -> list[tuple[int, float]] | str:
    """Scored (action, win chance) pairs best first, or a rejection reason."""

    evaluation = select_evaluation(payload.get("evals", []))
    if evaluation is None:
        return "shallow"
    candidates: dict[int, float] = {}
    for pv in evaluation.get("pvs", []):
        line = str(pv.get("line", "")).split()
        win = _candidate_win(pv, board.turn)
        if not line or win is None:
            continue
        try:
            action = move_to_action(board, board.parse_uci(line[0]))
        except ValueError:
            return "illegal_move"
        candidates.setdefault(action, win)
    if len(candidates) < MIN_CANDIDATES:
        return "few_candidates"
    return sorted(candidates.items(), key=lambda item: -item[1])


_EXCLUDED: frozenset[str] = frozenset()
# Sorted keys of the positions in skipped records.
_EARLIER: np.ndarray = np.zeros(0, dtype=np.uint64)


def _initialize_worker(excluded: frozenset[str], earlier: np.ndarray, low_priority: bool) -> None:
    global _EXCLUDED, _EARLIER
    _EXCLUDED = excluded
    _EARLIER = earlier
    if low_priority:
        lower_priority()


def _position_key(digest: str) -> int:
    return int(digest[:16], 16)


def _is_earlier(digest: str) -> bool:
    key = np.uint64(_position_key(digest))
    slot = int(np.searchsorted(_EARLIER, key))
    return slot < len(_EARLIER) and _EARLIER[slot] == key


def _position_keys(lines: list[str]) -> np.ndarray:
    """Keys of the positions in ``lines``, hashed as ``_encode_lines`` hashes them."""

    keys = []
    for line in lines:
        try:
            board = chess.Board(str(json.loads(line)["fen"]))
        except (ValueError, KeyError):
            continue
        keys.append(_position_key(_split_and_hash(board)[1]))
    return np.array(keys, dtype=np.uint64)


def _encode_lines(lines: list[str]) -> tuple[dict[str, ColumnChunk], Counter[str]]:
    chunks = {split: ColumnChunk.empty(FIXED, RAGGED) for split in SPLITS}
    rejected: Counter[str] = Counter()
    for line in lines:
        try:
            payload = json.loads(line)
            board = chess.Board(str(payload["fen"]))
        except (ValueError, KeyError):
            rejected["malformed"] += 1
            continue
        # The archive holds a few impossible positions (for example 17 white men)
        # that crash Stockfish.
        if board.status() != chess.STATUS_VALID:
            rejected["invalid_position"] += 1
            continue
        split, digest = _split_and_hash(board)
        if digest in _EXCLUDED:
            rejected["frozen_fen"] += 1
            continue
        if len(_EARLIER) and _is_earlier(digest):
            rejected["earlier_record"] += 1
            continue
        candidates = graded_candidates(payload, board)
        if isinstance(candidates, str):
            rejected[candidates] += 1
            continue
        encoded = encode_position(board)
        actions = legal_actions(board)
        chunk = chunks[split]
        chunk.fixed["pieces"].append(encoded.pieces)
        chunk.fixed["state"].append(state_vector(encoded))
        chunk.fixed["legal_counts"].append(len(actions))
        chunk.fixed["candidate_counts"].append(len(candidates))
        chunk.ragged["legal_actions"].extend(actions)
        chunk.ragged["candidate_actions"].extend(action for action, _ in candidates)
        chunk.ragged["candidate_win"].extend(win for _, win in candidates)
        if split in FEN_SPLITS:
            chunk.fens.append(board.fen())
    return chunks, rejected


def _records(source: Path) -> Iterator[str]:
    with source.open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        with io.TextIOWrapper(reader, encoding="utf-8") as text:
            yield from (line for line in text if line.strip())


def _chunks(lines: Iterator[str], size: int) -> Iterator[list[str]]:
    while chunk := list(islice(lines, size)):
        yield chunk


def _earlier_positions(
    chunks: Iterator[list[str]], workers: int, low_priority: bool
) -> tuple[np.ndarray, int]:
    """Sorted distinct keys of the positions in ``chunks``, and the number of records."""

    keys = [np.zeros(0, dtype=np.uint64)]
    records = 0
    if workers > 1:
        nothing: tuple[frozenset[str], np.ndarray, bool] = (frozenset(), keys[0], low_priority)
        with Pool(workers, initializer=_initialize_worker, initargs=nothing) as pool:
            pending: deque[AsyncResult[np.ndarray]] = deque()
            for chunk in chunks:
                records += len(chunk)
                pending.append(pool.apply_async(_position_keys, (chunk,)))
                if len(pending) >= workers * 2:
                    keys.append(pending.popleft().get())
            keys.extend(result.get() for result in pending)
    else:
        for chunk in chunks:
            records += len(chunk)
            keys.append(_position_keys(chunk))
    return np.unique(np.concatenate(keys)), records


def build_game_corpus(
    source: Path,
    destination: Path,
    *,
    limit: int,
    excluded_fens: Iterable[str] = (),
    skip_records: int = 0,
    workers: int = 1,
    chunk_size: int = 5000,
    low_priority: bool = True,
    minimum_free_gb: float = 10.0,
) -> dict[str, object]:
    """Stream the evaluation archive until ``limit`` positions are kept.

    The first ``skip_records`` records are read only to exclude their positions, so a
    later corpus shares no position with one built from them.
    """

    destination.mkdir(parents=True, exist_ok=True)
    free_gb = psutil.disk_usage(destination.resolve().anchor).free / 2**30
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"only {free_gb:.2f} GB free; refusing to build corpus")
    if low_priority:
        lower_priority()
    excluded = frozenset(
        sha256(" ".join(fen.split()[:4]).encode()).hexdigest() for fen in excluded_fens
    )
    writers = {
        split: ColumnWriter(destination / split, FIXED, RAGGED, keep_fens=split in FEN_SPLITS)
        for split in SPLITS
    }
    rejected: Counter[str] = Counter()
    candidate_histogram: Counter[int] = Counter()
    kept = 0
    read = 0

    def consume(result: tuple[dict[str, ColumnChunk], Counter[str]]) -> bool:
        nonlocal kept
        chunks, chunk_rejected = result
        rejected.update(chunk_rejected)
        for split, chunk in chunks.items():
            if kept >= limit:
                return False
            take = min(len(chunk), limit - kept)
            if take < len(chunk):
                chunk = ColumnChunk(
                    {name: values[:take] for name, values in chunk.fixed.items()},
                    _trim_ragged(chunk, take),
                    chunk.fens[:take],
                )
            candidate_histogram.update(
                np.asarray(chunk.fixed["candidate_counts"], dtype=np.int64).tolist()
            )
            writers[split].add(chunk)
            kept += take
        return kept < limit

    records = _records(source)
    earlier = np.zeros(0, dtype=np.uint64)
    if skip_records:
        skipped = _chunks(islice(records, skip_records), chunk_size)
        earlier, read = _earlier_positions(skipped, workers, low_priority)
    lines = _chunks(records, chunk_size)
    if workers > 1:
        with Pool(
            workers, initializer=_initialize_worker, initargs=(excluded, earlier, low_priority)
        ) as pool:
            pending: deque[AsyncResult[tuple[dict[str, ColumnChunk], Counter[str]]]] = deque()
            for chunk in lines:
                read += len(chunk)
                pending.append(pool.apply_async(_encode_lines, (chunk,)))
                if len(pending) >= workers * 2 and not consume(pending.popleft().get()):
                    break
            while pending and kept < limit:
                consume(pending.popleft().get())
    else:
        _initialize_worker(excluded, earlier, low_priority=False)
        for chunk in lines:
            read += len(chunk)
            if not consume(_encode_lines(chunk)):
                break

    manifest: dict[str, object] = {
        "schema": GAME_CORPUS_SCHEMA,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "limit": limit,
        # Counted from the start of the archive, skipped records included.
        "records_read": read,
        "skip_records": skip_records,
        "earlier_positions": len(earlier),
        "min_depth": MIN_DEPTH,
        "min_candidates": MIN_CANDIDATES,
        "excluded_fens": len(excluded),
        "rejected": dict(rejected),
        "candidate_counts": dict(sorted(candidate_histogram.items())),
        "splits": {split: writer.finish() for split, writer in writers.items()},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _trim_ragged(chunk: ColumnChunk, take: int) -> dict[str, list[object]]:
    legal = int(np.asarray(chunk.fixed["legal_counts"][:take], dtype=np.int64).sum())
    candidates = int(np.asarray(chunk.fixed["candidate_counts"][:take], dtype=np.int64).sum())
    return {
        "legal_actions": chunk.ragged["legal_actions"][:legal],
        "candidate_actions": chunk.ragged["candidate_actions"][:candidates],
        "candidate_win": chunk.ragged["candidate_win"][:candidates],
    }


class GameSplit:
    """One memory-mapped game-position split, served as graded-target batches."""

    def __init__(
        self, directory: Path, *, temperature: float = 4.0, labels: Path | None = None
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        # Win-chance points (0..100) per unit of log-probability in the target.
        self.temperature = temperature
        self.labels = labels is not None
        if labels is not None:
            # Every-move scores for a subset of positions; training then uses only that subset.
            self.label_indices = np.load(labels / "label_indices.npy")
            self.label_offsets = np.load(labels / "label_offsets.npy", mmap_mode="r")
            self.label_actions = np.load(labels / "label_actions.npy", mmap_mode="r")
            self.label_win = np.load(labels / "label_win.npy", mmap_mode="r")
            # Stockfish's line for the deep-best move, in the root's canonical frame.
            self.lines = (labels / "line_length.npy").exists()
            if self.lines:
                self.line_length = np.load(labels / "line_length.npy", mmap_mode="r")
                self.line_from = np.load(labels / "line_from.npy", mmap_mode="r")
                self.line_to = np.load(labels / "line_to.npy", mmap_mode="r")
                self.line_boards = np.load(labels / "line_boards.npy", mmap_mode="r")
        else:
            self.lines = False

        def column(name: str) -> np.ndarray:
            return np.load(directory / f"{name}.npy", mmap_mode="r")

        self.pieces = column("pieces")
        self.state = column("state")
        self.legal_offsets = column("legal_offsets")
        self.legal_actions = column("legal_actions")
        self.candidate_offsets = column("candidate_offsets")
        self.candidate_actions = column("candidate_actions")
        self.candidate_win = column("candidate_win")
        fens = directory / "fens.npy"
        self.fens = np.load(fens) if fens.exists() else None
        # Candidates are stored best first, so the first entry is the reference move.
        self.action = np.asarray(self.candidate_actions)[
            np.asarray(self.candidate_offsets[:-1])
        ].astype(np.int64)

    def __len__(self) -> int:
        return len(self.pieces)

    def legal_mask(self, indices: np.ndarray) -> np.ndarray:
        rows, actions = ragged_entries(self.legal_offsets, self.legal_actions, indices)
        mask = np.zeros((len(indices), ACTION_SIZE), dtype=np.bool_)
        mask[rows, actions.astype(np.int64)] = True
        return mask

    def trainable(self) -> np.ndarray:
        """Corpus indices training draws from: the labelled subset when labels exist."""

        return self.label_indices if self.labels else np.arange(len(self))

    def _win_table(self, indices: np.ndarray) -> np.ndarray:
        """Win chance per (row, action), NaN where unscored.

        Deep candidate scores are kept. Quick all-move scores fill the other legal
        moves, capped at the worst deep candidate so they never outrank one.
        """

        wins = np.full((len(indices), ACTION_SIZE), np.nan, dtype=np.float32)
        rows, actions = ragged_entries(self.candidate_offsets, self.candidate_actions, indices)
        _, deep = ragged_entries(self.candidate_offsets, self.candidate_win, indices)
        deep = deep.astype(np.float32)
        labelled, slots = self._labelled(indices)
        if len(labelled):
            worst = np.full(len(indices), np.inf, dtype=np.float32)
            np.minimum.at(worst, rows, deep)
            label_rows, label_actions = ragged_entries(
                self.label_offsets, self.label_actions, slots
            )
            _, quick = ragged_entries(self.label_offsets, self.label_win, slots)
            target_rows = labelled[label_rows]
            wins[target_rows, label_actions.astype(np.int64)] = np.minimum(
                quick.astype(np.float32), worst[target_rows]
            )
        wins[rows, actions.astype(np.int64)] = deep
        return wins

    def targets(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Dense graded policy, per-move values, their mask and the position value."""

        wins = self._win_table(indices)
        mask = ~np.isnan(wins)
        best = np.nanmax(wins, axis=1)
        weights = np.where(mask, np.exp(-(best[:, None] - wins) * 100 / self.temperature), 0.0)
        policy = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
        values = np.where(mask, 2 * wins - 1, 0.0).astype(np.float32)
        return {
            "policy_target": policy,
            "move_value_target": values,
            "move_value_mask": mask,
            "value": (2 * best - 1).astype(np.float32),
        }

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        # Imported here so corpus-building worker processes never load torch.
        import torch

        targets = {**self.targets(indices), **self.lookahead_targets(indices)}
        return {
            "pieces": torch.from_numpy(np.asarray(self.pieces[indices], dtype=np.int64)),
            "state": torch.from_numpy(np.asarray(self.state[indices], dtype=np.float32)),
            "action": torch.from_numpy(self.action[indices]),
            "legal_mask": torch.from_numpy(self.legal_mask(indices)),
            **{name: torch.from_numpy(array) for name, array in targets.items()},
        }

    def _labelled(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch rows that have every-move labels, and their rows in the label columns."""

        if not self.labels:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        slots = np.minimum(
            np.searchsorted(self.label_indices, indices), len(self.label_indices) - 1
        )
        labelled = np.flatnonzero(self.label_indices[slots] == indices)
        return labelled, slots[labelled]

    def lookahead_targets(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Reply squares (ply 2) and boards after plies 2 and 4; ``IGNORE`` where unknown.

        ``line_actions`` holds the line's action per ply in that ply's mover's frame, -1 past
        its end; an underpromotion in the line reads as a queen promotion.
        """

        from wombat.data.move_labels import LINE_PLIES

        size = len(indices)
        lookahead = {
            "reply_from": np.full(size, IGNORE, dtype=np.int64),
            "reply_to": np.full(size, IGNORE, dtype=np.int64),
            "board2": np.full((size, 64), IGNORE, dtype=np.int64),
            "board4": np.full((size, 64), IGNORE, dtype=np.int64),
            "line_actions": np.full((size, LINE_PLIES), -1, dtype=np.int64),
        }
        rows, slots = self._labelled(indices)
        if self.lines and len(rows):

            def usable(values: np.ndarray) -> np.ndarray:
                values = np.asarray(values, dtype=np.int64)
                return np.where(values == NO_LINE, IGNORE, values)

            lookahead["reply_from"][rows] = usable(self.line_from[slots, 1])
            lookahead["reply_to"][rows] = usable(self.line_to[slots, 1])
            lookahead["board2"][rows] = usable(self.line_boards[slots, 0])
            lookahead["board4"][rows] = usable(self.line_boards[slots, 1])
            lookahead["line_actions"][rows] = line_actions(
                self.line_from[slots], self.line_to[slots]
            )
        return lookahead

    def sparse_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Boards plus only the legal and scored moves, for ``densify`` on the training device."""

        legal_rows, legal_actions = ragged_entries(self.legal_offsets, self.legal_actions, indices)
        rows, actions = ragged_entries(self.candidate_offsets, self.candidate_actions, indices)
        _, deep = ragged_entries(self.candidate_offsets, self.candidate_win, indices)
        quick_rows = quick_actions = np.zeros(0, dtype=np.int64)
        quick = np.zeros(0, dtype=np.float32)
        labelled, slots = self._labelled(indices)
        if len(labelled):
            label_rows, quick_actions = ragged_entries(
                self.label_offsets, self.label_actions, slots
            )
            _, quick = ragged_entries(self.label_offsets, self.label_win, slots)
            quick_rows = labelled[label_rows]
        return {
            **self.lookahead_targets(indices),
            "pieces": np.asarray(self.pieces[indices], dtype=np.int64),
            "state": np.asarray(self.state[indices], dtype=np.float32),
            "action": self.action[indices],
            "legal_rows": legal_rows,
            "legal_actions": legal_actions.astype(np.int64),
            "deep_rows": rows,
            "deep_actions": actions.astype(np.int64),
            "deep_win": deep.astype(np.float32),
            "quick_rows": quick_rows,
            "quick_actions": quick_actions.astype(np.int64),
            "quick_win": quick.astype(np.float32),
        }

    def shuffled_batches(
        self, batch_size: int, generator: np.random.Generator, *, sparse: bool = False
    ) -> Iterator[Any]:
        """Yield full batches of trainable positions forever, reshuffling after each pass."""

        for chosen in _shuffled(self.trainable(), batch_size, generator):
            yield self.sparse_batch(chosen) if sparse else self.batch(chosen)


_SQUARE_PAIRS = square_pair_actions()


def line_actions(from_squares: np.ndarray, to_squares: np.ndarray) -> np.ndarray:
    """(rows, plies) actions of lines stored as squares in the root's frame, -1 past the end.

    Odd plies are the opponent's, so their squares turn to its frame; a move stops the line
    where it is not a queen-like or knight move.
    """

    origin = np.asarray(from_squares, dtype=np.int64)
    target = np.asarray(to_squares, dtype=np.int64)
    known = (origin != NO_LINE) & (target != NO_LINE)
    opponent = np.arange(origin.shape[1]) % 2 == 1
    origin = np.where(opponent, 63 - origin, origin).clip(0, 63)
    target = np.where(opponent, 63 - target, target).clip(0, 63)
    actions = np.where(known, _SQUARE_PAIRS[origin, target], -1)
    # Nothing after a missing ply belongs to the line.
    return np.where(np.cumprod(actions >= 0, axis=1) > 0, actions, -1)


def _shuffled(
    pool: np.ndarray, batch_size: int, generator: np.random.Generator
) -> Iterator[np.ndarray]:
    while True:
        order = pool[generator.permutation(len(pool))]
        for start in range(0, len(order) - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


class JoinedGameSplit:
    """Several train splits, each with its own labels, drawn from as one pool.

    Indices run through the first part, then the second, and so on.
    """

    def __init__(self, parts: Sequence[GameSplit]) -> None:
        self.parts = list(parts)
        self.starts = np.cumsum([0, *(len(part) for part in self.parts)])

    def __len__(self) -> int:
        return int(self.starts[-1])

    def trainable(self) -> np.ndarray:
        return np.concatenate(
            [
                part.trainable() + start
                for part, start in zip(self.parts, self.starts[:-1], strict=True)
            ]
        )

    def sparse_batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """``GameSplit.sparse_batch`` over the joined indices, rows in the order given."""

        owner = np.searchsorted(self.starts, indices, side="right") - 1
        batches = []
        for number, part in enumerate(self.parts):
            rows = np.flatnonzero(owner == number)
            if len(rows):
                batches.append((rows, part.sparse_batch(indices[rows] - self.starts[number])))
        joined: dict[str, np.ndarray] = {}
        for key, first in batches[0][1].items():
            if key.startswith(("legal_", "deep_", "quick_")):
                # Scored and legal move entries; their row columns point into the batch.
                joined[key] = np.concatenate(
                    [
                        rows[batch[key]] if key.endswith("_rows") else batch[key]
                        for rows, batch in batches
                    ]
                )
            else:
                joined[key] = np.empty((len(indices), *first.shape[1:]), dtype=first.dtype)
                for rows, batch in batches:
                    joined[key][rows] = batch[key]
        return joined

    def shuffled_batches(
        self, batch_size: int, generator: np.random.Generator, *, sparse: bool = True
    ) -> Iterator[dict[str, np.ndarray]]:
        if not sparse:
            raise ValueError("joined splits serve sparse batches only")
        return (
            self.sparse_batch(chosen)
            for chosen in _shuffled(self.trainable(), batch_size, generator)
        )


def with_children(
    sparse: dict[str, np.ndarray], share: float, generator: np.random.Generator
) -> dict[str, np.ndarray]:
    """Replace ``share`` of a sparse batch's rows by child positions trained on the value alone.

    A chosen row plays one of its scored moves, drawn uniformly so blunders come as often as
    good moves, by exact rules. The child's side to move gets the value 1 minus the move's win
    chance as ``densify`` scores it (-1..1 scale), -1 when mated and 0 in stalemate. Its policy,
    move-value and lookahead targets are empty. ``child_rows`` and ``child_value`` carry the
    value to ``densify``.
    """

    import torch

    from wombat.model.lines import cpu_rules

    size = len(sparse["action"])
    count = round(share * size)
    if not count:
        return sparse
    rows = np.sort(generator.choice(size, count, replace=False))
    # Every scored move with its ``densify`` win chance: deep scores win over capped quick ones.
    worst = np.full(size, np.inf, dtype=np.float32)
    np.minimum.at(worst, sparse["deep_rows"], sparse["deep_win"])
    quick = np.minimum(sparse["quick_win"], worst[sparse["quick_rows"]])
    keys = np.concatenate(
        [
            sparse["deep_rows"] * ACTION_SIZE + sparse["deep_actions"],
            sparse["quick_rows"] * ACTION_SIZE + sparse["quick_actions"],
        ]
    )
    scored, first = np.unique(keys, return_index=True)
    wins = np.concatenate([sparse["deep_win"], quick])[first]
    start = np.searchsorted(scored, rows * ACTION_SIZE)
    stop = np.searchsorted(scored, (rows + 1) * ACTION_SIZE)
    if np.any(stop == start):
        raise ValueError("a child position needs a parent with a scored move")
    picked = start + (generator.random(count) * (stop - start)).astype(np.int64)
    actions = scored[picked] % ACTION_SIZE
    rules = cpu_rules()
    with torch.no_grad():
        pieces, state = rules.child(
            torch.from_numpy(sparse["pieces"][rows]),
            torch.from_numpy(sparse["state"][rows]),
            torch.from_numpy(actions),
        )
        legal = rules.legal(pieces, state)
        mated = rules.in_check(pieces).numpy()
    playable = legal.any(dim=1).numpy()
    value = np.where(playable, 1 - 2 * wins[picked], np.where(mated, -1.0, 0.0))
    # A finished game has no legal move; allowing every action keeps its unused policy finite.
    legal[~legal.any(dim=1)] = True
    legal_rows, legal_actions = legal.nonzero(as_tuple=True)
    child = np.zeros(size, dtype=np.bool_)
    child[rows] = True
    replaced = dict(sparse)
    for prefix in ("deep", "quick"):
        kept = ~child[sparse[f"{prefix}_rows"]]
        for name in ("rows", "actions", "win"):
            replaced[f"{prefix}_{name}"] = sparse[f"{prefix}_{name}"][kept]
    kept = ~child[sparse["legal_rows"]]
    replaced["legal_rows"] = np.concatenate([sparse["legal_rows"][kept], rows[legal_rows.numpy()]])
    replaced["legal_actions"] = np.concatenate(
        [sparse["legal_actions"][kept], legal_actions.numpy()]
    )
    fills: dict[str, np.ndarray | int] = {
        "pieces": pieces.numpy(),
        "state": state.numpy(),
        "action": -1,
        "reply_from": IGNORE,
        "reply_to": IGNORE,
        "board2": IGNORE,
        "board4": IGNORE,
        "line_actions": -1,
    }
    for key, values in fills.items():
        replaced[key] = sparse[key].copy()
        replaced[key][rows] = values
    replaced["child_rows"] = rows.astype(np.int64)
    replaced["child_actions"] = actions.astype(np.int64)
    replaced["child_value"] = value.astype(np.float32)
    return replaced


def densify(
    sparse: dict[str, np.ndarray], temperature: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build ``GameSplit.batch``'s dense targets on ``device`` from a sparse batch.

    Moving a few hundred scored moves instead of three 4,672-wide arrays keeps
    the CPU out of the training loop's critical path.
    """

    import torch

    def tensor(name: str) -> torch.Tensor:
        return torch.from_numpy(sparse[name]).to(device, non_blocking=True)

    size = len(sparse["action"])
    wins = torch.full((size, ACTION_SIZE), float("nan"), device=device)
    deep_rows, deep_actions, deep = tensor("deep_rows"), tensor("deep_actions"), tensor("deep_win")
    quick_rows = tensor("quick_rows")
    if len(quick_rows):
        worst = torch.full((size,), float("inf"), device=device)
        worst = worst.scatter_reduce(0, deep_rows, deep, reduce="amin")
        wins[quick_rows, tensor("quick_actions")] = torch.minimum(
            tensor("quick_win"), worst[quick_rows]
        )
    wins[deep_rows, deep_actions] = deep
    mask = ~torch.isnan(wins)
    best = torch.where(mask, wins, float("-inf")).amax(dim=1)
    weights = torch.where(
        mask, torch.exp(-(best[:, None] - wins) * 100 / temperature), torch.zeros_like(wins)
    )
    legal = torch.zeros((size, ACTION_SIZE), dtype=torch.bool, device=device)
    legal[tensor("legal_rows"), tensor("legal_actions")] = True
    value = 2 * best - 1
    if "child_rows" in sparse:
        value = value.index_put((tensor("child_rows"),), tensor("child_value"))
    return {
        **{
            key: tensor(key)
            for key in ("reply_from", "reply_to", "board2", "board4", "line_actions")
        },
        "pieces": tensor("pieces"),
        "state": tensor("state"),
        "action": tensor("action"),
        "legal_mask": legal,
        # The best scored move weighs 1, so the clamp only keeps a child row's empty policy at 0.
        "policy_target": weights / weights.sum(dim=1, keepdim=True).clamp_min(1),
        "move_value_target": torch.where(mask, 2 * wins - 1, torch.zeros_like(wins)),
        "move_value_mask": mask,
        "value": value,
    }
