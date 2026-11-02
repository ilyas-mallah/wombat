"""Stockfish win chances for every legal move of game positions, in resumable chunks.

One multi-variation search per position scores all legal moves at once. Each
engine runs under its own worker process, because parsing an engine's output is
Python work and one process tops out near 30 positions per second. Chunks are
written atomically, so an interrupted run resumes where it stopped.

Each position also keeps the first plies of the deep-best move's principal
variation, as squares and boards in the root's canonical frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
from multiprocessing import Pool, util
from pathlib import Path
import shutil
from typing import Any, NamedTuple

import chess
import chess.engine
import numpy as np

from wombat.chess.encode import canonical_square, decode_position, encode_position
from wombat.chess.moves import move_to_action
from wombat.data.columns import file_sha256
from wombat.data.game_corpus import win_chance
from wombat.data.puzzle_corpus import lower_priority

MOVE_LABELS_SCHEMA = "wombat.move-labels.v1"
LINE_PLIES = 4
# Marks squares of plies past a line's end and boards the line never reached.
NO_SQUARE = 255
LINE_COLUMNS = ("line_length", "line_from", "line_to", "line_boards")


class Line(NamedTuple):
    """Up to ``LINE_PLIES`` plies of a principal variation in the root's canonical frame."""

    length: int
    from_squares: np.ndarray  # uint8 (LINE_PLIES,)
    to_squares: np.ndarray  # uint8 (LINE_PLIES,)
    boards: np.ndarray  # uint8 (LINE_PLIES // 2, 64): pieces after plies 2 and 4


def _analyse(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> dict[int, tuple[float, list[chess.Move]]]:
    """Win chance and principal variation for each legal move's action."""

    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=board.legal_moves.count(),
        # Parsing only the score and line keeps per-position Python work small.
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    scored: dict[int, tuple[float, list[chess.Move]]] = {}
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        cp = info["score"].pov(board.turn).score(mate_score=100_000)
        scored.setdefault(move_to_action(board, pv[0]), (win_chance(float(cp)), pv))
    return scored


def score_all_moves(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> list[tuple[int, float]]:
    """(action, win chance) for each legal move, from the side to move."""

    return sorted((action, win) for action, (win, _) in _analyse(engine, board, nodes).items())


def principal_line(board: chess.Board, pv: Sequence[chess.Move]) -> Line:
    """Play ``pv`` from ``board`` until ``LINE_PLIES``, an illegal move, or the game's end."""

    turn = board.turn
    played = board.copy(stack=False)
    from_squares = np.full(LINE_PLIES, NO_SQUARE, dtype=np.uint8)
    to_squares = np.full(LINE_PLIES, NO_SQUARE, dtype=np.uint8)
    boards = np.full((LINE_PLIES // 2, 64), NO_SQUARE, dtype=np.uint8)
    length = 0
    for move in pv[:LINE_PLIES]:
        if played.is_game_over() or not played.is_legal(move):
            break
        from_squares[length] = canonical_square(move.from_square, turn)
        to_squares[length] = canonical_square(move.to_square, turn)
        played.push(move)
        length += 1
        if length % 2 == 0:
            # The root's side moves again, so this encoding shares the root's frame.
            boards[length // 2 - 1] = encode_position(played).pieces
    return Line(length, from_squares, to_squares, boards)


def _missing_lines(rows: int) -> dict[str, np.ndarray]:
    return {
        "line_length": np.zeros(rows, dtype=np.uint8),
        "line_from": np.full((rows, LINE_PLIES), NO_SQUARE, dtype=np.uint8),
        "line_to": np.full((rows, LINE_PLIES), NO_SQUARE, dtype=np.uint8),
        "line_boards": np.full((rows, LINE_PLIES // 2, 64), NO_SQUARE, dtype=np.uint8),
    }


def _start_engine(command: Sequence[str]) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(list(command))
    try:
        engine.configure({"Threads": 1, "Hash": 16})
    except BaseException:
        # An engine left running keeps its event-loop thread, and the process, alive.
        _quit_quietly(engine)
        raise
    return engine


def _quit_quietly(engine: chess.engine.SimpleEngine) -> None:
    try:
        engine.quit()
    except (chess.engine.EngineError, TimeoutError):
        # A crashed engine cannot be asked to quit; its process is already gone.
        pass


_WORKER: dict[str, Any] = {}


def _initialize_worker(command: Sequence[str], split: str, nodes: int, low_priority: bool) -> None:
    if low_priority:
        # Stockfish children inherit the idle priority class on Windows.
        lower_priority()
    _WORKER.update(
        command=list(command),
        nodes=nodes,
        pieces=np.load(Path(split) / "pieces.npy", mmap_mode="r"),
        state=np.load(Path(split) / "state.npy", mmap_mode="r"),
        candidate_offsets=np.load(Path(split) / "candidate_offsets.npy", mmap_mode="r"),
        candidate_actions=np.load(Path(split) / "candidate_actions.npy", mmap_mode="r"),
        engine=_start_engine(command),
    )
    util.Finalize(None, _close_worker, exitpriority=10)


def _close_worker() -> None:
    engine = _WORKER.pop("engine", None)
    if engine is not None:
        _quit_quietly(engine)


def _label_index(index: int) -> tuple[list[tuple[int, float]], Line] | None:
    """All-move scores and the deep-best move's line, or None when the position is skipped.

    Impossible boards and repeated engine failures are skipped.
    """

    board = decode_position(_WORKER["pieces"][index], _WORKER["state"][index])
    if board.status() != chess.STATUS_VALID:
        return None
    # Candidates are stored best first.
    best = int(_WORKER["candidate_actions"][_WORKER["candidate_offsets"][index]])
    for _ in range(2):
        try:
            analysed = _analyse(_WORKER["engine"], board, _WORKER["nodes"])
        except chess.engine.EngineError:
            # EngineTerminatedError is a subclass; a crashed engine is replaced.
            _quit_quietly(_WORKER["engine"])
            _WORKER["engine"] = _start_engine(_WORKER["command"])
            continue
        scores = sorted((action, win) for action, (win, _) in analysed.items())
        _, pv = analysed.get(best, (0.0, []))
        return scores, principal_line(board, pv)
    return None


def _read_chunk(path: Path) -> dict[str, np.ndarray]:
    # Read eagerly so a thousand chunks never hold a thousand open files.
    with np.load(path) as chunk:
        return {name: chunk[name] for name in chunk.files}


def _write_chunk(path: Path, indices: np.ndarray, results: list[Any]) -> None:
    kept = [(int(i), result) for i, result in zip(indices, results, strict=True) if result]
    kept = [(index, result) for index, result in kept if result[0]]
    lines = [line for _, (_, line) in kept]
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        indices=np.array([index for index, _ in kept], dtype=np.int64),
        counts=np.array([len(moves) for _, (moves, _) in kept], dtype=np.int16),
        actions=np.array([action for _, (moves, _) in kept for action, _ in moves], dtype=np.int16),
        win=np.array([win for _, (moves, _) in kept for _, win in moves], dtype=np.float16),
        line_length=np.array([line.length for line in lines], dtype=np.uint8),
        line_from=np.array([line.from_squares for line in lines], dtype=np.uint8).reshape(
            -1, LINE_PLIES
        ),
        line_to=np.array([line.to_squares for line in lines], dtype=np.uint8).reshape(
            -1, LINE_PLIES
        ),
        line_boards=np.array([line.boards for line in lines], dtype=np.uint8).reshape(
            -1, LINE_PLIES // 2, 64
        ),
    )
    partial.replace(path)


def split_identity(split: Path) -> dict[str, object]:
    """The split a label directory belongs to, recognised by its pieces column."""

    pieces = np.load(split / "pieces.npy", mmap_mode="r")
    return {
        "path": split.as_posix(),
        "positions": len(pieces),
        "pieces_sha256": file_sha256(split / "pieces.npy"),
    }


def _same_split(first: object, second: object) -> bool:
    keys = ("positions", "pieces_sha256")
    return (
        isinstance(first, dict)
        and isinstance(second, dict)
        and all(first.get(key) is not None and first.get(key) == second.get(key) for key in keys)
    )


def _read_manifest(labels: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split") is None:
        raise ValueError(f"{labels} does not record the split it labels")
    return manifest


def _excluded(
    exclude: Sequence[Path], split: dict[str, object]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Corpus indices already labelled in ``exclude``, and a manifest entry for each directory."""

    indices = [np.empty(0, dtype=np.int64)]
    entries: list[dict[str, object]] = []
    for labels in exclude:
        manifest = _read_manifest(labels)
        if not _same_split(manifest["split"], split):
            raise ValueError(f"{labels} labels a different split")
        indices.append(np.load(labels / "label_indices.npy").astype(np.int64))
        entries.append(
            {
                "path": labels.as_posix(),
                "positions": manifest["positions"],
                "label_indices_sha256": manifest["sha256"]["label_indices"],
            }
        )
    return np.unique(np.concatenate(indices)), entries


def _check_selection(output: Path, chosen: np.ndarray, chunk_size: int) -> None:
    """Refuse to resume chunks that were cut from a different selection or chunk size."""

    selection = {
        "positions": len(chosen),
        "chunk_size": chunk_size,
        "chosen_sha256": sha256(np.ascontiguousarray(chosen, dtype=np.int64).tobytes()).hexdigest(),
    }
    path = output / "selection.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != selection:
            raise ValueError(f"{output} holds chunks of a different selection")
        return
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


def label_positions(
    split: Path,
    output: Path,
    *,
    count: int | None = None,
    nodes: int,
    engine_command: Sequence[str],
    workers: int = 12,
    chunk_size: int = 5_000,
    seed: int = 0,
    low_priority: bool = True,
    exclude: Sequence[Path] = (),
    reverse: bool = False,
) -> dict[str, object]:
    """Label a deterministic random subset of ``split``; resumes finished chunks.

    The subset is the first ``count`` of a seeded permutation (all when None) after
    dropping positions labelled in ``exclude``. With ``workers`` of 1 the single
    engine runs in this process. ``reverse`` works from the last chunk down, and a chunk
    that appears while the run is going is skipped, so two machines can share one output.
    """

    identity = split_identity(split)
    excluded, exclusions = _excluded(exclude, identity)
    positions = len(np.load(split / "pieces.npy", mmap_mode="r"))
    order = np.random.default_rng(seed).permutation(positions)
    if len(excluded):
        order = order[~np.isin(order, excluded)]
    chosen = np.sort(order[:count])
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    _check_selection(output, chosen, chunk_size)
    probe = _start_engine(engine_command)
    label = str(probe.id.get("name", "stockfish"))
    _quit_quietly(probe)
    pending = [
        (chunks / f"chunk-{number:05d}.npz", chosen[start : start + chunk_size])
        for number, start in enumerate(range(0, len(chosen), chunk_size))
    ]
    pending = [(path, indices) for path, indices in pending if not path.exists()]
    if reverse:
        pending.reverse()
    arguments = (list(engine_command), str(split), nodes, low_priority)
    if workers <= 1:
        _initialize_worker(*arguments)
        try:
            for path, indices in pending:
                if path.exists():
                    continue
                _write_chunk(path, indices, [_label_index(int(i)) for i in indices])
        finally:
            _close_worker()
    else:
        pool = Pool(workers, initializer=_initialize_worker, initargs=arguments)
        try:
            for path, indices in pending:
                if path.exists():
                    continue
                results = pool.map(_label_index, (int(i) for i in indices), chunksize=16)
                _write_chunk(path, indices, results)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
    return finalize_labels(
        output,
        nodes=nodes,
        engine=label,
        seed=seed,
        requested=len(chosen),
        split=identity,
        excluded=exclusions if exclude else None,
    )


def finalize_labels(
    output: Path,
    *,
    nodes: int,
    engine: str,
    seed: int,
    requested: int,
    split: dict[str, object] | None = None,
    excluded: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Merge chunks into ``label_*`` and ``line_*`` columns sorted by corpus index.

    Chunks written without lines give their rows empty lines.
    """

    # The digit pattern skips ``.partial.npz`` leftovers from an interrupted write.
    finished = sorted((output / "chunks").glob("chunk-[0-9][0-9][0-9][0-9][0-9].npz"))
    parts = [_read_chunk(path) for path in finished]
    indices = np.concatenate([part["indices"] for part in parts])
    counts = np.concatenate([part["counts"] for part in parts]).astype(np.int64)
    lines = [
        {name: part[name] for name in LINE_COLUMNS}
        if "line_length" in part
        else _missing_lines(len(part["indices"]))
        for part in parts
    ]
    columns = {
        "label_indices": indices.astype(np.int64),
        "label_offsets": np.concatenate(([0], np.cumsum(counts))),
        "label_actions": np.concatenate([part["actions"] for part in parts]),
        "label_win": np.concatenate([part["win"] for part in parts]),
    }
    for name in LINE_COLUMNS:
        columns[name] = np.concatenate([line[name] for line in lines])
    if np.any(np.diff(indices) <= 0):
        raise ValueError("label chunks must cover increasing, distinct corpus indices")
    hashes = {}
    for name, array in columns.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
        hashes[name] = file_sha256(output / f"{name}.npy")
    manifest: dict[str, object] = {
        "schema": MOVE_LABELS_SCHEMA,
        "positions": len(indices),
        # Impossible boards and positions an engine failed on twice are left unlabelled.
        "skipped": requested - len(indices),
        "moves": int(counts.sum()),
        "line_lengths": {
            length: int(np.count_nonzero(columns["line_length"] == length))
            for length in range(LINE_PLIES + 1)
        },
        "nodes": nodes,
        "engine": engine,
        "seed": seed,
        "sha256": hashes,
    }
    if split is not None:
        manifest["split"] = split
    if excluded is not None:
        manifest["excluded"] = excluded
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


RAGGED_COLUMNS = ("label_actions", "label_win")


def merge_labels(
    inputs: Sequence[Path],
    output: Path,
    *,
    allow_mixed_nodes: bool = False,
    block_rows: int = 500_000,
) -> dict[str, object]:
    """Combine label directories over one split into a single directory sorted by corpus index.

    Columns are streamed through memory maps block by block, and the directory is
    written beside ``output`` and renamed into place once complete. Inputs searched
    with different node counts merge only with ``allow_mixed_nodes``.
    """

    if len(inputs) < 2:
        raise ValueError("merging needs at least two label directories")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    manifests = [_read_manifest(labels) for labels in inputs]
    first = manifests[0]
    for labels, other in zip(inputs[1:], manifests[1:], strict=True):
        if not _same_split(other["split"], first["split"]):
            raise ValueError(f"{labels} labels a different split from {inputs[0]}")
        for key in ("schema", "engine") if allow_mixed_nodes else ("schema", "engine", "nodes"):
            if other.get(key) != first.get(key):
                raise ValueError(f"{labels} differs from {inputs[0]} in {key}")
    sources = [
        {
            name: np.load(labels / f"{name}.npy", mmap_mode="r")
            for name in ("label_indices", "label_offsets", *RAGGED_COLUMNS, *LINE_COLUMNS)
        }
        for labels in inputs
    ]
    indices = np.sort(np.concatenate([source["label_indices"] for source in sources]))
    if np.any(np.diff(indices) <= 0):
        raise ValueError("label directories overlap in corpus indices")
    # Each source row's position in the merged order.
    slots = [np.searchsorted(indices, source["label_indices"]) for source in sources]
    counts = np.empty(len(indices), dtype=np.int64)
    for source, slot in zip(sources, slots, strict=True):
        counts[slot] = np.diff(source["label_offsets"])
    offsets = np.concatenate(([0], np.cumsum(counts)))

    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.save(partial / "label_indices.npy", indices, allow_pickle=False)
    np.save(partial / "label_offsets.npy", offsets, allow_pickle=False)
    for name in RAGGED_COLUMNS:
        _scatter(partial / f"{name}.npy", name, sources, slots, offsets, block_rows)
    for name in LINE_COLUMNS:
        _scatter(partial / f"{name}.npy", name, sources, slots, None, block_rows)
    lengths = np.bincount(np.load(partial / "line_length.npy"), minlength=LINE_PLIES + 1)
    # A merged input records a list of counts.
    nodes = sorted({count for part in manifests for count in np.atleast_1d(part["nodes"]).tolist()})
    manifest: dict[str, object] = {
        "schema": MOVE_LABELS_SCHEMA,
        "positions": len(indices),
        "unlabelled": int(first["split"]["positions"]) - len(indices),
        "moves": int(offsets[-1]),
        "line_lengths": {length: int(lengths[length]) for length in range(LINE_PLIES + 1)},
        # One count when every input agrees, else the distinct counts in increasing order.
        "nodes": nodes[0] if len(nodes) == 1 else nodes,
        "engine": first["engine"],
        "split": first["split"],
        "inputs": [
            {
                "path": labels.as_posix(),
                "nodes": part["nodes"],
                "positions": part["positions"],
                "skipped": part.get("skipped"),
                "seed": part.get("seed"),
                "label_indices_sha256": part["sha256"]["label_indices"],
            }
            for labels, part in zip(inputs, manifests, strict=True)
        ],
        "sha256": {
            name: file_sha256(partial / f"{name}.npy")
            for name in ("label_indices", "label_offsets", *RAGGED_COLUMNS, *LINE_COLUMNS)
        },
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.replace(output)
    return manifest


def _scatter(
    path: Path,
    name: str,
    sources: list[dict[str, np.ndarray]],
    slots: list[np.ndarray],
    offsets: np.ndarray | None,
    block_rows: int,
) -> None:
    """Write column ``name`` of each source into its merged rows; ragged when ``offsets`` is set."""

    columns = [source[name] for source in sources]
    if len({(column.dtype, column.shape[1:]) for column in columns}) > 1:
        raise ValueError(f"label directories store {name} with different types or shapes")
    rows = int(offsets[-1]) if offsets is not None else sum(len(slot) for slot in slots)
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=columns[0].dtype, shape=(rows, *columns[0].shape[1:])
    )
    for source, slot, column in zip(sources, slots, columns, strict=True):
        for start in range(0, len(slot), block_rows):
            stop = min(start + block_rows, len(slot))
            if offsets is None:
                target[slot[start:stop]] = column[start:stop]
                continue
            own = np.asarray(source["label_offsets"][start : stop + 1], dtype=np.int64)
            # Each entry moves by its row's shift from the source's offsets to the merged ones.
            shift = offsets[slot[start:stop]] - own[:-1]
            destination = np.repeat(shift, np.diff(own)) + np.arange(own[0], own[-1])
            target[destination] = column[own[0] : own[-1]]
    target.flush()
    del target
