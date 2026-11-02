"""ChessBench's action-value data as a game corpus whose labels score every legal move.

DeepMind's bags hold (fen, move, win_prob) records, where win_prob is Stockfish 16's win
probability for the side that plays ``move``, from the same logistic as ``win_chance``. The
train shards are shuffled record by record, so one position's moves lie scattered over about
as many shards as it has moves, and only a scan of every shard completes a position.

Importing a shard keeps the records of a fixed hash-chosen fraction of FENs, sorted by FEN
key, and adds their unseen FENs to a store. Joining gathers each key range over every
imported shard, keeps positions whose labels cover exactly their legal moves, drops
evaluation positions and repeats, and writes a ``GameSplit`` train split with labels.
"""

from __future__ import annotations

from array import array
from base64 import b64encode
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
import csv
from functools import cache
from hashlib import blake2b, md5, sha256
import http.client
import json
import mmap
from multiprocessing import Pool
from pathlib import Path
import shutil
import struct
import time
from typing import Any, NamedTuple
import urllib.request

import chess
import numpy as np
import psutil

from wombat.chess.encode import play_encoding, state_vector
from wombat.chess.moves import uci_action
from wombat.data.columns import file_sha256
from wombat.data.move_labels import LINE_PLIES, MOVE_LABELS_SCHEMA, NO_SQUARE
from wombat.data.puzzle_corpus import lower_priority

CHESSBENCH_SCHEMA = "wombat.chessbench-corpus.v1"
TRAIN_SHARDS = 2148
TRAIN_URL = (
    "https://storage.googleapis.com/searchless_chess/data/train/"
    "action_value-{index:05d}-of-02148_data.bag"
)
TEST_URL = "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag"
# Best moves kept as candidates. Labels hold every move, so training targets are unchanged.
CANDIDATES = 5
KEY_SPACE = 1 << 64
_WIN = struct.Struct(">d")


def shard_name(index: int) -> str:
    return f"action_value-{index:05d}-of-{TRAIN_SHARDS:05d}"


def parse_shards(spec: str) -> list[int]:
    """Train shard indices from a list such as ``0-39,45``."""

    indices: set[int] = set()
    for part in filter(None, (piece.strip() for piece in spec.split(","))):
        first, _, last = part.partition("-")
        indices.update(range(int(first), int(last or first) + 1))
    if any(not 0 <= index < TRAIN_SHARDS for index in indices):
        raise ValueError(f"shard indices must lie in [0, {TRAIN_SHARDS})")
    return sorted(indices)


def bag_records(path: Path) -> Iterator[bytes]:
    """Records of an uncompressed Bagz file, whose tail lists each record's end offset."""

    with path.open("rb") as handle:
        if path.stat().st_size == 0:
            return
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if len(data) < 8:
                raise ValueError(f"{path} is too small to be a bag")
            (index_start,) = struct.unpack("<Q", data[-8:])
            if index_start > len(data) or (len(data) - index_start) % 8:
                raise ValueError(f"{path} has a malformed record index")
            ends = np.frombuffer(data[index_start:], dtype="<i8")
            start = 0
            for end in ends.tolist():
                yield data[start:end]
                start = end


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def decode_action_value(record: bytes) -> tuple[str, str, float]:
    """Beam's (StrUtf8, StrUtf8, Float) tuple: two varint-prefixed strings, a big-endian double."""

    length, offset = _varint(record, 0)
    fen = record[offset : offset + length]
    length, offset = _varint(record, offset + length)
    move = record[offset : offset + length]
    if len(record) - offset - length != _WIN.size:
        raise ValueError("malformed action-value record")
    return fen.decode(), move.decode(), _WIN.unpack_from(record, offset + length)[0]


def fen_key(fen: bytes) -> int:
    """64-bit hash of a whole FEN, which picks the sampled FENs and groups their records."""

    return int.from_bytes(blake2b(fen, digest_size=8).digest(), "big")


def move_code(uci: str) -> int:
    """A move's squares and promotion packed into 15 bits."""

    move = chess.Move.from_uci(uci)
    return move.from_square | move.to_square << 6 | (move.promotion or 0) << 12


@cache
def _spelling(code: int) -> str:
    return chess.Move(code & 63, code >> 6 & 63, code >> 12 or None).uci()


@cache
def _action(code: int, turn: chess.Color) -> int:
    try:
        return uci_action(_spelling(code), turn)
    except ValueError:
        return -1


class Selected(NamedTuple):
    """One bag's records of the sampled FENs, sorted by FEN key, and those FENs."""

    records: int
    keys: np.ndarray  # uint64 per kept record
    moves: np.ndarray  # uint16 move code per kept record
    win: np.ndarray  # float32 per kept record
    fen_keys: np.ndarray  # uint64, sorted and distinct
    fens: np.ndarray  # bytes, one per FEN key


def select_records(records: Iterable[bytes], threshold: int = KEY_SPACE) -> Selected:
    """Records whose FEN key is below ``threshold``: a ``threshold / 2**64`` share of FENs."""

    fens: dict[int, bytes] = {}
    codes: dict[bytes, int] = {}
    keys, moves, wins = array("Q"), array("H"), array("f")
    read = 0
    for record in records:
        read += 1
        split = record[0] + 1
        # One-byte lengths cover every real FEN and move; anything else takes the full decoder.
        slow = split > 0x80 or record[split] >= 0x80 or len(record) != split + record[split] + 9
        if slow:
            fen_text, move_text, win = decode_action_value(record)
            fen, move = fen_text.encode(), move_text.encode()
        else:
            fen = record[1:split]
        key = fen_key(fen)
        if key >= threshold:
            continue
        if not slow:
            move = record[split + 1 : -8]
            win = _WIN.unpack_from(record, len(record) - 8)[0]
        code = codes.get(move)
        if code is None:
            code = codes[move] = move_code(move.decode())
        keys.append(key)
        moves.append(code)
        wins.append(win)
        fens.setdefault(key, fen)
    key_column = np.asarray(keys, dtype=np.uint64)
    order = np.argsort(key_column, kind="stable")
    fen_keys = np.array(sorted(fens), dtype=np.uint64)
    return Selected(
        read,
        key_column[order],
        np.asarray(moves, dtype=np.uint16)[order],
        np.asarray(wins, dtype=np.float32)[order],
        fen_keys,
        np.array([fens[key] for key in fen_keys.tolist()], dtype=np.bytes_),
    )


def position_key(board: chess.Board) -> int:
    """First 64 bits of the SHA-256 of the board, turn, castling and en passant FEN fields."""

    return _fields_key(" ".join(board.fen().split()[:4]))


def _fields_key(fields: str) -> int:
    return int.from_bytes(sha256(fields.encode()).digest()[:8], "big")


def fen_keys(fens: Iterable[str]) -> np.ndarray:
    """Sorted distinct position keys of ``fens``."""

    return np.unique(np.array([position_key(chess.Board(fen)) for fen in fens], dtype=np.uint64))


def puzzle_line_fens(path: Path) -> Iterator[str]:
    """Every position of each line in a Lichess-format puzzle CSV, its start included."""

    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            board = chess.Board(row["FEN"])
            yield board.fen()
            for move in row["Moves"].split():
                board.push_uci(move)
                yield board.fen()


def _contains(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    if not len(sorted_keys):
        return np.zeros(len(keys), dtype=np.bool_)
    slots = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return np.asarray(sorted_keys[slots] == keys)


def encode_positions(
    keys: np.ndarray,
    moves: np.ndarray,
    win: np.ndarray,
    fen_keys: np.ndarray,
    fens: np.ndarray,
    excluded: set[int],
) -> tuple[dict[str, np.ndarray], Counter[str]]:
    """Chunk columns of every usable position, moves best first, and the rejection counts.

    Records are grouped by FEN key and looked up in the sorted ``fen_keys``. A position is
    usable when it is valid, its position key is not in ``excluded``, and its labelled moves
    are exactly its legal moves. Chunk keys are position keys, left for the join to
    deduplicate.
    """

    rejected: Counter[str] = Counter()
    # One record per FEN and move; a repeat keeps its first copy.
    order = np.lexsort((moves, keys))
    keys, moves, win = keys[order], moves[order], win[order]
    repeat = np.zeros(len(keys), dtype=np.bool_)
    repeat[1:] = (keys[1:] == keys[:-1]) & (moves[1:] == moves[:-1])
    if repeat.any():
        rejected["repeated_record"] += int(repeat.sum())
        keys, moves, win = keys[~repeat], moves[~repeat], win[~repeat]
    order = np.lexsort((moves, -win, keys))
    keys, moves, win = keys[order], moves[order], win[order]
    groups, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    slots = np.minimum(np.searchsorted(fen_keys, groups), max(len(fen_keys) - 1, 0))
    known = np.asarray(fen_keys[slots] == groups) if len(fen_keys) else np.zeros(0, np.bool_)
    kept = np.zeros(len(groups), dtype=np.bool_)
    black = np.zeros(len(groups), dtype=np.bool_)
    position_keys, pieces, states = [], [], []
    for index, (start, count) in enumerate(zip(starts.tolist(), counts.tolist(), strict=True)):
        if not known[index]:
            rejected["missing_fen"] += 1
            continue
        fen = bytes(fens[slots[index]]).decode()
        board = chess.Board(fen)
        if board.status() != chess.STATUS_VALID:
            rejected["invalid_position"] += 1
            continue
        fields = fen.split()
        # ChessBench writes python-chess FENs; only an en passant field may need normalizing.
        key = _fields_key(" ".join(fields[:4])) if fields[3] == "-" else position_key(board)
        if key in excluded:
            rejected["excluded"] += 1
            continue
        # Comparing spellings avoids encoding every legal move.
        labelled = [_spelling(code) for code in moves[start : start + count].tolist()]
        legal = {move.uci() for move in board.legal_moves}
        if not legal.issuperset(labelled):
            rejected["illegal_move"] += 1
            continue
        if len(labelled) < len(legal):
            rejected["missing_moves"] += 1
            continue
        # Games corpora and play both use halfmove bucket 0.
        encoded = play_encoding(board)
        kept[index] = True
        black[index] = board.turn == chess.BLACK
        position_keys.append(key)
        pieces.append(encoded.pieces)
        states.append(state_vector(encoded))
    taken = np.repeat(kept, counts)
    codes, inverse = np.unique(moves[taken], return_inverse=True)
    table = np.array(
        [[_action(code, chess.WHITE), _action(code, chess.BLACK)] for code in codes.tolist()],
        dtype=np.int16,
    ).reshape(-1, 2)
    columns: dict[str, np.ndarray] = {
        "keys": np.array(position_keys, dtype=np.uint64),
        "pieces": np.array(pieces, dtype=np.uint8).reshape(-1, 64),
        "state": np.array(states, dtype=np.uint8).reshape(-1, 26),
        "counts": counts[kept].astype(np.int16),
        "actions": table[
            inverse.reshape(-1), np.repeat(black[kept], counts[kept]).astype(np.int64)
        ],
        "win": win[taken].astype(np.float16),
    }
    return columns, rejected


def _digests(path: Path) -> tuple[str, str]:
    """Base64 MD5, as Cloud Storage reports it, and hex SHA-256 of a file."""

    first, second = md5(), sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            first.update(block)
            second.update(block)
    return b64encode(first.digest()).decode(), second.hexdigest()


def download(url: str, path: Path, *, attempts: int = 6, timeout: float = 60) -> dict[str, object]:
    """Fetch ``url`` into ``path`` through a resumable ``.part`` file, checking size and MD5."""

    partial = path.with_name(path.name + ".part")
    if path.exists():
        path.replace(partial)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=timeout
            ) as head:
                size = int(head.headers["Content-Length"])
                hashes = head.headers.get_all("x-goog-hash") or []
            break
        except (OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    # Cloud Storage sends ``x-goog-hash: md5=...`` beside the CRC32C.
    expected: dict[str, str] = {}
    for value in hashes:
        for item in value.split(","):
            kind, _, digest = item.strip().partition("=")
            expected[kind] = digest
    for attempt in range(attempts):
        have = partial.stat().st_size if partial.exists() else 0
        if have < size:
            request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    # A server that ignores the range sends the whole file again.
                    with partial.open("ab" if response.status == 206 else "wb") as target:
                        shutil.copyfileobj(response, target, 8 * 1024 * 1024)
            except (OSError, http.client.HTTPException):
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
                continue
        if partial.exists() and partial.stat().st_size == size:
            digest, sha = _digests(partial)
            if expected.get("md5", digest) == digest:
                partial.replace(path)
                return {"url": url, "bytes": size, "md5": digest, "sha256": sha}
        partial.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url} intact")


def _finished(directory: Path, pattern: str) -> list[Path]:
    """Files matching ``pattern``, leaving out ``.partial`` ones from an interrupted write."""

    return sorted(path for path in directory.glob(pattern) if ".partial." not in path.name)


def _check_free(directory: Path, minimum_gb: float) -> None:
    free_gb = psutil.disk_usage(str(directory.resolve())).free / 2**30
    if free_gb < minimum_gb:
        raise RuntimeError(f"only {free_gb:.2f} GB free for {directory}; refusing to continue")


_WORKER: dict[str, Any] = {}


def _initialize_worker(settings: dict[str, Any], low_priority: bool) -> None:
    if low_priority:
        lower_priority()
    _WORKER.clear()
    _WORKER.update(settings)


def _import_shard(task: tuple[str, str]) -> dict[str, Any]:
    """Keep one bag's sampled records in ``records/<name>``; a finished shard is skipped.

    ``location`` is a URL to download, then delete, or a local bag to read in place.
    """

    name, location = task
    records = Path(_WORKER["output"]) / "records"
    done = records / f"{name}.json"
    if done.exists():
        stats: dict[str, Any] = json.loads(done.read_text(encoding="utf-8"))
        return stats
    started = time.perf_counter()
    remote = location.startswith(("http://", "https://"))
    if remote:
        work_dir = Path(_WORKER["work_dir"])
        _check_free(work_dir, _WORKER["minimum_free_gb"])
        bag = work_dir / f"{name}.bag"
        source = download(location, bag)
    else:
        bag = Path(location)
        source = {"path": bag.as_posix(), "bytes": bag.stat().st_size, "sha256": file_sha256(bag)}
    fetched = time.perf_counter()
    selected = select_records(bag_records(bag), _WORKER["threshold"])
    if remote:
        bag.unlink()
    folder = records / name
    folder.mkdir(parents=True, exist_ok=True)
    for column in ("keys", "moves", "win", "fen_keys", "fens"):
        np.save(folder / f"{column}.npy", getattr(selected, column), allow_pickle=False)
    stats = {
        "name": name,
        "source": source,
        "threshold": _WORKER["threshold"],
        "records": selected.records,
        "selected": len(selected.keys),
        "fens": len(selected.fen_keys),
        "seconds": {
            "fetch": round(fetched - started, 1),
            "select": round(time.perf_counter() - fetched, 1),
        },
    }
    written = done.with_suffix(".partial.json")
    written.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    written.replace(done)
    return stats


def _merge_fens(output: Path, name: str, known: np.ndarray) -> np.ndarray:
    """Move a shard's unseen FENs into ``fens/<name>.npz``; returns the grown sorted keys."""

    folder = output / "records" / name
    if not (folder / "fens.npy").exists():
        return known
    target = output / "fens" / f"{name}.npz"
    if not target.exists():
        keys = np.load(folder / "fen_keys.npy")
        new = ~_contains(known, keys)
        partial = target.with_suffix(".partial.npz")
        np.savez(partial, keys=keys[new], fens=np.load(folder / "fens.npy")[new])
        partial.replace(target)
        # Two sorted runs, so the stable sort merges them in linear time.
        known = np.sort(np.concatenate([known, keys[new]]), kind="stable")
    (folder / "fen_keys.npy").unlink()
    (folder / "fens.npy").unlink()
    return known


def _known_fens(output: Path) -> np.ndarray:
    keys = [np.zeros(0, dtype=np.uint64)]
    for path in _finished(output / "fens", "*.npz"):
        with np.load(path) as stored:
            keys.append(stored["keys"])
    return np.sort(np.concatenate(keys))


def import_chessbench(
    sources: Sequence[tuple[str, str]],
    output: Path,
    *,
    work_dir: Path,
    fraction: float,
    excluded: np.ndarray,
    exclusion: dict[str, object],
    workers: int = 1,
    join: bool = True,
    partitions: int = 64,
    delete_records: bool = False,
    low_priority: bool = True,
    minimum_free_gb: float = 10.0,
) -> dict[str, object] | None:
    """Import each (name, URL or bag path) not yet imported, then join every imported shard.

    ``fraction`` of FENs is sampled by key, the same share in every shard. ``excluded`` holds
    sorted position keys to leave out and ``exclusion`` describes them.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    for directory in (output / "records", output / "fens", work_dir):
        directory.mkdir(parents=True, exist_ok=True)
        _check_free(directory, minimum_free_gb)
    if low_priority:
        lower_priority()
    threshold = min(int(fraction * KEY_SPACE), KEY_SPACE)
    for marker in _finished(output / "records", "*.json"):
        if json.loads(marker.read_text(encoding="utf-8"))["threshold"] != threshold:
            raise ValueError(f"{output} holds shards sampled at another fraction of FENs")
    settings = {
        "output": str(output),
        "work_dir": str(work_dir),
        "threshold": threshold,
        "minimum_free_gb": minimum_free_gb,
    }
    known = _known_fens(output)
    # Shards imported before an interruption may still hold unmerged FENs.
    for marker in _finished(output / "records", "*.json"):
        known = _merge_fens(output, marker.name[: -len(".json")], known)
    if workers <= 1:
        _initialize_worker(settings, low_priority=False)
        results: Iterator[dict[str, Any]] = map(_import_shard, sources)
        for stats in results:
            known = _merge_fens(output, stats["name"], known)
            print(json.dumps(_summary(stats)), flush=True)
    else:
        with Pool(
            workers, initializer=_initialize_worker, initargs=(settings, low_priority)
        ) as pool:
            for stats in pool.imap_unordered(_import_shard, sources):
                known = _merge_fens(output, stats["name"], known)
                print(json.dumps(_summary(stats)), flush=True)
    if not join:
        return None
    return join_shards(
        output,
        excluded=excluded,
        exclusion=exclusion,
        partitions=partitions,
        workers=workers,
        low_priority=low_priority,
        delete_records=delete_records,
    )


def _summary(stats: dict[str, Any]) -> dict[str, object]:
    return {key: stats[key] for key in ("name", "records", "selected", "seconds")}


def _join_partition(task: tuple[int, int, int]) -> dict[str, Any]:
    """Encode the positions whose FEN keys lie in [low, high) into ``join/part-<number>.npz``."""

    number, low, high = task
    output = Path(_WORKER["output"])

    def span(keys: np.ndarray) -> tuple[int, int]:
        # The last key below ``high``, since 2**64 itself does not fit in a key.
        start = np.searchsorted(keys, np.uint64(low), side="left")
        stop = np.searchsorted(keys, np.uint64(high - 1), side="right") if high > low else start
        return int(start), int(stop)

    parts: dict[str, list[np.ndarray]] = {"keys": [], "moves": [], "win": []}
    for name in _WORKER["names"]:
        folder = output / "records" / name
        keys = np.load(folder / "keys.npy", mmap_mode="r")
        start, stop = span(keys)
        for column in parts:
            values = keys if column == "keys" else np.load(folder / f"{column}.npy", mmap_mode="r")
            parts[column].append(np.array(values[start:stop]))
    store = output / "join"
    fen_keys = np.load(store / "fen_keys.npy", mmap_mode="r")
    start, stop = span(fen_keys)
    columns, rejected = encode_positions(
        np.concatenate(parts["keys"]),
        np.concatenate(parts["moves"]),
        np.concatenate(parts["win"]),
        np.array(fen_keys[start:stop]),
        np.array(np.load(store / "fens.npy", mmap_mode="r")[start:stop]),
        _WORKER["excluded"],
    )
    np.savez_compressed(
        store / f"part-{number:04d}.npz",
        keys=columns["keys"],
        pieces=columns["pieces"],
        state=columns["state"],
        counts=columns["counts"],
        actions=columns["actions"],
        win=columns["win"],
    )
    return {"number": number, "fens": stop - start, "kept": len(columns["keys"]), **rejected}


def join_shards(
    output: Path,
    *,
    excluded: np.ndarray,
    exclusion: dict[str, object],
    partitions: int = 64,
    workers: int = 1,
    low_priority: bool = True,
    delete_records: bool = False,
) -> dict[str, object]:
    """Write ``train`` and ``train-labels`` from every shard in ``output/records``.

    Positions come in FEN key order and each keeps its first occurrence. The new directories
    replace old ones only once they are complete. ``delete_records`` removes the imported
    records once every position is encoded.
    """

    markers = _finished(output / "records", "*.json")
    stats = [json.loads(path.read_text(encoding="utf-8")) for path in markers]
    if not stats:
        raise ValueError(f"{output / 'records'} holds no imported shards")
    thresholds = {int(entry["threshold"]) for entry in stats}
    if len(thresholds) != 1:
        raise ValueError("imported shards sampled different fractions of FENs")
    threshold = thresholds.pop()
    keys_path = output / "excluded_keys.npy"
    np.save(keys_path, np.asarray(excluded, dtype=np.uint64), allow_pickle=False)
    exclusion = {**exclusion, "positions": len(excluded), "sha256": file_sha256(keys_path)}
    store = output / "join"
    if store.exists():
        shutil.rmtree(store)
    store.mkdir()
    fen_parts = []
    for path in _finished(output / "fens", "*.npz"):
        with np.load(path) as stored:
            fen_parts.append((stored["keys"], stored["fens"]))
    all_keys = np.concatenate([keys for keys, _ in fen_parts])
    order = np.argsort(all_keys, kind="stable")
    np.save(store / "fen_keys.npy", all_keys[order], allow_pickle=False)
    np.save(store / "fens.npy", np.concatenate([fens for _, fens in fen_parts])[order])
    del fen_parts, all_keys, order
    edges = [threshold * part // partitions for part in range(partitions + 1)]
    tasks = [(part, edges[part], edges[part + 1]) for part in range(partitions)]
    settings = {
        "output": str(output),
        "names": [path.name[: -len(".json")] for path in markers],
        "excluded": set(np.asarray(excluded, dtype=np.uint64).tolist()),
    }
    if workers <= 1:
        _initialize_worker(settings, low_priority=False)
        joined = [_join_partition(task) for task in tasks]
    else:
        with Pool(
            workers, initializer=_initialize_worker, initargs=(settings, low_priority)
        ) as pool:
            joined = pool.map(_join_partition, tasks, chunksize=1)
    (store / "fen_keys.npy").unlink()
    (store / "fens.npy").unlink()
    if delete_records:
        # Frees the disk the output needs; a later import downloads every shard again.
        shutil.rmtree(output / "records")
        shutil.rmtree(output / "fens")
    chunks = [store / f"part-{part:04d}.npz" for part in range(partitions)]
    positions, fresh_rows = _first_occurrences(chunks)
    train, labels, moves = _write_columns(output, chunks, fresh_rows, positions)
    split = {
        "path": (output / "train").as_posix(),
        "positions": positions,
        "pieces_sha256": train["pieces"],
    }
    label_manifest = {
        "schema": MOVE_LABELS_SCHEMA,
        "positions": positions,
        "skipped": 0,
        "moves": moves,
        "line_lengths": {
            length: positions if length == 0 else 0 for length in range(LINE_PLIES + 1)
        },
        "nodes": None,
        "engine": "Stockfish 16 (ChessBench action values)",
        "seed": None,
        "sha256": labels,
        "split": split,
    }
    (output / "train-labels.partial" / "manifest.json").write_text(
        json.dumps(label_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for directory in ("train", "train-labels"):
        if (output / directory).exists():
            shutil.rmtree(output / directory)
        (output / f"{directory}.partial").replace(output / directory)
    shutil.rmtree(store)
    rejected: Counter[str] = Counter()
    for entry in joined:
        rejected.update({key: value for key, value in entry.items() if key not in _JOIN_COUNTS})
    kept = sum(int(entry["kept"]) for entry in joined)
    manifest: dict[str, object] = {
        "schema": CHESSBENCH_SCHEMA,
        "source": TRAIN_URL,
        "shards": stats,
        "records": sum(int(entry["records"]) for entry in stats),
        "selected_records": sum(int(entry["selected"]) for entry in stats),
        "fraction": threshold / KEY_SPACE,
        "fens": sum(int(entry["fens"]) for entry in joined),
        "rejected": dict(sorted(rejected.items())),
        # Repeats of a position kept earlier, differing at most in the move counters.
        "duplicates": kept - positions,
        "exclusion": exclusion,
        "candidates": CANDIDATES,
        "halfmove_bucket": 0,
        "splits": {"train": {"positions": positions, "moves": moves, "sha256": train}},
        "labels": (output / "train-labels").as_posix(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


_JOIN_COUNTS = ("number", "fens", "kept")


def _first_occurrences(chunks: Sequence[Path]) -> tuple[int, list[np.ndarray]]:
    """Rows of each chunk that are their position's first occurrence, and how many there are."""

    seen = np.zeros(0, dtype=np.uint64)
    fresh_rows = []
    for path in chunks:
        with np.load(path) as chunk:
            keys = chunk["keys"]
        fresh = np.zeros(len(keys), dtype=np.bool_)
        fresh[np.unique(keys, return_index=True)[1]] = True
        fresh &= ~_contains(seen, keys)
        # Two sorted runs, so the stable sort merges them in linear time.
        seen = np.sort(np.concatenate([seen, np.sort(keys[fresh])]), kind="stable")
        fresh_rows.append(fresh)
    positions = int(sum(int(fresh.sum()) for fresh in fresh_rows))
    if positions == 0:
        raise ValueError(
            "no complete positions survive; a position completes only once every shard "
            "holding its moves is imported"
        )
    return positions, fresh_rows


def _write_columns(
    output: Path, chunks: Sequence[Path], fresh_rows: Sequence[np.ndarray], positions: int
) -> tuple[dict[str, str], dict[str, str], int]:
    """Fill ``train.partial`` and ``train-labels.partial``; returns their hashes and the moves."""

    moves = candidates = 0
    for path, fresh in zip(chunks, fresh_rows, strict=True):
        with np.load(path) as chunk:
            counts = chunk["counts"][fresh].astype(np.int64)
        moves += int(counts.sum())
        candidates += int(np.minimum(counts, CANDIDATES).sum())
    directories = {"train": output / "train.partial", "labels": output / "train-labels.partial"}
    for directory in directories.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    shapes: dict[str, tuple[str, type[np.generic], tuple[int, ...]]] = {
        "pieces": ("train", np.uint8, (positions, 64)),
        "state": ("train", np.uint8, (positions, 26)),
        "legal_counts": ("train", np.int16, (positions,)),
        "candidate_counts": ("train", np.int8, (positions,)),
        "legal_offsets": ("train", np.int64, (positions + 1,)),
        "legal_actions": ("train", np.int16, (moves,)),
        "candidate_offsets": ("train", np.int64, (positions + 1,)),
        "candidate_actions": ("train", np.int16, (candidates,)),
        "candidate_win": ("train", np.float16, (candidates,)),
        "label_indices": ("labels", np.int64, (positions,)),
        "label_offsets": ("labels", np.int64, (positions + 1,)),
        "label_actions": ("labels", np.int16, (moves,)),
        "label_win": ("labels", np.float16, (moves,)),
        "line_length": ("labels", np.uint8, (positions,)),
        "line_from": ("labels", np.uint8, (positions, LINE_PLIES)),
        "line_to": ("labels", np.uint8, (positions, LINE_PLIES)),
        "line_boards": ("labels", np.uint8, (positions, LINE_PLIES // 2, 64)),
    }
    columns = {
        name: np.lib.format.open_memmap(
            directories[folder] / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (folder, dtype, shape) in shapes.items()
    }
    columns["label_indices"][:] = np.arange(positions)
    # Labels hold no Stockfish line.
    columns["line_length"][:] = 0
    for name in ("line_from", "line_to", "line_boards"):
        columns[name][:] = NO_SQUARE
    for name in ("legal_offsets", "candidate_offsets", "label_offsets"):
        columns[name][0] = 0
    row = entry = candidate = 0
    for path, fresh in zip(chunks, fresh_rows, strict=True):
        with np.load(path) as chunk:
            data = {key: chunk[key] for key in chunk.files}
        counts = data["counts"].astype(np.int64)
        taken = np.repeat(fresh, counts)
        counts = counts[fresh]
        actions, win = data["actions"][taken], data["win"][taken]
        rows, entries = len(counts), len(actions)
        # Moves are stored best first, so each position's leading entries are its candidates.
        within = np.arange(entries) - np.repeat(np.cumsum(counts) - counts, counts)
        top = within < CANDIDATES
        tops = np.minimum(counts, CANDIDATES)
        chosen = int(top.sum())
        span, moved = slice(row, row + rows), slice(entry, entry + entries)
        columns["pieces"][span] = data["pieces"][fresh]
        columns["state"][span] = data["state"][fresh]
        columns["legal_counts"][span] = counts
        columns["candidate_counts"][span] = tops
        for offsets in ("legal_offsets", "label_offsets"):
            columns[offsets][row + 1 : row + rows + 1] = entry + np.cumsum(counts)
        columns["candidate_offsets"][row + 1 : row + rows + 1] = candidate + np.cumsum(tops)
        columns["legal_actions"][moved] = actions
        columns["label_actions"][moved] = actions
        columns["label_win"][moved] = win
        columns["candidate_actions"][candidate : candidate + chosen] = actions[top]
        columns["candidate_win"][candidate : candidate + chosen] = win[top]
        row, entry, candidate = row + rows, entry + entries, candidate + chosen
    for column in columns.values():
        column.flush()
    del columns
    hashes: dict[str, dict[str, str]] = {"train": {}, "labels": {}}
    for name, (folder, _, _) in shapes.items():
        hashes[folder][name] = file_sha256(directories[folder] / f"{name}.npy")
    return hashes["train"], hashes["labels"], moves
