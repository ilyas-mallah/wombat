"""Create frozen puzzle suites and compact evaluation shards."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import heapq
import json
from pathlib import Path
from typing import Iterable

import psutil
import zstandard

from wombat.data.lichess import iter_evaluations, iter_puzzles
from wombat.data.splits import position_hash, split_for_fen
from wombat.eval.manifest import PositionCase, PositionManifest


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_puzzle_manifest(source: Path, destination: Path, limit: int) -> PositionManifest:
    """Select the exact deterministic lowest-hash sample without loading the archive."""

    if limit < 1:
        raise ValueError("limit must be positive")
    # Keep a small surplus so the expensive chess validation is performed only
    # on likely winners after the full hash scan, not on millions of records.
    candidate_limit = limit + max(100, limit // 100)
    selected: list[tuple[int, str, object]] = []
    for puzzle in iter_puzzles(source):
        key = int(sha256(puzzle.puzzle_id.encode()).hexdigest(), 16)
        entry = (-key, puzzle.puzzle_id, puzzle)
        if len(selected) < candidate_limit:
            heapq.heappush(selected, entry)
        elif key < -selected[0][0]:
            heapq.heapreplace(selected, entry)
    cases: list[PositionCase] = []
    for _, _, puzzle in sorted(selected, key=lambda item: -item[0]):
        try:
            fen, best_move = puzzle.critical_position()
        except (ValueError, KeyError):
            continue
        cases.append(
            PositionCase(
                id=puzzle.puzzle_id,
                fen=fen,
                best_move=best_move,
                tags=(f"rating:{puzzle.rating}", *puzzle.themes),
            )
        )
        if len(cases) == limit:
            break
    if len(cases) != limit:
        raise RuntimeError(f"only {len(cases)} valid puzzles found for requested {limit}")
    manifest = PositionManifest("lichess-puzzles-frozen-v1", tuple(cases))
    manifest.write(destination)
    return manifest


def build_evaluation_corpus(
    source: Path,
    destination: Path,
    limit: int,
    excluded_fens: Iterable[str] = (),
    minimum_free_gb: float = 10.0,
) -> dict[str, object]:
    if limit < 1:
        raise ValueError("limit must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    free_gb = psutil.disk_usage(destination.anchor).free / 2**30
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"only {free_gb:.2f} GB free; refusing to build corpus")

    exclusions = {position_hash(fen) for fen in excluded_fens}
    paths = {split: destination / f"{split}.jsonl.zst" for split in ("train", "validation", "test")}
    writers = {
        split: zstandard.open(path, mode="wt", encoding="utf-8") for split, path in paths.items()
    }
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    try:
        for record in iter_evaluations(source, lambda reason: rejected.update([reason])):
            key = position_hash(record.fen)
            if key in exclusions:
                rejected["puzzle_overlap"] += 1
                continue
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            split = split_for_fen(record.fen)
            writers[split].write(json.dumps(record.payload(), separators=(",", ":")) + "\n")
            counts[split] += 1
            if sum(counts.values()) >= limit:
                break
    finally:
        for writer in writers.values():
            writer.close()

    files = {
        split: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
            "records": counts[split],
        }
        for split, path in paths.items()
    }
    manifest: dict[str, object] = {
        "schema": "wombat.evaluation-corpus.v1",
        "source": str(source),
        "requested_records": limit,
        "written_records": sum(counts.values()),
        "exclusion_count": len(exclusions),
        "rejected": dict(rejected),
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
