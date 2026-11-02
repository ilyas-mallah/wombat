"""Directories of memory-mapped ``.npy`` columns, written by streaming.

Fixed columns hold one row per position. Ragged columns hold a variable number
of entries per position; each ragged group has a ``<group>_counts`` fixed
column and gets a ``<group>_offsets`` column when the directory is finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np

FixedSpec = dict[str, tuple[type[np.generic], tuple[int, ...]]]
RaggedSpec = dict[str, tuple[type[np.generic], str]]


@dataclass
class ColumnChunk:
    """Rows produced by one worker chunk for one split."""

    fixed: dict[str, list[object]]
    ragged: dict[str, list[object]]
    fens: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, fixed: FixedSpec, ragged: RaggedSpec) -> ColumnChunk:
        return cls({name: [] for name in fixed}, {name: [] for name in ragged})

    def __len__(self) -> int:
        return len(next(iter(self.fixed.values())))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npy(path: Path, array: np.ndarray) -> str:
    np.save(path, array, allow_pickle=False)
    return file_sha256(path)


class ColumnWriter:
    """Append chunks to raw files, then convert every column to ``.npy`` once."""

    def __init__(
        self, directory: Path, fixed: FixedSpec, ragged: RaggedSpec, *, keep_fens: bool
    ) -> None:
        self.directory = directory
        self.fixed = fixed
        self.ragged = ragged
        self.keep_fens = keep_fens
        self.scratch = directory / ".partial"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.handles = {
            name: (self.scratch / f"{name}.bin").open("wb") for name in (*fixed, *ragged)
        }
        self.lengths = {name: 0 for name in (*fixed, *ragged)}
        self.fens: list[str] = []

    def add(self, chunk: ColumnChunk) -> None:
        for name, (dtype, _) in self.fixed.items():
            values = np.asarray(chunk.fixed[name], dtype=dtype)
            self.handles[name].write(values.tobytes())
            self.lengths[name] += len(chunk.fixed[name])
        for name, (dtype, _) in self.ragged.items():
            self.handles[name].write(np.asarray(chunk.ragged[name], dtype=dtype).tobytes())
            self.lengths[name] += len(chunk.ragged[name])
        if self.keep_fens:
            self.fens.extend(chunk.fens)

    def _raw(self, name: str, dtype: type[np.generic], shape: tuple[int, ...]) -> np.ndarray:
        count = self.lengths[name]
        if count == 0:
            return np.zeros((0, *shape), dtype=dtype)
        return np.memmap(self.scratch / f"{name}.bin", dtype=dtype, mode="r", shape=(count, *shape))

    def finish(self) -> dict[str, object]:
        for handle in self.handles.values():
            handle.close()
        hashes: dict[str, str] = {}
        for name, (dtype, shape) in self.fixed.items():
            hashes[name] = _write_npy(self.directory / f"{name}.npy", self._raw(name, dtype, shape))
        for group in sorted({counts for _, counts in self.ragged.values()}):
            counts = self._raw(f"{group}_counts", self.fixed[f"{group}_counts"][0], ())
            offsets = np.concatenate(([0], np.cumsum(counts.astype(np.int64))))
            hashes[f"{group}_offsets"] = _write_npy(
                self.directory / f"{group}_offsets.npy", offsets
            )
            del counts
        for name, (dtype, _) in self.ragged.items():
            hashes[name] = _write_npy(self.directory / f"{name}.npy", self._raw(name, dtype, ()))
        if self.keep_fens:
            hashes["fens"] = _write_npy(self.directory / "fens.npy", np.array(self.fens, str))
        positions = next(iter(self.lengths[name] for name in self.fixed))
        shutil.rmtree(self.scratch)
        return {"positions": positions, "sha256": hashes}


def ragged_entries(
    offsets: np.ndarray, values: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batch row and value of every ragged entry belonging to ``indices``."""

    starts = offsets[indices]
    counts = offsets[indices + 1] - starts
    rows = np.repeat(np.arange(len(indices)), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return rows, np.asarray(values[np.repeat(starts, counts) + within])
