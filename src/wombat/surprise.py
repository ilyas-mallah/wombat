"""Per-position surprise of a trained model, and training draws weighted by it.

A game position's surprise is KL(target policy || model policy) over the legal moves,
with the training loss's target, plus the squared value error. A puzzle position's is
the cross-entropy of the solution move. Sampling follows Leela's diff focus and
KataGo's policy surprise weighting: part of each batch uniform, the rest by surprise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import torch

from wombat.data.game_corpus import GameSplit, densify
from wombat.data.puzzle_corpus import PuzzleSplit, lower_priority
from wombat.eval.puzzle_corpus_eval import load_checkpoint
from wombat.gpu_share import share_gpu
from wombat.model import WombatNetwork

SURPRISE_SCHEMA = "wombat.surprise.v1"
# Weights are capped at this quantile of their pool's surprise and floored at this share
# of the capped mean, so no position is left out of the weighted draws.
CAP_QUANTILE = 0.99
FLOOR_SHARE = 0.01
QUANTILES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)
# Arrays written per part, each aligned with that part's positions.
GAME_ARRAYS = ("games_policy_kl", "games_value_error")
PUZZLE_ARRAYS = ("puzzles_cross_entropy",)


def game_surprise(
    policy: torch.Tensor, value: torch.Tensor, targets: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(target || model) over legal moves and the squared value error, per position."""

    logits = policy.float().masked_fill(~targets["legal_mask"], float("-inf"))
    log_probs = torch.log_softmax(logits, dim=1).clamp_min(-1e4)
    target = targets["policy_target"]
    kl = torch.where(target > 0, target * (target.clamp_min(1e-30).log() - log_probs), 0.0)
    return kl.sum(dim=1), (value.float() - targets["value"]).square()


def puzzle_surprise(
    policy: torch.Tensor, legal: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy of the solution move over the legal moves, per position."""

    logits = policy.float().masked_fill(~legal, float("-inf"))
    log_probs = torch.log_softmax(logits, dim=1).clamp_min(-1e4)
    return -log_probs.gather(1, action[:, None]).squeeze(1)


def _readout(
    model: WombatNetwork, batch: dict[str, torch.Tensor], depth: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(batch["pieces"].to(device), batch["state"].to(device), depth)
    policy, value = cast(tuple[torch.Tensor, torch.Tensor], outputs)
    return policy, value


def _open_arrays(
    output: Path, sizes: dict[str, int], resume: bool
) -> dict[str, np.memmap[Any, Any]]:
    arrays: dict[str, np.memmap[Any, Any]] = {}
    for name, size in sizes.items():
        path = output / f"{name}.npy"
        if resume:
            arrays[name] = cast(Any, np.load(path, mmap_mode="r+"))
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(size,))
            array[:] = np.nan
            arrays[name] = array
    return arrays


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        **{f"q{quantile:g}": float(np.quantile(values, quantile)) for quantile in QUANTILES},
    }


def sampling_weights(surprise: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Surprise capped at its 99th percentile and floored above 0, with a summary."""

    if len(surprise) == 0 or not np.isfinite(surprise).all():
        raise ValueError("sampling weights need a finite surprise for every position")
    cap = float(np.quantile(surprise, CAP_QUANTILE))
    capped = np.minimum(surprise, cap)
    floor = max(FLOOR_SHARE * float(capped.mean()), 1e-12)
    weights = np.maximum(capped, floor).astype(np.float64)
    ordered = np.sort(weights)[::-1]
    total = float(ordered.sum())
    summary = {
        "positions": len(weights),
        "mean_surprise": float(surprise.mean()),
        "cap": cap,
        "floor": floor,
        "top_1_percent_weight": float(ordered[: max(1, len(ordered) // 100)].sum() / total),
        "top_10_percent_weight": float(ordered[: max(1, len(ordered) // 10)].sum() / total),
    }
    return weights, summary


def load_surprise(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Game corpus indices, game surprise, puzzle surprise and the manifest of a scored run."""

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SURPRISE_SCHEMA or not manifest.get("complete"):
        raise ValueError(f"{directory} holds no complete surprise scores")
    indices = np.load(directory / "games_indices.npy")
    games = np.load(directory / "games_policy_kl.npy") + np.load(
        directory / "games_value_error.npy"
    )
    puzzles = np.load(directory / "puzzles_cross_entropy.npy")
    return indices, games, puzzles, manifest


class SurpriseSampler:
    """Batch indices from ``pool``: a uniform part without replacement, the rest by weight.

    ``surprise`` is aligned with ``pool``; drawn surprise is totalled for the metrics.
    """

    def __init__(
        self,
        pool: np.ndarray,
        surprise: np.ndarray,
        share: float,
        batch_size: int,
        generator: np.random.Generator,
    ) -> None:
        if len(pool) != len(surprise):
            raise ValueError("surprise must be aligned with the pool")
        weights, self.summary = sampling_weights(surprise)
        self.cumulative = np.cumsum(weights)
        self.cumulative /= self.cumulative[-1]
        self.pool = pool
        self.surprise = surprise
        self.weighted = round(batch_size * share)
        self.uniform = batch_size - self.weighted
        self.generator = generator
        self.drawn = 0.0
        self.count = 0
        self._batches = self._draw()

    def _batch(self, uniform: np.ndarray) -> np.ndarray:
        drawn = self.generator.random(self.weighted)
        slots = np.concatenate((uniform, np.searchsorted(self.cumulative, drawn, side="right")))
        self.drawn += float(self.surprise[slots].sum())
        self.count += len(slots)
        return cast(np.ndarray, self.pool[slots])

    def _draw(self) -> Iterator[np.ndarray]:
        # The uniform part walks a fresh permutation per pass, as shuffled_batches does.
        if not self.uniform:
            while True:
                yield self._batch(np.zeros(0, dtype=np.int64))
        while True:
            order = self.generator.permutation(len(self.pool))
            for start in range(0, len(order) - self.uniform + 1, self.uniform):
                yield self._batch(order[start : start + self.uniform])

    def __iter__(self) -> SurpriseSampler:
        return self

    def __next__(self) -> np.ndarray:
        return next(self._batches)

    def drain_mean(self) -> float | None:
        """Mean surprise of the positions drawn since the last call."""

        mean = self.drawn / self.count if self.count else None
        self.drawn, self.count = 0.0, 0
        return mean


def _score_part(
    name: str,
    total: int,
    done: int,
    chunk_size: int,
    batch_size: int,
    score: Callable[[np.ndarray], dict[str, np.ndarray]],
    arrays: dict[str, np.memmap[Any, Any]],
    progress: Callable[[str, int], None],
) -> None:
    started = time.perf_counter()
    for chunk_start in range(done, total, chunk_size):
        chunk_stop = min(total, chunk_start + chunk_size)
        for start in range(chunk_start, chunk_stop, batch_size):
            rows = np.arange(start, min(chunk_stop, start + batch_size))
            for key, values in score(rows).items():
                arrays[key][rows] = values
        for key in arrays:
            arrays[key].flush()
        progress(name, chunk_stop)
        rate = (chunk_stop - done) / (time.perf_counter() - started)
        print(f"{name}: {chunk_stop:,} of {total:,} at {rate:,.0f} positions/s", flush=True)


def score_surprise(
    checkpoint: Path,
    labels: Path,
    output: Path,
    *,
    corpus: Path = Path("data/processed/games-v1"),
    puzzle_corpus: Path = Path("data/processed/puzzles-v1"),
    depth: int = 8,
    # 1,024 positions at 8 steps overflow a 1 GiB VRAM cap; 512 peaks near 0.76 GiB.
    batch_size: int = 512,
    chunk_size: int = 65_536,
    device_name: str = "cuda",
    max_vram_gb: float | None = 1.0,
) -> dict[str, object]:
    """Score every labelled training game position and every training puzzle position.

    Chunks are flushed as they finish, so a rerun with the same settings resumes.
    """

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    temperature = float(state.get("training_config", {}).get("temperature", 4.0))
    del state
    model = load_checkpoint(checkpoint, device)
    games = GameSplit(corpus / "train", temperature=temperature, labels=labels)
    puzzles = PuzzleSplit(puzzle_corpus / "train")
    indices = games.trainable()
    settings: dict[str, object] = {
        "schema": SURPRISE_SCHEMA,
        "checkpoint": str(checkpoint),
        "labels": str(labels),
        "games": str(corpus / "train"),
        "puzzles": str(puzzle_corpus / "train"),
        "depth": depth,
        "temperature": temperature,
        "game_positions": len(indices),
        "puzzle_positions": len(puzzles),
    }
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    done = {"games": 0, "puzzles": 0}
    seconds = 0.0
    resume = progress_path.exists()
    if resume:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved["settings"] != settings:
            raise ValueError(f"{output} was scored with other settings; use a new output")
        done, seconds = saved["done"], float(saved["seconds"])
    else:
        np.save(output / "games_indices.npy", indices)
    sizes = {**{key: len(indices) for key in GAME_ARRAYS}, "puzzles_cross_entropy": len(puzzles)}
    arrays = _open_arrays(output, sizes, resume)
    started = time.perf_counter()

    def record(part: str, count: int) -> None:
        done[part] = count
        elapsed = seconds + time.perf_counter() - started
        payload = {"settings": settings, "done": done, "seconds": elapsed}
        temporary = progress_path.with_suffix(".partial")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(progress_path)

    @torch.inference_mode()
    def score_games(rows: np.ndarray) -> dict[str, np.ndarray]:
        batch = densify(games.sparse_batch(indices[rows]), temperature, device)
        policy, value = _readout(model, batch, depth, device)
        kl, error = game_surprise(policy, value, batch)
        return {"games_policy_kl": kl.cpu().numpy(), "games_value_error": error.cpu().numpy()}

    @torch.inference_mode()
    def score_puzzles(rows: np.ndarray) -> dict[str, np.ndarray]:
        batch = puzzles.batch(rows)
        policy, _ = _readout(model, batch, depth, device)
        legal, action = batch["legal_mask"].to(device), batch["action"].to(device)
        return {"puzzles_cross_entropy": puzzle_surprise(policy, legal, action).cpu().numpy()}

    game_arrays = {key: arrays[key] for key in GAME_ARRAYS}
    puzzle_arrays = {key: arrays[key] for key in PUZZLE_ARRAYS}
    _score_part(
        "games",
        len(indices),
        done["games"],
        chunk_size,
        batch_size,
        score_games,
        game_arrays,
        record,
    )
    _score_part(
        "puzzles",
        len(puzzles),
        done["puzzles"],
        chunk_size,
        batch_size,
        score_puzzles,
        puzzle_arrays,
        record,
    )
    record("puzzles", len(puzzles))
    kl, error = (np.asarray(arrays[key]) for key in GAME_ARRAYS)
    cross_entropy = np.asarray(arrays["puzzles_cross_entropy"])
    remaining = np.asarray(puzzles.remaining)
    manifest: dict[str, object] = {
        **settings,
        "complete": True,
        "seconds": json.loads(progress_path.read_text(encoding="utf-8"))["seconds"],
        "games_policy_kl": _summary(kl),
        "games_value_error": _summary(error),
        "games_surprise": _summary(kl + error),
        "games_weights": sampling_weights(kl + error)[1],
        "puzzles_cross_entropy": _summary(cross_entropy),
        "puzzles_weights": sampling_weights(cross_entropy)[1],
        "puzzles_weights_by_remaining": {
            str(moves): sampling_weights(cross_entropy[remaining == moves])[1]
            for moves in np.unique(remaining).tolist()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
