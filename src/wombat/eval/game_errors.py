"""Move errors on held-out real-game positions whose every legal move Stockfish scored.

The set is ChessBench's test games as ``import-chessbench`` converts them: each legal move
carries Stockfish 16's win chance for the mover. A chosen move loses the best move's win
chance minus its own. Average accuracy barely separates players a thousand ladder Elo apart;
the rate of moves losing 20 or 40 points on live positions does.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import math
from pathlib import Path
import shutil
import time

import chess
import numpy as np
import torch

from wombat.chess.encode import decode_position, play_encoding, state_vector
from wombat.chess.moves import ACTION_SIZE, move_to_action
from wombat.data.columns import file_sha256, ragged_entries
from wombat.data.puzzle_corpus import lower_priority
from wombat.eval.puzzle_corpus_eval import paired_difference, wilson_interval

ERROR_SET_SCHEMA = "wombat.game-error-set.v1"
REPORT_SCHEMA = "wombat.game-errors.v1"
THRESHOLDS = (0.05, 0.10, 0.20, 0.40)
# Live positions: the best move's win chance lies in [0.10, 0.90].
LIVE = (0.10, 0.90)
# Knights, bishops, rooks and queens of both sides, kings and pawns left out.
PHASES = {"0-4": (0, 4), "5-8": (5, 8), "9-12": (9, 12), "13+": (13, 64)}
# Best win chance buckets, each [low, high) except the live edges, which stay live.
BALANCE = {
    "lost": (-1.0, LIVE[0]),
    "worse": (LIVE[0], 0.30),
    "level": (0.30, 0.70),
    "better": (0.70, LIVE[1]),
    "won": (LIVE[1], 2.0),
}
EXTERNAL_LABEL = "play"
_POSITION_COLUMNS = (
    "pieces",
    "state",
    "legal_counts",
    "legal_offsets",
    "legal_actions",
)
_LABEL_COLUMNS = ("label_offsets", "label_actions", "label_win")
_NON_PAWN = np.array([2, 3, 4, 5, 8, 9, 10, 11], dtype=np.uint8)
_MIX = np.uint64(0x9E3779B97F4A7C15)


def move_losses(
    offsets: np.ndarray,
    actions: np.ndarray,
    win: np.ndarray,
    rows: np.ndarray,
    chosen: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Win chance each chosen action gives up against the row's best labelled move, and that best.

    ``offsets``, ``actions`` and ``win`` are the ragged label columns; ``chosen`` holds one
    action per entry of ``rows``, and each must be one of that row's labelled moves.
    """

    rows = np.asarray(rows, dtype=np.int64)
    batch, labelled = ragged_entries(offsets, actions, rows)
    _, wins = ragged_entries(offsets, win, rows)
    wins = wins.astype(np.float64)
    best = np.full(len(rows), -np.inf)
    np.maximum.at(best, batch, wins)
    taken = np.full(len(rows), np.nan)
    hit = labelled.astype(np.int64) == np.asarray(chosen, dtype=np.int64)[batch]
    taken[batch[hit]] = wins[hit]
    missing = np.flatnonzero(np.isnan(taken))
    if len(missing):
        raise ValueError(f"row {int(rows[missing[0]])} has no label for the chosen action")
    return best - taken, best


def lichess_accuracy(loss: np.ndarray) -> np.ndarray:
    """Lichess per-move accuracy from win chance lost (0..1), as ``strength.move_accuracy``."""

    points = np.asarray(loss, dtype=np.float64) * 100
    return np.clip(103.1668 * np.exp(-0.04354 * points) - 3.1669, 0.0, 100.0)


def non_pawn_pieces(pieces: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(pieces), _NON_PAWN).sum(axis=1)


def live_mask(best: np.ndarray) -> np.ndarray:
    return (best >= LIVE[0]) & (best <= LIVE[1])


def balance_buckets(best: np.ndarray) -> dict[str, np.ndarray]:
    buckets = {name: (best >= low) & (best < high) for name, (low, high) in BALANCE.items()}
    buckets["better"] |= best == LIVE[1]
    buckets["won"] &= best != LIVE[1]
    return buckets


def phase_buckets(counts: np.ndarray) -> dict[str, np.ndarray]:
    return {name: (counts >= low) & (counts <= high) for name, (low, high) in PHASES.items()}


def _mean(values: np.ndarray) -> dict[str, float]:
    """Mean with a normal 95% interval."""

    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"mean": math.nan, "low": math.nan, "high": math.nan}
    mean = float(values.mean())
    spread = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0
    return {"mean": mean, "low": mean - spread, "high": mean + spread}


def rate_name(threshold: float) -> str:
    return f"over_{round(threshold * 100)}"


def error_summary(loss: np.ndarray) -> dict[str, object]:
    """Mean loss, implied Lichess accuracy and the rate of losses above each threshold."""

    loss = np.asarray(loss, dtype=np.float64)
    summary: dict[str, object] = {
        "positions": len(loss),
        "mean_loss": _mean(loss),
        "lichess_accuracy": _mean(lichess_accuracy(loss)),
    }
    for threshold in THRESHOLDS:
        count = int((loss > threshold).sum())
        low, high = wilson_interval(count, len(loss))
        summary[rate_name(threshold)] = {
            "rate": count / max(1, len(loss)),
            "low": low,
            "high": high,
            "count": count,
        }
    return summary


def depth_report(loss: np.ndarray, best: np.ndarray, phase: np.ndarray) -> dict[str, object]:
    """Live and overall summaries, live positions by phase, all positions by balance."""

    live = live_mask(best)
    return {
        "live": error_summary(loss[live]),
        "all": error_summary(loss),
        "phase_live": {
            name: error_summary(loss[live & members])
            for name, members in phase_buckets(phase).items()
        },
        "balance": {
            name: error_summary(loss[members]) for name, members in balance_buckets(best).items()
        },
    }


def paired_errors(
    first: np.ndarray, second: np.ndarray, best: np.ndarray
) -> dict[str, dict[str, object]]:
    """``second`` minus ``first`` on the same positions, live and overall, with 95% intervals.

    Rate differences use ``paired_difference`` on the error indicators, so a negative
    difference means ``second`` makes fewer such errors.
    """

    if first.shape != second.shape or first.shape != best.shape:
        raise ValueError("paired comparisons need identical positions")
    report: dict[str, dict[str, object]] = {}
    for name, members in (
        ("live", live_mask(best)),
        ("all", np.ones(len(best), np.bool_)),
    ):
        a, b = first[members], second[members]
        if not len(a):
            report[name] = {"positions": 0}
            continue
        block: dict[str, object] = {
            "positions": int(members.sum()),
            "mean_loss": _mean(b - a),
            "lichess_accuracy": _mean(lichess_accuracy(b) - lichess_accuracy(a)),
        }
        for threshold in THRESHOLDS:
            block[rate_name(threshold)] = {
                "first": float((a > threshold).mean()),
                "second": float((b > threshold).mean()),
                **paired_difference(a > threshold, b > threshold),
            }
        report[name] = block
    return report


class ErrorSet:
    """A prepared error set: positions, every move's label, FENs and the rows left out."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        positions, labels = directory / "positions", directory / "labels"

        def column(folder: Path, name: str) -> np.ndarray:
            return np.load(folder / f"{name}.npy", mmap_mode="r")

        self.pieces = column(positions, "pieces")
        self.state = column(positions, "state")
        self.legal_offsets = column(positions, "legal_offsets")
        self.legal_actions = column(positions, "legal_actions")
        self.fens = np.load(positions / "fens.npy")
        self.label_offsets = column(labels, "label_offsets")
        self.label_actions = column(labels, "label_actions")
        self.label_win = column(labels, "label_win")
        self.overlap = np.load(directory / "overlap.npy")
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.pieces)

    def rows(self) -> np.ndarray:
        """Every row not in any training set."""

        return np.setdiff1d(np.arange(len(self)), self.overlap)

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        rows, actions = ragged_entries(self.legal_offsets, self.legal_actions, indices)
        mask = np.zeros((len(indices), ACTION_SIZE), dtype=np.bool_)
        mask[rows, actions.astype(np.int64)] = True
        return {
            "pieces": torch.from_numpy(np.asarray(self.pieces[indices], dtype=np.int64)),
            "state": torch.from_numpy(np.asarray(self.state[indices], dtype=np.float32)),
            "legal_mask": torch.from_numpy(mask),
        }

    def losses(self, rows: np.ndarray, chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return move_losses(self.label_offsets, self.label_actions, self.label_win, rows, chosen)


def board_hashes(pieces: np.ndarray, state: np.ndarray) -> np.ndarray:
    """64-bit hash of the board, side to move and castling rights of encoded positions."""

    words = np.ascontiguousarray(pieces, dtype=np.uint8).view("<u8")
    flags = np.asarray(state[:, 1:6], dtype=np.uint64) << np.arange(5, dtype=np.uint64)
    hashes = flags.sum(axis=1, dtype=np.uint64) * _MIX
    for column in range(words.shape[1]):
        hashes ^= words[:, column]
        hashes *= _MIX
        hashes ^= hashes >> np.uint64(31)
    return hashes


def _position_fields(pieces: np.ndarray, state: np.ndarray) -> str:
    """Board, side, castling and legal en passant of an encoded position."""

    return " ".join(decode_position(pieces, state).fen(en_passant="legal").split()[:4])


def training_overlap(
    pieces: np.ndarray, state: np.ndarray, split: Path, *, chunk: int = 1 << 21
) -> np.ndarray:
    """Sorted rows of (``pieces``, ``state``) whose position is in the training ``split``.

    Positions match on board, side to move, castling and legal en passant; the halfmove
    bucket is ignored.
    """

    hashes = board_hashes(pieces, state)
    order = np.argsort(hashes, kind="stable")
    ordered = hashes[order]
    other_pieces = np.load(split / "pieces.npy", mmap_mode="r")
    other_state = np.load(split / "state.npy", mmap_mode="r")
    found: set[int] = set()
    for start in range(0, len(other_pieces), chunk):
        block_pieces = np.asarray(other_pieces[start : start + chunk])
        block_state = np.asarray(other_state[start : start + chunk])
        block = board_hashes(block_pieces, block_state)
        slots = np.minimum(np.searchsorted(ordered, block), len(ordered) - 1)
        for index in np.flatnonzero(ordered[slots] == block).tolist():
            fields = _position_fields(block_pieces[index], block_state[index])
            first = int(slots[index])
            while first < len(ordered) and ordered[first] == block[index]:
                row = int(order[first])
                if row not in found and _position_fields(pieces[row], state[row]) == fields:
                    found.add(row)
                first += 1
    return np.array(sorted(found), dtype=np.int64)


def _encoding_key(board: chess.Board) -> bytes:
    encoded = play_encoding(board)
    return encoded.pieces.tobytes() + state_vector(encoded).astype(np.uint8).tobytes()


def recover_fens(source: Path, pieces: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, int]:
    """Each row's full FEN from the import's FEN store, and how many rows had several.

    Positions repeated with other move counters keep the FEN with the smallest FEN key,
    the one the join kept.
    """

    chosen: dict[bytes, tuple[int, str]] = {}
    repeats: dict[bytes, int] = {}
    for path in sorted((source / "fens").glob("*.npz")):
        with np.load(path) as stored:
            pairs = zip(stored["keys"].tolist(), stored["fens"].tolist(), strict=True)
            for key, fen in pairs:
                text = fen.decode()
                code = _encoding_key(chess.Board(text))
                repeats[code] = repeats.get(code, 0) + 1
                if code not in chosen or key < chosen[code][0]:
                    chosen[code] = (key, text)
    fens, several = [], 0
    for row in range(len(pieces)):
        code = np.asarray(pieces[row], dtype=np.uint8).tobytes()
        code += np.asarray(state[row], dtype=np.uint8).tobytes()
        if code not in chosen:
            raise ValueError(f"row {row} has no FEN in {source / 'fens'}")
        fens.append(chosen[code][1])
        several += repeats[code] > 1
    return np.array(fens, dtype=np.str_), several


def prepare_error_set(
    source: Path, output: Path, training: Sequence[Path], *, low_priority: bool = True
) -> dict[str, object]:
    """Copy an ``import-chessbench`` corpus to ``output`` as a fixed error set.

    Adds each position's FEN and the rows whose position appears in any ``training`` split
    (``overlap.npy``), which the evaluation leaves out.
    """

    if low_priority:
        lower_priority()
    started = time.perf_counter()
    positions, labels = output / "positions", output / "labels"
    for folder in (positions, labels):
        folder.mkdir(parents=True, exist_ok=True)
    for name in _POSITION_COLUMNS:
        shutil.copyfile(source / "train" / f"{name}.npy", positions / f"{name}.npy")
    for name in _LABEL_COLUMNS:
        shutil.copyfile(source / "train-labels" / f"{name}.npy", labels / f"{name}.npy")
    imported = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    shutil.copyfile(source / "manifest.json", output / "import-manifest.json")
    pieces = np.load(positions / "pieces.npy")
    state = np.load(positions / "state.npy")
    fens, several = recover_fens(source, pieces, state)
    np.save(positions / "fens.npy", fens, allow_pickle=False)
    overlap: dict[str, int] = {}
    rows = np.zeros(0, dtype=np.int64)
    for split in training:
        found = training_overlap(pieces, state, split)
        overlap[split.as_posix()] = len(found)
        rows = np.union1d(rows, found)
    np.save(output / "overlap.npy", rows.astype(np.int64), allow_pickle=False)
    offsets = np.load(labels / "label_offsets.npy")
    win = np.load(labels / "label_win.npy").astype(np.float64)
    best = np.maximum.reduceat(win, offsets[:-1])
    kept = np.setdiff1d(np.arange(len(pieces)), rows)
    manifest: dict[str, object] = {
        "schema": ERROR_SET_SCHEMA,
        "source": source.as_posix(),
        "import_sha256": imported["splits"]["train"]["sha256"],
        "positions": len(pieces),
        "moves": int(offsets[-1]),
        "fens_with_other_counters": several,
        "training_overlap": {"splits": overlap, "positions": len(rows)},
        "evaluated": {"positions": len(kept), "live": int(live_mask(best[kept]).sum())},
        "sha256": {
            f"{folder.name}/{path.name}": file_sha256(path)
            for folder in (positions, labels)
            for path in sorted(folder.glob("*.npy"))
        }
        | {"overlap.npy": file_sha256(output / "overlap.npy")},
        "seconds": round(time.perf_counter() - started, 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _external_choices(
    error_set: ErrorSet,
    rows: np.ndarray,
    external: str,
    weights: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Actions an external player picks by its own rule, with no game history."""

    from wombat.external.chessbench import EXTERNAL_PLAYERS

    player = EXTERNAL_PLAYERS[external](weights, device)
    chosen = np.zeros(len(rows), dtype=np.int64)
    step = 4096
    for start in range(0, len(rows), step):
        boards = [chess.Board(str(error_set.fens[row])) for row in rows[start : start + step]]
        moves = player.best_moves(boards, batch_size)
        chosen[start : start + len(boards)] = [
            move_to_action(board, move) for board, move in zip(boards, moves, strict=True)
        ]
    return chosen


def evaluate_game_errors(
    error_set_path: Path,
    output: Path,
    *,
    checkpoint: Path | None = None,
    depths: Iterable[int] = (4, 8, 16),
    external: str | None = None,
    external_weights: Path | None = None,
    batch_size: int = 256,
    device_name: str = "cuda",
    max_vram_gb: float | None = 3.0,
    limit: int | None = None,
) -> dict[str, object]:
    """Score a checkpoint's argmax policy move at each depth, or an external player's move.

    Writes ``report.json`` and ``losses.npz`` (rows, best, and a loss and action column per
    depth, ``t<T>``, or ``play`` for an external player) for ``compare-game-errors``.
    """

    from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
    from wombat.gpu_share import share_gpu

    if (checkpoint is None) == (external is None):
        raise ValueError("give exactly one of a checkpoint and an external player")
    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    error_set = ErrorSet(error_set_path)
    rows = error_set.rows()
    if limit is not None:
        rows = rows[:limit]
    started = time.perf_counter()
    choices: dict[str, np.ndarray] = {}
    if external is not None:
        if external_weights is None:
            raise ValueError("an external player needs its weights")
        choices[EXTERNAL_LABEL] = _external_choices(
            error_set, rows, external, external_weights, device, batch_size
        )
    else:
        assert checkpoint is not None
        depths = tuple(sorted(set(depths)))
        model = load_checkpoint(checkpoint, device)
        recurrent = model.config.recurrent_blocks > 0
        predicted = predict_depths(
            model,
            error_set,
            rows,
            depths if recurrent else (0,),
            batch_size=batch_size,
            device=device,
        )
        for depth in depths:
            choices[f"t{depth}"] = predicted[depth if recurrent else 0]
    seconds = time.perf_counter() - started
    phase = non_pawn_pieces(np.asarray(error_set.pieces[rows]))
    arrays: dict[str, np.ndarray] = {"rows": rows}
    results: dict[str, object] = {}
    for label, chosen in choices.items():
        loss, best = error_set.losses(rows, chosen)
        arrays["best"] = best
        arrays[f"loss_{label}"] = loss.astype(np.float32)
        arrays[f"action_{label}"] = chosen.astype(np.int16)
        results[label] = depth_report(loss, best, phase)
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "player": str(checkpoint) if checkpoint is not None else external,
        "error_set": error_set_path.as_posix(),
        "error_set_sha256": error_set.manifest["sha256"],
        "positions": len(rows),
        "live": int(live_mask(arrays["best"]).sum()),
        "seconds": round(seconds, 1),
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "losses.npz", **arrays)  # type: ignore[arg-type]
    return report


def _losses_path(path: Path) -> Path:
    return path / "losses.npz" if path.is_dir() else path


def _loss_column(stored: np.lib.npyio.NpzFile, depth: int | None, path: Path) -> np.ndarray:
    """The depth's loss column, or the external player's single one."""

    if f"loss_{EXTERNAL_LABEL}" in stored.files:
        return np.asarray(stored[f"loss_{EXTERNAL_LABEL}"], dtype=np.float64)
    if depth is None or f"loss_t{depth}" not in stored.files:
        available = sorted(name[5:] for name in stored.files if name.startswith("loss_"))
        raise ValueError(f"{path} has no depth {depth}; it holds {available}")
    return np.asarray(stored[f"loss_t{depth}"], dtype=np.float64)


def compare_game_errors(
    first: Path, second: Path, depth: int | None, second_depth: int | None = None
) -> dict[str, object]:
    """Paired error differences, second minus first, between two stored evaluations."""

    second_depth = depth if second_depth is None else second_depth
    first_path, second_path = _losses_path(first), _losses_path(second)
    with np.load(first_path) as left, np.load(second_path) as right:
        if not np.array_equal(left["rows"], right["rows"]):
            raise ValueError("evaluations cover different positions")
        best = np.asarray(left["best"], dtype=np.float64)
        a = _loss_column(left, depth, first_path)
        b = _loss_column(right, second_depth, second_path)
    return {
        "first": first_path.as_posix(),
        "first_depth": depth,
        "second": second_path.as_posix(),
        "second_depth": second_depth,
        **paired_errors(a, b, best),
    }
