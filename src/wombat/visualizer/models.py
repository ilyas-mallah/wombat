"""Checkpoint discovery, a small CPU model cache, and per-position model analysis."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import pickle
import threading
from typing import Any

import chess
import numpy as np
import torch

from wombat.chess.encode import play_encoding, state_vector
from wombat.chess.moves import legal_actions
from wombat.eval.agreement import DRAW_GUARD_VALUE, guard_draws
from wombat.interpretability.explain import explain_move
from wombat.model import NetworkConfig, WombatNetwork

STRIP_DEPTHS = (1, 2, 4, 8, 16)
EXPLAIN_DEPTHS = (0, 1, 2, 4, 8, 16)
EXPLANATIONS = 64
MAX_STEPS = 32
CPU = torch.device("cpu")
PATTERNS = ("checkpoint*.pt", "snapshot-step-*.pt")
# Lichess's 0.00368208 per centipawn, expressed per pawn.
PAWN_SCALE = 0.368208
# The puzzle trainer's defaults, for configs written before these fields existed.
PUZZLE_TRAINER_DEFAULTS = {"task": "puzzles", "value_weight": 0.0, "move_value_weight": 0.0}
_CONFIG_FIELDS = frozenset(item.name for item in fields(NetworkConfig))


@dataclass(frozen=True)
class CheckpointInfo:
    id: str
    run: str
    name: str
    kind: str
    step: int | None
    readout: str
    parameters: int
    value_trained: bool
    move_values_trained: bool
    modified: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def value_to_pawns(value: float) -> float:
    """Pawns for a ``2W - 1`` value by inverting the Lichess win-chance curve."""

    win = min(0.999, max(0.001, (value + 1) / 2))
    return math.log(win / (1 - win)) / PAWN_SCALE


def training_config(raw: dict[str, Any], path: Path) -> dict[str, Any] | None:
    """The checkpoint's training config, else its run's ``config.json`` (snapshots)."""

    config = raw.get("training_config")
    if isinstance(config, dict):
        return config
    try:
        payload = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    config = payload.get("training_config") if isinstance(payload, dict) else None
    return config if isinstance(config, dict) else None


def trained_heads(training: dict[str, Any] | None, weights: Iterable[str]) -> tuple[bool, bool]:
    """Whether the value head and the per-move value head ever received a loss."""

    if training is None:
        return False, False
    if "corpus" in training:
        training = {**PUZZLE_TRAINER_DEFAULTS, **training}
    if training.get("task") == "puzzles":
        return False, False
    names = list(weights)
    value = bool(training.get("value_weight")) and any(
        name.startswith("value_head.") for name in names
    )
    move_values = bool(training.get("move_value_weight")) and any(
        name.startswith(("move_value_head.", "square_move_value_head.")) for name in names
    )
    return value, move_values


def _load_raw(path: Path) -> dict[str, Any]:
    # Memory mapping reads only the tensors that are touched; old formats cannot map.
    try:
        payload = torch.load(path, map_location=CPU, weights_only=False, mmap=True)
    except RuntimeError:
        payload = torch.load(path, map_location=CPU, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
        raise ValueError(f"{path} is not a Wombat checkpoint")
    return payload


def network_config(raw: dict[str, Any]) -> NetworkConfig:
    """Build a config, ignoring fields this version of the network no longer knows."""

    return NetworkConfig(**{k: v for k, v in raw.items() if k in _CONFIG_FIELDS})


class CheckpointIndex:
    """Lists checkpoints and step snapshots under a root, cached by file size and mtime."""

    def __init__(self, root: Path, base: Path | None = None) -> None:
        self.root = root
        self.base = base if base is not None else root.parent
        self._cache: dict[Path, tuple[tuple[float, int], CheckpointInfo | None]] = {}
        self._paths: dict[str, Path] = {}
        self._infos: dict[str, CheckpointInfo] = {}
        self._lock = threading.Lock()

    def _identifier(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.base.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _describe(self, path: Path) -> CheckpointInfo | None:
        try:
            raw = _load_raw(path)
        except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError):
            return None
        config = raw["model_config"]
        step = raw.get("step")
        run = path.parent.relative_to(self.root).as_posix() if path.parent != self.root else "."
        value, move_values = trained_heads(training_config(raw, path), raw["model"].keys())
        return CheckpointInfo(
            id=self._identifier(path),
            run=run,
            name=path.stem,
            kind="snapshot" if path.name.startswith("snapshot-") else "checkpoint",
            step=int(step) if isinstance(step, int) else None,
            readout=str(config.get("policy_readout", "global")),
            parameters=int(sum(tensor.numel() for tensor in raw["model"].values())),
            value_trained=value,
            move_values_trained=move_values,
            modified=path.stat().st_mtime,
        )

    def list(self) -> list[CheckpointInfo]:
        """Runs newest first; within a run the final checkpoint, then later steps first."""

        with self._lock:
            found: list[CheckpointInfo] = []
            paths = sorted(
                {path for pattern in PATTERNS for path in self.root.glob(f"**/{pattern}")}
            )
            for path in paths:
                stat = path.stat()
                key = (stat.st_mtime, stat.st_size)
                cached = self._cache.get(path)
                if cached is None or cached[0] != key:
                    cached = (key, self._describe(path))
                    self._cache[path] = cached
                if cached[1] is not None:
                    found.append(cached[1])
                    self._paths[cached[1].id] = path
                    self._infos[cached[1].id] = cached[1]
            newest: dict[str, float] = {}
            for info in found:
                newest[info.run] = max(newest.get(info.run, 0.0), info.modified)
            found.sort(
                key=lambda info: (
                    -newest[info.run],
                    info.run,
                    info.name != "checkpoint",
                    -(info.step if info.step is not None else -1),
                    -info.modified,
                    info.name,
                )
            )
            return found

    def info(self, identifier: str) -> CheckpointInfo:
        """Metadata for a discovered checkpoint; only discovered files are ever unpickled."""

        if identifier not in self._infos:
            self.list()
        if identifier not in self._infos:
            raise KeyError(f"unknown checkpoint: {identifier}")
        return self._infos[identifier]

    def path(self, identifier: str) -> Path:
        self.info(identifier)
        return self._paths[identifier]

    def default(self, preferred: Iterable[str] = ()) -> str | None:
        listed = self.list()
        ids = {info.id for info in listed}
        for item in preferred:
            if item in ids:
                return item
        return listed[0].id if listed else None


def load_network(path: Path) -> WombatNetwork:
    """Load weights on CPU; ``strict=False`` accepts both global and square readouts."""

    raw = _load_raw(path)
    model = WombatNetwork(network_config(raw["model_config"]))
    model.load_state_dict(raw["model"], strict=False)
    return model.to(CPU).eval()


class ModelCache:
    """Keeps recent networks and analyses in memory and serializes inference."""

    def __init__(self, index: CheckpointIndex, capacity: int = 3, results: int = 512) -> None:
        self.index = index
        self.capacity = capacity
        self.results = results
        self._models: OrderedDict[tuple[str, float], WombatNetwork] = OrderedDict()
        self._analyses: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._explanations: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, identifier: str) -> WombatNetwork:
        # Keying on mtime reloads a checkpoint that was overwritten while serving.
        key = (identifier, self.index.info(identifier).modified)
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]
            model = load_network(self.index.path(identifier))
            self._models[key] = model
            while len(self._models) > self.capacity:
                self._models.popitem(last=False)
            return model

    def explain(
        self,
        identifier: str,
        board: chess.Board,
        move: chess.Move,
        compare: chess.Move | None,
        steps: int,
    ) -> dict[str, Any]:
        """Why the checkpoint would play ``move``, with values also in pawns for White."""

        if not 0 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be in [0, {MAX_STEPS}]")
        info = self.index.info(identifier)
        key = (identifier, info.modified, board.fen(), move.uci(), str(compare), steps)
        with self._lock:
            if key in self._explanations:
                self._explanations.move_to_end(key)
                return copy.deepcopy(self._explanations[key])
        model = self.get(identifier)
        with self._lock:
            result = explain_move(
                model,
                board,
                move,
                compare=compare,
                steps=steps,
                depths=EXPLAIN_DEPTHS,
                zero_halfmove=True,
            )
            result.update(_evaluation(result["value"], board.turn))
            sign = 1 if board.turn == chess.WHITE else -1
            for item in result["moves"]:
                item["pawns_white"] = sign * value_to_pawns(item["value"])
            result["value_trained"] = info.value_trained
            result["move_values_trained"] = info.move_values_trained
            self._explanations[key] = result
            while len(self._explanations) > EXPLANATIONS:
                self._explanations.popitem(last=False)
            return copy.deepcopy(result)

    def analyse(self, identifier: str, board: chess.Board, steps: int) -> dict[str, Any]:
        """A private copy of the (cached) analysis, tagged with which heads are trained.

        With a trained value head the move is draw guarded, so the key includes the game.
        """

        info = self.index.info(identifier)
        history = (board.root().fen(), tuple(move.uci() for move in board.move_stack))
        key = (identifier, info.modified, board.fen(), history, steps)
        with self._lock:
            if key in self._analyses:
                self._analyses.move_to_end(key)
                return copy.deepcopy(self._analyses[key])
        model = self.get(identifier)
        draw_guard = DRAW_GUARD_VALUE if info.value_trained else None
        with self._lock:
            result = analyse(model, board, steps, draw_guard=draw_guard)
            result["value_trained"] = info.value_trained
            result["move_values_trained"] = info.move_values_trained
            self._analyses[key] = result
            while len(self._analyses) > self.results:
                self._analyses.popitem(last=False)
            return copy.deepcopy(result)


def _candidates(
    board: chess.Board,
    moves: list[chess.Move],
    actions: np.ndarray,
    logits: torch.Tensor,
    move_values: torch.Tensor,
) -> list[dict[str, Any]]:
    index = torch.from_numpy(actions)
    probabilities = torch.softmax(logits.float()[index], dim=0).numpy()
    values = move_values.float()[index].numpy()
    sign = 1 if board.turn == chess.WHITE else -1
    order = np.argsort(-probabilities, kind="stable")
    return [
        {
            "uci": moves[i].uci(),
            "san": board.san(moves[i]),
            "probability": float(probabilities[i]),
            "move_value": float(values[i]),
            "pawns_white": sign * value_to_pawns(float(values[i])),
        }
        for i in order
    ]


def _evaluation(value: float, turn: chess.Color) -> dict[str, float]:
    sign = 1 if turn == chess.WHITE else -1
    win = 100 * (value + 1) / 2
    return {
        "value": value,
        "pawns_white": sign * value_to_pawns(value),
        "win_white": win if turn == chess.WHITE else 100 - win,
    }


@torch.inference_mode()
def analyse(
    model: WombatNetwork,
    board: chess.Board,
    steps: int,
    *,
    top: int = 5,
    strip: tuple[int, ...] = STRIP_DEPTHS,
    zero_halfmove: bool = True,
    draw_guard: float | None = None,
) -> dict[str, Any]:
    """The move chosen after ``steps`` reasoning steps, its top candidates and evaluation.

    Values are ``2W - 1`` for the side to move (or the mover, per move); pawns and
    win percentages are from White's side. One pass also reads every ``strip`` depth.
    ``zero_halfmove`` encodes the board as ``play_encoding`` does for play, and ``draw_guard``
    is ``guard_draws``'s threshold for the chosen move.
    """

    if not 0 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [0, {MAX_STEPS}]")
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("the position has no legal moves")
    actions = np.asarray(legal_actions(board), dtype=np.int64)
    encoded = play_encoding(board, zero_halfmove=zero_halfmove)
    pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
    state = torch.from_numpy(state_vector(encoded))[None]
    depths = frozenset((*strip, steps))
    outputs = model(pieces, state, max(depths), intermediate_steps=depths, include_move_values=True)
    assert isinstance(outputs, dict)

    def readout(depth: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        policy, value, move_values = outputs[depth]
        ranked = _candidates(board, moves, actions, policy[0], move_values[0])
        return ranked, _evaluation(float(value[0]), board.turn)

    ranked, evaluation = readout(steps)
    by_uci = {move.uci(): move for move in moves}
    order = [by_uci[item["uci"]] for item in ranked]
    move = guard_draws(board, order, evaluation["value"], draw_guard)
    thinking = []
    for depth in strip:
        best, depth_evaluation = readout(depth)
        thinking.append({"steps": depth, **best[0], **depth_evaluation})
    return {
        "steps": steps,
        "move": move.uci(),
        "san": board.san(move),
        "avoided_draw": move != order[0],
        **evaluation,
        "candidates": ranked[:top],
        "strip": thinking,
    }
