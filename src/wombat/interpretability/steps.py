"""What the recurrent state stores at each step, and which parts of it make more steps help.

Linear probes read chess targets from the 64 square tokens of the input, of H0 and of H after
each probed step. Puzzles the model gets right only at the late depth, or only at the early one,
are compared with those it always gets right: how much each square's token changes between the
two depths, and which targets become decodable. Causal tests swap or freeze square tokens between
the two depths or remove the board features, and blunder attribution asks whether a probe knew
that a blundered move's destination loses material. Everything runs in fp32 without autocast.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any, cast

import chess
import numpy as np
import torch

from wombat.chess.encode import decode_position
from wombat.chess.moves import PLANES_PER_SQUARE, action_to_move
from wombat.data.game_corpus import IGNORE, GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit, lower_priority
from wombat.eval.puzzle_corpus_eval import (
    REMAINING_GROUPS,
    load_checkpoint,
    remaining_groups,
    sample_groups,
)
from wombat.eval.strength import BLUNDER_FROM
from wombat.gpu_share import share_gpu
from wombat.interpretability.labels import move_squares, static_exchange, step_labels
from wombat.interpretability.probes import (
    auroc,
    cross_fit_binary,
    cross_fit_square,
    folds,
    prior_accuracy,
    square_accuracy,
)
from wombat.model.network import (
    WombatNetwork,
    _move_destinations,
    board_consequences,
    board_phase,
)

DEFAULT_DEPTHS = (1, 2, 4, 8, 16)
BINARY_TARGETS = (
    "destination_loses_material",
    "opponent_free_capture",
    "own_mate_in_one",
    "opponent_mate_in_one",
)
SQUARE_TARGETS = ("best_move_origin", "best_move_destination", "reply_origin", "reply_destination")
_LABEL_FIELDS = {
    "destination_loses_material": ("destination_loses", "destination_mask"),
    "opponent_free_capture": ("free_capture", "free_capture_mask"),
    "own_mate_in_one": ("own_mate", "own_mate_mask"),
    "opponent_mate_in_one": ("opponent_mate", "opponent_mate_mask"),
}
# Destination square of every (origin, plane) action in the side-to-move frame.
DESTINATIONS = _move_destinations().numpy()


def action_squares(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Origin and destination squares of canonical actions."""

    actions = np.asarray(actions, dtype=np.int64)
    origin = actions // PLANES_PER_SQUARE
    return origin, DESTINATIONS[origin, actions % PLANES_PER_SQUARE]


def king_distance(squares: np.ndarray) -> np.ndarray:
    """(..., 64) king steps from each given square to every square."""

    squares = np.asarray(squares)[..., None]
    every = np.arange(64)
    return np.maximum(abs(squares % 8 - every % 8), abs(squares // 8 - every // 8))


@dataclass
class Sample:
    """Positions probed together, with their boards and each one's reference move.

    ``target`` is Stockfish's best move for game positions and the solution for puzzles;
    ``reply`` is the opponent's expected reply as (origin, destination), -1 where unknown.
    ``wins`` holds scored win chances per action (NaN where unscored) for move losses;
    ``lost`` replaces them with judged losses at ``lost_depth`` from a strength report.
    """

    name: str
    source: str
    indices: np.ndarray
    boards: list[chess.Board]
    pieces: np.ndarray
    state: np.ndarray
    legal: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    reply: np.ndarray
    puzzles: bool = False
    remaining: np.ndarray | None = None
    wins: np.ndarray | None = None
    lost: np.ndarray | None = None
    lost_depth: int | None = None
    binary: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    squares: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)

    def label(self) -> None:
        """Fill the binary and square probe targets."""

        labels = [step_labels(board) for board in self.boards]
        for name, (values, mask) in _LABEL_FIELDS.items():
            self.binary[name] = (
                np.stack([getattr(item, values) for item in labels]),
                np.stack([getattr(item, mask) for item in labels]),
            )
        origin, destination = action_squares(self.target)
        self.squares = {
            "best_move_origin": origin,
            "best_move_destination": destination,
            "reply_origin": self.reply[:, 0],
            "reply_destination": self.reply[:, 1],
        }

    def correct(self, predicted: np.ndarray) -> np.ndarray:
        """Predicted moves that match the target; for puzzles any mating move also counts."""

        correct = np.asarray(predicted) == self.target
        if self.puzzles:
            for row in np.flatnonzero(~correct):
                board = self.boards[row].copy(stack=False)
                board.push(action_to_move(board, int(predicted[row])))
                correct[row] = board.is_checkmate()
        return correct


def _board(split: GameSplit | PuzzleSplit, index: int) -> chess.Board:
    if split.fens is not None:
        return chess.Board(str(split.fens[index]))
    return decode_position(np.asarray(split.pieces[index]), np.asarray(split.state[index]))


def _valid(split: GameSplit | PuzzleSplit, candidates: np.ndarray, count: int) -> np.ndarray:
    """The first ``count`` candidates that are legal positions, sorted."""

    chosen: list[int] = []
    for index in candidates:
        if len(chosen) == count:
            break
        if _board(split, int(index)).status() == chess.STATUS_VALID:
            chosen.append(int(index))
    return np.sort(np.array(chosen, dtype=np.int64))


def _game_sample(
    name: str, split: GameSplit, source: Path, indices: np.ndarray, reply: np.ndarray
) -> Sample:
    return Sample(
        name=name,
        source=str(source),
        indices=indices,
        boards=[_board(split, int(index)) for index in indices],
        pieces=np.asarray(split.pieces[indices], dtype=np.int64),
        state=np.asarray(split.state[indices], dtype=np.float32),
        legal=split.legal_mask(indices),
        target=split.action[indices],
        groups=indices,
        reply=reply,
        wins=split._win_table(indices),
    )


def game_sample(directory: Path, count: int, seed: int = 0) -> Sample:
    """Up to ``count`` valid game positions of a split, Stockfish's deep-best move as target."""

    split = GameSplit(directory)
    order = np.random.default_rng(seed).permutation(len(split))
    indices = _valid(split, order, count)
    return _game_sample("games", split, directory, indices, np.full((len(indices), 2), -1))


def labelled_sample(directory: Path, labels: Path, count: int, seed: int = 0) -> Sample:
    """Labelled training positions: every move scored, and Stockfish's reply from the line."""

    split = GameSplit(directory, labels=labels)
    order = split.label_indices[np.random.default_rng(seed).permutation(len(split.label_indices))]
    indices = _valid(split, order, count)
    lookahead = split.lookahead_targets(indices)
    reply = np.stack((lookahead["reply_from"], lookahead["reply_to"]), axis=1)
    return _game_sample("labelled", split, directory, indices, np.where(reply == IGNORE, -1, reply))


def strength_sample(directory: Path, count: int) -> tuple[Sample, dict[str, object]]:
    """The positions of a strength report, with judged losses at its best depth.

    Returns the sample and a note naming the report's checkpoint, split and best depth.
    """

    report = json.loads((directory / "strength.json").read_text(encoding="utf-8"))
    curve = report["curve"]
    best = max(curve, key=lambda row: (row["lichess_accuracy"], -row["reasoning_steps"]))
    depth = int(best["reasoning_steps"])
    games = Path(report["games"])
    split = GameSplit(games)
    with np.load(directory / "strength.npz") as arrays:
        indices = np.asarray(arrays["game_indices"][:count], dtype=np.int64)
        lost = np.asarray(arrays[f"lost_t{depth}"][:count], dtype=np.float64)
    sample = _game_sample("games", split, games, indices, np.full((len(indices), 2), -1))
    sample.lost, sample.lost_depth = lost, depth
    note: dict[str, object] = {
        "report": str(directory),
        "checkpoint": report["checkpoint"],
        "games": str(games),
        "best_depth": depth,
        "lichess_accuracy": best["lichess_accuracy"],
    }
    return sample, note


def _opponent_reply(
    board: chess.Board, move: chess.Move, following: chess.Board
) -> tuple[int, int]:
    """The reply that leads from ``board`` after ``move`` to ``following``, in ``board``'s frame."""

    after = board.copy(stack=False)
    after.push(move)
    for reply in after.legal_moves:
        after.push(reply)
        reached = after.board_fen() == following.board_fen()
        after.pop()
        if reached:
            origin, destination = move_squares(board, reply)
            return origin, destination
    return -1, -1


def puzzle_sample(directory: Path, count: int, seed: int = 0) -> Sample:
    """About ``count`` puzzle positions, spread evenly over the moves-remaining groups."""

    split = PuzzleSplit(directory)
    per_group = max(1, count // len(REMAINING_GROUPS))
    chosen = sample_groups(remaining_groups(split.remaining), per_group, seed)
    indices = np.sort(np.concatenate(list(chosen.values()))).astype(np.int64)
    boards = [_board(split, int(index)) for index in indices]
    reply = np.full((len(indices), 2), -1, dtype=np.int64)
    puzzle = np.asarray(split.puzzle)
    line = np.asarray(split.line_index)
    for row, index in enumerate(indices):
        following = int(index) + 1
        if following >= len(split) or puzzle[following] != puzzle[index]:
            continue
        if line[following] != line[index] + 1:
            continue
        move = action_to_move(boards[row], int(split.action[index]))
        reply[row] = _opponent_reply(boards[row], move, _board(split, following))
    return Sample(
        name="puzzles",
        source=str(directory),
        indices=indices,
        boards=boards,
        pieces=np.asarray(split.pieces[indices], dtype=np.int64),
        state=np.asarray(split.state[indices], dtype=np.float32),
        legal=split.legal_mask(indices),
        target=split.action[indices],
        groups=puzzle[indices].astype(np.int64),
        reply=reply,
        puzzles=True,
        remaining=np.asarray(split.remaining[indices], dtype=np.int64),
    )


@contextmanager
def board_features_off(model: WombatNetwork) -> Iterator[None]:
    """Zero the board-feature projection's contribution to the input tokens."""

    if not model.config.board_features:
        yield
        return

    def zero(module: torch.nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(output)

    handle = model.board_feature_projection.register_forward_hook(zero)
    try:
        yield
    finally:
        handle.remove()


@dataclass
class Steps:
    """Square tokens per representation ("input", "t0", "t1", ...) and readouts per depth."""

    tokens: dict[str, np.ndarray]
    actions: dict[int, np.ndarray]
    replies: dict[int, np.ndarray]


def batch_size_for(model: WombatNetwork, max_vram_gb: float | None) -> int:
    """Positions per batch whose fp32 activations fit in half the VRAM cap."""

    if max_vram_gb is None:
        return 256
    config = model.config
    tokens = model.hidden_tokens
    width = 8 * config.d_model + 3 * config.mlp_width + 4 * config.attention_heads * tokens
    per_position = 4 * tokens * width
    return int(min(512, max(8, max_vram_gb * 2**30 / 2 / per_position)))


def _step(model: WombatNetwork, board_memory: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, model.recurrent_step(board_memory, hidden))


def _legal_argmax(logits: torch.Tensor, legal: torch.Tensor) -> np.ndarray:
    return logits.float().masked_fill(~legal, float("-inf")).argmax(dim=1).cpu().numpy()


def _reply_squares(predicted: dict[str, torch.Tensor]) -> np.ndarray:
    """(batch, 2) reply origin and destination from separate square logits or joint pairs."""

    if "reply_pair" in predicted:
        best = predicted["reply_pair"].flatten(1).argmax(-1)
        pair = torch.stack((best // 64, best % 64), dim=1)
    else:
        origin, destination = predicted["reply_from"], predicted["reply_to"]
        pair = torch.stack((origin.argmax(-1), destination.argmax(-1)), dim=1)
    return pair.cpu().numpy()


@torch.inference_mode()
def collect_steps(
    model: WombatNetwork,
    sample: Sample,
    depths: Sequence[int],
    device: torch.device,
    batch_size: int,
    *,
    keep_tokens: bool = True,
) -> Steps:
    """Run every position to the deepest depth once, keeping tokens and readouts per depth."""

    model.eval()
    depths = sorted(set(depths) | {0})
    tokens: dict[str, list[np.ndarray]] = {}
    actions: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    replies: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    lookahead = model.config.lookahead_heads

    def keep(name: str, hidden: torch.Tensor) -> None:
        if keep_tokens:
            tokens.setdefault(name, []).append(hidden[:, :64].half().cpu().numpy())

    for start in range(0, len(sample), batch_size):
        rows = slice(start, start + batch_size)
        pieces = torch.from_numpy(sample.pieces[rows]).to(device)
        state = torch.from_numpy(sample.state[rows]).to(device)
        legal = torch.from_numpy(sample.legal[rows]).to(device)
        keep("input", model._tokenize(pieces, state))
        board_memory = model.encode_board(pieces, state)
        hidden = model.initial_hidden(board_memory)
        for step in range(max(depths) + 1):
            if step:
                hidden = _step(model, board_memory, hidden)
            if step not in actions:
                continue
            keep(f"t{step}", hidden)
            policy = model.heads(
                hidden,
                board_memory.shape[1],
                consequences=board_consequences(board_memory),
                phase=board_phase(board_memory),
            )[0]
            actions[step].append(_legal_argmax(policy, legal))
            if lookahead:
                replies[step].append(_reply_squares(model.lookahead(hidden)))
    return Steps(
        tokens={name: np.concatenate(items) for name, items in tokens.items()},
        actions={depth: np.concatenate(items) for depth, items in actions.items()},
        replies={depth: np.concatenate(items) for depth, items in replies.items() if items},
    )


@dataclass
class ProbeScores:
    """Out-of-fold probe scores per target and representation, for subsets and attribution."""

    rows: dict[str, tuple[np.ndarray, np.ndarray]]
    binary: dict[str, dict[str, np.ndarray]]
    squares: dict[str, dict[str, np.ndarray]]


def _binary_metric(
    scores: np.ndarray, labels: np.ndarray, keep: np.ndarray | None = None
) -> float | None:
    chosen = ~np.isnan(scores) if keep is None else keep & ~np.isnan(scores)
    return auroc(scores[chosen], labels[chosen])


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _progress(message: str) -> None:
    """One timestamped progress line on stderr, so long probe runs show where they are."""

    print(f"{time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def probe_sample(
    sample: Sample,
    steps: Steps,
    device: torch.device,
    *,
    late: str,
    seed: int = 0,
) -> tuple[dict[str, object], ProbeScores]:
    """Fit every probe on every representation; report each and keep out-of-fold scores.

    The shuffled-label control fits the same probes on permuted labels at ``late``.
    """

    fold = folds(sample.groups, seed)
    generator = np.random.default_rng(seed)
    report: dict[str, object] = {}
    scores = ProbeScores({}, {}, {})
    groups = remaining_groups(sample.remaining) if sample.remaining is not None else {}
    for name in BINARY_TARGETS:
        _progress(f"{sample.name}: probing {name}")
        values, mask = sample.binary[name]
        positions, squares = np.nonzero(mask)
        labels = values[positions, squares]
        scores.rows[name] = (positions, squares)
        by_representation: dict[str, np.ndarray] = {}
        for representation, tokens in steps.tokens.items():
            by_representation[representation] = cross_fit_binary(
                tokens, positions, squares, labels, fold, device
            )
        scores.binary[name] = by_representation
        shuffled = cross_fit_binary(
            steps.tokens[late], positions, squares, labels, fold, device, shuffle=generator
        )
        entry: dict[str, object] = {
            "kind": "binary",
            "metric": "auroc",
            "positions": int(mask.any(axis=1).sum()),
            "rows": len(labels),
            "positives": int(labels.sum()),
            "by_representation": {
                key: _rounded(_binary_metric(value, labels))
                for key, value in by_representation.items()
            },
            f"shuffled_labels_{late}": _rounded(_binary_metric(shuffled, labels)),
        }
        if groups:
            entry["by_remaining"] = {
                group: {
                    key: _rounded(_binary_metric(value, labels, np.isin(positions, members)))
                    for key, value in by_representation.items()
                }
                for group, members in groups.items()
                if len(members)
            }
        report[name] = entry
    readout_squares = {depth: action_squares(actions) for depth, actions in steps.actions.items()}
    for name in SQUARE_TARGETS:
        targets = sample.squares[name]
        if not (targets >= 0).any():
            report[name] = None
            continue
        _progress(f"{sample.name}: probing {name}")
        logits = {
            representation: cross_fit_square(tokens, targets, fold, device)
            for representation, tokens in steps.tokens.items()
        }
        scores.squares[name] = logits
        shuffled_logits = cross_fit_square(
            steps.tokens[late], targets, fold, device, shuffle=generator
        )
        entry = {
            "kind": "square",
            "metric": "accuracy",
            "positions": int((targets >= 0).sum()),
            "prior": _rounded(prior_accuracy(targets, fold)),
            "by_representation": {
                key: _rounded(square_accuracy(value, targets)) for key, value in logits.items()
            },
            f"shuffled_labels_{late}": _rounded(square_accuracy(shuffled_logits, targets)),
        }
        known = targets >= 0
        if name.startswith("best_move"):
            part = 0 if name.endswith("origin") else 1
            entry["model_policy"] = {
                f"t{depth}": _rounded(float((pair[part][known] == targets[known]).mean()))
                for depth, pair in readout_squares.items()
            }
        elif steps.replies:
            part = 0 if name.endswith("origin") else 1
            entry["model_reply_head"] = {
                f"t{depth}": _rounded(float((pair[known, part] == targets[known]).mean()))
                for depth, pair in steps.replies.items()
            }
        if groups:
            entry["by_remaining"] = {
                group: {
                    key: _rounded(square_accuracy(value[members], targets[members]))
                    for key, value in logits.items()
                }
                for group, members in groups.items()
                if len(members)
            }
        report[name] = entry
    return report, scores


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def improvement(
    sample: Sample, steps: Steps, scores: ProbeScores, early: int, late: int
) -> dict[str, object]:
    """Puzzles gained and lost between ``early`` and ``late`` steps, against steady ones.

    Token change is the L2 distance of each square token between the two depths; ranks count
    1 for the square that changed most. Probe metrics use the out-of-fold scores.
    """

    right_early = sample.correct(steps.actions[early])
    right_late = sample.correct(steps.actions[late])
    sets = {
        "gained": ~right_early & right_late,
        "lost": right_early & ~right_late,
        "both_right": right_early & right_late,
        "both_wrong": ~right_early & ~right_late,
    }
    difference = steps.tokens[f"t{late}"].astype(np.float32) - steps.tokens[f"t{early}"]
    change = np.linalg.norm(difference, axis=-1)
    relative = change / change.mean(axis=1, keepdims=True).clip(1e-12)
    ranks = (-change).argsort(axis=1).argsort(axis=1) + 1
    origin, destination = action_squares(sample.target)
    rows = np.arange(len(sample))
    king = (sample.pieces == 12).argmax(axis=1)
    distance = np.minimum(king_distance(origin), king_distance(destination))
    categories = {
        "origin": (rows, origin),
        "destination": (rows, destination),
        "opponent_king": (rows, king),
    }
    report: dict[str, object] = {
        "early": early,
        "late": late,
        "accuracy_early": float(right_early.mean()),
        "accuracy_late": float(right_late.mean()),
    }
    for set_name, members in sets.items():
        chosen = np.flatnonzero(members)
        entry: dict[str, object] = {"positions": len(chosen)}
        if not len(chosen):
            report[set_name] = entry
            continue
        entry["mean_change"] = _mean(change[chosen])
        entry["relative_change"] = {
            name: _mean(relative[index][members]) for name, index in categories.items()
        }
        near = {"1": distance == 1, "2": distance == 2, "3": distance == 3, "4+": distance >= 4}
        entry["relative_change_by_king_steps_from_move"] = {
            name: _mean(relative[chosen][region[chosen]]) for name, region in near.items()
        }
        entry["rank"] = {
            name: _mean(ranks[index][members].astype(np.float64))
            for name, index in categories.items()
        }
        entry["in_top_5"] = {
            name: _mean((ranks[index][members] <= 5).astype(np.float64))
            for name, index in categories.items()
        }
        probes: dict[str, object] = {}
        for name, by_representation in scores.binary.items():
            positions, squares = scores.rows[name]
            values = sample.binary[name][0][positions, squares]
            keep = members[positions]
            first = _binary_metric(by_representation[f"t{early}"], values, keep)
            second = _binary_metric(by_representation[f"t{late}"], values, keep)
            probes[name] = _change(first, second)
        for name, logits in scores.squares.items():
            targets = sample.squares[name]
            first = square_accuracy(logits[f"t{early}"][chosen], targets[chosen])
            second = square_accuracy(logits[f"t{late}"][chosen], targets[chosen])
            probes[name] = _change(first, second)
        entry["probes"] = probes
        report[set_name] = entry
    return report


def _change(first: float | None, second: float | None) -> dict[str, float | None]:
    gain = None if first is None or second is None else second - first
    return {"early": _rounded(first), "late": _rounded(second), "gain": _rounded(gain)}


def square_mask(squares: np.ndarray, tokens: int) -> torch.Tensor:
    """(batch, tokens) mask over H marking each row's listed squares; -1 lists nothing."""

    mask = np.zeros((len(squares), tokens + 1), dtype=np.bool_)
    rows = np.repeat(np.arange(len(squares)), squares.shape[1])
    mask[rows, np.where(squares < 0, tokens, squares).reshape(-1)] = True
    return torch.from_numpy(mask[:, :tokens])


def swap_tokens(early: torch.Tensor, late: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``early`` with the masked tokens replaced by their ``late`` values."""

    return torch.where(mask.to(early.device)[..., None], late, early)


def freeze_mask(
    origin: np.ndarray, destination: np.ndarray, radius: int, tokens: int, *, inside: bool = False
) -> torch.Tensor:
    """(batch, tokens) squares farther than ``radius`` king steps from both of a move's squares.

    ``inside`` marks the squares within the radius instead. Other tokens are never marked.
    """

    distance = np.minimum(king_distance(origin), king_distance(destination))
    far = distance <= radius if inside else distance > radius
    mask = np.zeros((len(origin), tokens), dtype=np.bool_)
    mask[:, :64] = far
    return torch.from_numpy(mask)


def run_frozen(
    model: WombatNetwork,
    board_memory: torch.Tensor,
    hidden: torch.Tensor,
    steps: int,
    frozen: torch.Tensor,
) -> torch.Tensor:
    """Run ``steps`` more steps, holding the ``frozen`` tokens at their values in ``hidden``."""

    kept = hidden
    frozen = frozen.to(hidden.device)[..., None]
    for _ in range(steps):
        hidden = torch.where(frozen, kept, _step(model, board_memory, hidden))
    return hidden


def _variant(
    sample: Sample,
    actions: np.ndarray,
    late_actions: np.ndarray,
    right_early: np.ndarray,
    gained: np.ndarray,
    changed: np.ndarray,
) -> dict[str, object]:
    correct = sample.correct(actions)
    agrees = actions == late_actions
    return {
        "accuracy": _rounded(float(correct.mean())),
        "change_from_early": _rounded(float(correct.mean() - right_early.mean())),
        "agrees_with_late_move": _rounded(float(agrees.mean())),
        "agrees_with_late_move_where_it_changed": _rounded(_mean(agrees[changed].astype(float))),
        "right_on_gained_puzzles": _rounded(_mean(correct[gained].astype(float))),
    }


@torch.inference_mode()
def causal_tests(
    model: WombatNetwork,
    sample: Sample,
    early: int,
    late: int,
    device: torch.device,
    batch_size: int,
    radii: Sequence[int] = (1, 2),
) -> dict[str, object]:
    """Swap and freeze parts of H between ``early`` and ``late`` and re-read the policy.

    Swaps replace square tokens of the early state with their late values: the late move's
    origin and destination, every other square, or all 64. Freezes hold the squares outside
    (or inside) ``radius`` king steps of the target move's squares at their early values while
    the other tokens run on to ``late``.
    """

    if not 0 <= early < late:
        raise ValueError("causal tests need 0 <= early < late")
    model.eval()
    tokens = model.hidden_tokens
    swaps = ("move_squares", "other_squares", "all_squares")
    freezes = [
        (f"{where}_{radius}", radius, where == "inside")
        for radius in radii
        for where in ("outside", "inside")
    ]
    target_origin, target_destination = action_squares(sample.target)
    results: dict[str, list[np.ndarray]] = {
        name: [] for name in ("early", "late", *swaps, *(name for name, _, _ in freezes))
    }
    frozen_counts: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in freezes}
    for start in range(0, len(sample), batch_size):
        rows = slice(start, start + batch_size)
        pieces = torch.from_numpy(sample.pieces[rows]).to(device)
        state = torch.from_numpy(sample.state[rows]).to(device)
        legal = torch.from_numpy(sample.legal[rows]).to(device)
        board_memory = model.encode_board(pieces, state)
        board_tokens = board_memory.shape[1]
        consequences, phase = board_consequences(board_memory), board_phase(board_memory)

        def read(hidden: torch.Tensor) -> np.ndarray:
            policy = model.heads(hidden, board_tokens, consequences=consequences, phase=phase)[0]
            return _legal_argmax(policy, legal)

        hidden = model.initial_hidden(board_memory)
        for _ in range(early):
            hidden = _step(model, board_memory, hidden)
        at_early = hidden
        for _ in range(late - early):
            hidden = _step(model, board_memory, hidden)
        at_late = hidden
        late_actions = read(at_late)
        results["early"].append(read(at_early))
        results["late"].append(late_actions)
        origin, destination = action_squares(late_actions)
        move = square_mask(np.stack((origin, destination), axis=1), tokens)
        every = square_mask(np.tile(np.arange(64), (len(late_actions), 1)), tokens)
        masks = {"move_squares": move, "other_squares": every & ~move, "all_squares": every}
        for name, mask in masks.items():
            results[name].append(read(swap_tokens(at_early, at_late, mask)))
        for name, radius, inside in freezes:
            frozen = freeze_mask(
                target_origin[rows], target_destination[rows], radius, tokens, inside=inside
            )
            frozen_counts[name].append(frozen.sum(1).numpy())
            results[name].append(
                read(run_frozen(model, board_memory, at_early, late - early, frozen))
            )
    actions = {name: np.concatenate(items) for name, items in results.items()}
    right_early = sample.correct(actions["early"])
    right_late = sample.correct(actions["late"])
    gained = ~right_early & right_late
    changed = actions["early"] != actions["late"]

    def variant(name: str) -> dict[str, object]:
        return _variant(sample, actions[name], actions["late"], right_early, gained, changed)

    freeze_report: dict[str, object] = {}
    for name, radius, inside in freezes:
        entry = variant(name)
        gain = float(right_late.mean() - right_early.mean())
        change = entry["change_from_early"]
        assert isinstance(change, float)
        entry["gain_kept"] = _rounded(change / gain) if gain else None
        entry["frozen_squares"] = _rounded(float(np.concatenate(frozen_counts[name]).mean()))
        entry["radius"] = radius
        entry["frozen"] = "within radius" if inside else "beyond radius"
        freeze_report[name] = entry
    return {
        "early": early,
        "late": late,
        "positions": len(sample),
        "accuracy_early": _rounded(float(right_early.mean())),
        "accuracy_late": _rounded(float(right_late.mean())),
        "gained": int(gained.sum()),
        "late_move_differs": int(changed.sum()),
        "policy_readout": model.config.policy_readout,
        "swap": {name: variant(name) for name in swaps},
        "freeze": freeze_report,
    }


def board_features_ablation(
    model: WombatNetwork,
    sample: Sample,
    normal: dict[int, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, object] | None:
    """Accuracy by depth with the board-feature projection zeroed at the input, or None."""

    if not model.config.board_features:
        return None
    with board_features_off(model):
        steps = collect_steps(model, sample, sorted(normal), device, batch_size, keep_tokens=False)
    return {
        f"t{depth}": {
            "accuracy": _rounded(float(sample.correct(normal[depth]).mean())),
            "accuracy_features_off": _rounded(float(sample.correct(steps.actions[depth]).mean())),
            "same_move": _rounded(float((steps.actions[depth] == normal[depth]).mean())),
        }
        for depth in sorted(normal)
    }


def best_depth(sample: Sample, actions: dict[int, np.ndarray]) -> int:
    """The depth whose moves most often match the target, the shallowest on ties."""

    return max(sorted(actions), key=lambda depth: (sample.correct(actions[depth]).mean(), -depth))


def move_losses(sample: Sample, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Win-chance points lost by each move, and whether that is exact or a lower bound.

    An unscored move is no better than the worst scored one, which bounds its loss.
    """

    if sample.lost is not None:
        return sample.lost, np.ones(len(sample), dtype=np.bool_)
    if sample.wins is None:
        raise ValueError(f"the {sample.name} sample has no move scores")
    wins = sample.wins
    chosen = wins[np.arange(len(sample)), actions]
    exact = ~np.isnan(chosen)
    best, worst = np.nanmax(wins, axis=1), np.nanmin(wins, axis=1)
    return np.where(exact, best - chosen, best - worst) * 100, exact


def blunder_attribution(
    sample: Sample,
    actions: np.ndarray,
    scores: ProbeScores,
    representation: str,
    threshold: float = BLUNDER_FROM,
) -> dict[str, object]:
    """Whether the destination probe says a blundered move loses material.

    "Present" means the out-of-fold probe on ``representation`` gives the move's destination a
    probability of at least one half of losing material, so the readout ignored knowledge the
    state held; "absent" means it did not. Blunders lose ``threshold`` or more win-chance
    points, exactly or as a lower bound; moves outside the scores with a smaller bound are
    unknown.
    """

    lost, exact = move_losses(sample, actions)
    blunder = lost >= threshold
    positions, squares = scores.rows["destination_loses_material"]
    lookup = np.full((len(sample), 64), -1, dtype=np.int64)
    lookup[positions, squares] = np.arange(len(positions))
    logits = scores.binary["destination_loses_material"][representation]
    _, destination = action_squares(actions)
    row = lookup[np.arange(len(sample)), destination]
    logit = np.where(row >= 0, logits[row.clip(0)], np.nan)
    scored = ~np.isnan(logit)
    says = scored & (logit >= 0)
    label = sample.binary["destination_loses_material"][0][np.arange(len(sample)), destination]
    losing = np.zeros(len(sample), dtype=np.bool_)
    for index in np.flatnonzero(blunder):
        board = sample.boards[index]
        losing[index] = static_exchange(board, action_to_move(board, int(actions[index]))) < 0

    def split(members: np.ndarray) -> dict[str, int]:
        return {
            "blunders": int(members.sum()),
            "knowledge_present": int((members & says).sum()),
            "knowledge_absent": int((members & scored & ~says).sum()),
        }

    judged_fine = exact & ~blunder
    return {
        "representation": representation,
        "threshold": threshold,
        "positions": len(sample),
        "exact_losses": int(exact.sum()),
        "unknown": int((~exact & ~blunder).sum()),
        **split(blunder),
        "blunders_losing_material_by_exchange": split(blunder & losing),
        "other_blunders": split(blunder & ~losing),
        "blunders_onto_squares_labelled_losing": int((blunder & label).sum()),
        "probe_says_loses_on_other_moves": _rounded(_mean(says[judged_fine].astype(float))),
    }


def _describe(sample: Sample) -> dict[str, object]:
    description: dict[str, object] = {
        "source": sample.source,
        "positions": len(sample),
        "known_replies": int((sample.reply >= 0).all(axis=1).sum()),
        "positives": {
            name: {"positive": int(values[mask].sum()), "rows": int(mask.sum())}
            for name, (values, mask) in sample.binary.items()
        },
    }
    if sample.remaining is not None:
        description["by_remaining"] = {
            group: len(members) for group, members in remaining_groups(sample.remaining).items()
        }
    return description


def probe_steps(
    checkpoint: Path,
    output: Path,
    *,
    games: Path,
    puzzles: Path,
    labelled_games: Path | None = None,
    labels: Path | None = None,
    strength: Path | None = None,
    positions: int = 2_000,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    early: int = 4,
    late: int = 16,
    radii: Sequence[int] = (1, 2),
    device_name: str = "cuda",
    max_vram_gb: float | None = 1.0,
    batch_size: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Probe, compare and intervene on one checkpoint; write ``report.json`` to ``output``.

    Game positions come from ``games``, or from a strength report's directory, whose judged
    losses and best depth then drive blunder attribution. Labelled training positions are added
    when ``labels`` is given; the model trained on them, but only they carry Stockfish's reply.
    """

    started = time.perf_counter()
    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    model = load_checkpoint(checkpoint, device)
    cap = max_vram_gb if device.type == "cuda" else None
    batch = batch_size or batch_size_for(model, cap)
    notes: dict[str, object] = {}
    if strength is not None:
        games_sample, note = strength_sample(strength, positions)
        judged = Path(str(note["checkpoint"])).resolve()
        note["same_checkpoint"] = judged == checkpoint.resolve()
        notes["strength"] = note
    else:
        games_sample = game_sample(games, positions, seed)
    puzzles_sample = puzzle_sample(puzzles, positions, seed)
    samples = [games_sample, puzzles_sample]
    if labels is not None and labelled_games is not None:
        samples.append(labelled_sample(labelled_games, labels, positions, seed))
        notes["labelled"] = "training positions: the checkpoint was trained on them"
    step_depths = sorted({0, *depths, early, late})
    best = games_sample.lost_depth
    if best is not None:
        step_depths = sorted({*step_depths, best})
    probes: dict[str, object] = {}
    blunders: dict[str, object] = {}
    ablation: dict[str, object] = {}
    improvement_report: dict[str, object] = {}
    for sample in samples:
        _progress(f"{sample.name}: labelling {len(sample.indices)} positions")
        sample.label()
        _progress(f"{sample.name}: collecting steps {step_depths}")
        steps = collect_steps(model, sample, step_depths, device, batch)
        probes[sample.name], scores = probe_sample(
            sample, steps, device, late=f"t{late}", seed=seed
        )
        if sample.puzzles:
            improvement_report = improvement(sample, steps, scores, early, late)
        else:
            if best is None:
                best = best_depth(sample, steps.actions)
            blunders[sample.name] = blunder_attribution(
                sample, steps.actions[best], scores, f"t{best}"
            )
        if sample.name in ("games", "puzzles"):
            _progress(f"{sample.name}: board-feature ablation")
            ablation[sample.name] = board_features_ablation(
                model, sample, steps.actions, device, batch
            )
        del steps, scores
    _progress("causal tests on puzzles")
    causal = causal_tests(model, puzzles_sample, early, late, device, batch, radii)
    causal["board_features_off"] = ablation if model.config.board_features else None
    report: dict[str, object] = {
        "schema": "wombat.probe-steps.v1",
        "checkpoint": str(checkpoint),
        "config": asdict(model.config),
        "device": device_name,
        "batch_size": batch,
        "seed": seed,
        "depths": step_depths,
        "representations": ["input", *(f"t{depth}" for depth in step_depths)],
        "early": early,
        "late": late,
        "best_depth": best,
        "best_depth_source": "strength report"
        if games_sample.lost_depth is not None
        else "agreement with Stockfish's best move on the game sample",
        "samples": {sample.name: _describe(sample) for sample in samples},
        "probes": probes,
        "improvement": improvement_report,
        "causal": causal,
        "blunders": blunders,
        "notes": notes,
    }
    if device.type == "cuda":
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    report["seconds"] = round(time.perf_counter() - started, 1)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _progress(f"wrote {output / 'report.json'} after {report['seconds']} s")
    return report


def _cell(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "  -  "


def format_summary(report: dict[str, Any]) -> str:
    """A short plain-text account of a ``probe_steps`` report."""

    representations = report["representations"]
    early, late = report["early"], report["late"]
    lines = ["Probes: AUROC for binary targets, top-1 square accuracy for the others."]
    header = " ".join(f"{name:>6}" for name in representations)
    for sample, targets in report["probes"].items():
        lines.append(f"[{sample}] {'target':<27}{header}  shuffled")
        for name, entry in targets.items():
            if entry is None:
                continue
            row = " ".join(
                f"{_cell(entry['by_representation'][key]):>6}" for key in representations
            )
            shuffled = _cell(entry[f"shuffled_labels_t{late}"])
            lines.append(f"  {name:<{27 + len(sample) + 1}}{row}  {shuffled}")
    change = report["improvement"]
    lines.append("")
    lines.append(
        f"Puzzles T{early} to T{late}: accuracy {change['accuracy_early']:.3f} to"
        f" {change['accuracy_late']:.3f}; gained {change['gained']['positions']},"
        f" lost {change['lost']['positions']}, both right {change['both_right']['positions']}"
    )
    for name in ("gained", "both_right"):
        entry = change[name]
        if not entry["positions"]:
            continue
        relative = entry["relative_change"]
        lines.append(
            f"  {name}: token change against the position mean, origin"
            f" {_cell(relative['origin'])}, destination {_cell(relative['destination'])},"
            f" opponent king {_cell(relative['opponent_king'])}"
        )
        gains = ", ".join(
            f"{target} {_cell(values['gain'])}" for target, values in entry["probes"].items()
        )
        lines.append(f"  {name}: probe gain {gains}")
    causal = report["causal"]
    lines.append("")
    lines.append(
        f"Causal, puzzles (T{early} {causal['accuracy_early']:.3f}, T{late}"
        f" {causal['accuracy_late']:.3f}; T{late} move differs on {causal['late_move_differs']}):"
    )
    for name, entry in causal["swap"].items():
        lines.append(
            f"  swap {name} into T{early}: picks the T{late} move on"
            f" {_cell(entry['agrees_with_late_move_where_it_changed'])} of changed positions,"
            f" accuracy {_cell(entry['accuracy'])}"
        )
    for name, entry in causal["freeze"].items():
        lines.append(
            f"  freeze {entry['frozen']} {entry['radius']} after T{early}: accuracy"
            f" {_cell(entry['accuracy'])}, gain kept {_cell(entry['gain_kept'])},"
            f" gained puzzles kept {_cell(entry['right_on_gained_puzzles'])}"
        )
    for sample, depths in (causal["board_features_off"] or {}).items():
        if depths is None:
            continue
        cells = ", ".join(
            f"{key} {_cell(value['accuracy'])}/{_cell(value['accuracy_features_off'])}"
            for key, value in depths.items()
        )
        lines.append(f"  board features on/off, {sample}: {cells}")
    lines.append("")
    for sample, entry in report["blunders"].items():
        lines.append(
            f"Blunders at T{report['best_depth']} ({sample}): {entry['blunders']} of"
            f" {entry['positions']}, knowledge present {entry['knowledge_present']}, absent"
            f" {entry['knowledge_absent']}, unknown losses {entry['unknown']}"
        )
    strength = report["notes"].get("strength")
    if strength is not None and not strength["same_checkpoint"]:
        lines.append(f"Warning: the strength report judged {strength['checkpoint']}")
    return "\n".join(lines)
