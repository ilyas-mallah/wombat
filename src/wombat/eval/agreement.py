"""Move-agreement evaluation with strict legal masking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess
import numpy as np

from wombat.chess.moves import ACTION_SIZE, action_to_move, legal_action_mask
from wombat.eval.manifest import PositionManifest
from wombat.policy import PolicyProvider, ValuedPolicyProvider

# Play-time draw guard threshold on the model's -1..1 value for the side to move.
DRAW_GUARD_VALUE = 0.3


@dataclass(frozen=True)
class AgreementResult:
    manifest_sha256: str
    reasoning_steps: int
    positions: int
    correct: int
    accuracy: float
    choices: tuple[dict[str, object], ...]


def draws_immediately(board: chess.Board, move: chess.Move) -> bool:
    """Whether ``move`` stalemates the opponent or lets them claim a draw on their turn.

    The opponent can claim when the position occurs a third time or one of their replies makes it
    occur a third time, or under the fifty-move rule; a losing opponent will.
    """

    board.push(move)
    try:
        return board.is_stalemate() or board.can_claim_draw()
    finally:
        board.pop()


def guard_draws(
    board: chess.Board,
    ranked: Sequence[chess.Move],
    value: float | None,
    threshold: float | None,
) -> chess.Move:
    """The first of ``ranked`` (best first) that does not draw at once, when winning.

    Winning means ``value`` is above ``threshold``; otherwise, or when every move draws, it is
    the first move. A threshold of None disables the guard.
    """

    if threshold is None or value is None or value <= threshold:
        return ranked[0]
    for move in ranked:
        if not draws_immediately(board, move):
            return move
    return ranked[0]


def select_move(
    provider: PolicyProvider,
    board: chess.Board,
    reasoning_steps: int,
    *,
    draw_guard: float | None = None,
) -> chess.Move:
    """The highest-logit legal move.

    With ``draw_guard`` set and a provider that reports its value, a side to move valued above it
    skips moves that stalemate or repeat a third time (``guard_draws``).
    """

    return select_move_and_value(provider, board, reasoning_steps, draw_guard=draw_guard)[0]


def select_move_and_value(
    provider: PolicyProvider,
    board: chess.Board,
    reasoning_steps: int,
    *,
    draw_guard: float | None = None,
) -> tuple[chess.Move, float | None]:
    """``select_move`` with the position value it read, or None when it read none."""

    value: float | None = None
    if draw_guard is not None and isinstance(provider, ValuedPolicyProvider):
        raw, value = provider.policy_and_value(board, reasoning_steps)
    else:
        raw = provider.policy_logits(board, reasoning_steps)
    logits = np.asarray(raw)
    if logits.shape != (ACTION_SIZE,):
        raise ValueError(f"policy logits must have shape ({ACTION_SIZE},), got {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("policy logits contain non-finite values")
    mask = legal_action_mask(board)
    if not mask.any():
        raise ValueError("cannot select a move in a terminal position")
    masked = np.where(mask, logits, -np.inf)
    if value is None or draw_guard is None or value <= draw_guard:
        return action_to_move(board, int(np.argmax(masked))), value
    legal = np.flatnonzero(mask)
    order = legal[np.argsort(-logits[legal], kind="stable")]
    ranked = [action_to_move(board, int(action)) for action in order]
    return guard_draws(board, ranked, value, draw_guard), value


def evaluate_agreement(
    provider: PolicyProvider, manifest: PositionManifest, reasoning_steps: int
) -> AgreementResult:
    if reasoning_steps < 1:
        raise ValueError("reasoning_steps must be positive")
    manifest.validate()
    choices: list[dict[str, object]] = []
    correct = 0
    for case in manifest.positions:
        board = chess.Board(case.fen)
        chosen = select_move(provider, board, reasoning_steps).uci()
        is_correct = chosen == case.best_move
        correct += int(is_correct)
        choices.append(
            {"id": case.id, "expected": case.best_move, "chosen": chosen, "correct": is_correct}
        )
    count = len(manifest.positions)
    return AgreementResult(
        manifest_sha256=manifest.sha256,
        reasoning_steps=reasoning_steps,
        positions=count,
        correct=correct,
        accuracy=correct / count if count else 0.0,
        choices=tuple(choices),
    )
