"""Common policy interface used by evaluators and the UCI engine."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, runtime_checkable

import chess
import numpy as np

from wombat.chess.moves import ACTION_SIZE, move_to_action


class PolicyProvider(Protocol):
    def policy_logits(self, board: chess.Board, reasoning_steps: int) -> np.ndarray:
        """Return one logit for every Wombat action."""


@runtime_checkable
class ValuedPolicyProvider(PolicyProvider, Protocol):
    def policy_and_value(
        self, board: chess.Board, reasoning_steps: int
    ) -> tuple[np.ndarray, float]:
        """Return the action logits and the position value in -1..1 for the side to move."""


@runtime_checkable
class SelectingPolicyProvider(PolicyProvider, Protocol):
    select: str
    beta: float

    def set_selection(self, select: str, beta: float) -> None:
        """Rank moves by ``wombat.selection.selection_scores`` under ``select`` and ``beta``."""


class DeterministicRandomPolicy:
    """Order-independent fixture policy for end-to-end infrastructure tests."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def policy_logits(self, board: chess.Board, reasoning_steps: int) -> np.ndarray:
        material = f"{self.seed}:{reasoning_steps}:{board.fen()}".encode()
        seed = int.from_bytes(sha256(material).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(ACTION_SIZE).astype(np.float32)


class FixturePolicy:
    """Policy assigning the largest logit to a configured legal UCI move."""

    def __init__(self, moves_by_fen: dict[str, str]) -> None:
        self.moves_by_fen = moves_by_fen

    def policy_logits(self, board: chess.Board, reasoning_steps: int) -> np.ndarray:
        del reasoning_steps
        logits = np.zeros(ACTION_SIZE, dtype=np.float32)
        uci = self.moves_by_fen.get(board.fen())
        if uci is not None:
            move = chess.Move.from_uci(uci)
            logits[move_to_action(board, move)] = 1.0
        return logits

