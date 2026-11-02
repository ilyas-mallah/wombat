"""Connect a torch Wombat network to the shared policy interface."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from wombat.chess.encode import play_encoding, state_vector
from wombat.chess.moves import legal_action_mask
from wombat.model.network import WombatNetwork
from wombat.selection import MIX_BETA, SELECT_MODES, require_move_values, selection_scores

__all__ = ["TorchPolicyProvider", "state_vector"]


class TorchPolicyProvider:
    """Plays a network; ``zero_halfmove`` encodes every board in the halfmove bucket of training.

    ``select`` and ``beta`` rank moves as ``wombat.selection.selection_scores`` does. Given its
    ``checkpoint``, value and mix selection first check that its move values were trained.
    """

    def __init__(
        self,
        model: WombatNetwork,
        device: torch.device | str = "cpu",
        *,
        zero_halfmove: bool = True,
        select: str = "policy",
        beta: float = MIX_BETA,
        checkpoint: Path | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.zero_halfmove = zero_halfmove
        self.checkpoint = checkpoint
        self.select, self.beta = "policy", beta
        self.set_selection(select, beta)

    def set_selection(self, select: str, beta: float) -> None:
        if select not in SELECT_MODES:
            raise ValueError(f"unknown move selection {select!r}; choose one of {SELECT_MODES}")
        # Value and mix states were checked when entered.
        if select != "policy" and self.select == "policy" and self.checkpoint is not None:
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            require_move_values(saved, self.checkpoint, self.model.state_dict(), select)
        self.select, self.beta = select, beta

    def inputs(self, board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
        """The (1, 64) piece ids and (1, 26) state vector the network reads for ``board``."""

        encoded = play_encoding(board, zero_halfmove=self.zero_halfmove)
        pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None].to(self.device)
        state = torch.from_numpy(state_vector(encoded))[None].to(self.device)
        return pieces, state

    @torch.inference_mode()
    def policy_logits(self, board: chess.Board, reasoning_steps: int) -> np.ndarray:
        return self.policy_and_value(board, reasoning_steps)[0]

    @torch.inference_mode()
    def policy_and_value(
        self, board: chess.Board, reasoning_steps: int
    ) -> tuple[np.ndarray, float]:
        """Action scores and the position value; outside policy mode the scores are not logits."""

        if self.select == "policy":
            policy, value = self.model(*self.inputs(board), reasoning_steps)
            return policy[0].float().cpu().numpy(), float(value[0])
        policy, value, move_values = self.model(
            *self.inputs(board), reasoning_steps, include_move_values=True
        )
        legal = torch.from_numpy(legal_action_mask(board))[None].to(self.device)
        scores = selection_scores(policy, move_values, legal, self.select, self.beta)
        # select_move wants finite scores and masks illegal actions itself.
        finite = scores.masked_fill(~legal, torch.finfo(torch.float32).min)
        return finite[0].cpu().numpy(), float(value[0])
