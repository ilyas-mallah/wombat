"""Exact chess rules for imagined lines, for a whole batch with tensor ops.

Positions follow ``wombat.chess.encode``: (batch, 64) piece ids in the side to move's frame and
(batch, 26) state vectors. A child is the position after one action, in its own mover's frame:
the board rotated 180 degrees with colours swapped. Action -1 is a null move, which passes the
turn; lines use it when a position has no legal move.
"""

from __future__ import annotations

from functools import cache

import torch
from torch import Tensor, nn

from wombat.chess.encode import NO_EN_PASSANT_FILE
from wombat.chess.moves import PLANES_PER_SQUARE, square_pair_actions
from wombat.model.move_consequences import MoveConsequences


class LineRules(nn.Module):
    """Legal actions and child positions; its tables are buffers and nothing is learned."""

    pairs: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.moves = MoveConsequences()
        self.register_buffer("pairs", torch.from_numpy(square_pair_actions()), persistent=False)

    def legal(self, pieces: Tensor, state: Tensor) -> Tensor:
        """(batch, 4672) bool legal actions."""

        return self.moves.legal(pieces, state)

    def in_check(self, pieces: Tensor) -> Tensor:
        """(batch,) bool: whether the side to move's king is attacked."""

        boards = torch.cat((pieces.long(), pieces.new_zeros(pieces.shape[0], 1).long()), dim=1)
        king = self.moves._king(boards, 6)
        return self.moves._attacked(boards, king[:, None], theirs=True)[:, 0]

    def child(self, pieces: Tensor, state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Pieces and state after each row's action, in the child's frame.

        The halfmove bucket drops to 0 after a pawn move or capture and is kept otherwise, since
        a bucket does not record the clock within it.
        """

        rules = self.moves
        batch = pieces.shape[0]
        rows = torch.arange(batch, device=pieces.device)
        moving = actions >= 0
        action = actions.clamp_min(0)
        origin, plane = action // PLANES_PER_SQUARE, action % PLANES_PER_SQUARE
        boards = torch.cat((pieces.long(), pieces.new_zeros(batch, 1).long()), dim=1)
        after, kind, _, victim, _, _ = rules._play(boards, rows, origin, plane)
        after = torch.where(moving[:, None], after, boards)
        child = rules.swap[after[:, :64].flip(1)]
        target = rules.destinations[origin, plane]
        rights = state[:, 2:6] > 0.5
        king = moving & (kind == 6)
        # Rights with the rook on file 0 and file 7, the mover's first, as ``encode`` orders them.
        own_west = rights[:, 0] & ~(king | (moving & (origin == 0)))
        own_east = rights[:, 1] & ~(king | (moving & (origin == 7)))
        their_west = rights[:, 2] & ~(moving & (target == 56))
        their_east = rights[:, 3] & ~(moving & (target == 63))
        double = moving & (kind == 1) & (target - origin == 16)
        # Rotation moves file f to 7 - f, so each side's rights swap between the files.
        castling = torch.stack((their_east, their_west, own_east, own_west), dim=1)
        ep_file = torch.where(double, 7 - origin % 8, NO_EN_PASSANT_FILE)
        bucket = state[:, 15:26].argmax(dim=1)
        bucket = torch.where(moving & ((kind == 1) | (victim != 0)), 0, bucket)
        after_state = torch.zeros_like(state)
        after_state[:, 0], after_state[:, 1] = state[:, 1], state[:, 0]
        after_state[:, 2:6] = castling.to(state.dtype)
        after_state[rows, 6 + ep_file] = 1
        after_state[rows, 15 + bucket] = 1
        return child, after_state

    def reply(self, pieces: Tensor, state: Tensor, actions: Tensor, following: Tensor) -> Tensor:
        """The opponent's action between a position, its ``actions`` and ``following``.

        ``following`` holds the pieces two plies later, back in the first mover's frame. -1
        where no single legal-looking reply joins them. Castling is read from the king's squares
        and an underpromotion from the piece that appears.
        """

        child, child_state = self.child(pieces, state, actions)
        # The later position seen from the replying side.
        seen = self.moves.swap[following.long().flip(1)]
        own_before = (child >= 1) & (child <= 6)
        own_after = (seen >= 1) & (seen <= 6)
        left = own_before & (seen != child)
        arrived = own_after & (seen != child)
        # Castling moves two pieces; the king's squares name the move.
        origin = (left.long() * (1 + 8 * (child == 6).long())).argmax(dim=1)
        target = (arrived.long() * (1 + 8 * (seen == 6).long())).argmax(dim=1)
        rows = torch.arange(len(child), device=child.device)
        pawn = child[rows, origin] == 1
        promoted = seen[rows, target]
        under = pawn & (target // 8 == 7) & (promoted >= 2) & (promoted <= 4)
        step = target % 8 - origin % 8 + 1
        underpromotion = origin * PLANES_PER_SQUARE + 64 + (promoted - 2) * 3 + step
        found = torch.where(under, underpromotion, self.pairs[origin, target])
        found = torch.where(left.any(dim=1) & arrived.any(dim=1), found, -1)
        replayed, _ = self.child(child, child_state, found)
        exact = (replayed == following.long()).all(dim=1) & (found >= 0)
        return torch.where(exact, found, -1)


@cache
def cpu_rules() -> LineRules:
    """One CPU instance for the data pipeline, since building the tables takes a moment."""

    return LineRules()


def one_hot_actions(actions: Tensor, size: int) -> Tensor:
    """(batch, size) bool marking each row's action; nothing for -1."""

    marked = nn.functional.one_hot(actions.clamp_min(0), size).bool()
    return marked & (actions >= 0)[:, None]


def own_moves(logits: Tensor, legal: Tensor) -> Tensor:
    """Each row's highest scoring legal action, -1 where it has none."""

    masked = logits.float().masked_fill(~legal, float("-inf"))
    return torch.where(legal.any(dim=1), masked.argmax(dim=1), -1)
