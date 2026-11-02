"""Reasoner-style core: a writable board memory M, a small workspace H and imagined lines.

Lines of ``line_plies`` plies follow one another, each starting from the present position: M
is reset to the encoded present board B (64 square tokens and the state token) and the model
picks each ply's action itself. A played action is a write that should turn M into the child's
encoding; after each ply M is rotated into the child's frame. H is a few learned tokens that
read B, M and themselves every step and persist across lines, so they can compare them. The
answer is read from B's tokens after they read H.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from wombat.chess.moves import PLANES_PER_SQUARE
from wombat.model.lines import LineRules

if TYPE_CHECKING:
    from wombat.model.network import NetworkConfig

# Destination marks: no promotion or a queen, then knight, bishop and rook underpromotions.
PROMOTION_KINDS = 4


class WorkspaceBlock(nn.Module):
    """Pre-norm attention of queries over keys with QK-norm, then a SwiGLU; identity at start.

    With ``square_bias`` the first 64 queries and keys are squares and share a learned per-head
    64 x 64 logit bias, starting at zero.
    """

    def __init__(self, config: NetworkConfig, *, square_bias: bool) -> None:
        super().__init__()
        from wombat.model.network import SwiGLU

        width, heads = config.d_model, config.attention_heads
        self.heads = heads
        self.query_input_norm = nn.RMSNorm(width)
        self.key_input_norm = nn.RMSNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.query_norm = nn.RMSNorm(width // heads)
        self.key_norm = nn.RMSNorm(width // heads)
        self.output = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.RMSNorm(width)
        self.mlp = SwiGLU(width, config.mlp_width)
        self.square_bias = nn.Parameter(torch.zeros(heads, 64, 64)) if square_bias else None
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.mlp.output.weight)

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """Queries updated by reading ``keys``, which include the queries to self-attend."""

        normed = self.key_input_norm(keys)
        q = self.query(self.query_input_norm(queries)).unflatten(-1, (self.heads, -1))
        k = self.key(normed).unflatten(-1, (self.heads, -1))
        v = self.value(normed).unflatten(-1, (self.heads, -1))
        q, k = self.query_norm(q), self.key_norm(k)
        bias = None
        if self.square_bias is not None:
            rows, columns = queries.shape[1] - 64, keys.shape[1] - 64
            bias = nn.functional.pad(self.square_bias, (0, columns, 0, rows)).to(q.dtype)
        mixed = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias
        )
        queries = queries + self.output(mixed.transpose(1, 2).flatten(-2))
        return queries + self.mlp(self.mlp_norm(queries))


class WorkspaceCore(nn.Module):
    """The parts of M and H that the network's encoder and heads do not provide."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        width = config.d_model
        self.steps_per_ply = config.steps_per_ply
        self.initial_workspace = nn.Parameter(torch.empty(config.workspace_tokens, width))
        # Added to the logit of a first move an earlier line already tried; from zero.
        self.repeat_bias = nn.Parameter(torch.zeros(()))
        blocks = range(config.recurrent_blocks)
        square_bias = config.square_attention_bias
        self.workspace_blocks = nn.ModuleList(
            WorkspaceBlock(config, square_bias=False) for _ in blocks
        )
        self.memory_blocks = nn.ModuleList(
            WorkspaceBlock(config, square_bias=square_bias) for _ in blocks
        )
        self.readout_blocks = nn.ModuleList(
            WorkspaceBlock(config, square_bias=square_bias) for _ in range(config.readout_blocks)
        )
        self.workspace_norm = nn.RMSNorm(width) if config.normalize_state else None
        # Marks written on the moving piece's square and its destination, at embedding scale.
        self.move_origin = nn.Parameter(torch.randn(width))
        self.move_destination = nn.Parameter(torch.randn(PROMOTION_KINDS, width))
        self.move_workspace = nn.Linear(2 * width, width)
        self.rules = LineRules()
        nn.init.normal_(self.initial_workspace, std=0.02)

    def start(self, board_memory: Tensor) -> Tensor:
        """H0 for each position."""

        batch = board_memory.shape[0]
        return self.initial_workspace.unsqueeze(0).expand(batch, -1, -1)

    def phase(self, step: int, line_plies: int) -> tuple[bool, int, bool, bool]:
        """Where 0-based ``step`` falls in lines of ``line_plies`` plies: whether it starts a
        line, its ply in the line, and whether it starts and ends that ply."""

        within = step % (line_plies * self.steps_per_ply)
        ply, offset = divmod(within, self.steps_per_ply)
        return within == 0, ply, offset == 0, offset == self.steps_per_ply - 1

    def first_logits(self, logits: Tensor, tried: Tensor) -> Tensor:
        """A line's first-move logits, with the repeat bias on the moves earlier lines tried."""

        return logits + self.repeat_bias * tried

    def play(self, memory: Tensor, workspace: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Mark each row's action on M and tell H its two squares; -1 plays nothing."""

        moving = (actions >= 0)[:, None, None]
        action = actions.clamp_min(0)
        origin, plane = action // PLANES_PER_SQUARE, action % PLANES_PER_SQUARE
        target = self.rules.moves.destinations[origin, plane]
        promotion = torch.where(plane >= 64, 1 + (plane - 64) // 3, 0)
        rows = torch.arange(len(actions), device=actions.device)
        squares = torch.cat((memory[rows, origin], memory[rows, target]), dim=-1)
        workspace = workspace + moving * self.move_workspace(squares).unsqueeze(1)
        marks = memory.new_zeros(memory.shape)
        marks = marks.index_put((rows, origin), self.move_origin.expand(len(rows), -1).to(marks))
        marks = marks.index_put((rows, target), self.move_destination[promotion].to(marks))
        return memory + moving * marks, workspace

    def step(
        self, board_memory: Tensor, memory: Tensor, workspace: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One shared step: H reads B, M and H, then M reads M and H, per block."""

        for workspace_block, memory_block in zip(self.workspace_blocks, self.memory_blocks):
            keys = torch.cat((board_memory.to(memory.dtype), memory, workspace.to(memory.dtype)), 1)
            workspace = workspace_block(workspace, keys)
            memory = memory_block(memory, torch.cat((memory, workspace.to(memory.dtype)), 1))
        if self.workspace_norm is not None:
            workspace = self.workspace_norm(workspace)
        return memory, workspace

    def rotate(self, memory: Tensor, square_embedding: Tensor) -> Tensor:
        """M in the child's frame: squares rotated 180 degrees, each re-addressed to its square.

        The token landing on square j came from 63 - j, so it swaps that square's embedding for
        j's; the state token stays.
        """

        moved = square_embedding - square_embedding.flip(0)
        squares = memory[:, :64].flip(1) + moved.to(memory.dtype)
        return torch.cat((squares, memory[:, 64:]), dim=1)

    def read(self, tokens: Tensor, workspace: Tensor) -> Tensor:
        """Board tokens (B or M) after they read H through the readout blocks."""

        for block in self.readout_blocks:
            tokens = block(tokens, torch.cat((tokens, workspace.to(tokens.dtype)), dim=1))
        return tokens
