"""Small weight-tied recurrent transformer for Wombat v1."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from wombat.chess.moves import (
    KNIGHT_DELTAS,
    PLANES_PER_SQUARE,
    RAY_DIRECTIONS,
    UNDERPROMOTION_PIECES,
)
from wombat.model.board_features import ATTACK_CODES, AttackPairs, BoardFeatures, MoveGraph
from wombat.model.lines import one_hot_actions, own_moves
from wombat.model.move_consequences import CONSEQUENCES, MoveConsequences
from wombat.model.workspace import WorkspaceCore

# Width of the origin queries and destination keys of the from_to move and pair reply readouts.
MOVE_KEY_WIDTH = 64
# Local neighbourhoods computed per position from the pieces.
MOVE_NEIGHBOURHOODS = ("move_graph", "move_lines")
# Relative attention's key and value vectors: the nine king offsets, then non-square keys.
RELATIVE_OFFSETS = 10
# Channels per square token the state attention bias compresses the board to (smolgen's 32).
STATE_BIAS_SQUARE_WIDTH = 8
# Non-pawn piece counts the phase buckets split evenly; the last bucket also takes any above.
PHASE_PIECES = 16
# (file, rank) offsets between two squares, -7..7 on each axis.
KING_OFFSETS = 15 * 15


@dataclass(frozen=True)
class NetworkConfig:
    d_model: int = 192
    attention_heads: int = 6
    prelude_blocks: int = 3
    recurrent_blocks: int = 2
    mlp_width: int = 768
    latent_tokens: int = 32
    # None disables the per-token residual cap.
    max_residual_update: float | None = 0.25
    # "global" reads all 4,672 logits from the state token; "square" has each
    # square token emit the 73 logits of moves leaving that square.
    policy_readout: str = "global"
    # With the square readout, "from_to" adds a query-key score between each move's origin and
    # destination square tokens to its logit and its move value.
    move_readout: str = "square"
    # RMS-normalize the whole hidden state after every recurrent step.
    normalize_state: bool = False
    use_update_gate: bool = False
    use_scratchpad_backup: bool = False
    use_future_workspaces: bool = False
    future_lines: int = 8
    future_horizon: int = 4
    dropout: float = 0.0
    # Learned per-head 64 x 64 attention bias between square tokens in the recurrent core.
    square_attention_bias: bool = False
    # Heads on the square tokens predicting the line's reply squares and future boards.
    lookahead_heads: bool = False
    # "squares" predicts the reply's origin and destination separately; "pair" scores all
    # 64 x 64 (origin, destination) pairs jointly (needs lookahead_heads).
    reply_readout: str = "squares"
    # Boards of 64 position-independent scratch tokens appended to H; no head reads them.
    scratch_boards: int = 0
    # The same extra tokens without square embeddings or square bias (the v8-flat control).
    flat_scratch: bool = False
    # Reset the square and state tokens to the board encoding before every step, so only
    # latent and scratch tokens carry anything from one step to the next.
    refresh_board: bool = False
    # Recurrent square tokens see only squares within one king step, and no other H token,
    # so information crosses the board one square per block (needs the square bias).
    local_attention: bool = False
    # Squares local attention reaches: "king" steps, "king_knight" adding knight jumps,
    # "move_graph" adding each position's attacks and pushes in both directions (see
    # WombatNetwork.local_edges), or "move_lines" whose sliders see through blockers, with a
    # learned per-head bias for 0, 1 or 2+ pieces between the two squares of a line.
    local_neighbourhood: str = "king"
    # Add a learned projection of each square's attack, pin and check features to its token.
    board_features: bool = False
    # Also project sole defenders, checking squares and lines, discovered checks, x-rays and
    # material (needs board_features).
    board_features_long_range: bool = False
    # Add each move's consequences one move later (see ``MoveConsequences``) to its logit and move
    # value, weighted per move from its origin and destination tokens (needs the square readout).
    move_consequences: bool = False
    # Shaw relative positions in local king attention: square queries add q . a_k[offset] to each
    # logit and the a_v[offset] of the keys they read, per block and head, from zero. The offsets
    # are the nine king steps between two squares of the board memory or H; every non-square key
    # a square query reads (the board's state token) shares a tenth vector.
    relative_attention: bool = False
    # Smolgen (Leela) or GAB (Chessformer): each recurrent block turns the square tokens of H into
    # per-head weights over 64 x 64 templates shared by all blocks, added to the square pair logits
    # wherever the square table applies, at every step. Under local attention only unmasked pairs
    # (a square's king neighbours) are affected. The templates start at zero.
    state_attention_bias: bool = False
    # Width of each block's compressed board vector and of each head's template weights.
    state_attention_bias_width: int = 32
    # RMS-normalize each head's queries and keys before their dot product, which bounds the
    # attention logits (ViT-22B); applied through the square attention bias path.
    qk_norm: bool = False
    # With K > 0 the position value and every move value are K logits over equal win-chance bins
    # on [0, 1] (ChessBench), read out as their expected value on the -1..1 scale; 0 keeps tanh.
    value_bins: int = 0
    # Per block and head a learned logit, from zero, on each pair of square tokens where the piece
    # on one attacks or defends the piece on the other, indexed by both pieces (see
    # ``AttackPairs``), one table for the attacker's query and one for the target's. Local
    # attention also unmasks those pairs, and relative attention gives them no offset term.
    attack_pairs: bool = False
    # With K > 0 the last layer of every policy, move value and value readout of the square
    # readout holds K copies, all starting equal, and each position reads the copy of its phase
    # bucket by non-pawn pieces (see ``phase_buckets``); the layers below are shared.
    phase_heads: int = 0
    # Add to each square token a learned embedding of its (file, rank) offset from the side to
    # move's king and one of its offset from the opponent's king (see ``king_offsets``), from zero.
    king_relative: bool = False
    # "workspace" replaces the recurrent core with a writable board memory M, a workspace of
    # ``workspace_tokens`` tokens and imagined lines of ``line_plies`` plies, ``steps_per_ply``
    # steps each, one after another from the present position (see ``wombat.model.workspace``);
    # ``recurrent_blocks`` then counts its block pairs per step.
    core: str = "recurrent"
    workspace_tokens: int = 12
    steps_per_ply: int = 2
    line_plies: int = 2
    # Blocks in which the board tokens read the workspace before the heads.
    readout_blocks: int = 1


CORES = ("recurrent", "workspace")
# Options of the recurrent core that the workspace core has no part for; its blocks always
# normalise queries and keys.
RECURRENT_ONLY = (
    "use_update_gate",
    "use_scratchpad_backup",
    "use_future_workspaces",
    "refresh_board",
    "local_attention",
    "relative_attention",
    "state_attention_bias",
    "qk_norm",
    "flat_scratch",
    "attack_pairs",
    "phase_heads",
)


def _check_core(config: NetworkConfig) -> None:
    if config.core not in CORES:
        raise ValueError(f"unknown core: {config.core}")
    if config.core == "recurrent":
        return
    if config.move_readout != "from_to":
        raise ValueError("the workspace core reads moves with the from_to readout")
    if config.latent_tokens or config.scratch_boards:
        raise ValueError("the workspace core keeps its plan in workspace tokens only")
    for name in RECURRENT_ONLY:
        if getattr(config, name):
            raise ValueError(f"{name} belongs to the recurrent core")
    sizes = (config.workspace_tokens, config.steps_per_ply, config.line_plies)
    if min(*sizes, config.readout_blocks, config.recurrent_blocks) < 1:
        raise ValueError("the workspace core needs tokens, steps per ply and blocks")


def value_bin_values(bins: int) -> Tensor:
    """(bins,) centres of ``bins`` equal win-chance bins on [0, 1], on the -1..1 value scale."""

    return (torch.arange(bins, dtype=torch.float32) + 0.5) / bins * 2 - 1


def expected_value(logits: Tensor, values: Tensor) -> Tensor:
    """Float32 expected value of (..., bins) logits whose bins hold ``values``."""

    # Elementwise, so autocast keeps the sum in float32.
    return (logits.float().softmax(dim=-1) * values).sum(dim=-1)


def phase_buckets(pieces: Tensor, buckets: int) -> Tensor:
    """(batch,) phase bucket of each position of (batch, 64) piece ids.

    A position with n knights, bishops, rooks and queens of both sides is in bucket
    min(n * buckets // 16, buckets - 1): with 4 buckets 0 to 3, 4 to 7, 8 to 11, and 12 or more.
    """

    kind = pieces - 6 * (pieces > 6).long()
    count = ((kind >= 2) & (kind <= 5)).sum(dim=1)
    return (count * buckets // PHASE_PIECES).clamp_max(buckets - 1)


def king_offsets(pieces: Tensor) -> Tensor:
    """(batch, 2, 64) offset of every square from the side to move's king, then the opponent's.

    Offsets are in the side-to-move frame, coded 15 (rank step + 7) + file step + 7 so each axis
    spans -7..7. A position without that king measures from square 0.
    """

    squares = torch.arange(64, device=pieces.device)
    kings = torch.stack(((pieces == 6).long().argmax(1), (pieces == 12).long().argmax(1)), dim=1)
    file_steps = squares % 8 - (kings % 8).unsqueeze(-1)
    rank_steps = squares // 8 - (kings // 8).unsqueeze(-1)
    return 15 * (rank_steps + 7) + file_steps + 7


def _king_neighbourhood(knight: bool = False) -> Tensor:
    """(64, 64) additive mask: 0 between squares at most one king step apart, else -inf.

    With ``knight`` squares a knight's jump apart are also 0.
    """

    files, ranks = torch.arange(64) % 8, torch.arange(64) // 8
    file_steps = (files[:, None] - files[None]).abs()
    rank_steps = (ranks[:, None] - ranks[None]).abs()
    near = (file_steps <= 1) & (rank_steps <= 1)
    if knight:
        near |= ((file_steps == 1) & (rank_steps == 2)) | ((file_steps == 2) & (rank_steps == 1))
    return torch.zeros(64, 64).masked_fill(~near, float("-inf"))


def _relative_tables() -> tuple[Tensor, Tensor, Tensor]:
    """Tables letting one attention call add relative terms for king offsets.

    A square's class is (rank mod 3, file mod 3), and 9 is every non-square token; within one
    king step a key's class fixes its offset from the query. Offset (file step df, rank step dr)
    is 3 * (dr + 1) + df + 1, so 4 is the square itself and 9 is a non-square key. Returns the
    (65, 10) one-hot class codes of the squares and then of a non-square token, and per query
    square the (64, 10) offset of each class and class of each offset.
    """

    codes = torch.zeros(65, RELATIVE_OFFSETS)
    class_offsets = torch.full((64, RELATIVE_OFFSETS), RELATIVE_OFFSETS - 1)
    offset_classes = torch.full((64, RELATIVE_OFFSETS), RELATIVE_OFFSETS - 1)
    for square in range(64):
        file, rank = square % 8, square // 8
        codes[square, 3 * (rank % 3) + file % 3] = 1
        for rank_step in (-1, 0, 1):
            for file_step in (-1, 0, 1):
                offset = 3 * (rank_step + 1) + file_step + 1
                target = 3 * ((rank + rank_step) % 3) + (file + file_step) % 3
                offset_classes[square, offset] = target
                class_offsets[square, target] = offset
    codes[64, RELATIVE_OFFSETS - 1] = 1
    return codes, class_offsets, offset_classes


def _square_relative_tables() -> tuple[Tensor, Tensor, Tensor]:
    """Relative attention tables by key square, for attack pairs that open far square pairs.

    Returns the (65, 65) one-hot codes of the squares and then of a non-square token, the
    (64, 65) offset (as in ``_relative_tables``) of each code from each query square, 10 past
    one king step, and those offsets one-hot as (64, 65, 10), zero past one king step.
    """

    files, ranks = torch.arange(64) % 8, torch.arange(64) // 8
    file_steps = files[None] - files[:, None]
    rank_steps = ranks[None] - ranks[:, None]
    near = (file_steps.abs() <= 1) & (rank_steps.abs() <= 1)
    offsets = torch.where(near, 3 * (rank_steps + 1) + file_steps + 1, RELATIVE_OFFSETS)
    offsets = torch.cat((offsets, torch.full((64, 1), RELATIVE_OFFSETS - 1)), dim=1)
    one_hot = nn.functional.one_hot(offsets, RELATIVE_OFFSETS + 1)[..., :RELATIVE_OFFSETS]
    return torch.eye(65), offsets, one_hot.float()


def _move_destinations() -> Tensor:
    """(64, 73) destination square of each action in the side-to-move frame.

    Off-board destinations point at the origin square; those actions are never legal.
    """

    table = torch.arange(64).unsqueeze(1).repeat(1, PLANES_PER_SQUARE)
    deltas = [
        (unit_file * distance, unit_rank * distance)
        for unit_file, unit_rank in RAY_DIRECTIONS
        for distance in range(1, 8)
    ]
    deltas += list(KNIGHT_DELTAS)
    deltas += [(file_step, 1) for _ in UNDERPROMOTION_PIECES for file_step in (-1, 0, 1)]
    for square in range(64):
        file, rank = square % 8, square // 8
        for plane, (file_step, rank_step) in enumerate(deltas):
            if 0 <= file + file_step < 8 and 0 <= rank + rank_step < 8:
                table[square, plane] = square + file_step + 8 * rank_step
    return table


def _tile_squares(table: Tensor, segments: tuple[tuple[bool, int], ...], dim: int) -> Tensor:
    """Lay ``segments`` out along ``dim``: square spans repeat ``table``, the rest are zero.

    ``table`` spans the 64 squares along ``dim``; a square segment's length is a multiple of 64.
    """

    parts = []
    for square, length in segments:
        if not length:
            continue
        if square:
            repeats = [1] * table.ndim
            repeats[dim] = length // 64
            parts.append(table.repeat(repeats))
        else:
            shape = list(table.shape)
            shape[dim] = length
            parts.append(table.new_zeros(shape))
    return torch.cat(parts, dim=dim)


@dataclass(frozen=True)
class ReasoningStepTrace:
    step: int
    hidden: Tensor
    policy_logits: Tensor
    value: Tensor
    attention_maps: tuple[Tensor, ...]


@dataclass(frozen=True)
class MoveValueBins:
    """Every move's value logits over the value bins, kept factored.

    A move's logits are its origin's and its destination's rows of the (batch, 64, bins)
    ``origin`` and ``destination`` plus its scalar move value score (before the tanh of the
    scalar head) times ``tilt``. ``destinations`` is the (64, 73) destination of each action.
    Two (width, bins) projections stand in for a per-plane (width, 73 x bins) one; with the
    value head's wider output K bins add 3 (width + 1) K - (width + 1) parameters, 73,535 at
    width 384 and 64 bins.
    """

    origin: Tensor
    destination: Tensor
    scores: Tensor
    tilt: Tensor
    destinations: Tensor
    values: Tensor

    def logits(self, actions: Tensor | None = None) -> Tensor:
        """(batch, moves, bins) logits of the (batch, moves) ``actions``, or of all 4,672."""

        if actions is None:
            return torch.cat([self._rank_logits(rank) for rank in range(8)], dim=1)
        rows = torch.arange(actions.shape[0], device=actions.device).unsqueeze(1)
        origin = self.origin[rows, actions // PLANES_PER_SQUARE]
        destination = self.destination[rows, self.destinations.flatten()[actions]]
        scores = self.scores.flatten(1).gather(1, actions)
        return origin + destination + scores.unsqueeze(-1) * self.tilt

    def _rank_logits(self, rank: int) -> Tensor:
        """(batch, 8 x 73, bins) logits of the moves leaving one rank's squares."""

        squares = slice(8 * rank, 8 * rank + 8)
        destination = self.destination[:, self.destinations[squares]]
        pairs = (self.origin[:, squares].unsqueeze(2) + destination).flatten(1, 2)
        return pairs + self.scores[:, squares].flatten(1).unsqueeze(-1) * self.tilt

    def expected(self) -> Tensor:
        """(batch, 4672) expected move values on the -1..1 scale."""

        # A rank of origins at a time, so a large batch never holds every move's bins at once.
        ranks = [expected_value(self._rank_logits(rank), self.values) for rank in range(8)]
        return torch.cat(ranks, dim=1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, width: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, width * 2, bias=False)
        self.output = nn.Linear(width, d_model, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, values = self.input(inputs).chunk(2, dim=-1)
        return self.output(torch.nn.functional.silu(gate) * values)


class StateAttentionBias(nn.Module):
    """A block's smolgen: per-head weights for the shared 64 x 64 templates, from the squares."""

    def __init__(self, d_model: int, heads: int, width: int) -> None:
        super().__init__()
        self.compress = nn.Linear(d_model, STATE_BIAS_SQUARE_WIDTH, bias=False)
        board = nn.Linear(64 * STATE_BIAS_SQUARE_WIDTH, width)
        self.board = nn.Sequential(board, nn.GELU(), nn.LayerNorm(width))
        weights = nn.Linear(width, heads * width)
        self.weights = nn.Sequential(weights, nn.GELU(), nn.LayerNorm(heads * width))
        self.heads = heads

    def forward(self, squares: Tensor, templates: Tensor) -> Tensor:
        """(batch, heads, 64, 64) square pair logits.

        ``squares`` are the (batch, 64, d_model) square tokens; ``templates`` is the shared
        (4096, width) weight whose rows are the flattened (query, key) square pairs.
        """

        board = self.board(self.compress(squares).flatten(1))
        weights = self.weights(board).unflatten(-1, (self.heads, -1))
        return nn.functional.linear(weights, templates).unflatten(-1, (64, 64))


class RecurrentBlock(nn.Module):
    """Pre-norm residual transformer implementing one part of F(B, H)."""

    king_mask: Tensor
    relative_codes: Tensor
    class_offsets: Tensor
    offset_classes: Tensor
    square_offsets: Tensor
    square_offset_codes: Tensor

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_norm = nn.RMSNorm(config.d_model)
        self.memory_norm = nn.RMSNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.mlp_width)
        self.update_gate = nn.Linear(config.d_model, 2)
        if config.square_attention_bias:
            # Chess topology as an additive logit per head and square pair, starting neutral.
            self.square_bias = nn.Parameter(torch.zeros(config.attention_heads, 64, 64))
        if config.local_attention and config.local_neighbourhood not in MOVE_NEIGHBOURHOODS:
            # A buffer moves to the model's device, so graph capture never copies from the CPU.
            knight = config.local_neighbourhood == "king_knight"
            self.register_buffer("king_mask", _king_neighbourhood(knight), persistent=False)
        if config.local_neighbourhood == "move_lines":
            # Per head, a logit for 0, 1 or 2+ pieces between two squares of a line, from neutral.
            self.blocker_bias = nn.Parameter(torch.zeros(config.attention_heads, 3))
        if config.relative_attention:
            heads = config.attention_heads
            shape = (heads, RELATIVE_OFFSETS, config.d_model // heads)
            self.relative_keys = nn.Parameter(torch.zeros(shape))
            self.relative_values = nn.Parameter(torch.zeros(shape))
            names = ("relative_codes", "class_offsets", "offset_classes")
            tables = _relative_tables()
            if config.attack_pairs:
                names = ("relative_codes", "square_offsets", "square_offset_codes")
                tables = _square_relative_tables()
            for name, table in zip(names, tables):
                self.register_buffer(name, table, persistent=False)
        if config.attack_pairs:
            # Per head, a logit per attack code for the attacker's query reading the target and
            # for the target's reading the attacker, from neutral.
            self.attack_bias = nn.Parameter(torch.zeros(2, config.attention_heads, ATTACK_CODES))
        if config.state_attention_bias:
            self.state_bias = StateAttentionBias(
                config.d_model, config.attention_heads, config.state_attention_bias_width
            )
        if config.qk_norm:
            head_width = config.d_model // config.attention_heads
            self.query_norm = nn.RMSNorm(head_width)
            self.key_norm = nn.RMSNorm(head_width)
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        """Make the residual branch initially preserve H exactly."""

        nn.init.zeros_(self.attention.out_proj.weight)
        if self.attention.out_proj.bias is not None:
            nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.mlp.output.weight)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.constant_(self.update_gate.bias, -1.0)

    def _bound_update(self, update: Tensor) -> Tensor:
        """Smoothly cap each token's RMS update without changing its direction."""

        limit = self.config.max_residual_update
        if limit is None:
            return update
        if limit <= 0:
            raise ValueError("max_residual_update must be positive")
        # Compute the norm in fp32: the recurrent core commonly runs under
        # fp16 autocast and squaring a large activation there can overflow.
        # Clamp before sqrt so an exact identity initialization has a finite
        # derivative (sqrt'(0) is otherwise infinite and yields NaN gradients).
        rms = update.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        scale = torch.tanh(rms / limit) * limit / rms
        return update * scale.to(update.dtype)

    def attention_bias(
        self,
        board_tokens: int,
        hidden_tokens: int,
        dtype: torch.dtype | None = None,
        local_edges: Tensor | None = None,
        state_bias: Tensor | None = None,
    ) -> Tensor:
        """The (heads, H tokens, memory tokens) logit bias, the square table on every square pair.

        Square tokens are the board memory's squares, H's squares and the scratch boards' squares.
        The move graph's or the attack pairs' (batch, 64, 64) ``local_edges`` makes it (batch,
        heads, H, memory), as does a (batch, heads, 64, 64) ``state_bias`` added to the table
        before any mask.
        """

        config = self.config
        scratch = 64 * config.scratch_boards
        if hidden_tokens < 64 + scratch:
            raise ValueError(f"H has {hidden_tokens} tokens, fewer than its squares and boards")
        # H is its squares, then the state and latent tokens, then the scratch boards.
        rows = (
            (True, 64),
            (False, hidden_tokens - 64 - scratch),
            (not config.flat_scratch, scratch),
        )
        columns = ((True, 64), (False, board_tokens - 64), *rows)
        square = self.square_bias if dtype is None else self.square_bias.to(dtype)
        if state_bias is not None:
            square = square + state_bias.to(square.dtype)
        if config.attack_pairs:
            if local_edges is None:
                raise ValueError("attack pairs need the attack codes of their positions")
            square = square + self._pair_bias(local_edges, square.dtype)
        if config.local_attention:
            if config.local_neighbourhood in MOVE_NEIGHBOURHOODS:
                if local_edges is None:
                    raise ValueError("move graph attention needs the local edges of its positions")
                square = square + self._edge_bias(local_edges, square.dtype)
            else:
                mask = self.king_mask.to(square.dtype)
                if local_edges is not None and config.attack_pairs:
                    # Either piece of a pair reads the other's squares, however far apart.
                    paired = (local_edges > 0) | (local_edges.transpose(1, 2) > 0)
                    mask = mask.masked_fill(paired.unsqueeze(1), 0)
                square = square + mask
        bias = _tile_squares(_tile_squares(square, columns, -1), rows, -2)
        if config.local_attention:
            # Square queries may not read H's state or latent tokens, a global shortcut.
            bias = bias.clone()
            latent = slice(board_tokens + 64, board_tokens + hidden_tokens - scratch)
            bias[..., :64, latent] = float("-inf")
        return bias

    def _edge_bias(self, local_edges: Tensor, dtype: torch.dtype) -> Tensor:
        """(batch, heads, 64, 64) logits: -inf off the move graph, the blocker bias on lines."""

        blocked = torch.zeros(local_edges.shape, dtype=dtype, device=local_edges.device)
        blocked = blocked.masked_fill(local_edges == 0, float("-inf")).unsqueeze(1)
        if self.config.local_neighbourhood != "move_lines":
            return blocked
        # Codes 0 and 1 add nothing; line codes 2, 3 and 4 read the heads' blocker bias.
        table = nn.functional.pad(self.blocker_bias.to(dtype), (2, 0))
        return blocked + nn.functional.embedding(local_edges, table.T).permute(0, 3, 1, 2)

    def _pair_bias(self, codes: Tensor, dtype: torch.dtype) -> Tensor:
        """(batch, heads, 64, 64) logits of the (batch, attacker, target) attack ``codes``."""

        # Code 0, no pair, reads a zero column.
        table = nn.functional.pad(self.attack_bias.to(dtype), (1, 0))
        attacker = nn.functional.embedding(codes, table[0].T)
        target = nn.functional.embedding(codes.transpose(1, 2), table[1].T)
        return (attacker + target).permute(0, 3, 1, 2)

    def _biased_attention(
        self,
        queries: Tensor,
        memory: Tensor,
        capture_attention: bool,
        local_edges: Tensor | None,
        state_templates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """The block's multi-head attention with the square bias, reusing its projections.

        With the state attention bias ``state_templates`` is the network's shared template weight.
        """

        attention = self.attention
        heads = attention.num_heads
        width = queries.shape[-1]
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        q, k, v = (
            nn.functional.linear(
                source,
                weight[index * width : (index + 1) * width],
                None if bias is None else bias[index * width : (index + 1) * width],
            )
            for index, source in enumerate((queries, memory, memory))
        )

        head_width = width // heads
        # (batch, tokens, heads, head width); the attention reads the heads-first transposes.
        q, k, v = (tensor.unflatten(-1, (heads, head_width)) for tensor in (q, k, v))
        if self.config.qk_norm:
            q, k = self.query_norm(q), self.key_norm(k)
        hidden_tokens = queries.shape[1]
        board_tokens = memory.shape[1] - hidden_tokens
        state_bias = None
        if self.config.state_attention_bias:
            if state_templates is None:
                raise ValueError("the state attention bias needs the network's shared templates")
            state_bias = self.state_bias(queries[:, :64], state_templates)
        logits_bias = self.attention_bias(
            board_tokens, hidden_tokens, q.dtype, local_edges, state_bias
        )
        scale = None
        if self.config.relative_attention:
            q, k, v = self._with_relative_codes(q, k, v, board_tokens)
            scale = 1 / math.sqrt(head_width)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if capture_attention:
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_width) + logits_bias
            probabilities = scores.softmax(dim=-1)
            mixed = probabilities @ v
        else:
            # CPU SDPA sends a per-position mask to its flash kernel and a shared one to math;
            # math keeps a zero state bias bit-identical to the model without it.
            backend = SDPBackend.MATH if state_bias is not None and q.device.type == "cpu" else None
            with nullcontext() if backend is None else sdpa_kernel(backend):
                mixed = nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=logits_bias, scale=scale
                )
            probabilities = None
        if self.config.relative_attention:
            mixed = self._add_relative_values(mixed, head_width)
        return attention.out_proj(mixed.transpose(1, 2).flatten(-2)), probabilities

    def _with_relative_codes(
        self, q: Tensor, k: Tensor, v: Tensor, board_tokens: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Q, K and V, (batch, tokens, heads, width), widened so one attention call adds the
        relative key term.

        Keys and values gain their class code (see ``_relative_tables``) and square queries their
        a_k scores in class order, so each square pair's logit gains q . a_k[offset] / sqrt(width)
        and the mixed code columns hold each query's attention per class. Zero columns pad the
        head width to a multiple of 8. With attack pairs, whose far keys a class cannot place,
        the codes are the key's square (see ``_square_relative_tables``) instead of its class.
        """

        batch, hidden_tokens, heads, width = q.shape
        code_width = self.relative_codes.shape[1]
        padding = -(width + code_width) % 8
        other = self.relative_codes[64]
        columns = ((True, 64), (False, board_tokens - 64), (True, 64), (False, hidden_tokens - 64))
        codes = _tile_squares(self.relative_codes[:64] - other, columns, 0) + other
        codes = nn.functional.pad(codes.to(q.dtype), (0, padding))
        codes = codes.unsqueeze(1).expand(batch, -1, heads, -1)
        keys = self.relative_keys.to(q.dtype)
        scores = torch.einsum("bshd,hcd->bshc", q[:, :64], keys)
        if self.config.attack_pairs:
            # Offset 10, past one king step, reads a zero score.
            scores = nn.functional.pad(scores, (0, 1))
            index = self.square_offsets.unsqueeze(1).expand(batch, -1, heads, -1)
        else:
            index = self.class_offsets.unsqueeze(1).expand(batch, -1, heads, -1)
        rows = (0, padding, 0, 0, 0, hidden_tokens - 64)
        by_code = nn.functional.pad(scores.gather(-1, index), rows)
        return torch.cat((q, by_code), -1), torch.cat((k, codes), -1), torch.cat((v, codes), -1)

    def _add_relative_values(self, mixed: Tensor, width: int) -> Tensor:
        """The attention output without its code columns, square rows gaining sum p * a_v."""

        mixed = mixed.transpose(1, 2)
        batch, _, heads, _ = mixed.shape
        if self.config.attack_pairs:
            by_square = mixed[:, :64, :, width : width + 65]
            codes = self.square_offset_codes.to(mixed.dtype)
            by_offset = torch.einsum("bshk,sko->bsho", by_square, codes)
        else:
            index = self.offset_classes.unsqueeze(1).expand(batch, -1, heads, -1)
            by_offset = mixed[:, :64, :, width : width + RELATIVE_OFFSETS].gather(-1, index)
        values = torch.einsum("bshc,hcd->bshd", by_offset, self.relative_values.to(mixed.dtype))
        squares = mixed[:, :64, :, :width] + values
        return torch.cat((squares, mixed[:, 64:, :, :width]), dim=1).transpose(1, 2)

    def forward(
        self,
        board_memory: Tensor,
        hidden: Tensor,
        *,
        capture_attention: bool = False,
        local_edges: Tensor | None = None,
        state_templates: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        queries = self.hidden_norm(hidden)
        memory = torch.cat((self.memory_norm(board_memory), queries), dim=1)
        attention: Tensor | None
        if self.config.square_attention_bias:
            update, attention = self._biased_attention(
                queries, memory, capture_attention, local_edges, state_templates
            )
        else:
            update, attention = self.attention(
                queries,
                memory,
                memory,
                need_weights=capture_attention,
                average_attn_weights=False,
            )
        # Bound the whole per-token delta, rather than each coordinate. A
        # coordinate-wise cap still permits an O(sqrt(d_model)) state jump.
        bounded_attention = self._bound_update(update)
        if self.config.use_update_gate:
            attention_gate = torch.sigmoid(self.update_gate(queries)[..., :1])
            bounded_attention = bounded_attention * attention_gate
        hidden = hidden + bounded_attention
        normalized_hidden = self.mlp_norm(hidden)
        mlp_update = self.mlp(normalized_hidden)
        bounded_mlp = self._bound_update(mlp_update)
        if self.config.use_update_gate:
            mlp_gate = torch.sigmoid(self.update_gate(normalized_hidden)[..., 1:2])
            bounded_mlp = bounded_mlp * mlp_gate
        hidden = hidden + bounded_mlp
        if capture_attention:
            assert attention is not None
            return hidden, attention
        return hidden


class WombatNetwork(nn.Module):
    """Encode once, then reuse one recurrent core for every reasoning step."""

    move_destinations: Tensor
    value_bin_values: Tensor

    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        if config.use_future_workspaces and (
            config.future_lines * config.future_horizon > config.latent_tokens
        ):
            raise ValueError("future workspaces require future_lines * future_horizon latent tokens")
        if config.policy_readout not in ("global", "square"):
            raise ValueError(f"unknown policy_readout: {config.policy_readout}")
        if config.move_readout not in ("square", "from_to"):
            raise ValueError(f"unknown move_readout: {config.move_readout}")
        if config.move_readout == "from_to" and config.policy_readout != "square":
            raise ValueError("the from_to move readout needs the square policy readout")
        if config.reply_readout not in ("squares", "pair"):
            raise ValueError(f"unknown reply_readout: {config.reply_readout}")
        if config.reply_readout == "pair" and not (
            config.lookahead_heads and config.policy_readout == "square"
        ):
            raise ValueError("the pair reply readout needs lookahead heads and the square readout")
        if config.local_neighbourhood not in ("king", "king_knight", *MOVE_NEIGHBOURHOODS):
            raise ValueError(f"unknown local_neighbourhood: {config.local_neighbourhood}")
        if config.local_neighbourhood != "king" and not config.local_attention:
            raise ValueError("local_neighbourhood applies only to local attention")
        if config.local_attention and not config.square_attention_bias:
            raise ValueError("local attention is applied through the square attention bias")
        if config.relative_attention and not (
            config.local_attention and config.local_neighbourhood == "king"
        ):
            # Past one king step two keys of a square query could share a class.
            raise ValueError("relative attention needs local attention over king steps")
        if config.relative_attention and config.scratch_boards:
            raise ValueError("relative attention covers H's squares, not scratch boards")
        if config.state_attention_bias and not config.square_attention_bias:
            raise ValueError("the state attention bias is added through the square attention bias")
        if config.qk_norm and not config.square_attention_bias:
            raise ValueError("query and key normalization is applied through the square bias")
        if config.state_attention_bias_width < 1:
            raise ValueError("state_attention_bias_width must be positive")
        if config.attack_pairs and not config.square_attention_bias:
            raise ValueError("attack pairs are added through the square attention bias")
        if config.attack_pairs and config.local_neighbourhood in MOVE_NEIGHBOURHOODS:
            raise ValueError("attack pairs need a king neighbourhood, not the move graph")
        square_readout = config.policy_readout == "square"
        if square_readout and (config.use_future_workspaces or config.use_scratchpad_backup):
            raise ValueError("future workspaces and scratchpad backup need the global readout")
        if config.scratch_boards < 0:
            raise ValueError("scratch_boards cannot be negative")
        if config.flat_scratch and not config.scratch_boards:
            raise ValueError("flat_scratch needs scratch_boards")
        if config.board_features_long_range and not config.board_features:
            raise ValueError("board_features_long_range needs board_features")
        if config.move_consequences and not square_readout:
            raise ValueError("move consequences need the square policy readout")
        if config.value_bins < 0 or config.value_bins == 1:
            raise ValueError("value_bins must be 0 (scalar value heads) or at least 2")
        if config.value_bins and not square_readout:
            raise ValueError("value bins need the square policy readout")
        if config.phase_heads < 0:
            raise ValueError("phase_heads cannot be negative")
        if config.phase_heads and not square_readout:
            raise ValueError("phase heads need the square policy readout")
        _check_core(config)
        recurrent = config.core == "recurrent"
        self.config = config
        d_model = config.d_model
        self.piece_embedding = nn.Embedding(13, d_model)
        self.square_embedding = nn.Embedding(64, d_model)
        self.state_projection = nn.Sequential(nn.Linear(26, d_model), nn.GELU(), nn.LayerNorm(d_model))
        if config.board_features:
            # Its tables are buffers left out of the state dict; only the projection is learned.
            self.board_features = BoardFeatures(config.board_features_long_range)
            width = self.board_features.width
            self.board_feature_projection = nn.Linear(width, d_model, bias=False)
        if config.local_neighbourhood in MOVE_NEIGHBOURHOODS:
            self.move_graph = MoveGraph(lines=config.local_neighbourhood == "move_lines")
        if config.attack_pairs:
            self.attack_pairs = AttackPairs()
        if config.king_relative:
            # A vector per (file, rank) offset from each king; zero, so a fresh model is unchanged.
            self.own_king_offset = nn.Embedding(KING_OFFSETS, d_model)
            self.enemy_king_offset = nn.Embedding(KING_OFFSETS, d_model)
            nn.init.zeros_(self.own_king_offset.weight)
            nn.init.zeros_(self.enemy_king_offset.weight)
        self.prelude = nn.ModuleList(self._block() for _ in range(config.prelude_blocks))
        self.recurrent_core = nn.ModuleList(
            RecurrentBlock(config) for _ in range(config.recurrent_blocks if recurrent else 0)
        )
        if not recurrent:
            self.workspace_core = WorkspaceCore(config)
        if config.state_attention_bias:
            # Shared by every block, as in smolgen; zero, so a fresh model is unchanged.
            templates = nn.Linear(config.state_attention_bias_width, 64 * 64, bias=False)
            self.state_bias_templates = templates
            nn.init.zeros_(self.state_bias_templates.weight)
        self.initial_scratchpad = nn.Parameter(torch.empty(config.latent_tokens, d_model))
        if config.scratch_boards:
            self.scratch_board_embedding = nn.Parameter(torch.empty(config.scratch_boards, d_model))
            self.scratch_content = nn.Parameter(torch.empty(d_model))
            if config.flat_scratch:
                # Stands in for the square embedding, one vector per extra token.
                self.flat_scratch_embedding = nn.Parameter(
                    torch.empty(64 * config.scratch_boards, d_model)
                )
        if config.normalize_state and recurrent:
            self.state_norm = nn.RMSNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        # The phase heads' output layers stack their copies along the outputs.
        copies = max(config.phase_heads, 1)
        if square_readout:
            self.square_policy_head = nn.Linear(d_model, 73 * copies)
            move_value = nn.Linear(d_model, 73 * copies)
            self.square_move_value_head = nn.Sequential(move_value, nn.Tanh())
            if config.move_readout == "from_to":
                self.policy_query = nn.Linear(d_model, MOVE_KEY_WIDTH * copies)
                self.policy_key = nn.Linear(d_model, MOVE_KEY_WIDTH * copies)
                self.move_value_query = nn.Linear(d_model, MOVE_KEY_WIDTH * copies)
                self.move_value_key = nn.Linear(d_model, MOVE_KEY_WIDTH * copies)
            if config.move_readout == "from_to" or config.move_consequences or config.value_bins:
                # A buffer moves to the model's device, so graph capture never copies from the CPU.
                self.register_buffer("move_destinations", _move_destinations(), persistent=False)
            if config.value_bins:
                # Each move's bin logits: origin and destination rows plus the scalar score tilting
                # them, the tilt starting at the bin values so a higher score means a higher value.
                bins = config.value_bins
                self.move_value_bin_origin = nn.Linear(d_model, bins * copies)
                self.move_value_bin_destination = nn.Linear(d_model, bins * copies, bias=False)
                self.move_value_bin_tilt = nn.Parameter(value_bin_values(bins))
                self.register_buffer("value_bin_values", value_bin_values(bins), persistent=False)
            if config.move_consequences:
                self.move_consequences = MoveConsequences()
                # Per-move feature weights from the origin (with a bias) and destination tokens,
                # one set for the logit and one for the move value, all starting at zero.
                width = CONSEQUENCES * copies
                self.policy_consequence_origin = nn.Linear(d_model, width)
                self.policy_consequence_destination = nn.Linear(d_model, width, bias=False)
                self.value_consequence_origin = nn.Linear(d_model, width)
                self.value_consequence_destination = nn.Linear(d_model, width, bias=False)
                for layer in (
                    self.policy_consequence_origin,
                    self.policy_consequence_destination,
                    self.value_consequence_origin,
                    self.value_consequence_destination,
                ):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        else:
            self.policy_head = nn.Linear(d_model, 64 * 73)
            self.move_value_head = nn.Sequential(nn.Linear(d_model, 64 * 73), nn.Tanh())
        if config.lookahead_heads:
            if not square_readout:
                raise ValueError("lookahead heads read the square tokens of the square readout")
            if config.reply_readout == "pair":
                self.reply_query = nn.Linear(d_model, MOVE_KEY_WIDTH)
                self.reply_key = nn.Linear(d_model, MOVE_KEY_WIDTH)
                self.reply_origin_bias = nn.Linear(d_model, 1)
            else:
                self.reply_from_head = nn.Linear(d_model, 1)
                self.reply_to_head = nn.Linear(d_model, 1)
            # Scored against the encoder's own piece embeddings, so imagined squares share
            # the space of real ones.
            self.ply2_projection = nn.Linear(d_model, d_model)
            self.ply4_head = nn.Linear(d_model, 13)
        if config.use_future_workspaces:
            # Each candidate line contains one compact token per future ply.
            # The token predicts both the action and the exact resulting board.
            self.future_piece_head = nn.Linear(d_model, 64 * 13)
            self.future_state_head = nn.Linear(d_model, 26)
            self.future_value_head = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
        if config.use_scratchpad_backup:
            self.scratch_value_head = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
            self.scratch_policy_scale = nn.Parameter(torch.tensor(0.1))
            self.scratch_move_value_scale = nn.Parameter(torch.tensor(0.1))
        if not square_readout:
            # The continuation decoder is tied to the global action embedding.
            self.continuation_action_projection = nn.Linear(d_model, d_model, bias=False)
            self.continuation_cell = nn.GRUCell(d_model, d_model)
        if config.value_bins:
            self.value_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, config.value_bins * copies),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, copies),
                nn.Tanh(),
            )
        if config.phase_heads:
            self._equal_phase_copies()
        nn.init.normal_(self.initial_scratchpad, std=0.02)
        if config.scratch_boards:
            nn.init.normal_(self.scratch_board_embedding, std=0.02)
            nn.init.normal_(self.scratch_content, std=0.02)
            if config.flat_scratch:
                # The scale of the nn.Embedding square vectors it replaces.
                nn.init.normal_(self.flat_scratch_embedding)

    def phase_layers(self) -> list[nn.Linear]:
        """The output layers holding one copy per phase bucket (with phase heads)."""

        names = (
            "square_policy_head",
            "policy_query",
            "policy_key",
            "move_value_query",
            "move_value_key",
            "move_value_bin_origin",
            "move_value_bin_destination",
            "policy_consequence_origin",
            "policy_consequence_destination",
            "value_consequence_origin",
            "value_consequence_destination",
        )
        layers = [getattr(self, name) for name in names if hasattr(self, name)]
        return [*layers, self.square_move_value_head[0], self.value_head[2]]

    @torch.no_grad()
    def _equal_phase_copies(self) -> None:
        for layer in self.phase_layers():
            for tensor in (layer.weight, layer.bias):
                if tensor is not None:
                    copies = tensor.unflatten(0, (self.config.phase_heads, -1))
                    copies.copy_(copies[:1].expand_as(copies))

    def _phase(self, outputs: Tensor, phase: Tensor | None) -> Tensor:
        """A phase layer's (batch, ..., K x width) ``outputs`` at each position's (batch,) bucket."""

        buckets = self.config.phase_heads
        if not buckets:
            return outputs
        if phase is None:
            raise ValueError("this model has phase heads; pass the board's phase buckets")
        copies = outputs.unflatten(-1, (buckets, -1))
        index = phase.view(-1, *[1] * (copies.ndim - 1))
        index = index.expand(*copies.shape[:-2], 1, copies.shape[-1])
        return copies.gather(-2, index).squeeze(-2)

    def phase(self, pieces: Tensor) -> Tensor | None:
        """(batch,) phase buckets of the positions (see ``phase_buckets``), None without them."""

        if not self.config.phase_heads:
            return None
        return phase_buckets(pieces, self.config.phase_heads)

    @property
    def hidden_tokens(self) -> int:
        """Tokens in H: 65 board tokens, the latent tokens and the scratch boards."""

        return 65 + self.config.latent_tokens + 64 * self.config.scratch_boards

    def reset_recurrent_to_identity(self) -> None:
        """Reset untrained recurrent output projections without touching H0."""

        for block in self.recurrent_core:
            block.reset_to_identity()

    def _block(self) -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.mlp_width,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def _tokenize(self, pieces: Tensor, state: Tensor) -> Tensor:
        if pieces.ndim != 2 or pieces.shape[1] != 64:
            raise ValueError(f"pieces must have shape (batch, 64), got {tuple(pieces.shape)}")
        if state.ndim != 2 or state.shape[1] != 26:
            raise ValueError(f"state must have shape (batch, 26), got {tuple(state.shape)}")
        squares = torch.arange(64, device=pieces.device)
        square_tokens = self.piece_embedding(pieces) + self.square_embedding(squares)[None, :, :]
        if self.config.board_features:
            features = self.board_features(pieces)
            square_tokens = square_tokens + self.board_feature_projection(features)
        if self.config.king_relative:
            own, enemy = king_offsets(pieces).unbind(1)
            offsets = self.own_king_offset(own) + self.enemy_king_offset(enemy)
            square_tokens = square_tokens + offsets
        state_token = self.state_projection(state).unsqueeze(1)
        return torch.cat((square_tokens, state_token), dim=1)

    def encode_board(self, pieces: Tensor, state: Tensor, *, consequences: bool = True) -> Tensor:
        """Produce immutable board memory B for all recurrent iterations.

        With the move graph or attack pairs B carries its positions' ``local_edges`` as an
        attribute of that name, computed once and read by every ``recurrent_step``; a copy of B needs them passed explicitly. Likewise with
        move consequences B carries ``move_consequences`` for the readouts (see
        ``board_consequences``), unless ``consequences`` is False because nothing reads out,
        and with phase heads its ``phase`` buckets (see ``board_phase``).
        """

        board_memory = self._tokenize(pieces, state)
        for block in self.prelude:
            board_memory = block(board_memory)
        edges = self.local_edges(pieces)
        if edges is not None:
            board_memory.local_edges = edges  # type: ignore[attr-defined]
        if consequences and self.config.move_consequences:
            features = self.consequences(pieces, state)
            board_memory.move_consequences = features  # type: ignore[attr-defined]
        if self.config.phase_heads:
            board_memory.phase = self.phase(pieces)  # type: ignore[attr-defined]
        return board_memory

    def consequences(self, pieces: Tensor, state: Tensor) -> Tensor | None:
        """(batch, 64, 73, CONSEQUENCES) features of every move, None without move consequences."""

        if not self.config.move_consequences:
            return None
        with torch.no_grad():
            return self.move_consequences(pieces, state)

    def local_edges(self, pieces: Tensor) -> Tensor | None:
        """(batch, 64, 64) per-position square pair codes the recurrent blocks read, or None.

        With the move graph (see ``MoveGraph.codes``) 0 blocks square i from reading square j
        and 1 allows it; move_lines codes line pairs 2 plus the pieces between them, clipped at 2.
        With attack pairs they are the (attacker, target) codes of ``AttackPairs``.
        """

        if self.config.attack_pairs:
            return self.attack_pairs(pieces)
        if self.config.local_neighbourhood not in MOVE_NEIGHBOURHOODS:
            return None
        return self.move_graph.codes(pieces)

    def initial_hidden(self, board_memory: Tensor) -> Tensor:
        """Build H0 from board tokens and learned, position-independent scratch tokens."""

        batch = board_memory.shape[0]
        scratchpad = self.initial_scratchpad.unsqueeze(0).expand(batch, -1, -1)
        if not self.config.scratch_boards:
            return torch.cat((board_memory, scratchpad), dim=1)
        boards = self.scratch_board_tokens().unsqueeze(0).expand(batch, -1, -1)
        return torch.cat((board_memory, scratchpad, boards), dim=1)

    def scratch_board_tokens(self) -> Tensor:
        """Initial scratch-board tokens, (boards * 64, width), board by board in square order.

        Each is its square's embedding plus its board's embedding plus the shared content vector.
        """

        config = self.config
        if not config.scratch_boards:
            raise RuntimeError("the model has no scratch boards")
        if config.flat_scratch:
            cells = self.flat_scratch_embedding.unflatten(0, (config.scratch_boards, 64))
        else:
            cells = self.square_embedding.weight.unsqueeze(0)
        tokens = cells + self.scratch_board_embedding.unsqueeze(1) + self.scratch_content
        return tokens.flatten(0, 1)

    def recurrent_step(
        self,
        board_memory: Tensor,
        hidden: Tensor,
        *,
        capture_attention: bool = False,
        local_edges: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Apply the shared F(B, H) once and return an H with identical shape.

        ``local_edges`` (see ``local_edges``) defaults to the one ``encode_board`` attached to B.
        """

        if self.config.core != "recurrent":
            raise RuntimeError("the workspace core steps through workspace_core.step")
        expected_tokens = (
            board_memory.shape[1] + self.config.latent_tokens + 64 * self.config.scratch_boards
        )
        if hidden.ndim != 3 or hidden.shape[1:] != (expected_tokens, self.config.d_model):
            raise ValueError(
                "hidden must have shape "
                f"(batch, {expected_tokens}, {self.config.d_model}), got {tuple(hidden.shape)}"
            )
        updated = hidden
        if self.config.refresh_board:
            board_tokens = board_memory.shape[1]
            updated = torch.cat((board_memory.to(hidden.dtype), hidden[:, board_tokens:]), dim=1)
        if local_edges is None:
            local_edges = getattr(board_memory, "local_edges", None)
        attention_maps: list[Tensor] = []
        templates = None
        if self.config.state_attention_bias:
            templates = self.state_bias_templates.weight
        for block in self.recurrent_core:
            block_output = block(
                board_memory,
                updated,
                capture_attention=capture_attention,
                local_edges=local_edges,
                state_templates=templates,
            )
            if capture_attention:
                updated, attention = block_output
                attention_maps.append(attention)
            else:
                updated = block_output
        if self.config.normalize_state:
            updated = self.state_norm(updated)
        if updated.shape != hidden.shape:
            raise AssertionError("F(B, H) must preserve the exact hidden-state shape")
        if capture_attention:
            return updated, tuple(attention_maps)
        return updated

    def heads(
        self,
        hidden: Tensor,
        board_tokens: int = 65,
        *,
        include_move_values: bool = False,
        consequences: Tensor | None = None,
        phase: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Policy logits, value and optionally move values read from H.

        A model with move consequences needs the position's ``consequences`` (see
        ``board_consequences``), and one with phase heads its ``phase`` buckets (see
        ``board_phase``); other models ignore them. With value bins the values are the
        expected values of their bins, on the same -1..1 scale as the tanh heads.
        """

        hidden, consequences = self._readout_tokens(hidden, board_tokens, consequences)
        # The global board-state slot exists in H0, so G(H0) is a genuine
        # position-aware non-recurrent baseline. The same slot remains mutable
        # and is read by exactly the same heads after every recurrent step.
        global_token = hidden[:, board_tokens - 1]
        value = self._phase(self.value_head(global_token), phase)
        if self.config.value_bins:
            value = expected_value(value, self.value_bin_values)
        else:
            value = value.squeeze(-1)
        if self.config.policy_readout == "square":
            squares = hidden[:, : board_tokens - 1]
            policy = self._square_policy(squares, consequences, phase)
            if not include_move_values:
                return policy, value
            scores = self._move_value_scores(squares, consequences, phase)
            if self.config.value_bins:
                return policy, value, self._move_value_bins(squares, scores, phase).expected()
            return policy, value, self.square_move_value_head[1](scores).flatten(1)
        policy = self.policy_head(global_token)
        if self.config.use_future_workspaces:
            count = self.config.future_lines * self.config.future_horizon
            trajectories = hidden[:, board_tokens : board_tokens + count].reshape(
                hidden.shape[0],
                self.config.future_lines,
                self.config.future_horizon,
                self.config.d_model,
            )
            roots = trajectories[:, :, 0]
            root_logits = nn.functional.linear(
                roots, self.policy_head.weight, self.policy_head.bias
            )
            root_values = self.future_value_head(roots).squeeze(-1)
            # The played move is read directly from the competing imagined
            # futures.  Consequently continuation tokens cannot be ignored by
            # an independent global policy head.
            policy = torch.logsumexp(
                root_logits + root_values.unsqueeze(-1), dim=1
            ) - math.log(self.config.future_lines)
            value = root_values.mean(dim=1)
        backed_move_values: Tensor | None = None
        if self.config.use_scratchpad_backup:
            scratchpad = hidden[:, board_tokens:]
            # Every interchangeable scratch token proposes moves and carries a
            # scalar backed-up value. The normal action head is tied across the
            # global policy and all latent branches.
            slot_policy = nn.functional.linear(
                scratchpad, self.policy_head.weight, self.policy_head.bias
            )
            slot_values = self.scratch_value_head(scratchpad).squeeze(-1)
            backed_policy = torch.logsumexp(
                slot_policy + slot_values.unsqueeze(-1), dim=1
            ) - math.log(scratchpad.shape[1])
            policy = policy + self.scratch_policy_scale * backed_policy
            slot_assignment = torch.softmax(slot_policy.transpose(1, 2), dim=-1)
            backed_move_values = torch.einsum(
                "bas,bs->ba", slot_assignment, slot_values
            )
        if include_move_values:
            move_values = self.move_value_head(global_token)
            if backed_move_values is not None:
                move_values = move_values + (
                    self.scratch_move_value_scale * backed_move_values
                )
            return policy, value, move_values
        return policy, value

    def value_bin_heads(
        self,
        hidden: Tensor,
        board_tokens: int = 65,
        *,
        consequences: Tensor | None = None,
        phase: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, MoveValueBins]:
        """Policy logits, the (batch, bins) position value logits and every move's value bins.

        The training readout of a model with value bins; ``heads`` reads their expected values.
        """

        if not self.config.value_bins:
            raise RuntimeError("the model has scalar value heads")
        hidden, consequences = self._readout_tokens(hidden, board_tokens, consequences)
        squares = hidden[:, : board_tokens - 1]
        scores = self._move_value_scores(squares, consequences, phase)
        return (
            self._square_policy(squares, consequences, phase),
            self._phase(self.value_head(hidden[:, board_tokens - 1]), phase),
            self._move_value_bins(squares, scores, phase),
        )

    def _readout_tokens(
        self, hidden: Tensor, board_tokens: int, consequences: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """The normalised board and latent tokens the heads read, and the consequences they use."""

        if not self.config.move_consequences:
            consequences = None
        elif consequences is None:
            raise ValueError("this model reads move consequences; pass the board's consequences")
        # Heads read the board and latent tokens, never the scratch boards.
        return self.final_norm(hidden[:, : board_tokens + self.config.latent_tokens]), consequences

    def _square_policy(
        self, squares: Tensor, consequences: Tensor | None, phase: Tensor | None = None
    ) -> Tensor:
        """(batch, 4672) logits of the square readout."""

        policy = self._phase(self.square_policy_head(squares), phase)
        if self.config.move_readout == "from_to":
            policy = policy + self._destination_scores(
                squares, self.policy_query, self.policy_key, phase
            )
        if consequences is not None:
            policy = policy + self._consequence_scores(
                squares,
                consequences,
                self.policy_consequence_origin,
                self.policy_consequence_destination,
                phase,
            )
        # Flattening (square, plane) reproduces action = from_square * 73 + plane.
        return policy.flatten(1)

    def _move_value_scores(
        self, squares: Tensor, consequences: Tensor | None, phase: Tensor | None = None
    ) -> Tensor:
        """(batch, 64, 73) move value scores of the square readout, before the tanh."""

        scores = self._phase(self.square_move_value_head[0](squares), phase)
        if self.config.move_readout == "from_to":
            scores = scores + self._destination_scores(
                squares, self.move_value_query, self.move_value_key, phase
            )
        if consequences is not None:
            scores = scores + self._consequence_scores(
                squares,
                consequences,
                self.value_consequence_origin,
                self.value_consequence_destination,
                phase,
            )
        return scores

    def _move_value_bins(
        self, squares: Tensor, scores: Tensor, phase: Tensor | None = None
    ) -> MoveValueBins:
        return MoveValueBins(
            origin=self._phase(self.move_value_bin_origin(squares), phase),
            destination=self._phase(self.move_value_bin_destination(squares), phase),
            scores=scores,
            tilt=self.move_value_bin_tilt,
            destinations=self.move_destinations,
            values=self.value_bin_values,
        )

    def _destination_scores(
        self, squares: Tensor, query: nn.Linear, key: nn.Linear, phase: Tensor | None = None
    ) -> Tensor:
        """(batch, 64, 73) scaled dot product of each move's origin query and destination key."""

        queries, keys = self._phase(query(squares), phase), self._phase(key(squares), phase)
        # Equals gathering a key per move, without holding a (batch, 64, 73, width) key tensor.
        scores = queries @ keys.transpose(1, 2) / math.sqrt(MOVE_KEY_WIDTH)
        index = self.move_destinations.expand(squares.shape[0], -1, -1)
        return scores.gather(2, index)

    def _consequence_scores(
        self,
        squares: Tensor,
        consequences: Tensor,
        origin: nn.Linear,
        destination: nn.Linear,
        phase: Tensor | None = None,
    ) -> Tensor:
        """(batch, 64, 73) each move's consequences dotted with weights from its two tokens."""

        from_origin = self._phase(origin(squares), phase)
        from_destination = self._phase(destination(squares), phase)
        weights = from_origin.unsqueeze(2) + from_destination[:, self.move_destinations]
        return (weights * consequences.to(weights.dtype)).sum(-1)

    def workspace_readout(self, board_memory: Tensor, workspace: Tensor) -> Tensor:
        """B's 65 tokens after reading H, which the heads read in the workspace core."""

        return self.workspace_core.read(board_memory, workspace)

    def line_policy(self, memory: Tensor, workspace: Tensor) -> Tensor:
        """(batch, 4672) logits of the line's next action, read from M and H."""

        squares = self.workspace_core.read(memory, workspace)[:, :64]
        return self._square_policy(self.final_norm(squares), None)

    def rotate_memory(self, memory: Tensor) -> Tensor:
        """M after a ply, in the child's frame."""

        return self.workspace_core.rotate(memory, self.square_embedding.weight)

    def lookahead(self, hidden: Tensor) -> dict[str, Tensor]:
        """Reply squares and future boards predicted from the square tokens.

        ``reply_from`` and ``reply_to`` are logits over the 64 squares, or with the pair reply
        readout ``reply_pair`` holds (batch, origin, destination) logits over all square pairs;
        a promotion is its pawn's two squares, whatever the piece. ``board2`` and ``board4``
        are 13-way piece logits per square after plies 2 and 4.
        """

        if not self.config.lookahead_heads:
            raise RuntimeError("lookahead heads are disabled")
        squares = self.final_norm(hidden[:, :64])
        boards = {
            "board2": self.ply2_projection(squares) @ self.piece_embedding.weight.T,
            "board4": self.ply4_head(squares),
        }
        if self.config.reply_readout == "pair":
            scores = self.reply_query(squares) @ self.reply_key(squares).transpose(1, 2)
            pair = scores / math.sqrt(MOVE_KEY_WIDTH) + self.reply_origin_bias(squares)
            return {"reply_pair": pair, **boards}
        return {
            "reply_from": self.reply_from_head(squares).squeeze(-1),
            "reply_to": self.reply_to_head(squares).squeeze(-1),
            **boards,
        }

    def future_tokens(self, hidden: Tensor, board_tokens: int = 65) -> Tensor:
        """Return candidate-line tokens as (batch, lines, horizon, width)."""

        count = self.config.future_lines * self.config.future_horizon
        scratch = self.final_norm(hidden)[:, board_tokens : board_tokens + count]
        return scratch.reshape(
            hidden.shape[0],
            self.config.future_lines,
            self.config.future_horizon,
            self.config.d_model,
        )

    def future_predictions(
        self, hidden: Tensor, board_tokens: int = 65
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict actions, resulting boards, and backed-up values per line/ply."""

        if not self.config.use_future_workspaces:
            raise RuntimeError("future workspaces are disabled")
        tokens = self.future_tokens(hidden, board_tokens)
        action_logits = nn.functional.linear(
            tokens, self.policy_head.weight, self.policy_head.bias
        )
        piece_logits = self.future_piece_head(tokens).reshape(
            *tokens.shape[:3], 64, 13
        )
        state = self.future_state_head(tokens)
        values = self.future_value_head(tokens).squeeze(-1)
        return action_logits, piece_logits, state, values

    def continuation_loss(
        self, hidden: Tensor, actions: Tensor, mask: Tensor, board_tokens: int = 65
    ) -> Tensor:
        """Predict each next PV move from H and the preceding candidate sequence.

        The action embedding and output classifier are tied to the policy head,
        keeping this a lightweight diagnostic/training decoder rather than a
        second search engine embedded in the network.
        """

        if self.config.policy_readout != "global":
            raise RuntimeError("the continuation decoder requires the global policy readout")
        if actions.shape != mask.shape or actions.ndim != 3:
            raise ValueError("continuation actions/mask must have matching (B, lines, plies)")
        if not bool(mask[:, :, 1:].any()):
            return hidden.float().sum(dim=(1, 2)) * 0
        batch, lines, plies = actions.shape
        normalized = self.final_norm(hidden)
        global_context = normalized[:, board_tokens - 1]
        scratchpad = normalized[:, board_tokens : board_tokens + self.config.latent_tokens]
        # Each root candidate retrieves its own context from the unordered
        # scratchpad. This forces continuation supervision to reach the latent
        # workspace while avoiding fixed "branch 1/branch 2" slot semantics.
        root_actions = actions[:, :, 0]
        root_embeddings = self.policy_head.weight[root_actions]
        candidate_queries = self.continuation_action_projection(root_embeddings)
        retrieval_logits = torch.einsum(
            "bld,bsd->bls", candidate_queries, scratchpad
        ) / (self.config.d_model**0.5)
        retrieval = torch.softmax(retrieval_logits.float(), dim=-1).to(hidden.dtype)
        candidate_context = torch.einsum("bls,bsd->bld", retrieval, scratchpad)
        context = global_context[:, None, :] + candidate_context
        context = context.reshape(batch * lines, -1)
        losses = torch.zeros(batch, device=hidden.device, dtype=torch.float32)
        counts = torch.zeros(batch, device=hidden.device, dtype=torch.float32)
        for ply in range(plies - 1):
            prefix_action = actions[:, :, ply].reshape(-1)
            action_embedding = self.policy_head.weight[prefix_action]
            context = self.continuation_cell(
                self.continuation_action_projection(action_embedding), context
            )
            logits = nn.functional.linear(
                context, self.policy_head.weight, self.policy_head.bias
            )
            targets = actions[:, :, ply + 1].reshape(-1)
            valid = mask[:, :, ply + 1].reshape(-1)
            token_loss = nn.functional.cross_entropy(logits, targets, reduction="none").float()
            token_loss = (token_loss * valid).reshape(batch, lines).sum(dim=1)
            token_count = valid.reshape(batch, lines).sum(dim=1)
            losses = losses + token_loss
            counts = counts + token_count
        return losses / counts.clamp_min(1)

    @torch.inference_mode()
    def trace_reasoning(
        self,
        pieces: Tensor,
        state: Tensor,
        reasoning_steps: int,
        *,
        capture_attention: bool = True,
        move_to_cpu: bool = True,
    ) -> tuple[ReasoningStepTrace, ...]:
        """Capture H0..Hn and readouts without affecting the training path."""

        if reasoning_steps < 0:
            raise ValueError("reasoning_steps cannot be negative")
        board_memory = self.encode_board(pieces, state)
        consequences = board_consequences(board_memory)
        phase = board_phase(board_memory)
        hidden = self.initial_hidden(board_memory)
        traces: list[ReasoningStepTrace] = []

        def snapshot(
            step: int, attention_maps: tuple[Tensor, ...] = ()
        ) -> ReasoningStepTrace:
            policy, value = self.heads(
                hidden, board_memory.shape[1], consequences=consequences, phase=phase
            )

            def store(tensor: Tensor) -> Tensor:
                tensor = tensor.detach()
                return tensor.cpu() if move_to_cpu else tensor

            return ReasoningStepTrace(
                step=step,
                hidden=store(hidden),
                policy_logits=store(policy),
                value=store(value),
                attention_maps=tuple(store(attention) for attention in attention_maps),
            )

        traces.append(snapshot(0))
        for step in range(1, reasoning_steps + 1):
            if capture_attention:
                hidden, attention_maps = self.recurrent_step(
                    board_memory, hidden, capture_attention=True
                )
            else:
                hidden = self.recurrent_step(board_memory, hidden)
                attention_maps = ()
            traces.append(snapshot(step, attention_maps))
        return tuple(traces)

    def forward(
        self,
        pieces: Tensor,
        state: Tensor,
        reasoning_steps: int = 1,
        *,
        intermediate_steps: frozenset[int] = frozenset(),
        include_move_values: bool = False,
        include_hidden: bool = False,
    ) -> tuple[Tensor, ...] | dict[int, tuple[Tensor, ...]]:
        if reasoning_steps < 0:
            raise ValueError("reasoning_steps cannot be negative")
        if self.config.core == "workspace":
            return self._workspace_forward(
                pieces,
                state,
                reasoning_steps,
                intermediate_steps=intermediate_steps,
                include_move_values=include_move_values,
                include_hidden=include_hidden,
            )
        board_memory = self.encode_board(pieces, state)
        consequences = board_consequences(board_memory)
        phase = board_phase(board_memory)
        hidden = self.initial_hidden(board_memory)

        outputs: dict[int, tuple[Tensor, Tensor]] = {}

        def readout() -> tuple[Tensor, ...]:
            result = self.heads(
                hidden,
                board_memory.shape[1],
                include_move_values=include_move_values,
                consequences=consequences,
                phase=phase,
            )
            return (*result, hidden) if include_hidden else result

        if 0 in intermediate_steps:
            outputs[0] = readout()
        for step in range(1, reasoning_steps + 1):
            hidden = self.recurrent_step(board_memory, hidden)
            if step in intermediate_steps or step == reasoning_steps:
                outputs[step] = readout()
        if reasoning_steps == 0:
            outputs[0] = readout()
        if intermediate_steps:
            return outputs
        return outputs[reasoning_steps]

    def _workspace_forward(
        self,
        pieces: Tensor,
        state: Tensor,
        reasoning_steps: int,
        *,
        intermediate_steps: frozenset[int],
        include_move_values: bool,
        include_hidden: bool,
    ) -> tuple[Tensor, ...] | dict[int, tuple[Tensor, ...]]:
        """Readouts after 0 to ``reasoning_steps`` steps of lines of the model's own moves.

        The steps run lines of ``config.line_plies`` plies from the present position, the last
        one possibly unfinished. Each ply starts by playing the line policy's best legal action;
        after its steps M turns to the child's frame.
        """

        core, plies = self.workspace_core, self.config.line_plies
        board_memory = self.encode_board(pieces, state)
        consequences = board_consequences(board_memory)
        workspace = core.start(board_memory)
        memory = board_memory
        wanted = intermediate_steps | {reasoning_steps}
        outputs: dict[int, tuple[Tensor, ...]] = {}

        def readout() -> tuple[Tensor, ...]:
            tokens = self.workspace_readout(board_memory, workspace)
            result = self.heads(
                tokens, 65, include_move_values=include_move_values, consequences=consequences
            )
            return (*result, tokens) if include_hidden else result

        if 0 in wanted:
            outputs[0] = readout()
        root = board = child = (pieces, state)
        root_legal = core.rules.legal(pieces, state) if reasoning_steps else None
        tried = torch.zeros_like(root_legal) if root_legal is not None else None
        for step in range(reasoning_steps):
            starts_line, ply, starts_ply, ends_ply = core.phase(step, plies)
            if starts_line:
                memory, board = board_memory, root
            if starts_ply:
                logits = self.line_policy(memory, workspace)
                if ply == 0:
                    assert root_legal is not None and tried is not None
                    actions = own_moves(core.first_logits(logits, tried), root_legal)
                    tried = tried | one_hot_actions(actions, tried.shape[1])
                else:
                    actions = own_moves(logits, core.rules.legal(*board))
                memory, workspace = core.play(memory, workspace, actions)
                child = core.rules.child(*board, actions)
            memory, workspace = core.step(board_memory, memory, workspace)
            if ends_ply:
                memory, board = self.rotate_memory(memory), child
            if step + 1 in wanted:
                outputs[step + 1] = readout()
        if intermediate_steps:
            return outputs
        return outputs[reasoning_steps]


def board_consequences(board_memory: Tensor) -> Tensor | None:
    """The move consequences ``encode_board`` attached to B, None when it attached none."""

    return getattr(board_memory, "move_consequences", None)


def board_phase(board_memory: Tensor) -> Tensor | None:
    """The phase buckets ``encode_board`` attached to B, None when it attached none."""

    return getattr(board_memory, "phase", None)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def noisy_squares(hidden: Tensor, std: float) -> Tensor:
    """``hidden`` with Gaussian noise of this std added to its 64 square tokens."""

    squares = hidden[:, :64]
    return torch.cat((squares + std * torch.randn_like(squares), hidden[:, 64:]), dim=1)
