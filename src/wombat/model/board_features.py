"""Per-square attack, pin and check features computed from the canonical piece ids.

Piece ids follow ``wombat.chess.encode``: 1..6 are the side to move's P N B R Q K, 7..12 the
opponent's, 0 is empty, and the side to move's pawns advance toward higher ranks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wombat.chess.moves import KNIGHT_DELTAS, RAY_DIRECTIONS

# Attacker counts one-hot over 0..7 (clipped), then the cheapest attacker's class one-hot over
# none, pawn, knight or bishop, rook, queen, king.
COUNT_CLASSES = 8
ATTACKER_CLASSES = 6
# Per side (side to move, then opponent) count and class; then the side to move's pinned pieces,
# the opponent's pinned pieces and whether the side to move is in check, on every square.
BOARD_FEATURES = 2 * (COUNT_CLASSES + ATTACKER_CLASSES) + 3

# X-ray attackers (sliders behind exactly one piece) one-hot over 0, 1, 2 or more.
XRAY_CLASSES = 3
# Material per side for pawns, knights, bishops, rooks and queens, over its starting count.
MATERIAL_SCALE = (8.0, 2.0, 2.0, 2.0, 1.0)
# The long-range set, appended after BOARD_FEATURES; each pair is side to move, then opponent:
# sole defenders (2); squares where the side to move's knight, diagonal slider, orthogonal slider
# or pawn would check the opponent's king, then the opponent's against ours (8); squares on the
# lines checking our king and theirs (2); discovered-check pieces (2); x-ray counts (6); material.
LONG_RANGE_FEATURES = 2 + 8 + 2 + 2 + 2 * XRAY_CLASSES + 2 * len(MATERIAL_SCALE)

_OFF_BOARD = 64
# Attack pair codes 1..144, one per (attacker id, target id) of the 12 piece ids; 0 is no pair.
ATTACK_CODES = 12 * 12
# Squares holding a pawn that attacks the target: the side to move's pawns sit a rank below it.
_OWN_PAWN_SOURCES = ((-1, -1), (1, -1))
_THEIR_PAWN_SOURCES = ((-1, 1), (1, 1))
# King steps are the ray directions at distance one.
_STEP_SOURCES = (*KNIGHT_DELTAS, *RAY_DIRECTIONS, *_OWN_PAWN_SOURCES, *_THEIR_PAWN_SOURCES)
# Step columns of each side's attacking pawns; a pawn standing on a square attacks the squares
# in the other side's columns.
_PAWN_COLUMNS = (slice(16, 18), slice(18, 20))
# Rays along which the side to move's and the opponent's pawns push.
_PUSH_RAYS = (RAY_DIRECTIONS.index((0, 1)), RAY_DIRECTIONS.index((0, -1)))


def _offset(square: int, file_step: int, rank_step: int) -> int:
    file, rank = square % 8 + file_step, square // 8 + rank_step
    return file + 8 * rank if 0 <= file < 8 and 0 <= rank < 8 else _OFF_BOARD


def step_sources() -> Tensor:
    """(64, 20) knight, king, own pawn and opponent pawn source squares; 64 is off the board."""

    return torch.tensor(
        [[_offset(square, *step) for step in _STEP_SOURCES] for square in range(64)]
    )


def ray_squares() -> Tensor:
    """(64, 8, 7) squares along each ray direction by distance; 64 is off the board."""

    return torch.tensor(
        [
            [
                [_offset(square, df * distance, dr * distance) for distance in range(1, 8)]
                for df, dr in RAY_DIRECTIONS
            ]
            for square in range(64)
        ]
    )


class BoardFeatures(nn.Module):
    """Map (batch, 64) piece ids to (batch, 64, width) floats with batched gathers.

    Attacks respect blocking and ignore en passant and castling; a pin is an absolute pin of the
    piece on the square to its own king. Its tables are buffers, so they move with the model.
    ``long_range`` appends the LONG_RANGE_FEATURES set to the BOARD_FEATURES one.
    """

    steps: Tensor
    rays: Tensor
    orthogonal: Tensor
    material_scale: Tensor

    def __init__(self, long_range: bool = False) -> None:
        super().__init__()
        self.long_range = long_range
        self.width = BOARD_FEATURES + (LONG_RANGE_FEATURES if long_range else 0)
        self.register_buffer("steps", step_sources(), persistent=False)
        # An eighth, always empty square ends every ray, the result of a gather past its last piece.
        rays = torch.cat((ray_squares(), torch.full((64, 8, 1), _OFF_BOARD)), dim=-1)
        self.register_buffer("rays", rays, persistent=False)
        orthogonal = torch.tensor([0 in direction for direction in RAY_DIRECTIONS])
        self.register_buffer("orthogonal", orthogonal, persistent=False)
        if long_range:
            scale = torch.tensor(MATERIAL_SCALE).repeat(2)
            self.register_buffer("material_scale", scale, persistent=False)

    def forward(self, pieces: Tensor) -> Tensor:
        # Small integers keep the (batch, 64, 8, 7) ray tensors cheap.
        padded = torch.cat((pieces, pieces.new_zeros(pieces.shape[0], 1)), dim=1).byte()
        steps = padded[:, self.steps]
        rays = padded[:, self.rays]
        seen = (rays != 0).cumsum(-1, dtype=torch.uint8)
        # Distance index of the first and second piece along each ray, 7 where there is none.
        first_index = (seen == 0).sum(-1, keepdim=True).clamp_max(7)
        second_index = (seen <= 1).sum(-1, keepdim=True).clamp_max(7)
        first_piece = rays.gather(-1, first_index).squeeze(-1)
        second_piece = rays.gather(-1, second_index).squeeze(-1)
        first_square = self.rays.expand(pieces.shape[0], -1, -1, -1).gather(-1, first_index)
        first_square = first_square.squeeze(-1)
        # Rook on orthogonal rays, bishop on diagonal ones.
        line_piece = 3 + self.orthogonal.long()
        counts = torch.arange(COUNT_CLASSES, device=pieces.device)
        parts: list[Tensor] = []
        attackers: list[Tensor] = []
        for offset, pawn_steps in ((0, steps[..., 16:18]), (6, steps[..., 18:20])):
            line = first_piece == line_piece + offset
            classes = torch.stack(
                (
                    (pawn_steps == 1 + offset).sum(-1),
                    (steps[..., :8] == 2 + offset).sum(-1) + (line & ~self.orthogonal).sum(-1),
                    (line & self.orthogonal).sum(-1),
                    (first_piece == 5 + offset).sum(-1),
                    (steps[..., 8:16] == 6 + offset).sum(-1),
                ),
                dim=-1,
            )
            total = classes.sum(-1)
            attackers.append(total)
            present = classes > 0
            cheapest = present & (present.cumsum(-1) == 1)
            parts.append(total.clamp_max(COUNT_CLASSES - 1).unsqueeze(-1) == counts)
            parts.append(torch.cat((~present.any(-1, keepdim=True), cheapest), dim=-1))
        for offset in (0, 6):
            enemy = 6 - offset
            own_first = (first_piece > offset) & (first_piece <= offset + 6)
            pinner = (second_piece == line_piece + enemy) | (second_piece == 5 + enemy)
            king = (pieces == 6 + offset).unsqueeze(-1)
            # Rays leaving other squares, or pinning nothing, mark the off-board column.
            pinned = first_square.masked_fill(~(own_first & pinner & king), _OFF_BOARD)
            marks = padded.new_zeros(padded.shape).scatter_(1, pinned.flatten(1), 1)
            parts.append(marks[:, :64, None] > 0)
        check = ((attackers[1] > 0) & (pieces == 6)).any(1)
        parts.append(check[:, None, None].expand(-1, 64, 1))
        features = torch.cat(parts, dim=-1).float()
        if not self.long_range:
            return features
        long_range = self._long_range(
            pieces, padded, steps, first_piece, second_piece, first_index, first_square
        )
        return torch.cat((features, long_range), dim=-1)

    def _long_range(
        self,
        pieces: Tensor,
        padded: Tensor,
        steps: Tensor,
        first_piece: Tensor,
        second_piece: Tensor,
        first_index: Tensor,
        first_square: Tensor,
    ) -> Tensor:
        """The LONG_RANGE_FEATURES columns from the ray and step gathers of ``forward``."""

        batch = pieces.shape[0]
        device = pieces.device
        line_piece = 3 + self.orthogonal.long()
        offsets = (0, 6)

        def mark(squares: Tensor) -> Tensor:
            # A (batch, 64, 1) flag on every listed square; 64 lists nothing.
            marks = padded.new_zeros(padded.shape).scatter_(1, squares.flatten(1), 1)
            return marks[:, :64, None] > 0

        # Each side's attackers of every square: knight, king and pawn steps, and first sliders.
        step_attacks: list[Tensor] = []
        step_squares: list[Tensor] = []
        ray_attacks: list[Tensor] = []
        counts: list[Tensor] = []
        for offset, pawns in zip(offsets, _PAWN_COLUMNS):
            step_attack = torch.cat(
                (
                    steps[..., :8] == 2 + offset,
                    steps[..., 8:16] == 6 + offset,
                    steps[..., pawns] == 1 + offset,
                ),
                dim=-1,
            )
            ray_attack = (first_piece == line_piece + offset) | (first_piece == 5 + offset)
            step_attacks.append(step_attack)
            step_squares.append(torch.cat((self.steps[:, :16], self.steps[:, pawns]), dim=-1))
            ray_attacks.append(ray_attack)
            counts.append(step_attack.sum(-1) + ray_attack.sum(-1))
        flags: list[Tensor] = []
        for side, offset in enumerate(offsets):
            # A lone defender's square is the sum of its source squares; kings are never defended.
            defended = (pieces > offset) & (pieces < offset + 6)
            sole = defended & (counts[side] == 1) & (counts[1 - side] > 0)
            source = (step_attacks[side] * step_squares[side]).sum(-1)
            source = source + (ray_attacks[side] * first_square).sum(-1)
            flags.append(mark(source.masked_fill(~sole, _OFF_BOARD)))
        for side, offset in enumerate(offsets):
            # A piece of this side standing on the square would check the enemy king.
            king = 12 - offset
            on_line = first_piece == king
            flags += [
                (steps[..., :8] == king).any(-1, keepdim=True),
                (on_line & ~self.orthogonal).any(-1, keepdim=True),
                (on_line & self.orthogonal).any(-1, keepdim=True),
                (steps[..., _PAWN_COLUMNS[1 - side]] == king).any(-1, keepdim=True),
            ]
        distance = torch.arange(8, device=device)
        for side, offset in enumerate(offsets):
            # Checkers of this side's king, and the squares between a checking slider and it.
            own_king = (pieces == 6 + offset).unsqueeze(-1)
            enemy = 1 - side
            checkers = step_squares[enemy].expand(batch, -1, -1)
            checkers = checkers.masked_fill(~(step_attacks[enemy] & own_king), _OFF_BOARD)
            along = (ray_attacks[enemy] & own_king).unsqueeze(-1) & (distance <= first_index)
            line = self.rays.expand(batch, -1, -1, -1).masked_fill(~along, _OFF_BOARD)
            flags.append(mark(torch.cat((checkers.flatten(1), line.flatten(1)), dim=1)))
        for offset in offsets:
            # From the enemy king: first one of this side's pieces, then its slider behind it.
            enemy_king = (pieces == 12 - offset).unsqueeze(-1)
            own_first = (first_piece > offset) & (first_piece <= offset + 6)
            behind = (second_piece == line_piece + offset) | (second_piece == 5 + offset)
            discovered = first_square.masked_fill(~(own_first & behind & enemy_king), _OFF_BOARD)
            flags.append(mark(discovered))
        xray_classes = torch.arange(XRAY_CLASSES, device=device)
        for offset in offsets:
            behind = (second_piece == line_piece + offset) | (second_piece == 5 + offset)
            xrays = behind.sum(-1).clamp_max(XRAY_CLASSES - 1)
            flags.append(xrays.unsqueeze(-1) == xray_classes)
        # Local attention keeps square tokens from reading the state token, so material reaches
        # them as the same values broadcast to every square.
        types = torch.arange(1, 6, device=device)
        material = torch.cat(
            ((pieces.unsqueeze(-1) == types).sum(1), (pieces.unsqueeze(-1) == types + 6).sum(1)),
            dim=-1,
        )
        material = (material / self.material_scale).unsqueeze(1).expand(-1, 64, -1)
        return torch.cat((torch.cat(flags, dim=-1).float(), material), dim=-1)


class AttackPairs(nn.Module):
    """Map (batch, 64) piece ids to (batch, 64, 64) codes of which piece attacks which.

    Entry (a, b) is 0 unless the piece on a attacks b (the pseudo-attacks of ``BoardFeatures``)
    and b holds a piece; then it is 1 + 12 (attacker id - 1) + target id - 1, so the code names
    both pieces and, by their colours, an attack or a defence.
    """

    steps: Tensor
    rays: Tensor
    orthogonal: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("steps", step_sources(), persistent=False)
        rays = torch.cat((ray_squares(), torch.full((64, 8, 1), _OFF_BOARD)), dim=-1)
        self.register_buffer("rays", rays, persistent=False)
        orthogonal = torch.tensor([0 in direction for direction in RAY_DIRECTIONS])
        self.register_buffer("orthogonal", orthogonal, persistent=False)

    def forward(self, pieces: Tensor) -> Tensor:
        batch = pieces.shape[0]
        padded = torch.cat((pieces, pieces.new_zeros(batch, 1)), dim=1)
        # From each target square: the pieces a step away and the first piece along each ray.
        steps = padded[:, self.steps]
        rays = padded[:, self.rays]
        first_index = ((rays != 0).cumsum(-1) == 0).sum(-1, keepdim=True).clamp_max(7)
        first_piece = rays.gather(-1, first_index).squeeze(-1)
        first_square = self.rays.expand(batch, -1, -1, -1).gather(-1, first_index).squeeze(-1)
        kind = first_piece - 6 * (first_piece > 6).long()
        queen = kind == 5
        slider = torch.where(self.orthogonal, (kind == 4) | queen, (kind == 3) | queen)
        step_kind = steps - 6 * (steps > 6).long()
        attacks = torch.cat(
            (
                step_kind[..., :8] == 2,
                step_kind[..., 8:16] == 6,
                steps[..., _PAWN_COLUMNS[0]] == 1,
                steps[..., _PAWN_COLUMNS[1]] == 7,
                slider,
            ),
            dim=-1,
        ) & (pieces != 0).unsqueeze(-1)
        sources = torch.cat((self.steps.expand(batch, -1, -1), first_square), dim=-1)
        attackers = torch.cat((steps, first_piece), dim=-1)
        codes = (1 + 12 * (attackers - 1) + pieces.unsqueeze(-1) - 1) * attacks
        # A source square attacks a target in at most one way; misses land in column 64.
        index = sources.masked_fill(~attacks, _OFF_BOARD)
        by_target = codes.new_zeros(batch, 64, 65).scatter_(2, index, codes)[..., :64]
        return by_target.transpose(1, 2)


class MoveGraph(nn.Module):
    """Map (batch, 64) piece ids to the (batch, 64, 64) squares each square may attend to.

    Square i sees j when they are at most one king step apart, when the piece on i attacks j or
    pushes there, or when the piece on j does so to i. Attacks are those of ``BoardFeatures``;
    pushes are single steps and double steps from the start rank over two empty squares. With
    ``lines`` bishops, rooks and queens reach every square of their lines, through blockers.
    """

    rays: Tensor
    orthogonal: Tensor
    double_push: Tensor
    targets: Tensor
    near: Tensor

    def __init__(self, lines: bool = False) -> None:
        super().__init__()
        self.lines = lines
        rays = ray_squares()
        self.register_buffer("rays", rays, persistent=False)
        orthogonal = torch.tensor([0 in direction for direction in RAY_DIRECTIONS])
        self.register_buffer("orthogonal", orthogonal, persistent=False)
        # Per side, the start rank squares and the ray their pawns push along.
        ranks = torch.arange(64) // 8
        double_push = torch.zeros(2, 64, len(RAY_DIRECTIONS), dtype=torch.bool)
        double_push[0, ranks == 1, _PUSH_RAYS[0]] = True
        double_push[1, ranks == 6, _PUSH_RAYS[1]] = True
        self.register_buffer("double_push", double_push, persistent=False)
        # Every ray square by direction and distance, then the knight jumps; 64 is off the board.
        targets = torch.cat((rays.flatten(1), step_sources()[:, : len(KNIGHT_DELTAS)]), dim=1)
        self.register_buffer("targets", targets, persistent=False)
        # King steps, which also cover king moves, pawn captures and single pushes.
        near = torch.eye(64, 65, dtype=torch.bool)
        near[torch.arange(64)[:, None], rays[:, :, 0]] = True
        self.register_buffer("near", near[:, :64], persistent=False)

    def forward(self, pieces: Tensor) -> Tensor:
        return self.codes(pieces) > 0

    def codes(self, pieces: Tensor) -> Tensor:
        """(batch, 64, 64) long: 0 where square i may not read j, 1 where it may.

        With ``lines`` a pair on a line the piece on either square slides along is instead 2 plus
        the pieces strictly between them, clipped at 2 (so 2, 3 or 4).
        """

        batch = pieces.shape[0]
        padded = torch.cat((pieces, pieces.new_zeros(batch, 1)), dim=1).byte()
        occupied = padded[:, self.rays] != 0
        # Pieces strictly before each ray square; a slider without lines stops on its blocker.
        between = occupied.cumsum(-1, dtype=torch.uint8) - occupied.byte()
        kind = pieces - 6 * (pieces > 6).long()
        queen = kind == 5
        straight, diagonal = (kind == 4) | queen, (kind == 3) | queen
        slides = torch.where(self.orthogonal, straight.unsqueeze(-1), diagonal.unsqueeze(-1))
        reach = slides.unsqueeze(-1) if self.lines else slides.unsqueeze(-1) & (between == 0)
        ray_codes = reach.byte() * (2 + between.clamp_max(2))
        # Pawns never slide, so their double push takes the empty code slot at distance two.
        pawns = torch.stack((pieces == 1, pieces == 7), dim=1).unsqueeze(-1)
        push = (pawns & self.double_push).any(1) & ~occupied[..., :2].any(-1)
        ray_codes[..., 1] |= push.byte()
        knights = (kind == 2).unsqueeze(-1).expand(-1, -1, len(KNIGHT_DELTAS)).byte()
        edges = torch.cat((ray_codes.flatten(2), knights), dim=-1)
        # Off-board targets all land in column 64, which is dropped.
        index = self.targets.expand(batch, -1, -1)
        reached = edges.new_zeros(batch, 64, 65).scatter_(2, index, edges)[..., :64]
        # Both ends of a line count the same pieces between them, so the larger code is the line's.
        codes = torch.maximum(torch.maximum(reached, reached.transpose(1, 2)), self.near.byte())
        return codes.long() if self.lines else (codes > 0).long()
