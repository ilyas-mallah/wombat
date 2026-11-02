"""What each legal move allows one move later, computed for a whole batch with tensor ops.

Piece ids follow ``wombat.chess.encode``: 1..6 are the side to move's P N B R Q K, 7..12 the
opponent's, 0 is empty, and the side to move's pawns advance toward higher ranks. The opponent's
replies are generated in its own frame: the board rotated 180 degrees with colours swapped. Boards
carry a 65th, always empty square that off-board lookups land on.

Per move, on the position after it and from the mover's point of view: the static exchange of the
moved piece on its destination, the opponent's best static-exchange gain over its legal captures
(0 when none gains), whether the opponent has a legal mate in one, and whether the move gives check.
Exchanges follow ``wombat.interpretability.labels.static_exchange`` exactly, down to its tie break
between equally cheap attackers. Legality of the opponent's replies follows python-chess, pins,
checks, en passant and castling included, and so does the legality of the side to move's moves.
Mate in one is searched over the replies that give check, which every mating move does. Illegal
moves get zeros.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wombat.chess.moves import KNIGHT_DELTAS, PLANES_PER_SQUARE, RAY_DIRECTIONS
from wombat.model.board_features import ray_squares, step_sources

# Static exchange of the moved piece, the opponent's best exchange gain, the opponent's mate in
# one, and check, in that order.
CONSEQUENCES = 4
# Material features are pawns over a queen's 9, clipped to [-1, 1].
MATERIAL_SCALE = 9.0
# Pawn units by piece id, as ``labels.PIECE_VALUES``; a king only has to dominate.
_VALUES = (0, 1, 3, 3, 5, 9, 100)
# Every capture in an exchange spends one of the at most 31 pieces off the target square.
_MAX_ROUNDS = 31
# Under CUDA graph capture every stage has a fixed length per position of the batch: moves of the
# side to move (pseudo-legal), the opponent's replies to the legal ones, their en passant and
# castling replies, and the legal captures and legal checks among the replies. Entries past a cap
# are dropped, leaving 0 features on the batch's last positions; eager runs size every stage
# exactly. Batches of 128 games-v1 and puzzles-v1 positions averaged at most 34, 1160, 7, 91 and
# 56 per position.
CAPS = {"moves": 48, "replies": 1792, "special": 16, "captures": 160, "checks": 128}
# Mate tests run in chunks of this many replies to bound their gathers.
_MATE_CHUNK = 4096
_OFF = 64
_BIG = 1 << 20
# Planes of a king's two-square castling step east and west, and of a pawn's diagonal steps.
_CASTLE_EAST = RAY_DIRECTIONS.index((1, 0)) * 7 + 1
_CASTLE_WEST = RAY_DIRECTIONS.index((-1, 0)) * 7 + 1
_PAWN_DIAGONALS = (RAY_DIRECTIONS.index((1, 1)) * 7, RAY_DIRECTIONS.index((-1, 1)) * 7)
_DOUBLE_PUSH = RAY_DIRECTIONS.index((0, 1)) * 7 + 1


def _destinations() -> Tensor:
    """(65, 73) destination of each action; 64 where it leaves the board or starts off it."""

    deltas = [
        (df * distance, dr * distance) for df, dr in RAY_DIRECTIONS for distance in range(1, 8)
    ]
    deltas += list(KNIGHT_DELTAS)
    deltas += [(file_step, 1) for _ in range(3) for file_step in (-1, 0, 1)]
    table = torch.full((65, PLANES_PER_SQUARE), _OFF)
    for square in range(64):
        file, rank = square % 8, square // 8
        for plane, (df, dr) in enumerate(deltas):
            if 0 <= file + df < 8 and 0 <= rank + dr < 8:
                table[square, plane] = square + df + 8 * dr
    return table


def _capable() -> Tensor:
    """(13, 9, 8) whether piece p at distance index i along ray r from a square attacks it.

    Ray 8 and index 7 stand for "not on a ray" and are never capable.
    """

    table = torch.zeros(13, 9, 8, dtype=torch.bool)
    for piece in range(1, 13):
        kind, theirs = (piece - 1) % 6 + 1, piece > 6
        for ray, (df, dr) in enumerate(RAY_DIRECTIONS):
            diagonal = df != 0 and dr != 0
            slides = kind == 5 or (kind == 3 and diagonal) or (kind == 4 and not diagonal)
            table[piece, ray, :7] = slides
            # The side to move's pawns attack upward, so they stand a rank below the square.
            pawn = kind == 1 and dr == (1 if theirs else -1) and df != 0
            table[piece, ray, 0] |= kind == 6 or pawn
    return table


def _lines() -> tuple[Tensor, Tensor, Tensor]:
    """(65, 65) ray from a to b and b's distance index on it (8 and 7 when unaligned), and
    whether b is a knight's jump from a."""

    rays = ray_squares()
    knights = step_sources()[:, : len(KNIGHT_DELTAS)]
    ray_of = torch.full((65, 65), 8)
    index_of = torch.full((65, 65), 7)
    near = torch.zeros(65, 65, dtype=torch.bool)
    for square in range(64):
        for ray in range(8):
            for index in range(7):
                target = int(rays[square, ray, index])
                if target != _OFF:
                    ray_of[square, target] = ray
                    index_of[square, target] = index
        for target in knights[square].tolist():
            if target != _OFF:
                near[square, target] = True
    return ray_of, index_of, near


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _compact(mask: Tensor, cap: int | None) -> tuple[Tensor, Tensor]:
    """Flat indices of ``mask``'s set entries and which of them are real.

    Without a cap the list is exact; with one it has ``cap`` entries, padded with index 0.
    """

    flat = mask.flatten()
    if cap is None:
        index = flat.nonzero().squeeze(1)
        return index, torch.ones_like(index, dtype=torch.bool)
    position = flat.cumsum(0, dtype=torch.int32) - 1
    slot = torch.where(flat & (position < cap), position, cap).long()
    order = torch.arange(flat.numel(), device=flat.device)
    index = order.new_zeros(cap + 1).scatter_(0, slot, order)[:cap]
    return index, torch.arange(cap, device=flat.device) < flat.sum()


class MoveConsequences(nn.Module):
    """Map (batch, 64) piece ids and (batch, 26) states to (batch, 64, 73, CONSEQUENCES) floats.

    ``raw`` gives the unscaled integers per action. Its tables are buffers, so they move with the
    model; nothing here is learned.
    """

    rays: Tensor
    knights: Tensor
    destinations: Tensor
    capable: Tensor
    ray_of: Tensor
    index_of: Tensor
    knight_near: Tensor
    values: Tensor
    swap: Tensor
    upper: Tensor

    def __init__(self) -> None:
        super().__init__()
        # Each ray gets an eighth, always empty square, and square 64 has only empty rays.
        rays = torch.cat((ray_squares(), torch.full((64, 8, 1), _OFF)), dim=-1)
        rays = torch.cat((rays, torch.full((1, 8, 8), _OFF)))
        self.register_buffer("rays", rays, persistent=False)
        knights = step_sources()[:, : len(KNIGHT_DELTAS)]
        knights = torch.cat((knights, torch.full((1, len(KNIGHT_DELTAS)), _OFF)))
        self.register_buffer("knights", knights, persistent=False)
        self.register_buffer("destinations", _destinations(), persistent=False)
        self.register_buffer("capable", _capable(), persistent=False)
        ray_of, index_of, near = _lines()
        self.register_buffer("ray_of", ray_of, persistent=False)
        self.register_buffer("index_of", index_of, persistent=False)
        self.register_buffer("knight_near", near, persistent=False)
        self.register_buffer("values", torch.tensor(_VALUES + _VALUES[1:]), persistent=False)
        swap = torch.tensor([0, *range(7, 13), *range(1, 7)])
        self.register_buffer("swap", swap, persistent=False)
        self.register_buffer("upper", torch.triu(torch.ones(8, 8)), persistent=False)
        self.static: bool | None = None

    def forward(self, pieces: Tensor, state: Tensor) -> Tensor:
        raw = self.raw(pieces, state).float()
        material = (raw[..., :2] / MATERIAL_SCALE).clamp(-1.0, 1.0)
        features = torch.cat((material, raw[..., 2:]), dim=-1)
        return features.view(pieces.shape[0], 64, PLANES_PER_SQUARE, CONSEQUENCES)

    def raw(self, pieces: Tensor, state: Tensor) -> Tensor:
        """(batch, 4672, CONSEQUENCES) long: exchange and best reply gain in pawns, mate, check.

        Illegal actions get zeros. ``static`` (None follows graph capture) chooses the fixed
        CAPS over exact sizes.
        """

        batch, device = pieces.shape[0], pieces.device
        static = _capturing(device) if self.static is None else self.static
        sizes = {stage: size * batch if static else None for stage, size in CAPS.items()}
        boards = torch.cat((pieces.long(), pieces.new_zeros(batch, 1).long()), dim=1)
        black = state[:, 1] > 0.5
        rows, origin, plane, played, _, valid = self._legal_moves(boards, state, sizes["moves"])
        after, kind, new, victim, en_passant, castle = played
        target = self.destinations[origin, plane]
        gains = self.values[victim] + en_passant.long() + self.values[new] - self.values[kind]
        their_king = self._king(after, 12)
        check = self._attacked(after, their_king[:, None], theirs=False)[:, 0]
        mover = {"origin": origin, "plane": plane, "kind": kind, "valid": valid}
        replies = self._replies(after, mover, black[rows], state[rows], sizes)
        exchange = self._exchange(
            torch.cat((after, replies["boards"])),
            torch.cat((target, replies["targets"])),
            torch.cat((gains, replies["gains"])),
            torch.cat((black[rows], ~black[rows][replies["rows"]])),
            static,
        )
        count = len(target)
        own = exchange[:count].masked_fill(castle, 0)
        best = torch.zeros_like(own).scatter_reduce(
            0, replies["rows"], exchange[count:] * replies["valid"], "amax"
        )
        features = torch.stack((own, best, replies["mate"], check.long()), dim=1)
        actions = 64 * PLANES_PER_SQUARE
        # Padding entries land on one extra row, which is dropped.
        slots = torch.where(
            valid, rows * actions + origin * PLANES_PER_SQUARE + plane, batch * actions
        )
        dense = features.new_zeros(batch * actions + 1, CONSEQUENCES)
        dense[slots] = features
        return dense[:-1].view(batch, actions, CONSEQUENCES)

    def legal(self, pieces: Tensor, state: Tensor) -> Tensor:
        """(batch, 4672) bool: the legal actions of each position, as python-chess finds them."""

        batch = pieces.shape[0]
        static = _capturing(pieces.device) if self.static is None else self.static
        boards = torch.cat((pieces.long(), pieces.new_zeros(batch, 1).long()), dim=1)
        size = CAPS["moves"] * batch if static else None
        rows, origin, plane, _, _, valid = self._legal_moves(boards, state, size)
        actions = 64 * PLANES_PER_SQUARE
        slots = torch.where(valid, rows * actions + origin * PLANES_PER_SQUARE + plane, -1)
        mask = torch.zeros(batch * actions + 1, dtype=torch.bool, device=pieces.device)
        mask[slots] = True
        return mask[:-1].view(batch, actions)

    def _legal_moves(
        self, boards: Tensor, state: Tensor, size: int | None
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...], dict[str, Tensor], Tensor]:
        """The side to move's moves as rows, origins and planes, the ``_play`` result of each,
        the positions' ``_defence``, and which moves are real and legal."""

        black = state[:, 1] > 0.5
        ep_file = state[:, 6:15].argmax(dim=1)
        ep = torch.where(ep_file < 8, 40 + ep_file, _OFF)
        king_file = 4 - black.long()
        origins = self._own_squares(boards)
        moves = self._pseudo_moves(boards, origins, state[:, 2:4] > 0.5, ep, king_file)
        index, valid = _compact(moves, size)
        rows = index // (16 * PLANES_PER_SQUARE)
        plane = index % PLANES_PER_SQUARE
        origin = origins[rows, (index // PLANES_PER_SQUARE) % 16]
        played = self._play(boards, rows, origin, plane)
        after, castle = played[0], played[5]
        # Only legal moves go on: they may not leave the king attacked, nor castle out of check
        # or across an attacked square.
        own_king = self._king(after, 6)
        valid &= ~self._attacked(after, own_king[:, None], theirs=True)[:, 0]
        info = self._defence(boards)
        passed = torch.where(plane == _CASTLE_EAST, 2, 6)
        calm = (info["checks"][rows] == 0) & ~info["unsafe"][rows, passed]
        valid &= ~castle | calm
        return rows, origin, plane, played, info, valid

    def _replies(
        self,
        after: Tensor,
        mover: dict[str, Tensor],
        black: Tensor,
        state: Tensor,
        sizes: dict[str, int | None],
    ) -> dict[str, Tensor]:
        """The opponent's legal captures, as exchange instances, and its mate in one per move.

        ``after`` holds the positions after each move in the mover's frame, ``mover`` that move's
        origin, plane, piece and whether it is real; ``black`` and ``state`` are the mover's.
        """

        count = len(after)
        boards = torch.cat((self.swap[after[:, :64].flip(1)], after[:, 64:]), dim=1)
        # Their castling rights: the mover's opponent rook at file 0 lands on file 7 once rotated.
        castle = torch.stack((state[:, 5], state[:, 4]), dim=1) > 0.5
        double = (mover["kind"] == 1) & (mover["plane"] == _DOUBLE_PUSH)
        ep = torch.where(double, 55 - mover["origin"], _OFF)
        origins = self._own_squares(boards)
        moves = self._pseudo_moves(boards, origins, castle, ep, 3 + black.long())
        moves &= mover["valid"][:, None, None]
        kinds = boards.gather(1, origins)
        # Integer indices only: a list index is a CPU tensor, which graph capture cannot copy.
        special = torch.zeros_like(moves)
        special[..., _CASTLE_EAST] = kinds == 6
        special[..., _CASTLE_WEST] = kinds == 6
        for plane in _PAWN_DIAGONALS:
            landing = self.destinations[:, plane][origins] == ep[:, None]
            special[..., plane] = (kinds == 1) & landing
        special &= moves
        info = self._defence(boards)

        # Ordinary replies: legality and check from per-position tables, no boards needed.
        index, valid = _compact(moves & ~special, sizes["replies"])
        reply_rows = index // (16 * PLANES_PER_SQUARE)
        reply_plane = index % PLANES_PER_SQUARE
        reply_origin = origins[reply_rows, (index // PLANES_PER_SQUARE) % 16]
        legal, gives, capture = self._classify(boards, info, reply_rows, reply_origin, reply_plane)
        captures, captured = _compact(valid & legal & capture, sizes["captures"])
        checks, checking = _compact(valid & legal & gives, sizes["checks"])

        # En passant and castling replies are played out and tested on their boards.
        index, special_valid = _compact(special, sizes["special"])
        special_rows = index // (16 * PLANES_PER_SQUARE)
        special_plane = index % PLANES_PER_SQUARE
        special_origin = origins[special_rows, (index // PLANES_PER_SQUARE) % 16]
        played, _, _, _, passing, castling = self._play(
            boards, special_rows, special_origin, special_plane
        )
        king = self._king(played, 6)
        special_legal = ~self._attacked(played, king[:, None], theirs=True)[:, 0]
        # Castling also needs no check and a safe square to pass; the step east is ray 2's first.
        passed = torch.where(special_plane == _CASTLE_EAST, 2, 6)
        calm = (info["checks"][special_rows] == 0) & ~info["unsafe"][special_rows, passed]
        special_legal &= special_valid & (~castling | calm)
        special_check = self._attacked(played, self._king(played, 12)[:, None], theirs=False)[:, 0]

        capture_rows = reply_rows[captures]
        taken, moved, promoted, victim, _, _ = self._play(
            boards, capture_rows, reply_origin[captures], reply_plane[captures]
        )
        gains = self.values[victim] + self.values[promoted] - self.values[moved]
        targets = self.destinations[reply_origin[captures], reply_plane[captures]]
        ep_capture = special_legal & passing

        check_rows = reply_rows[checks]
        check_origin = reply_origin[checks]
        checked, check_kind, _, _, _, _ = self._play(
            boards, check_rows, check_origin, reply_plane[checks]
        )
        # The mover may answer a checking double push en passant, on the square it skipped.
        double = (check_kind == 1) & (reply_plane[checks] == _DOUBLE_PUSH)
        their_ep = torch.where(double, check_origin + 8, _OFF)
        mated = self._mated(
            torch.cat((checked, played)),
            torch.cat((their_ep, torch.full_like(special_rows, _OFF))),
        )
        mated &= torch.cat((checking, special_legal & special_check))
        mate = torch.zeros(count, dtype=torch.long, device=after.device).scatter_reduce(
            0, torch.cat((check_rows, special_rows)), mated.long(), "amax"
        )
        return {
            "boards": torch.cat((taken, played)),
            "targets": torch.cat((targets, self.destinations[special_origin, special_plane])),
            "gains": torch.cat((gains, torch.ones_like(special_rows))),
            "rows": torch.cat((capture_rows, special_rows)),
            "valid": torch.cat((captured, ep_capture)).long(),
            "mate": mate,
        }

    def _own_squares(self, boards: Tensor) -> Tensor:
        """(n, 16) squares of the side to move's pieces in order, padded with 64."""

        squares = torch.arange(64, device=boards.device)
        own = (boards[:, :64] >= 1) & (boards[:, :64] <= 6)
        keys = torch.where(own, squares, 64 + squares)
        return keys.sort(dim=1).values[:, :16].clamp_max(_OFF)

    def _king(self, boards: Tensor, king: int) -> Tensor:
        return (boards[:, :64] == king).long().argmax(dim=1)

    def _along(self, boards: Tensor, squares: Tensor) -> tuple[Tensor, Tensor]:
        """Pieces (long) and squares along the eight rays from each of ``squares``."""

        lines = self.rays[squares]
        pieces = boards.gather(1, lines.flatten(1)).view(lines.shape)
        return pieces, lines

    def _running(self, flags: Tensor) -> Tensor:
        """Float counts of set flags up to each of the 8 entries along the last dimension.

        A matmul, because CUDA scans along a dimension this short are slow.
        """

        return flags.float() @ self.upper

    def _fronts(self, pieces: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Index and piece of the first and second pieces along each ray; index 7 is none."""

        seen = self._running(pieces != 0)
        first = (seen == 0).sum(-1, keepdim=True).clamp_max(7)
        second = (seen <= 1).sum(-1, keepdim=True).clamp_max(7)
        return (
            first.squeeze(-1),
            pieces.gather(-1, first).squeeze(-1),
            second.squeeze(-1),
            pieces.gather(-1, second).squeeze(-1),
        )

    def _attacked(
        self, boards: Tensor, squares: Tensor, *, theirs: bool, without: int | None = None
    ) -> Tensor:
        """(n, k) whether the opponent (``theirs``) or the side to move attacks each square.

        ``without`` lifts every piece with that id first.
        """

        pieces, _ = self._along(boards, squares)
        if without is not None:
            pieces = pieces.masked_fill(pieces == without, 0)
        first, front, _, _ = self._fronts(pieces)
        rays = torch.arange(8, device=boards.device)
        side = front >= 7 if theirs else (front >= 1) & (front <= 6)
        hits = (self.capable[front, rays, first] & side).any(-1)
        jumps = boards.gather(1, self.knights[squares].flatten(1)).view(
            *squares.shape, len(KNIGHT_DELTAS)
        )
        return hits | (jumps == (8 if theirs else 2)).any(-1)

    def _pseudo_moves(
        self, boards: Tensor, origins: Tensor, castle: Tensor, ep: Tensor, king_file: Tensor
    ) -> Tensor:
        """(n, k, 73) pseudo-legal moves of the side to move from each of ``origins``.

        ``castle`` holds the rights with the rook on file 0 and file 7, ``ep`` the en passant
        square (64 for none), ``king_file`` the file of the king's starting square.
        """

        kind = boards.gather(1, origins)
        ahead, lines = self._along(boards, origins)
        occupied = ahead != 0
        seen = self._running(occupied)
        clear = seen == occupied.float()
        empty = ~occupied
        landing = (lines != _OFF) & clear & (empty | (ahead >= 7))
        shape = kind[..., None, None]
        rays = torch.arange(8, device=boards.device)[:, None]
        steps = torch.arange(8, device=boards.device)
        diagonal = rays % 2 == 1
        reach = (((shape == 3) | (shape == 5)) & diagonal) | (
            ((shape == 4) | (shape == 5)) & ~diagonal
        )
        reach = reach | ((shape == 6) & (steps == 0))
        rank = (origins // 8)[..., None, None]
        forward = rays == 0
        pawn = (forward & (steps == 0) & empty) | (forward & (steps == 1) & empty & (rank == 1))
        sideways = ((rays == 1) | (rays == 7)) & (steps == 0)
        pawn = pawn | (sideways & ((ahead >= 7) | (lines == ep[:, None, None, None])))
        along = landing & torch.where(shape == 1, pawn, reach)
        along = along[..., :7].flatten(-2)
        jumps = self.knights[origins]
        target = boards.gather(1, jumps.flatten(1)).view(jumps.shape)
        jumping = (kind == 2).unsqueeze(-1) & (jumps != _OFF) & ((target == 0) | (target >= 7))
        # Underpromotions share the queen promotion's squares: north-west, north, north-east.
        promoting = ((kind == 1) & (origins // 8 == 6)).unsqueeze(-1)
        under = torch.stack((along[..., 49], along[..., 0], along[..., 7]), dim=-1) & promoting
        moves = torch.cat((along, jumping, under.repeat(1, 1, 3)), dim=-1)
        # Castling: the king on its start square, the right, and only empty squares to the rook.
        first = (seen == 0).sum(-1, keepdim=True).clamp_max(7)
        first_piece = ahead.gather(-1, first).squeeze(-1)
        first_square = lines.gather(-1, first).squeeze(-1)
        home = (kind == 6) & (origins == king_file[:, None])
        east = home & castle[:, 1:] & (first_piece[..., 2] == 4) & (first_square[..., 2] == 7)
        west = home & castle[:, :1] & (first_piece[..., 6] == 4) & (first_square[..., 6] == 0)
        moves[..., _CASTLE_EAST] |= east
        moves[..., _CASTLE_WEST] |= west
        return moves & ((kind >= 1) & (kind <= 6)).unsqueeze(-1)

    def _play(
        self, boards: Tensor, rows: Tensor, origins: Tensor, planes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Boards after each move, with the moving piece, the piece it becomes, the piece on its
        destination, and whether it was en passant or castling."""

        board = boards[rows].clone()
        kind = board.gather(1, origins[:, None]).squeeze(1)
        target = self.destinations[origins, planes]
        victim = board.gather(1, target[:, None]).squeeze(1)
        promotion = torch.where(planes >= 64, 2 + (planes - 64) // 3, 5)
        new = torch.where((kind == 1) & (target >= 56) & (target < _OFF), promotion, kind)
        en_passant = (kind == 1) & ((target - origins) % 8 != 0) & (victim == 0)
        castle = (kind == 6) & ((target - origins).abs() == 2)
        corner = origins - origins % 8 + torch.where(target > origins, 7, 0)
        board.scatter_(1, origins[:, None], 0)
        board.scatter_(1, target[:, None], new[:, None])
        board.scatter_(1, torch.where(en_passant, target - 8, _OFF)[:, None], 0)
        board.scatter_(1, torch.where(castle, corner, _OFF)[:, None], 0)
        rook = torch.where(castle, (origins + target) // 2, _OFF)
        board.scatter_(1, rook[:, None], torch.where(castle, 4, 0)[:, None])
        return board, kind, new, victim, en_passant, castle

    def _defence(self, boards: Tensor) -> dict[str, Tensor]:
        """Per position, with the side to move's king K and the opponent's king J: checks on K,
        squares that answer them, pins toward K, attacks on K's steps with K lifted, and the rays
        from J."""

        rays = torch.arange(8, device=boards.device)
        king = self._king(boards, 6)
        pieces, lines = self._along(boards, king[:, None])
        pieces, lines = pieces[:, 0], lines[:, 0]
        first, front, second, back = self._fronts(pieces)
        checking = (front >= 7) & self.capable[front, rays, first]
        jumps = self.knights[king]
        jumping = boards.gather(1, jumps) == 8
        checks = checking.sum(1) + jumping.sum(1)
        # Squares that capture the checker or block its line; all of them while not in check.
        line = checking.unsqueeze(-1) & (torch.arange(8, device=boards.device) <= first[..., None])
        answers = torch.zeros_like(boards, dtype=torch.bool)
        answers.scatter_(1, torch.where(line, lines, _OFF).flatten(1), True)
        answers.scatter_(1, torch.where(jumping, jumps, _OFF), True)
        answers = torch.where((checks == 0)[:, None], True, answers & (checks == 1)[:, None])
        pinned = (front >= 1) & (front <= 6) & (back >= 7) & self.capable[back, rays, second]
        blocker = lines.gather(-1, first[..., None]).squeeze(-1)
        pin = torch.full_like(boards, 8).scatter_(
            1, torch.where(pinned, blocker, _OFF), torch.where(pinned, rays, 8)
        )
        # K's steps, then the castling squares two steps east and west.
        steps = torch.cat((lines[:, :, 0], lines[:, 2:3, 1], lines[:, 6:7, 1]), dim=1)
        unsafe = self._attacked(boards, steps, theirs=True, without=6)
        their_king = self._king(boards, 12)
        pieces, lines = self._along(boards, their_king[:, None])
        occupied = pieces[:, 0] != 0
        first, _, second, back = self._fronts(pieces[:, 0])
        revealing = (back >= 1) & (back <= 6) & self.capable[back, rays, second]
        return {
            "king": king,
            "checks": checks,
            "answers": answers,
            "pin": pin,
            "unsafe": unsafe,
            "their_king": their_king,
            "before": self._running(occupied) - occupied.float(),
            "first": first,
            "second": second,
            "revealing": revealing,
        }

    def _classify(
        self, boards: Tensor, info: dict[str, Tensor], rows: Tensor, origins: Tensor, planes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Legality, check and capture of ordinary moves (no en passant or castling)."""

        target = self.destinations[origins, planes]
        kind = boards[rows, origins]
        victim = boards[rows, target]
        promotion = torch.where(planes >= 64, 2 + (planes - 64) // 3, 5)
        new = torch.where((kind == 1) & (target >= 56), promotion, kind)
        king = info["king"][rows]
        pin = info["pin"][rows, origins]
        # A king steps along the ray of its plane; ``unsafe`` lists its steps by ray.
        stepping = ~info["unsafe"][rows, (planes // 7).clamp_max(7)]
        staying = (pin == 8) | (self.ray_of[king, target] == pin)
        legal = torch.where(kind == 6, stepping, info["answers"][rows, target] & staying)
        their_king = info["their_king"][rows]
        ray, index = self.ray_of[their_king, target], self.index_of[their_king, target]
        left, gap = self.ray_of[their_king, origins], self.index_of[their_king, origins]
        # Pieces between the destination and J, not counting the square the mover left.
        between = (
            info["before"][rows, ray.clamp_max(7), index] - ((left == ray) & (gap < index)).long()
        )
        direct = self.capable[new, ray, index] & (between == 0)
        direct |= (new == 2) & self.knight_near[their_king, target]
        opened = left.clamp_max(7)
        discovered = (
            (left < 8)
            & (gap == info["first"][rows, opened])
            & info["revealing"][rows, opened]
            & ~((ray == left) & (index < info["second"][rows, opened]))
        )
        return legal, direct | discovered, victim >= 7

    def _mated(self, boards: Tensor, ep: Tensor) -> Tensor:
        """Whether the side to move is checkmated, in the opponent's frame of each board.

        Boards are rotated to the checked side's frame first; ``ep`` is in the opponent's frame.
        """

        if len(boards) <= _MATE_CHUNK:
            return self._mated_chunk(boards, ep)
        return torch.cat(
            [
                self._mated_chunk(
                    boards[start : start + _MATE_CHUNK], ep[start : start + _MATE_CHUNK]
                )
                for start in range(0, len(boards), _MATE_CHUNK)
            ]
        )

    def _mated_chunk(self, boards: Tensor, ep: Tensor) -> Tensor:
        boards = torch.cat((self.swap[boards[:, :64].flip(1)], boards[:, 64:]), dim=1)
        device = boards.device
        rays = torch.arange(8, device=device)
        king = self._king(boards, 6)
        pieces, lines = self._along(boards, king[:, None])
        pieces, lines = pieces[:, 0], lines[:, 0]
        first, front, second, back = self._fronts(pieces)
        checking = (front >= 7) & self.capable[front, rays, first]
        jumps = self.knights[king]
        jumping = boards.gather(1, jumps) == 8
        checks = checking.sum(1) + jumping.sum(1)
        pinned_line = (front >= 1) & (front <= 6) & (back >= 7) & self.capable[back, rays, second]
        blocker = lines.gather(-1, first[..., None]).squeeze(-1)
        pinned = torch.zeros_like(boards, dtype=torch.bool)
        pinned.scatter_(1, torch.where(pinned_line, blocker, _OFF), True)
        pinned[:, _OFF] = False
        # King steps to squares it does not hold, not attacked once the king is lifted.
        steps = lines[:, :, 0]
        free = (steps != _OFF) & ~((boards.gather(1, steps) >= 1) & (boards.gather(1, steps) <= 6))
        escape = (free & ~self._attacked(boards, steps, theirs=True, without=6)).any(1)
        # With one checker: its square and, for a slider, the squares between it and the king.
        ray = checking.long().argmax(1)
        reach = first.gather(1, ray[:, None])
        on_line = lines.gather(1, ray[:, None, None].expand(-1, 1, 8))[:, 0, :7]
        near = checking.any(1, keepdim=True) & (torch.arange(7, device=device) <= reach)
        jumper = jumps.gather(1, jumping.long().argmax(1, keepdim=True))
        answers = torch.cat((on_line, jumper), dim=1)
        usable = torch.cat((near, jumping.any(1, keepdim=True)), dim=1)
        usable &= (answers != _OFF) & (checks == 1)[:, None]
        occupant = boards.gather(1, answers)
        pieces, lines = self._along(boards, answers)
        first, front, _, _ = self._fronts(pieces)
        source = lines.gather(-1, first[..., None]).squeeze(-1)
        movers = (front >= 1) & (front <= 5) & self.capable[front, rays, first]
        # Pawns take diagonally and push straight; the pushes are counted below.
        movers &= (front != 1) | (occupant != 0)[..., None]
        movers &= ~pinned.gather(1, source.flatten(1)).view(source.shape)
        knights = self.knights[answers]
        leapers = boards.gather(1, knights.flatten(1)).view(knights.shape) == 2
        leapers &= ~pinned.gather(1, knights.flatten(1)).view(knights.shape)
        below = (answers - 8).clamp_min(0)
        further = (answers - 16).clamp_min(0)
        vacant = (occupant == 0) & (answers >= 8) & (answers < _OFF)
        push = vacant & (boards.gather(1, below) == 1) & ~pinned.gather(1, below)
        double = vacant & (answers // 8 == 3) & (boards.gather(1, below) == 0)
        double &= (boards.gather(1, further) == 1) & ~pinned.gather(1, further)
        block = usable & (movers.any(-1) | leapers.any(-1) | push | double)
        # En passant, rotated into the checked side's frame, is played out and tested.
        square = torch.where(ep < _OFF, 63 - ep, _OFF)
        file = square % 8
        passers = torch.stack((square - 9, square - 7), dim=1)
        ready = (square[:, None] < _OFF) & torch.stack((file > 0, file < 7), dim=1)
        passers = torch.where(ready, passers, _OFF)
        ready &= boards.gather(1, passers) == 1
        taken = boards.repeat_interleave(2, dim=0)
        flat = passers.flatten()
        landing = torch.where(ready.flatten(), square.repeat_interleave(2), _OFF)
        taken.scatter_(1, flat[:, None], 0)
        taken.scatter_(1, landing[:, None], torch.where(ready.flatten(), 1, 0)[:, None])
        victim = torch.where(ready.flatten(), landing - 8, _OFF)
        taken.scatter_(1, victim[:, None], 0)
        taken[:, _OFF] = 0
        exposed = self._attacked(taken, king.repeat_interleave(2)[:, None], theirs=True)
        passing = (ready & ~exposed.view(-1, 2)).any(1)
        return (checks > 0) & ~escape & ~block.any(1) & ~passing

    def _exchange(
        self, boards: Tensor, target: Tensor, gains: Tensor, black: Tensor, static: bool
    ) -> Tensor:
        """``static_exchange`` for boards on which the side to move just took on ``target``.

        ``gains`` is what that first capture netted; the opponent recaptures first. ``black``
        marks boards whose side to move is Black, which orders equally cheap attackers by their
        absolute squares as python-chess does.
        """

        device = boards.device
        rays = torch.arange(8, device=device)[:, None]
        steps = torch.arange(8, device=device)
        pieces, lines = self._along(boards, target[:, None])
        pieces, lines = pieces[:, 0], lines[:, 0]
        # Along each ray the attackers are the unbroken run of pieces that attack the target.
        capable = self.capable[pieces, rays, steps]
        occupied = pieces != 0
        run = occupied & capable & (self._running(occupied & ~capable) == 0)
        order = torch.where(run, self._running(run).long() - 1, 7)
        chain = torch.zeros_like(pieces).scatter_(-1, order, torch.where(run, pieces, 0))
        chain_squares = torch.full_like(lines, _OFF)
        chain_squares.scatter_(-1, order, torch.where(run, lines, _OFF))
        jumps = self.knights[target]
        knights = boards.gather(1, jumps)
        knights = torch.where((knights == 2) | (knights == 8), knights, 0)
        chain_absolute = torch.where(black[:, None, None], 63 - chain_squares, chain_squares)
        knight_absolute = torch.where(black[:, None], 63 - jumps, jumps)
        slots = torch.arange(16, device=device)
        if static:
            rounds = _MAX_ROUNDS
        elif len(target):
            rounds = int((run.sum((1, 2)) + (knights != 0).sum(1)).max())
        else:
            rounds = 0
        count = len(target)
        pointer = torch.zeros(count, 8, dtype=torch.long, device=device)
        used = torch.zeros(count, 8, dtype=torch.bool, device=device)
        theirs = torch.ones(count, dtype=torch.bool, device=device)
        active = torch.ones(count, dtype=torch.bool, device=device)
        on_target = self.values[boards.gather(1, target[:, None]).squeeze(1)]
        history = torch.zeros(count, rounds + 2, dtype=torch.long, device=device)
        history[:, 0] = gains
        length = torch.ones(count, dtype=torch.long, device=device)

        def candidates() -> tuple[Tensor, Tensor]:
            front = chain.gather(-1, pointer[..., None]).squeeze(-1)
            where = chain_absolute.gather(-1, pointer[..., None]).squeeze(-1)
            return (
                torch.cat((front, knights.masked_fill(used, 0)), dim=1),
                torch.cat((where, knight_absolute), dim=1),
            )

        for _ in range(rounds):
            pieces, where = candidates()
            side = torch.where(theirs[:, None], pieces >= 7, (pieces >= 1) & (pieces <= 6))
            key = torch.where(side, self.values[pieces] * 64 + where, _BIG)
            best_key, best = key.min(dim=1)
            found = best_key < _BIG
            taken = (best[:, None] == slots) & (found & active)[:, None]
            pointer = (pointer + taken[:, :8].long()).clamp_max(7)
            used |= taken[:, 8:]
            value = best_key // 64
            # A king may not recapture onto a square the other side still attacks.
            pieces, _ = candidates()
            other = torch.where(theirs[:, None], (pieces >= 1) & (pieces <= 6), pieces >= 7)
            go = active & found & ~((value == 100) & other.any(1))
            previous = history.gather(1, (length - 1)[:, None]).squeeze(1)
            slot = torch.where(go, length, rounds + 1)
            history.scatter_(1, slot[:, None], (on_target - previous)[:, None])
            on_target = torch.where(go, value, on_target)
            length = length + go.long()
            theirs = theirs ^ go
            active = go
        for depth in range(rounds, 0, -1):
            folded = torch.minimum(history[:, depth - 1], -history[:, depth])
            history[:, depth - 1] = torch.where(depth < length, folded, history[:, depth - 1])
        return history[:, 0]
