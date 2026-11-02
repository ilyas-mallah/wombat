from __future__ import annotations

import random

import chess
import numpy as np
import torch

from wombat.chess.encode import canonical_square, encode_position
from wombat.model.board_features import (
    ATTACK_CODES,
    ATTACKER_CLASSES,
    BOARD_FEATURES,
    COUNT_CLASSES,
    LONG_RANGE_FEATURES,
    MATERIAL_SCALE,
    XRAY_CLASSES,
    AttackPairs,
    BoardFeatures,
    MoveGraph,
)

# Cheapest attacker class: 0 is none, then pawn, knight or bishop, rook, queen, king.
PIECE_CLASS = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
SIDE_WIDTH = COUNT_CLASSES + ATTACKER_CLASSES

# Pins along ranks, files and diagonals by each slider, for either side, blocked pins, and checks
# by every piece type against each colour.
HAND_MADE = (
    "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1",
    "4k3/8/8/8/b7/8/2N5/3K4 w - - 0 1",
    "3qk3/8/8/1B6/8/8/8/4K2R b K - 0 1",
    "k7/1p6/8/8/8/5Q2/8/7K b - - 0 1",
    "4k3/4r3/8/4p3/8/8/4N3/4K3 w - - 0 1",
    "4k3/4n3/8/8/8/8/8/4RK2 w - - 0 1",
    "4k3/4n3/8/8/8/8/8/4RK2 b - - 0 1",
    "r3k3/8/8/8/8/8/8/rR2K3 w - - 0 1",
    "1k6/8/8/8/8/8/8/qP1QK3 w - - 0 1",
    "4k3/8/8/8/8/5n2/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1",
    "4k3/3P4/8/8/8/8/8/4K3 b - - 0 1",
    "4k3/8/8/8/8/8/5p2/4K3 w - - 0 1",
    "8/8/8/8/8/2k5/8/K1q5 w - - 0 1",
    "4k3/8/8/1b6/8/8/4R3/5K2 w - - 0 1",
    "6k1/5P2/8/8/8/8/8/K7 b - - 0 1",
    "7k/8/8/8/8/8/8/K5Rq b - - 0 1",
    # Nine attackers on d4, more than the clipped count shows, for either side to move.
    "4k3/8/2N1N3/1N3N2/8/1N3N2/2N1N3/3RK3 w - - 0 1",
    "4k3/8/2N1N3/1N3N2/8/1N3N2/2N1N3/3RK3 b - - 0 1",
)


def _played_positions(count: int, seed: int) -> list[chess.Board]:
    randomizer = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(randomizer.randrange(0, 160)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        boards.append(board)
    return boards


def _scattered_positions(count: int, seed: int) -> list[chess.Board]:
    """One king each and up to 24 other pieces on random squares, often in check or pinned."""

    randomizer = random.Random(seed)
    others = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    boards: list[chess.Board] = []
    for _ in range(count):
        board = chess.Board.empty()
        squares = randomizer.sample(range(64), 2 + randomizer.randrange(0, 25))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for square in squares[2:]:
            color = randomizer.random() < 0.5
            board.set_piece_at(square, chess.Piece(randomizer.choice(others), color))
        board.turn = randomizer.random() < 0.5
        boards.append(board)
    return boards


def _features(boards: list[chess.Board]) -> torch.Tensor:
    pieces = np.stack([encode_position(board).pieces for board in boards])
    with torch.no_grad():
        return BoardFeatures()(torch.from_numpy(pieces).long())


def _cheapest(board: chess.Board, color: chess.Color, square: chess.Square) -> int:
    pieces = board.piece_map()
    attackers = board.attackers(color, square)
    return min((PIECE_CLASS[pieces[source].piece_type] for source in attackers), default=0)


def _check_against_python_chess(boards: list[chess.Board]) -> dict[str, int]:
    features = _features(boards)
    assert features.shape == (len(boards), 64, BOARD_FEATURES)
    assert features.dtype == torch.float32
    assert set(features.unique().tolist()) <= {0.0, 1.0}
    totals = {"attacked": 0, "pinned": 0, "check": 0, "black": 0, "crowded": 0}
    for board, position in zip(boards, features):
        us, them = board.turn, not board.turn
        in_check = board.is_check()
        totals["check"] += in_check
        totals["black"] += us == chess.BLACK
        for square in chess.SQUARES:
            row = position[canonical_square(square, us)]
            for index, color in enumerate((us, them)):
                side = row[index * SIDE_WIDTH : (index + 1) * SIDE_WIDTH]
                count, category = side[:COUNT_CLASSES], side[COUNT_CLASSES:]
                assert count.sum() == 1 and category.sum() == 1
                attackers = len(board.attackers(color, square))
                totals["attacked"] += attackers > 0
                totals["crowded"] += attackers > COUNT_CLASSES - 1
                where = (board.fen(), square)
                assert int(count.argmax()) == min(attackers, COUNT_CLASSES - 1), where
                assert int(category.argmax()) == _cheapest(board, color, square), where
                piece = board.piece_at(square)
                pinned = piece is not None and piece.color == color
                pinned = pinned and board.is_pinned(color, square)
                totals["pinned"] += pinned
                assert bool(row[2 * SIDE_WIDTH + index]) == pinned, where
            assert bool(row[-1]) == in_check, board.fen()
    return totals


def test_hand_made_pins_and_checks_match_python_chess() -> None:
    boards = [chess.Board(fen) for fen in HAND_MADE]
    totals = _check_against_python_chess(boards)
    assert totals["pinned"] >= 8 and totals["check"] >= 7
    assert totals["crowded"] >= 2


def test_played_positions_match_python_chess() -> None:
    boards = _played_positions(400, seed=11)
    totals = _check_against_python_chess(boards)
    assert 150 <= totals["black"] <= 250
    assert totals["check"] >= 10 and totals["pinned"] >= 10


def test_scattered_positions_match_python_chess() -> None:
    boards = _scattered_positions(400, seed=12)
    totals = _check_against_python_chess(boards)
    assert 150 <= totals["black"] <= 250
    assert totals["check"] >= 100 and totals["pinned"] >= 50


def test_a_blocked_slider_attacks_its_blocker_but_not_past_it() -> None:
    board = chess.Board("4k3/8/8/8/4p3/8/8/4R1K1 w - - 0 1")
    row = _features([board])[0]
    for square, attacked in ((chess.E4, True), (chess.E5, False), (chess.E2, True)):
        count = row[canonical_square(square, chess.WHITE), :COUNT_CLASSES]
        assert int(count.argmax()) == int(attacked)


def test_tables_are_buffers_left_out_of_the_state_dict() -> None:
    features = BoardFeatures()
    assert not list(features.state_dict())
    assert not list(features.parameters())
    assert {name for name, _ in features.named_buffers()} == {"steps", "rays", "orthogonal"}


SLIDERS = {chess.BISHOP: (True, False), chess.ROOK: (False, True), chess.QUEEN: (True, True)}
# Long-range column offsets after BOARD_FEATURES, as documented in wombat.model.board_features.
SOLE, CHECKING, CHECK_LINE, DISCOVERED, XRAY, MATERIAL = 0, 2, 10, 12, 14, 20
# An overloaded defender (d6) owned by either side to move; discovered checks by a rook (e4) for
# either side to move, blocked twice or by an enemy piece, and by a bishop for each colour;
# batteries on a file and a diagonal; two x-rays on one square for either side to move; the side
# not to move in check; a bare pair of kings.
LONG_RANGE_HAND_MADE = (
    "4k3/8/3q3n/8/1n6/8/8/1R2K2Q w - - 0 1",
    "4k3/8/3q3n/8/1n6/8/8/1R2K2Q b - - 0 1",
    "4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1",
    "4k3/8/8/8/4N3/8/8/K3R3 b - - 0 1",
    "4k3/8/4p3/8/4N3/8/8/K3R3 w - - 0 1",
    "4k3/8/8/8/4n3/8/8/K3R3 w - - 0 1",
    "8/6k1/8/8/3N4/8/1B6/K7 w - - 0 1",
    "8/1b6/8/3n4/8/8/6K1/k7 b - - 0 1",
    "4k3/8/8/8/8/4R3/4R3/4Q1K1 w - - 0 1",
    "k7/8/8/8/4B3/8/2Q5/7K w - - 0 1",
    "4k3/8/8/8/8/2BR4/1Q1R4/4K3 b - - 0 1",
    "4k3/8/8/8/8/2BR4/1Q1R4/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4RK2 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
)


def _line_kind(source: chess.Square, target: chess.Square) -> tuple[bool, bool]:
    """Whether two distinct squares share a diagonal, and whether they share a rank or file."""

    files = chess.square_file(source) - chess.square_file(target)
    ranks = chess.square_rank(source) - chess.square_rank(target)
    if source == target:
        return False, False
    return abs(files) == abs(ranks), files == 0 or ranks == 0


def _pieces_between(board: chess.Board, source: chess.Square, target: chess.Square) -> int:
    return len(chess.SquareSet(chess.between(source, target)) & board.occupied)


def _slides_to(board: chess.Board, slider: chess.Square, target: chess.Square) -> bool:
    diagonal, orthogonal = _line_kind(slider, target)
    piece_type = board.piece_type_at(slider)
    assert piece_type is not None
    moves_diagonally, moves_orthogonally = SLIDERS[piece_type]
    return (diagonal and moves_diagonally) or (orthogonal and moves_orthogonally)


def _sliders(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    return [square for kind in SLIDERS for square in board.pieces(kind, color)]


def _sole_defenders(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    """``color``'s pieces that alone defend an attacked non-king piece of their own."""

    marked: set[chess.Square] = set()
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        defenders = board.attackers(color, square)
        if board.attackers(not color, square) and len(defenders) == 1:
            marked.add(next(iter(defenders)))
    return marked


def _checking(board: chess.Board, color: chess.Color, square: chess.Square) -> list[bool]:
    """Whether a knight, diagonal slider, orthogonal slider or pawn of ``color`` would check."""

    king = board.king(not color)
    assert king is not None
    diagonal, orthogonal = _line_kind(square, king)
    clear = _pieces_between(board, square, king) == 0
    return [
        bool(chess.BB_KNIGHT_ATTACKS[king] & chess.BB_SQUARES[square]),
        diagonal and clear,
        orthogonal and clear,
        bool(chess.BB_PAWN_ATTACKS[color][square] & chess.BB_SQUARES[king]),
    ]


def _check_line(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    """Every checker of ``color``'s king and the squares between it and the king."""

    king = board.king(color)
    assert king is not None
    squares: set[chess.Square] = set()
    for checker in board.attackers(not color, king):
        squares.add(checker)
        squares.update(chess.SquareSet(chess.between(checker, king)))
    return squares


def _discoverers(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    """``color``'s pieces that alone stand between one of its sliders and the enemy king."""

    king = board.king(not color)
    assert king is not None
    found: set[chess.Square] = set()
    for slider in _sliders(board, color):
        blockers = chess.SquareSet(chess.between(slider, king)) & board.occupied
        if _slides_to(board, slider, king) and len(blockers) == 1:
            blocker = next(iter(blockers))
            if board.color_at(blocker) == color:
                found.add(blocker)
    return found


def _xrays(board: chess.Board, color: chess.Color, square: chess.Square) -> int:
    """``color``'s sliders reaching ``square`` through exactly one piece of either colour."""

    return sum(
        _slides_to(board, slider, square) and _pieces_between(board, slider, square) == 1
        for slider in _sliders(board, color)
    )


def _long_range_reference(board: chess.Board) -> np.ndarray:
    """The long-range columns square by square in the real frame, stored at canonical squares."""

    us, them = board.turn, not board.turn
    sole = [_sole_defenders(board, color) for color in (us, them)]
    lines = [_check_line(board, color) for color in (us, them)]
    discoverers = [_discoverers(board, color) for color in (us, them)]
    material = [
        float(np.float32(len(board.pieces(kind, color))) / np.float32(scale))
        for color in (us, them)
        for kind, scale in zip(chess.PIECE_TYPES, MATERIAL_SCALE)
    ]
    rows = np.zeros((64, LONG_RANGE_FEATURES), dtype=np.float32)
    for square in chess.SQUARES:
        values: list[float] = [square in sole[0], square in sole[1]]
        values += _checking(board, us, square) + _checking(board, them, square)
        values += [square in lines[0], square in lines[1]]
        values += [square in discoverers[0], square in discoverers[1]]
        for color in (us, them):
            xrays = min(_xrays(board, color, square), XRAY_CLASSES - 1)
            values += [float(xrays == count) for count in range(XRAY_CLASSES)]
        rows[canonical_square(square, us)] = values + material
    return rows


def _long_range_features(boards: list[chess.Board]) -> torch.Tensor:
    pieces = torch.from_numpy(np.stack([encode_position(board).pieces for board in boards]))
    with torch.no_grad():
        features = BoardFeatures(long_range=True)(pieces.long())
    assert features.shape == (len(boards), 64, BOARD_FEATURES + LONG_RANGE_FEATURES)
    assert features.dtype == torch.float32
    assert torch.equal(features[..., :BOARD_FEATURES], _features(boards))
    return features[..., BOARD_FEATURES:]


def _check_long_range(boards: list[chess.Board]) -> np.ndarray:
    """Compare every long-range column with python-chess; return each flag column's count."""

    features = _long_range_features(boards).numpy()
    for board, position in zip(boards, features):
        expected = _long_range_reference(board)
        wrong = np.argwhere(position != expected)
        if len(wrong):
            index, column = (int(value) for value in wrong[0])
            square = chess.square_name(canonical_square(index, board.turn))
            found, wanted = position[index, column], expected[index, column]
            raise AssertionError(f"{board.fen()} {square} column {column}: {found} != {wanted}")
    assert set(np.unique(features[..., :MATERIAL]).tolist()) <= {0.0, 1.0}
    return (features[..., :MATERIAL] > 0).sum(axis=(0, 1))


def _enough(counts: np.ndarray, least: int, legal: bool = False) -> None:
    """Every flag, and each x-ray class above zero, occurs at least ``least`` times.

    In ``legal`` positions the side not to move is never in check, so its check line is empty.
    """

    xrays = [XRAY + side * XRAY_CLASSES + count for side in (0, 1) for count in (1, 2)]
    for column in (*range(SOLE, XRAY), *xrays):
        if legal and column == CHECK_LINE + 1:
            assert counts[column] == 0
        else:
            assert counts[column] >= least, (column, counts.tolist())


def test_long_range_hand_made_positions_match_python_chess() -> None:
    boards = [chess.Board(fen) for fen in (*HAND_MADE, *LONG_RANGE_HAND_MADE)]
    _enough(_check_long_range(boards), 1)


def test_long_range_played_positions_match_python_chess() -> None:
    boards = _played_positions(400, seed=13)
    assert 150 <= sum(board.turn == chess.BLACK for board in boards) <= 250
    _enough(_check_long_range(boards), 10, legal=True)


def test_long_range_scattered_positions_match_python_chess() -> None:
    boards = _scattered_positions(400, seed=14)
    assert 150 <= sum(board.turn == chess.BLACK for board in boards) <= 250
    _enough(_check_long_range(boards), 40)


def _row(fen: str, square: chess.Square) -> torch.Tensor:
    board = chess.Board(fen)
    return _long_range_features([board])[0, canonical_square(square, board.turn)]


def _xray_count(fen: str, square: chess.Square, side: int = 0) -> int:
    start = XRAY + side * XRAY_CLASSES
    return int(_row(fen, square)[start : start + XRAY_CLASSES].argmax())


def test_an_overloaded_defender_is_flagged_for_its_side() -> None:
    white_to_move, black_to_move = LONG_RANGE_HAND_MADE[:2]
    # Black's queen on d6 alone defends both attacked knights.
    assert _row(white_to_move, chess.D6)[SOLE : SOLE + 2].tolist() == [0.0, 1.0]
    assert _row(black_to_move, chess.D6)[SOLE : SOLE + 2].tolist() == [1.0, 0.0]
    assert _row(white_to_move, chess.E8)[SOLE : SOLE + 2].sum() == 0


def test_discovered_checks_need_one_own_piece_before_the_slider() -> None:
    assert _row(LONG_RANGE_HAND_MADE[2], chess.E4)[DISCOVERED : DISCOVERED + 2].tolist() == [1, 0]
    assert _row(LONG_RANGE_HAND_MADE[3], chess.E4)[DISCOVERED : DISCOVERED + 2].tolist() == [0, 1]
    for blocked in LONG_RANGE_HAND_MADE[4:6]:
        assert _row(blocked, chess.E4)[DISCOVERED : DISCOVERED + 2].sum() == 0
    assert _row(LONG_RANGE_HAND_MADE[6], chess.D4)[DISCOVERED] == 1
    assert _row(LONG_RANGE_HAND_MADE[7], chess.D5)[DISCOVERED] == 1


def test_batteries_count_the_rear_slider_as_an_xray() -> None:
    file, diagonal, double = LONG_RANGE_HAND_MADE[8:11]
    # The queen on e1 is third in line, so only the rook on e2 x-rays e4.
    assert [_xray_count(file, square) for square in (chess.E4, chess.E3, chess.E8)] == [1, 1, 1]
    assert [_xray_count(diagonal, square) for square in (chess.F5, chess.H7)] == [1, 1]
    assert _xray_count(diagonal, chess.A8) == 0
    # Black to move: White's queen behind c3 and rook behind d3 both x-ray d4.
    assert _xray_count(double, chess.D4, side=1) == 2


def test_checking_squares_for_every_piece_type() -> None:
    kings = LONG_RANGE_HAND_MADE[-1]
    ours, theirs = slice(CHECKING, CHECKING + 4), slice(CHECKING + 4, CHECKING + 8)
    # Knight, diagonal slider, orthogonal slider and pawn against the king on e8.
    expected = {
        chess.D6: [1, 0, 0, 0],
        chess.F7: [0, 1, 0, 1],
        chess.D7: [0, 1, 0, 1],
        chess.A4: [0, 1, 0, 0],
        chess.E2: [0, 0, 1, 0],
        chess.H8: [0, 0, 1, 0],
        chess.C5: [0, 0, 0, 0],
    }
    for square, flags in expected.items():
        assert _row(kings, square)[ours].tolist() == flags, chess.square_name(square)
    # A black pawn or bishop on d2 would check the king on e1.
    assert _row(kings, chess.D2)[theirs].tolist() == [0, 1, 0, 1]


def test_material_is_broadcast_and_scaled_by_starting_counts() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    features = _long_range_features([board])[0, :, MATERIAL:]
    assert torch.equal(features, features[:1].expand(64, -1))
    assert features[0].tolist() == [1.0] * 10
    promoted = _row("4k3/8/8/8/8/8/PPP5/QQ2K3 b - - 0 1", chess.A1)[MATERIAL:]
    assert promoted.tolist() == [0.0] * 5 + [3 / 8, 0.0, 0.0, 0.0, 2.0]


def test_long_range_tables_are_buffers_only_when_asked() -> None:
    features = BoardFeatures(long_range=True)
    assert not list(features.state_dict()) and not list(features.parameters())
    names = {name for name, _ in features.named_buffers()}
    assert names == {"steps", "rays", "orthogonal", "material_scale"}
    assert features.width == BOARD_FEATURES + LONG_RANGE_FEATURES == 61
    assert BoardFeatures().width == BOARD_FEATURES


KING_STEPS = torch.tensor(
    [[chess.square_distance(a, b) <= 1 for b in chess.SQUARES] for a in chess.SQUARES]
)


def _move_graph_reference(board: chess.Board, lines: bool = False) -> tuple[torch.Tensor, int]:
    """Canonical (64, 64) codes of ``MoveGraph(lines)``, and the position's double pushes."""

    edges = torch.zeros(64, 64, dtype=torch.bool)
    line = torch.zeros(64, 64, dtype=torch.bool)
    doubles = 0
    for square, piece in board.piece_map().items():
        targets = set(board.attacks(square))
        if lines and piece.piece_type in SLIDERS:
            targets = set()
            for target in chess.SQUARES:
                line[square, target] = _slides_to(board, square, target)
        if piece.piece_type == chess.PAWN:
            step, start = (8, 1) if piece.color == chess.WHITE else (-8, 6)
            if 0 <= square + step < 64 and board.piece_at(square + step) is None:
                targets.add(square + step)
                if chess.square_rank(square) == start and board.piece_at(square + 2 * step) is None:
                    targets.add(square + 2 * step)
                    doubles += 1
        for target in targets:
            edges[square, target] = True
    line = line | line.T
    codes = (KING_STEPS | edges | edges.T | line).long()
    for first, second in line.nonzero().tolist():
        codes[first, second] = 2 + min(_pieces_between(board, first, second), 2)
    order = torch.tensor([canonical_square(square, board.turn) for square in chess.SQUARES])
    canonical = torch.zeros_like(codes)
    canonical[order[:, None], order[None]] = codes
    return canonical, doubles


def _check_move_graph(boards: list[chess.Board], lines: bool = False) -> dict[str, int]:
    pieces = torch.from_numpy(np.stack([encode_position(board).pieces for board in boards]))
    graph = MoveGraph(lines)
    with torch.no_grad():
        codes = graph.codes(pieces.long())
        allowed = graph(pieces.long())
    assert allowed.shape == codes.shape == (len(boards), 64, 64) and allowed.dtype == torch.bool
    assert torch.equal(allowed, codes > 0)
    totals = {"black": 0, "far": 0, "doubles": 0, "one": 0, "more": 0}
    for board, row in zip(boards, codes):
        expected, doubles = _move_graph_reference(board, lines)
        assert torch.equal(row, expected), board.fen()
        totals["black"] += board.turn == chess.BLACK
        totals["far"] += int(((row > 0) & ~KING_STEPS).sum())
        totals["doubles"] += doubles
        totals["one"] += int((row == 3).sum())
        totals["more"] += int((row == 4).sum())
    return totals


def test_move_graph_matches_python_chess_on_played_positions() -> None:
    totals = _check_move_graph(_played_positions(400, seed=21))
    assert 150 <= totals["black"] <= 250
    assert totals["far"] >= 10_000 and totals["doubles"] >= 500


def test_move_graph_matches_python_chess_on_scattered_positions() -> None:
    totals = _check_move_graph(_scattered_positions(400, seed=22))
    assert 150 <= totals["black"] <= 250
    assert totals["far"] >= 10_000


def test_move_graph_edges_stop_at_blockers_and_run_both_ways() -> None:
    boards = [
        # A rook on a1 blocked on a5, and a free white double push on h2.
        chess.Board("4k3/8/8/p7/8/8/7P/R3K3 w - - 0 1"),
        # Black to move: b7 pushes two squares, g7 cannot since the knight stands on g5.
        chess.Board("4k3/1p4p1/8/6N1/8/8/8/4K3 b - - 0 1"),
    ]
    _check_move_graph(boards)
    pieces = np.stack([encode_position(board).pieces for board in boards])
    allowed = MoveGraph()(torch.from_numpy(pieces).long())

    def sees(index: int, source: chess.Square, target: chess.Square) -> bool:
        turn = boards[index].turn
        return bool(allowed[index, canonical_square(source, turn), canonical_square(target, turn)])

    assert sees(0, chess.A1, chess.A5) and sees(0, chess.A5, chess.A1)
    assert not sees(0, chess.A1, chess.A6)
    assert sees(0, chess.H2, chess.H4) and sees(0, chess.H4, chess.H2)
    assert sees(1, chess.B7, chess.B5) and not sees(1, chess.G7, chess.G5)
    # The knight on g5 reaches e6, f7, h7 and e4, so each of them sees g5.
    assert all(sees(1, square, chess.G5) for square in (chess.E6, chess.F7, chess.H7, chess.E4))


def test_move_graph_tables_are_buffers_left_out_of_the_state_dict() -> None:
    for lines in (False, True):
        graph = MoveGraph(lines)
        assert not list(graph.state_dict()) and not list(graph.parameters())
        names = {name for name, _ in graph.named_buffers()}
        assert names == {"rays", "orthogonal", "double_push", "targets", "near"}


def test_move_lines_and_blocker_counts_match_python_chess_on_played_positions() -> None:
    totals = _check_move_graph(_played_positions(400, seed=23), lines=True)
    assert 150 <= totals["black"] <= 250
    assert totals["one"] >= 5_000 and totals["more"] >= 5_000 and totals["doubles"] >= 500


def test_move_lines_and_blocker_counts_match_python_chess_on_scattered_positions() -> None:
    totals = _check_move_graph(_scattered_positions(400, seed=24), lines=True)
    assert 150 <= totals["black"] <= 250
    assert totals["one"] >= 5_000 and totals["more"] >= 5_000


def test_move_lines_see_through_a_pinned_piece_and_count_it() -> None:
    # The bishop on b5 pins the knight on c6 to the king on e8; the rook on a1 sees up the file.
    board = chess.Board("4k3/8/2n5/1B6/p7/8/8/R3K3 w - - 0 1")
    _check_move_graph([board], lines=True)
    pieces = torch.from_numpy(encode_position(board).pieces).long()[None]
    lines, graph = MoveGraph(lines=True).codes(pieces)[0], MoveGraph().codes(pieces)[0]
    assert (lines[chess.B5, chess.E8], graph[chess.B5, chess.E8]) == (3, 0)
    assert (lines[chess.B5, chess.C6], graph[chess.B5, chess.C6]) == (2, 1)
    assert lines[chess.A1, chess.A8] == lines[chess.A8, chess.A1] == 3
    assert graph[chess.A1, chess.A8] == 0
    # Knights keep their jumps; the king's neighbours are allowed without a line bias.
    assert lines[chess.C6, chess.E5] == graph[chess.C6, chess.E5] == 1
    assert lines[chess.E1, chess.F2] == 1


def _attack_pairs_reference(board: chess.Board) -> torch.Tensor:
    """Canonical (64, 64) ``AttackPairs`` codes from python-chess pseudo-attacks."""

    ids = encode_position(board).pieces
    codes = torch.zeros(64, 64, dtype=torch.long)
    for square in board.piece_map():
        source = canonical_square(square, board.turn)
        for target in board.attacks(square):
            victim = canonical_square(target, board.turn)
            if board.piece_at(target) is not None:
                codes[source, victim] = 1 + 12 * (int(ids[source]) - 1) + int(ids[victim]) - 1
    return codes


def _check_attack_pairs(boards: list[chess.Board]) -> dict[str, int]:
    pieces = torch.from_numpy(np.stack([encode_position(board).pieces for board in boards]))
    codes = AttackPairs()(pieces.long())
    assert codes.shape == (len(boards), 64, 64) and codes.dtype == torch.long
    assert int(codes.min()) >= 0 and int(codes.max()) <= ATTACK_CODES
    totals = {"black": 0, "far": 0, "defences": 0, "pinned": 0}
    for board, row in zip(boards, codes):
        assert torch.equal(row, _attack_pairs_reference(board)), board.fen()
        totals["black"] += board.turn == chess.BLACK
        totals["far"] += int(((row > 0) & ~KING_STEPS).sum())
        attacker, target = (row[row > 0] - 1) // 12, (row[row > 0] - 1) % 12
        totals["defences"] += int(((attacker < 6) == (target < 6)).sum())
        totals["pinned"] += sum(
            board.is_pinned(piece.color, square) and bool(board.attacks(square) & board.occupied)
            for square, piece in board.piece_map().items()
        )
    return totals


def test_attack_pairs_match_python_chess_on_played_positions() -> None:
    totals = _check_attack_pairs(_played_positions(300, seed=31))
    assert 100 <= totals["black"] <= 200
    assert totals["far"] >= 1_000 and totals["defences"] >= 1_000


def test_attack_pairs_match_python_chess_on_scattered_positions() -> None:
    # Pinned pieces still attack: the pairs are pseudo-attacks, as python-chess ``attacks``.
    totals = _check_attack_pairs(_scattered_positions(400, seed=32))
    assert 150 <= totals["black"] <= 250
    assert totals["far"] >= 1_000 and totals["pinned"] >= 20


def test_attack_pairs_name_both_pieces_and_stop_at_blockers() -> None:
    # The rook on a1 attacks the pawn on a5, not the rook behind it, and defends its king; the
    # knight on e2, pinned by the rook on e8, attacks the pawn on d4 and defends the g1 knight.
    board = chess.Board("r3r1k1/8/8/p7/3p4/8/4N3/R3K1N1 w - - 0 1")
    _check_attack_pairs([board])
    codes = AttackPairs()(torch.from_numpy(encode_position(board).pieces).long()[None])[0]
    assert codes[chess.A1, chess.A5] == 1 + 12 * (4 - 1) + 7 - 1
    assert codes[chess.A1, chess.A8] == 0
    assert codes[chess.A1, chess.E1] == 1 + 12 * (4 - 1) + 6 - 1
    assert codes[chess.E2, chess.D4] == 1 + 12 * (2 - 1) + 7 - 1
    assert codes[chess.E2, chess.G1] == 1 + 12 * (2 - 1) + 2 - 1
    assert codes[chess.E8, chess.E2] == 1 + 12 * (10 - 1) + 2 - 1
    assert codes[chess.D4, chess.E2] == 0
    assert not list(AttackPairs().state_dict()) and not list(AttackPairs().parameters())
