from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
import threading
from typing import Any
import urllib.error
import urllib.request

import chess
import chess.engine
import numpy as np
import pytest
import torch
import zstandard

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import legal_actions
from wombat.data.lichess import PuzzleRecord
from wombat.eval.agreement import draws_immediately
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.visualizer.engine import (
    MAX_ELO,
    MIN_ELO,
    Evaluation,
    Stockfish,
    classify,
    move_accuracy,
    strength,
    win_percent,
)
from wombat.visualizer.models import (
    EXPLAIN_DEPTHS,
    STRIP_DEPTHS,
    CheckpointIndex,
    ModelCache,
    analyse,
    trained_heads,
    value_to_pawns,
)
from wombat.visualizer.positions import PositionPool
from wombat.visualizer.puzzles import PuzzleBook, find_reply, is_solution, reconstruct_line
from wombat.visualizer.server import VisualizerApp, make_server

SQUARE = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=1,
    recurrent_blocks=1,
    latent_tokens=4,
    policy_readout="square",
    normalize_state=True,
    max_residual_update=None,
)
GLOBAL = replace(SQUARE, policy_readout="global", normalize_state=False, max_residual_update=0.25)
BLACK_TO_MOVE = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3"


def _save(path: Path, config: NetworkConfig, seed: int, **extra: object) -> WombatNetwork:
    torch.manual_seed(seed)
    model = WombatNetwork(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(config), **extra}, path)
    return model


# A pre-``task`` puzzle trainer config, as stored by the v6 runs.
PUZZLE_TRAINING = {"corpus": "data/processed/puzzles-v1", "steps": 40, "max_depth": 16}
GAMES_TRAINING = {"dataset": "games.jsonl", "value_weight": 0.25, "move_value_weight": 0.25}
SNAPSHOTS = ("runs/games/snapshot-step-005000.pt", "runs/games/snapshot-step-010000.pt")


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    _save(
        root / "square-run" / "checkpoint.pt", SQUARE, 1, step=40, training_config=PUZZLE_TRAINING
    )
    # A legacy config: no readout field (so global) and a field this network no longer has.
    legacy = {k: v for k, v in asdict(GLOBAL).items() if k != "policy_readout"}
    torch.manual_seed(2)
    model = WombatNetwork(GLOBAL)
    (root / "old" / "nested").mkdir(parents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {**legacy, "retired_option": 3},
            "training_config": GAMES_TRAINING,
        },
        root / "old" / "nested" / "checkpoint-epoch-1.pt",
    )
    # Model-only snapshots of a train-games run; the run's config.json holds its losses.
    games = {"corpus": "games", "task": "games", "value_weight": 1.0, "move_value_weight": 0.0}
    (root / "games").mkdir()
    (root / "games" / "config.json").write_text(json.dumps({"training_config": games}))
    for number, identifier in enumerate(SNAPSHOTS):
        _save(tmp_path / identifier, SQUARE, 10 + number, step=5000 * (number + 1))
    (root / "broken").mkdir()
    (root / "broken" / "checkpoint.pt").write_bytes(b"not a checkpoint")
    return root


def test_discovery_lists_readouts_snapshots_and_trained_heads(runs: Path) -> None:
    index = CheckpointIndex(runs)
    ordered = index.list()
    listed = {info.id: info for info in ordered}
    assert set(listed) == {
        "runs/square-run/checkpoint.pt",
        "runs/old/nested/checkpoint-epoch-1.pt",
        *SNAPSHOTS,
    }
    square = listed["runs/square-run/checkpoint.pt"]
    assert (square.run, square.name, square.kind, square.step, square.readout) == (
        "square-run",
        "checkpoint",
        "checkpoint",
        40,
        "square",
    )
    assert square.parameters == count_parameters(WombatNetwork(SQUARE))
    assert (square.value_trained, square.move_values_trained) == (False, False)
    legacy = listed["runs/old/nested/checkpoint-epoch-1.pt"]
    assert (legacy.run, legacy.step, legacy.readout) == ("old/nested", None, "global")
    assert (legacy.value_trained, legacy.move_values_trained) == (True, True)
    snapshots = [info for info in ordered if info.kind == "snapshot"]
    assert [info.step for info in snapshots] == [10000, 5000]
    assert all(info.run == "games" for info in snapshots)
    assert all(
        (info.value_trained, info.move_values_trained) == (True, False) for info in snapshots
    )
    assert index.default(preferred=("runs/square-run/checkpoint.pt",)) == square.id
    with pytest.raises(KeyError):
        index.path("../../etc/passwd")


def test_trained_heads_follow_the_training_losses() -> None:
    heads = ["value_head.0.weight", "square_move_value_head.0.weight"]
    assert trained_heads(None, heads) == (False, False)
    assert trained_heads(PUZZLE_TRAINING, heads) == (False, False)
    games = {"corpus": "g", "task": "games", "value_weight": 1.0, "move_value_weight": 0.5}
    assert trained_heads(games, heads) == (True, True)
    assert trained_heads({**games, "task": "puzzles"}, heads) == (False, False)
    assert trained_heads({**games, "value_weight": 0.0}, heads) == (False, True)
    # Older game trainers without a move-value loss, and heads missing from the weights.
    assert trained_heads({"dataset": "d", "value_weight": 0.25}, heads) == (True, False)
    assert trained_heads(GAMES_TRAINING, ["policy_head.weight"]) == (False, False)


def test_values_convert_to_pawns_on_the_lichess_curve() -> None:
    assert value_to_pawns(0.0) == 0.0
    assert value_to_pawns(0.5) == pytest.approx(np.log(3) / 0.368208)
    assert value_to_pawns(-0.5) == pytest.approx(-value_to_pawns(0.5))
    assert value_to_pawns(1.0) == pytest.approx(np.log(999) / 0.368208)
    assert value_to_pawns(-1.0) == value_to_pawns(-0.999999)
    for value in (-0.9, -0.3, 0.2, 0.8):
        # Round trip through the Lichess win-percent curve used for Stockfish.
        assert win_percent(100 * value_to_pawns(value)) == pytest.approx(50 * (value + 1))


def test_model_cache_loads_weights_and_evicts_the_oldest(runs: Path) -> None:
    _save(runs / "third" / "checkpoint.pt", SQUARE, 3)
    cache = ModelCache(CheckpointIndex(runs), capacity=2)
    square = cache.get("runs/square-run/checkpoint.pt")
    torch.manual_seed(1)
    expected = WombatNetwork(SQUARE).state_dict()
    for name, tensor in square.state_dict().items():
        assert torch.equal(tensor, expected[name])
    assert next(square.parameters()).device.type == "cpu"
    legacy = cache.get("runs/old/nested/checkpoint-epoch-1.pt")
    assert legacy.config.policy_readout == "global"
    assert cache.get("runs/square-run/checkpoint.pt") is square
    cache.get("runs/third/checkpoint.pt")
    assert cache.get("runs/square-run/checkpoint.pt") is square
    assert cache.get("runs/old/nested/checkpoint-epoch-1.pt") is not legacy


def test_cached_analyses_carry_head_flags_and_are_private_copies(runs: Path) -> None:
    cache = ModelCache(CheckpointIndex(runs))
    board = chess.Board()
    puzzle = cache.analyse("runs/square-run/checkpoint.pt", board, 2)
    assert (puzzle["value_trained"], puzzle["move_values_trained"]) == (False, False)
    games = cache.analyse(SNAPSHOTS[0], board, 2)
    assert (games["value_trained"], games["move_values_trained"]) == (True, False)
    games["candidates"][0]["solves"] = True
    again = cache.analyse(SNAPSHOTS[0], board, 2)
    assert "solves" not in again["candidates"][0]
    assert again["candidates"] == [
        {k: v for k, v in item.items() if k != "solves"} for item in games["candidates"]
    ]
    assert len(cache._analyses) == 2


@pytest.mark.parametrize("config", [SQUARE, GLOBAL], ids=["square", "global"])
@pytest.mark.parametrize("fen", [chess.STARTING_FEN, BLACK_TO_MOVE])
def test_analysis_picks_the_best_legal_move_at_every_depth(config: NetworkConfig, fen: str) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(config).eval()
    board = chess.Board(fen)
    legal = {move.uci() for move in board.legal_moves}
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
    state = torch.from_numpy(state_vector(encoded))[None]
    actions = legal_actions(board)
    moves = list(board.legal_moves)
    sign = 1 if board.turn == chess.WHITE else -1
    for steps in (0, 1, 3, 32):
        result = analyse(model, board, steps)
        assert result["steps"] == steps
        assert result["move"] in legal
        with torch.inference_mode():
            logits, value, move_values = model(pieces, state, steps, include_move_values=True)
        assert result["move"] == moves[int(torch.argmax(logits[0][actions]))].uci()
        assert result["value"] == pytest.approx(float(value[0]), abs=1e-5)
        assert result["pawns_white"] == pytest.approx(sign * value_to_pawns(result["value"]))
        mover_win = 50 * (result["value"] + 1)
        assert result["win_white"] == pytest.approx(mover_win if sign == 1 else 100 - mover_win)
        for item in result["candidates"]:
            action = actions[[move.uci() for move in moves].index(item["uci"])]
            expected = float(move_values[0][action])
            assert item["move_value"] == pytest.approx(expected, abs=1e-5)
            assert item["pawns_white"] == pytest.approx(sign * value_to_pawns(expected), abs=1e-4)
        probabilities = [item["probability"] for item in result["candidates"]]
        assert len(probabilities) == 5 and probabilities == sorted(probabilities, reverse=True)
        assert 0 < sum(probabilities) <= 1 + 1e-6
        assert -1 <= result["value"] <= 1
        assert [item["steps"] for item in result["strip"]] == list(STRIP_DEPTHS)
        assert all(item["uci"] in legal for item in result["strip"])
    with pytest.raises(ValueError):
        analyse(model, board, 33)
    with pytest.raises(ValueError):
        analyse(model, chess.Board("7k/5QQ1/8/8/8/8/8/K7 b - - 0 1"), 1)


def test_analysis_draw_guard_avoids_a_stalemate_when_winning() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(SQUARE).eval()
    with torch.no_grad():
        model.square_policy_head.weight.zero_()
        model.square_policy_head.bias.zero_()
        # Plane 15 moves two squares east; here only Qe6-g6, which stalemates.
        model.square_policy_head.bias[15] = 10.0
        model.value_head[2].weight.zero_()
        model.value_head[2].bias.fill_(3.0)
    board = chess.Board("7k/5K2/4Q3/8/8/8/8/8 w - - 0 1")
    plain = analyse(model, board, 2)
    assert plain["move"] == "e6g6" and plain["avoided_draw"] is False
    guarded = analyse(model, board, 2, draw_guard=0.3)
    assert guarded["value"] > 0.3 and guarded["avoided_draw"] is True
    assert guarded["candidates"] == plain["candidates"]
    moves = [chess.Move.from_uci(item["uci"]) for item in plain["candidates"]]
    expected = next(move for move in moves if not draws_immediately(board, move))
    assert guarded["move"] == expected.uci() and guarded["san"] == board.san(expected)
    assert not board.move_stack


def test_puzzle_lines_match_the_source_records(
    puzzle_corpus: Path, puzzle_archive: tuple[Path, list[PuzzleRecord]]
) -> None:
    records = {record.puzzle_id: record for record in puzzle_archive[1]}
    book = PuzzleBook(puzzle_corpus / "test")
    assert book.available and len(book) == 3
    seen = set()
    for number in range(len(book)):
        puzzle = book.puzzle(number)
        record = records[puzzle.puzzle_id]
        seen.add(puzzle.puzzle_id)
        assert puzzle.rating == record.rating
        assert [step.solution for step in puzzle.steps] == list(record.moves[1::2])
        assert [step.reply for step in puzzle.steps] == [*record.moves[2::2], None]
        board = chess.Board(record.fen)
        board.push_uci(record.moves[0])
        for step in puzzle.steps:
            assert step.fen == board.fen()
            assert step.solution_san == board.san(chess.Move.from_uci(step.solution))
            board.push_uci(step.solution)
            if step.reply is not None:
                board.push_uci(step.reply)
    assert len(seen) == 3
    generator = random.Random(0)
    ratings = sorted(records[book.puzzle(n).puzzle_id].rating for n in range(3))
    for _ in range(10):
        assert book.random(generator, ratings[1], ratings[1]).rating == ratings[1]
    with pytest.raises(LookupError):
        book.random(generator, 9000)


def test_line_reconstruction_and_the_checkmate_rule() -> None:
    board = chess.Board()
    fens, actions = [], []
    for solver, reply in (("e2e4", "e7e5"), ("g1f3", "b8c6"), ("f1c4", None)):
        fens.append(board.fen())
        actions.append(
            legal_actions(board)[list(board.legal_moves).index(chess.Move.from_uci(solver))]
        )
        board.push_uci(solver)
        if reply:
            board.push_uci(reply)
    steps = reconstruct_line(fens, actions)
    assert [(s.solution, s.reply) for s in steps] == [
        ("e2e4", "e7e5"),
        ("g1f3", "b8c6"),
        ("f1c4", None),
    ]
    assert [s.solution_san for s in steps] == ["e4", "Nf3", "Bc4"]
    with pytest.raises(ValueError):
        find_reply(chess.Board(), chess.STARTING_FEN)
    two_mates = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R2R2K1 w - - 0 1")
    solution = chess.Move.from_uci("a1a8")
    assert is_solution(two_mates, solution, solution)
    assert is_solution(two_mates, chess.Move.from_uci("d1d8"), solution)
    assert not is_solution(two_mates, chess.Move.from_uci("g1f1"), solution)


def test_lichess_win_percent_and_accuracy_formulas() -> None:
    assert win_percent(0) == pytest.approx(50)
    assert win_percent(100) == pytest.approx(59.1, abs=0.05)
    assert win_percent(-300) == pytest.approx(100 - win_percent(300))
    assert win_percent(5000) == win_percent(1000) == pytest.approx(97.54, abs=0.01)
    assert move_accuracy(0) == pytest.approx(100, abs=1e-3)
    assert move_accuracy(10) == pytest.approx(103.1668 * np.exp(-0.4354) - 3.1669)
    assert move_accuracy(100) == 0
    labels = [classify(lost) for lost in (0, 4.99, 5, 9.99, 10, 14.99, 15, 60)]
    assert (
        labels
        == ["good", "good", "inaccuracy", "inaccuracy", "mistake", "mistake"] + ["blunder"] * 2
    )


def test_judging_compares_the_played_move_with_the_best(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Stockfish(None)
    assert not engine.available
    board = chess.Board()
    scores = {board.fen(): Evaluation(50, None, "e2e4")}
    after = board.copy()
    after.push_uci("g1h3")
    scores[after.fen()] = Evaluation(200, None, "e7e5")
    monkeypatch.setattr(engine, "evaluate", lambda b, nodes=None: scores[b.fen()])
    verdict = engine.judge(board, chess.Move.from_uci("g1h3"))
    lost = win_percent(50) - win_percent(-200)
    assert verdict["lost"] == pytest.approx(lost)
    assert verdict["label"] == classify(lost) == "blunder"
    assert verdict["best_san"] == "e4"
    assert verdict["eval_after"]["cp_white"] == -200
    mated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert Stockfish(None).evaluate(mated).cp < -10_000


def test_strength_limits_map_to_uci_options() -> None:
    limit, options = strength(nodes=1500)
    assert (limit.nodes, limit.time, options) == (1500, None, {})
    limit, options = strength(elo=1800, movetime=0.1)
    assert (limit.nodes, limit.time) == (None, 0.1)
    assert options == {"UCI_LimitStrength": True, "UCI_Elo": 1800}
    assert strength(elo=900)[1]["UCI_Elo"] == MIN_ELO == 1320
    assert strength(elo=9000)[1]["UCI_Elo"] == MAX_ELO
    limit, options = strength(elo=None, movetime=0.0)
    assert options == {"UCI_LimitStrength": False} and limit.time == 0.01


class _FakeEngine:
    """Records play calls in place of a Stockfish process."""

    def __init__(self) -> None:
        self.calls: list[tuple[chess.engine.Limit, dict[str, object]]] = []

    def play(
        self, board: chess.Board, limit: chess.engine.Limit, *, options: dict[str, object]
    ) -> chess.engine.PlayResult:
        self.calls.append((limit, options))
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)


def test_engine_moves_at_an_elo_or_a_node_budget(
    runs: Path, puzzle_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any existing file makes the engine "available"; the process itself is faked.
    engine = Stockfish(sys.executable)
    fake = _FakeEngine()
    monkeypatch.setattr(engine, "_started", lambda: fake)
    app = VisualizerApp(
        checkpoints=CheckpointIndex(runs),
        puzzles=PuzzleBook(puzzle_corpus / "test"),
        positions=PositionPool(tmp_path / "missing.jsonl.zst"),
        stockfish=engine,
    )
    board = chess.Board()
    reply = app.engine_move({"fen": board.fen(), "elo": "1500", "movetime": 0.1})
    assert chess.Move.from_uci(reply["move"]) in board.legal_moves
    app.engine_move({"fen": board.fen(), "elo": "full"})
    app.engine_move({"fen": board.fen(), "nodes": 2000})
    (elo_limit, elo_options), (full_limit, full_options), (node_limit, node_options) = fake.calls
    assert elo_limit.time == 0.1 and elo_options == {"UCI_LimitStrength": True, "UCI_Elo": 1500}
    assert full_limit.time == 0.1 and full_options == {"UCI_LimitStrength": False}
    assert node_limit.nodes == 2000 and node_options == {}


@pytest.fixture()
def server(runs: Path, puzzle_corpus: Path, tmp_path: Path) -> Iterator[str]:
    positions = tmp_path / "positions.jsonl.zst"
    lines = [json.dumps({"fen": BLACK_TO_MOVE}), json.dumps({"fen": chess.STARTING_FEN})]
    positions.write_bytes(zstandard.ZstdCompressor().compress("\n".join(lines).encode()))
    app = VisualizerApp(
        checkpoints=CheckpointIndex(runs),
        puzzles=PuzzleBook(puzzle_corpus / "test"),
        positions=PositionPool(positions),
        stockfish=Stockfish(None),
    )
    app.warm()
    assert len(app.models._models) == 1
    httpd = make_server(app, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _request(url: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            status = response.status
            kind = response.headers["Content-Type"]
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    return status, json.loads(raw) if kind == "application/json" else raw.decode()


def test_http_endpoints(server: str) -> None:
    status, page = _request(server + "/")
    assert status == 200 and "<title>Wombat Visualizer</title>" in page
    assert _request(server + "/api/status")[1]["stockfish"] is False
    listing = _request(server + "/api/checkpoints")[1]
    assert len(listing["checkpoints"]) == 4 and listing["default"] in {
        item["id"] for item in listing["checkpoints"]
    }
    trained = {item["id"]: item["value_trained"] for item in listing["checkpoints"]}
    assert trained["runs/square-run/checkpoint.pt"] is False and trained[SNAPSHOTS[1]] is True
    fen = _request(server + "/api/position/random")[1]["fen"]
    assert fen in {chess.Board(BLACK_TO_MOVE).fen(), chess.STARTING_FEN}

    status, state = _request(server + "/api/state", {"fen": None, "moves": ["f2f3", "e7e5"]})
    assert status == 200 and state["sans"] == ["f3", "e5"] and state["turn"] == "white"
    mate = _request(server + "/api/state", {"moves": ["f2f3", "e7e5", "g2g4", "d8h4"]})[1]
    assert mate["outcome"] == {"result": "0-1", "termination": "checkmate"}
    assert mate["check"] == "e1" and mate["legal"] == []
    assert _request(server + "/api/state", {"moves": ["e2e5"]})[0] == 400

    for checkpoint in [item["id"] for item in listing["checkpoints"]]:
        for steps in (0, 4):
            status, result = _request(
                server + "/api/analyse", {"checkpoint": checkpoint, "fen": fen, "steps": steps}
            )
            assert status == 200
            assert result["move"] in {move.uci() for move in chess.Board(fen).legal_moves}
            assert result["value_trained"] is trained[checkpoint]
            assert {"pawns_white", "win_white"} <= set(result)
            assert all("pawns_white" in item for item in result["candidates"])
    status, error = _request(server + "/api/analyse", {"checkpoint": "runs/../x.pt", "fen": fen})
    assert status == 400 and "unknown checkpoint" in error["error"]

    puzzle = _request(server + "/api/puzzle/random")[1]
    first = puzzle["steps"][0]
    status, marked = _request(
        server + "/api/analyse", {"fen": first["fen"], "steps": 2, "solution": first["solution"]}
    )
    board = chess.Board(first["fen"])
    for item in (*marked["candidates"], *marked["strip"]):
        after = board.copy()
        after.push_uci(item["uci"])
        assert item["solves"] == (item["uci"] == first["solution"] or after.is_checkmate())
    assert marked["solves"] == marked["candidates"][0]["solves"]
    wrong = {"fen": first["fen"], "solution": "a1a1"}
    assert _request(server + "/api/analyse", wrong)[0] == 400
    assert _request(server + "/api/puzzle/random?min_rating=99999")[0] == 400
    assert _request(server + "/api/eval", {"fen": fen})[0] == 400
    assert _request(server + "/api/nowhere")[0] == 404


def test_explain_endpoint(server: str) -> None:
    body = {"checkpoint": SNAPSHOTS[0], "fen": BLACK_TO_MOVE, "move": "g8f6", "steps": 2}
    status, result = _request(server + "/api/explain", {**body, "compare": "f8c5"})
    assert status == 200
    keys = {"depths", "value", "pawns_white", "win_white", "moves", "reply", "attention"}
    assert keys | {"occlusion", "value_trained", "move_values_trained"} <= set(result)
    assert (result["move"]["san"], result["compare"]["san"]) == ("Nf6", "Bc5")
    assert (result["value_trained"], result["move_values_trained"]) == (True, False)
    assert [depth["steps"] for depth in result["depths"]] == list(EXPLAIN_DEPTHS)
    assert result["pawns_white"] == pytest.approx(-value_to_pawns(result["value"]))
    for item in result["moves"]:
        assert item["pawns_white"] == pytest.approx(-value_to_pawns(item["value"]))
    # The square readout without lookahead heads has no expected reply.
    assert result["reply"] is None and len(result["attention"]["weights"]) == 64
    assert {row["square"] for row in result["occlusion"]["pieces"]}.isdisjoint({"e1", "e8", "g8"})
    assert len(json.dumps(result)) < 20_000
    assert _request(server + "/api/explain", {**body, "compare": "f8c5"})[1] == result

    assert _request(server + "/api/explain", {**body, "move": "g8g6"})[0] == 400
    assert _request(server + "/api/explain", {**body, "steps": 99})[0] == 400
    status, error = _request(server + "/api/explain", {**body, "checkpoint": "runs/../x.pt"})
    assert status == 400 and "unknown checkpoint" in error["error"]


def test_analyse_endpoint_replays_the_game(server: str) -> None:
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    body = {"checkpoint": SNAPSHOTS[0], "fen": board.fen(), "steps": 1, "moves": moves}
    status, result = _request(server + "/api/analyse", body)
    assert status == 200 and result["move"] in {move.uci() for move in board.legal_moves}
    assert result["move"] != "f6g8" or result["value"] <= 0.3
    assert _request(server + "/api/analyse", {**body, "moves": moves[:-1]})[0] == 400
    assert _request(server + "/api/analyse", {**body, "moves": [*moves, "e2e5"]})[0] == 400


def test_position_pool_reads_only_up_to_its_limit(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl.zst"
    lines = [json.dumps({"fen": chess.STARTING_FEN, "best_move": "e2e4"})] * 50
    path.write_bytes(zstandard.ZstdCompressor().compress("\n".join(lines).encode()))
    pool = PositionPool(path, limit=7)
    assert len(pool.fens) == 7
    assert PositionPool(tmp_path / "missing.jsonl.zst").random(random.Random(0)) == (
        chess.STARTING_FEN
    )
