"""A stdlib JSON server for the visualizer page."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import random
from typing import Any
from urllib.parse import parse_qs, urlparse

import chess

from wombat.visualizer.engine import Stockfish
from wombat.visualizer.models import MAX_STEPS, STRIP_DEPTHS, CheckpointIndex, ModelCache
from wombat.visualizer.positions import PositionPool
from wombat.visualizer.puzzles import PuzzleBook, is_solution

STATIC = Path(__file__).with_name("static")
MAX_BODY = 1 << 20
LOG = logging.getLogger(__name__)


class RequestError(ValueError):
    """A client mistake, reported as HTTP 400 with its message."""


@dataclass
class VisualizerApp:
    checkpoints: CheckpointIndex
    puzzles: PuzzleBook
    positions: PositionPool
    stockfish: Stockfish
    models: ModelCache = field(init=False)
    generator: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.models = ModelCache(self.checkpoints)

    def warm(self) -> None:
        """Index checkpoints, puzzles and positions ahead of the first request."""

        default = self.checkpoints.default()
        if default is not None:
            self.models.get(default)
        if self.puzzles.available:
            len(self.puzzles)
        len(self.positions.fens)

    def status(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "stockfish": self.stockfish.available,
            "puzzles": self.puzzles.available,
            "strip_depths": list(STRIP_DEPTHS),
            "max_steps": MAX_STEPS,
        }

    def list_checkpoints(self, _: dict[str, Any]) -> dict[str, Any]:
        listed = self.checkpoints.list()
        return {
            "checkpoints": [info.to_json() for info in listed],
            "default": self.checkpoints.default(),
        }

    def random_position(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"fen": self.positions.random(self.generator)}

    def random_puzzle(self, query: dict[str, Any]) -> dict[str, Any]:
        def rating(name: str) -> int | None:
            value = query.get(name)
            return int(value) if value not in (None, "") else None

        try:
            puzzle = self.puzzles.random(self.generator, rating("min_rating"), rating("max_rating"))
        except LookupError as error:
            raise RequestError(str(error)) from error
        return puzzle.to_json()

    def state(self, body: dict[str, Any]) -> dict[str, Any]:
        """Replay ``moves`` from ``fen``; returns legal moves, SAN history and any outcome."""

        board = _board(body.get("fen"))
        sans: list[str] = []
        for uci in body.get("moves", []):
            move = _move(board, uci)
            sans.append(board.san(move))
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        king = board.king(board.turn)
        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "legal": [move.uci() for move in board.legal_moves],
            "sans": sans,
            "check": chess.square_name(king) if board.is_check() and king is not None else None,
            "outcome": None
            if outcome is None
            else {"result": outcome.result(), "termination": outcome.termination.name.lower()},
        }

    def analyse(self, body: dict[str, Any]) -> dict[str, Any]:
        """The model's move at ``fen``; ``start`` and ``moves`` give the game for its draw guard."""

        board = _board(body.get("fen"))
        if body.get("moves") is not None:
            played = _board(body.get("start"))
            for uci in body["moves"]:
                played.push(_move(played, uci))
            if body.get("fen") and played.fen() != board.fen():
                raise RequestError("moves do not reach fen")
            board = played
        steps = int(body.get("steps", 8))
        checkpoint = body.get("checkpoint") or self.checkpoints.default()
        if checkpoint is None:
            raise RequestError("no checkpoint found")
        try:
            result = self.models.analyse(str(checkpoint), board, steps)
        except KeyError as error:
            raise RequestError(str(error)) from error
        if body.get("solution"):
            # Marks which candidates and which depths would solve a puzzle position.
            solution = _move(board, body["solution"])
            for item in (*result["candidates"], *result["strip"]):
                item["solves"] = is_solution(board, chess.Move.from_uci(item["uci"]), solution)
            result["solves"] = result["candidates"][0]["solves"]
        return result

    def explain(self, body: dict[str, Any]) -> dict[str, Any]:
        """Why the checkpoint plays ``move``: depth strip, values, reply, attention, occlusion."""

        board = _board(body.get("fen"))
        move = _move(board, body.get("move"))
        compare = _move(board, body["compare"]) if body.get("compare") else None
        steps = int(body.get("steps", 8))
        checkpoint = body.get("checkpoint") or self.checkpoints.default()
        if checkpoint is None:
            raise RequestError("no checkpoint found")
        try:
            return self.models.explain(str(checkpoint), board, move, compare, steps)
        except KeyError as error:
            raise RequestError(str(error)) from error

    def _engine(self) -> Stockfish:
        if not self.stockfish.available:
            raise RequestError("Stockfish is not configured")
        return self.stockfish

    def evaluate(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._engine().position(_board(body.get("fen")))

    def judge(self, body: dict[str, Any]) -> dict[str, Any]:
        board = _board(body.get("fen"))
        return self._engine().judge(board, _move(board, body.get("move")))

    def engine_move(self, body: dict[str, Any]) -> dict[str, Any]:
        """Stockfish's move: ``elo`` (a number or "full") for timed play, else ``nodes``."""

        board = _board(body.get("fen"))
        engine = self._engine()
        if "elo" in body:
            elo = None if body["elo"] in (None, "full") else int(body["elo"])
            movetime = max(0.01, min(5.0, float(body.get("movetime", 0.1))))
            move = engine.play(board, elo=elo, movetime=movetime)
        else:
            move = engine.play(board, nodes=max(1, min(1_000_000, int(body.get("nodes", 1000)))))
        return {"move": move.uci(), "san": board.san(move)}

    def routes(self) -> dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]]:
        return {
            ("GET", "/api/status"): self.status,
            ("GET", "/api/checkpoints"): self.list_checkpoints,
            ("GET", "/api/position/random"): self.random_position,
            ("GET", "/api/puzzle/random"): self.random_puzzle,
            ("POST", "/api/state"): self.state,
            ("POST", "/api/analyse"): self.analyse,
            ("POST", "/api/explain"): self.explain,
            ("POST", "/api/eval"): self.evaluate,
            ("POST", "/api/judge"): self.judge,
            ("POST", "/api/stockfish/move"): self.engine_move,
        }


def _board(fen: object) -> chess.Board:
    try:
        board = chess.Board(str(fen)) if fen else chess.Board()
    except ValueError as error:
        raise RequestError(f"invalid FEN: {fen}") from error
    if not board.is_valid():
        raise RequestError(f"invalid position: {fen}")
    return board


def _move(board: chess.Board, uci: object) -> chess.Move:
    try:
        move = chess.Move.from_uci(str(uci))
    except ValueError as error:
        raise RequestError(f"invalid move: {uci}") from error
    if move not in board.legal_moves:
        raise RequestError(f"illegal move: {uci}")
    return move


def make_handler(app: VisualizerApp) -> type[BaseHTTPRequestHandler]:
    routes = app.routes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            if method == "GET" and url.path in ("/", "/index.html"):
                page = (STATIC / "index.html").read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
                return
            handler = routes.get((method, url.path))
            if handler is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {method} {url.path}"})
                return
            try:
                if method == "GET":
                    payload = {key: values[-1] for key, values in parse_qs(url.query).items()}
                else:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_BODY:
                        raise RequestError("request body too large")
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise RequestError("expected a JSON object")
                self._json(HTTPStatus.OK, handler(payload))
            except (RequestError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:
                LOG.exception("%s %s failed", method, url.path)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(error)})

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return Handler


def make_server(
    app: VisualizerApp, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server
