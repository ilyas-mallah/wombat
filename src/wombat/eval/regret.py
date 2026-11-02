"""Fixed-node Stockfish regret evaluation with a content-addressed cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine

from wombat.eval.agreement import select_move
from wombat.eval.manifest import PositionManifest
from wombat.policy import PolicyProvider


@dataclass(frozen=True)
class MoveEvaluation:
    fen: str
    move: str
    score_cp: int
    nodes: int
    engine: str
    cache_key: str


class StockfishRegretEvaluator:
    def __init__(self, executable: Path, nodes: int, cache_dir: Path) -> None:
        if nodes < 1:
            raise ValueError("nodes must be positive")
        self.executable = executable
        self.nodes = nodes
        self.cache_dir = cache_dir
        self.engine: chess.engine.SimpleEngine | None = None
        self.engine_label = "unknown"

    def __enter__(self) -> "StockfishRegretEvaluator":
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.executable))
        self.engine_label = " ".join(
            str(self.engine.id.get(key, "")) for key in ("name", "author") if self.engine.id.get(key)
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def _key_payload(self, board: chess.Board, move: chess.Move) -> dict[str, Any]:
        return {
            "schema": "wombat.stockfish-cache.v1",
            "engine": self.engine_label,
            "nodes": self.nodes,
            "fen": board.fen(),
            "move": move.uci(),
        }

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveEvaluation:
        if move not in board.legal_moves:
            raise ValueError(f"cannot evaluate illegal move {move.uci()}")
        payload = self._key_payload(board, move)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        key = sha256(canonical).hexdigest()
        cache_path = self.cache_dir / key[:2] / f"{key}.json"
        if cache_path.exists():
            return MoveEvaluation(**json.loads(cache_path.read_text(encoding="utf-8")))
        if self.engine is None:
            raise RuntimeError("StockfishRegretEvaluator must be used as a context manager")

        info = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes),
            root_moves=[move],
            info=chess.engine.INFO_SCORE,
        )
        # Normalize to the player at the root, regardless of whose turn follows the move.
        score = info["score"].pov(board.turn).score(mate_score=100_000)
        if score is None:
            raise RuntimeError("Stockfish returned no scalar score")
        result = MoveEvaluation(
            fen=board.fen(),
            move=move.uci(),
            score_cp=int(score),
            nodes=self.nodes,
            engine=self.engine_label,
            cache_key=key,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
        return result


def evaluate_regret(
    provider: PolicyProvider,
    manifest: PositionManifest,
    evaluator: StockfishRegretEvaluator,
    reasoning_steps: int,
) -> dict[str, object]:
    positions: list[dict[str, object]] = []
    total = 0
    for case in manifest.positions:
        board = chess.Board(case.fen)
        reference_move = chess.Move.from_uci(case.best_move)
        chosen_move = select_move(provider, board, reasoning_steps)
        reference = evaluator.evaluate_move(board, reference_move)
        chosen = evaluator.evaluate_move(board, chosen_move)
        regret = reference.score_cp - chosen.score_cp
        total += regret
        positions.append(
            {
                "id": case.id,
                "reference_move": reference_move.uci(),
                "chosen_move": chosen_move.uci(),
                "reference_cp": reference.score_cp,
                "chosen_cp": chosen.score_cp,
                "regret_cp": regret,
            }
        )
    return {
        "manifest_sha256": manifest.sha256,
        "reasoning_steps": reasoning_steps,
        "positions": positions,
        "mean_regret_cp": total / len(positions) if positions else 0.0,
    }

