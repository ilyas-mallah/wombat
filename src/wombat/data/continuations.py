"""Annotate corpus positions with several scored Stockfish continuation lines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

import chess
import chess.engine
import zstandard


@dataclass(frozen=True)
class ContinuationConfig:
    source: Path
    destination: Path
    stockfish: Path
    nodes: int = 5_000
    multipv: int = 8
    plies: int = 8
    threads: int = 4
    hash_mb: int = 512
    limit: int | None = None
    field_name: str = "continuations"


def _score_payload(score: chess.engine.PovScore, turn: chess.Color) -> dict[str, int | None]:
    relative = score.pov(turn)
    return {
        "score_cp": relative.score(),
        "score_mate": relative.mate(),
    }


def annotate_continuations(config: ContinuationConfig) -> dict[str, object]:
    if config.nodes < 1 or config.multipv < 1 or config.plies < 2:
        raise ValueError("nodes/multipv must be positive and plies must be at least two")
    if config.field_name not in {"continuations", "deep_continuations"}:
        raise ValueError("field_name must be continuations or deep_continuations")
    config.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.destination.with_suffix(config.destination.suffix + ".partial")
    counts: Counter[str] = Counter()
    started = time.perf_counter()

    def start_engine() -> chess.engine.SimpleEngine:
        instance = chess.engine.SimpleEngine.popen_uci(str(config.stockfish))
        instance.configure({"Threads": config.threads, "Hash": config.hash_mb})
        return instance

    engine = start_engine()
    engine_name = str(engine.id.get("name", "Stockfish"))
    try:
        with (
            zstandard.open(config.source, mode="rt", encoding="utf-8") as source,
            zstandard.open(temporary, mode="wt", encoding="utf-8") as destination,
        ):
            for line in source:
                payload = json.loads(line)
                board = chess.Board(payload["fen"])
                if not board.is_valid():
                    payload[config.field_name] = []
                    payload[f"{config.field_name}_error"] = (
                        f"invalid_board_status:{int(board.status())}"
                    )
                    counts["failed"] += 1
                    counts["invalid_boards"] += 1
                    destination.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    if config.limit is not None and counts["positions"] + counts["failed"] >= config.limit:
                        break
                    continue
                requested = min(config.multipv, board.legal_moves.count())
                analysis = None
                for attempt in range(2):
                    try:
                        analysis = engine.analyse(
                            board,
                            chess.engine.Limit(nodes=config.nodes),
                            multipv=requested,
                            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                        )
                        break
                    except chess.engine.EngineTerminatedError:
                        counts["engine_restarts"] += 1
                        try:
                            engine.close()
                        except Exception:
                            pass
                        engine = start_engine()
                    except (chess.engine.EngineError, KeyError):
                        break
                try:
                    if analysis is None:
                        raise RuntimeError("engine analysis failed after retry")
                    if isinstance(analysis, dict):
                        analysis = [analysis]
                    continuations = []
                    for item in analysis:
                        pv = item.get("pv", ())[: config.plies]
                        if len(pv) < 2 or "score" not in item:
                            continue
                        continuations.append(
                            {
                                "move": pv[0].uci(),
                                **_score_payload(item["score"], board.turn),
                                "pv": [move.uci() for move in pv],
                            }
                        )
                    payload[config.field_name] = continuations
                    counts["positions"] += 1
                    counts["lines"] += len(continuations)
                    counts["plies"] += sum(len(item["pv"]) for item in continuations)
                except (RuntimeError, KeyError):
                    payload[config.field_name] = []
                    counts["failed"] += 1
                destination.write(json.dumps(payload, separators=(",", ":")) + "\n")
                if config.limit is not None and counts["positions"] + counts["failed"] >= config.limit:
                    break
    finally:
        try:
            engine.quit()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            pass
    temporary.replace(config.destination)
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "schema": "wombat.continuation-corpus.v2",
        "source": str(config.source),
        "destination": str(config.destination),
        "engine": engine_name,
        "nodes": config.nodes,
        "multipv": config.multipv,
        "max_plies": config.plies,
        "field_name": config.field_name,
        "threads": config.threads,
        "positions": counts["positions"],
        "lines": counts["lines"],
        "plies": counts["plies"],
        "failed": counts["failed"],
        "engine_restarts": counts["engine_restarts"],
        "invalid_boards": counts["invalid_boards"],
        "seconds": elapsed,
        "positions_per_second": counts["positions"] / elapsed,
    }
    report_path = config.destination.with_suffix(config.destination.suffix + ".manifest.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
