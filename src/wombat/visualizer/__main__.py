"""Launch the visualizer: ``python -m wombat.visualizer --port 8765``."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

import torch

from wombat.data.puzzle_corpus import lower_priority
from wombat.settings import load_executables
from wombat.visualizer.engine import Stockfish
from wombat.visualizer.models import CheckpointIndex
from wombat.visualizer.positions import PositionPool
from wombat.visualizer.puzzles import PuzzleBook
from wombat.visualizer.server import VisualizerApp, make_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m wombat.visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--puzzles", type=Path, default=Path("data/processed/puzzles-v1/test"))
    parser.add_argument(
        "--positions", type=Path, default=Path("data/processed/main-1m/test.jsonl.zst")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--stockfish-nodes", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    lower_priority()
    torch.set_num_threads(args.threads)
    stockfish = Stockfish(load_executables(args.config)["stockfish"], nodes=args.stockfish_nodes)
    app = VisualizerApp(
        checkpoints=CheckpointIndex(args.runs),
        puzzles=PuzzleBook(args.puzzles),
        positions=PositionPool(args.positions),
        stockfish=stockfish,
    )
    threading.Thread(target=app.warm, name="warm", daemon=True).start()
    server = make_server(app, args.host, args.port)
    print(
        f"Wombat visualizer on http://{args.host}:{server.server_address[1]}/ (CPU only)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stockfish.close()


if __name__ == "__main__":
    main()
