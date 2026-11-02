"""Stockfish-judged move quality for moves chosen outside Wombat, on the strength positions.

The positions are the ones ``evaluate_strength`` samples, so the metrics line up with its
reports. Moves arrive as a JSON ``{fen: uci}`` map keyed by the exported FEN strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

from wombat.chess.moves import move_to_action
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import lower_priority
from wombat.eval.strength import StockfishJudge, _summary, judge_positions, valid_sample


def strength_fens(games: Path, positions: int = 2_000) -> tuple[np.ndarray, list[str]]:
    """Corpus indices and FENs of the strength report's positions, in report order."""

    split = GameSplit(games)
    if split.fens is None:
        raise ValueError("strength positions need a split with FENs")
    indices = valid_sample(split, positions)
    return indices, [str(split.fens[index]) for index in indices]


def export_strength_fens(games: Path, output: Path, positions: int = 2_000) -> int:
    _, fens = strength_fens(games, positions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{fen}\n" for fen in fens), encoding="utf-8")
    return len(fens)


def evaluate_external_strength(
    moves: Path,
    *,
    games: Path,
    output: Path,
    stockfish: Path,
    positions: int = 2_000,
    nodes: int = 20_000,
    engines: int = 4,
    cache_dir: Path = Path("runs/stockfish-cache"),
) -> dict[str, object]:
    """Lichess accuracy and error rates of externally chosen moves on the strength positions."""

    lower_priority()
    chosen: dict[str, str] = json.loads(moves.read_text(encoding="utf-8"))
    split = GameSplit(games)
    indices, fens = strength_fens(games, positions)
    actions = []
    for fen in fens:
        if fen not in chosen:
            raise ValueError(f"no move for {fen}")
        board = chess.Board(fen)
        move = chess.Move.from_uci(chosen[fen])
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} for {fen}")
        actions.append(move_to_action(board, move))
    with StockfishJudge(stockfish, nodes, cache_dir, engines=engines) as judge:
        lost = judge_positions(split, indices, {0: np.array(actions)}, judge, workers=engines)[0]
    report: dict[str, object] = {
        "schema": "wombat.external-strength.v1",
        "moves": str(moves),
        "games": str(games),
        "positions": len(indices),
        "nodes": nodes,
        **_summary(lost),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "strength.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "strength.npz", game_indices=indices, lost=lost)
    return report
