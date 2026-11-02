from pathlib import Path

import chess

from wombat.eval.regret import StockfishRegretEvaluator


def test_cache_key_changes_with_budget(tmp_path: Path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    low = StockfishRegretEvaluator(Path("stockfish"), 256, tmp_path)
    high = StockfishRegretEvaluator(Path("stockfish"), 1024, tmp_path)
    low.engine_label = high.engine_label = "Stockfish fixture"
    assert low._key_payload(board, move) != high._key_payload(board, move)

