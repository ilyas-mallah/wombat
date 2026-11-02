"""A minimal UCI engine for tests: scores every legal move by its target square.

Each line continues with the best reply and one more best move, by the same score.
If the file named by ``FAKE_UCI_CRASH_ONCE`` exists, the engine deletes it and
exits with status 1 on its first search, like Stockfish on a bad position.
``FAKE_UCI_NO_ELO`` hides the strength-limiting options. Search limits are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import chess

LINE_PLIES = 3


def _best(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=lambda move: -move.to_square)


def principal_variation(board: chess.Board, first: chess.Move) -> list[chess.Move]:
    """``first`` and then best moves, up to ``LINE_PLIES`` plies or the game's end."""

    played = board.copy(stack=False)
    line = [first]
    played.push(first)
    while len(line) < LINE_PLIES and (replies := _best(played)):
        line.append(replies[0])
        played.push(replies[0])
    return line


def main() -> None:
    board = chess.Board()
    multipv = 1
    for line in sys.stdin:
        words = line.split()
        if not words:
            continue
        command = words[0]
        if command == "uci":
            print("id name FakeFish")
            for option in ("Threads", "Hash", "MultiPV"):
                print(f"option name {option} type spin default 1 min 1 max 1024")
            if not os.environ.get("FAKE_UCI_NO_ELO"):
                print("option name UCI_LimitStrength type check default false")
                print("option name UCI_Elo type spin default 1320 min 1320 max 3190")
            print("uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "setoption" and words[2].lower() == "multipv":
            multipv = int(words[4])
        elif command == "position":
            if words[1] == "startpos":
                board = chess.Board()
                rest = words[2:]
            else:
                end = words.index("moves") if "moves" in words else len(words)
                board = chess.Board(" ".join(words[2:end]))
                rest = words[end:]
            for uci in rest[1:]:
                board.push_uci(uci)
        elif command == "go":
            marker = os.environ.get("FAKE_UCI_CRASH_ONCE")
            if marker and Path(marker).exists():
                Path(marker).unlink()
                sys.exit(1)
            moves = _best(board)[:multipv]
            for rank, move in enumerate(moves, start=1):
                cp = move.to_square * 4 - 100
                pv = " ".join(ply.uci() for ply in principal_variation(board, move))
                print(f"info depth 1 multipv {rank} score cp {cp} nodes 1 pv {pv}")
            print(f"bestmove {moves[0].uci()}", flush=True)
        elif command == "quit":
            return


if __name__ == "__main__":
    main()
