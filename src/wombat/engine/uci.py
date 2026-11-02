"""Minimal synchronous UCI loop for evaluation matches."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

import chess

from wombat.eval.agreement import DRAW_GUARD_VALUE, select_move
from wombat.policy import DeterministicRandomPolicy, PolicyProvider, SelectingPolicyProvider
from wombat.selection import SELECT_MODES

MAX_REASONING_STEPS = 128
CHECKPOINT_REASONING_STEPS = 8


def load_provider(checkpoint: Path) -> PolicyProvider:
    """A trained checkpoint served on two CPU threads."""

    import torch

    from wombat.eval.puzzle_corpus_eval import load_checkpoint
    from wombat.model.adapter import TorchPolicyProvider

    torch.set_num_threads(2)
    device = torch.device("cpu")
    return TorchPolicyProvider(load_checkpoint(checkpoint, device), device, checkpoint=checkpoint)


def _set_position(board: chess.Board, tokens: list[str]) -> chess.Board:
    if not tokens:
        raise ValueError("position requires startpos or fen")
    if tokens[0] == "startpos":
        board = chess.Board()
        move_index = 1
    elif tokens[0] == "fen":
        try:
            move_marker = tokens.index("moves")
        except ValueError:
            move_marker = len(tokens)
        fen = " ".join(tokens[1:move_marker])
        board = chess.Board(fen)
        move_index = move_marker
    else:
        raise ValueError("position requires startpos or fen")

    if move_index < len(tokens) and tokens[move_index] == "moves":
        for uci in tokens[move_index + 1 :]:
            board.push_uci(uci)
    return board


def run_uci(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    provider: PolicyProvider | None = None,
    reasoning_steps: int = 1,
    draw_guard: float | None = DRAW_GUARD_VALUE,
) -> int:
    """Serve ``provider``; ``draw_guard`` is ``select_move``'s stalemate and repetition guard."""

    provider = provider or DeterministicRandomPolicy()
    board = chess.Board()
    default_steps = reasoning_steps

    def send(line: str) -> None:
        print(line, file=output_stream or sys.stdout, flush=True)

    for raw_line in input_stream or sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]
        try:
            if command == "uci":
                send("id name Wombat 0.1.0")
                send("id author Wombat contributors")
                send(
                    f"option name ReasoningSteps type spin default {default_steps}"
                    f" min 1 max {MAX_REASONING_STEPS}"
                )
                if isinstance(provider, SelectingPolicyProvider):
                    modes = " ".join(f"var {mode}" for mode in SELECT_MODES)
                    send(f"option name MoveSelection type combo default {provider.select} {modes}")
                    send(f"option name SelectBeta type string default {provider.beta:g}")
                send("uciok")
            elif command == "isready":
                send("readyok")
            elif command == "ucinewgame":
                board.reset()
            elif command == "setoption" and "name" in tokens and "value" in tokens:
                name_start = tokens.index("name") + 1
                value_start = tokens.index("value")
                name = " ".join(tokens[name_start:value_start])
                value = " ".join(tokens[value_start + 1 :])
                if name == "ReasoningSteps":
                    reasoning_steps = min(MAX_REASONING_STEPS, max(1, int(value)))
                elif name == "MoveSelection" and isinstance(provider, SelectingPolicyProvider):
                    provider.set_selection(value, provider.beta)
                elif name == "SelectBeta" and isinstance(provider, SelectingPolicyProvider):
                    provider.set_selection(provider.select, float(value))
            elif command == "position":
                board = _set_position(board, tokens[1:])
            elif command == "go":
                if board.is_game_over():
                    send("bestmove 0000")
                else:
                    move = select_move(provider, board, reasoning_steps, draw_guard=draw_guard)
                    send(f"bestmove {move.uci()}")
            elif command == "stop":
                continue
            elif command == "quit":
                return 0
        except (ValueError, IndexError) as error:
            send(f"info string error {error}")
    return 0


def main(checkpoint: Path | None = None) -> int:
    """Serve ``checkpoint`` when given, otherwise the deterministic random policy."""

    if checkpoint is None:
        return run_uci()
    return run_uci(provider=load_provider(checkpoint), reasoning_steps=CHECKPOINT_REASONING_STEPS)


if __name__ == "__main__":
    raise SystemExit(main())
