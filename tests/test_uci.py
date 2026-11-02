from __future__ import annotations

from dataclasses import asdict
from io import StringIO
from pathlib import Path

import chess
import pytest
import torch

from wombat.cli import main
from wombat.engine.uci import run_uci
from wombat.model import NetworkConfig, WombatNetwork
from wombat.policy import FixturePolicy

TINY = NetworkConfig(
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


def test_minimal_uci_session() -> None:
    board = chess.Board()
    commands = StringIO(
        "uci\nisready\nsetoption name ReasoningSteps value 8\nposition startpos\ngo nodes 1\nquit\n"
    )
    output = StringIO()
    assert run_uci(commands, output, FixturePolicy({board.fen(): "e2e4"})) == 0
    lines = output.getvalue().splitlines()
    assert "uciok" in lines
    assert "readyok" in lines
    assert "bestmove e2e4" in lines


def test_reasoning_steps_option_reaches_128() -> None:
    output = StringIO()
    run_uci(StringIO("uci\nquit\n"), output)
    assert "option name ReasoningSteps type spin default 1 min 1 max 128" in output.getvalue()


def test_checkpoint_engine_answers_go_with_legal_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {"model": WombatNetwork(TINY).state_dict(), "model_config": asdict(TINY)}, checkpoint
    )
    monkeypatch.setattr(
        "sys.stdin",
        StringIO(
            "uci\n"
            "setoption name ReasoningSteps value 128\n"
            "position startpos moves e2e4\n"
            "go movetime 100\n"
            "quit\n"
        ),
    )
    assert main(["uci", "--checkpoint", str(checkpoint)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "option name ReasoningSteps type spin default 8 min 1 max 128" in lines
    best = [line.split()[1] for line in lines if line.startswith("bestmove")]
    board = chess.Board()
    board.push_uci("e2e4")
    assert len(best) == 1
    assert chess.Move.from_uci(best[0]) in board.legal_moves
