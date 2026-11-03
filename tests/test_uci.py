from __future__ import annotations

from dataclasses import asdict
from io import StringIO
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from wombat.cli import main
from wombat.engine.uci import load_provider, run_uci
from wombat.eval.puzzle_corpus_eval import load_checkpoint
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


def test_uci_device_option_reaches_the_checkpoint_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[tuple[Path, str]] = []

    def fake_loader(checkpoint: Path, device_name: str = "cpu") -> FixturePolicy:
        loaded.append((checkpoint, device_name))
        return FixturePolicy({})

    monkeypatch.setattr("wombat.engine.uci.load_provider", fake_loader)
    monkeypatch.setattr("sys.stdin", StringIO("quit\n"))
    assert main(["uci", "--checkpoint", str(tmp_path / "m.pt"), "--device", "cuda"]) == 0
    monkeypatch.setattr("sys.stdin", StringIO("quit\n"))
    assert main(["uci", "--checkpoint", str(tmp_path / "m.pt")]) == 0
    assert loaded == [(tmp_path / "m.pt", "cuda"), (tmp_path / "m.pt", "cpu")]


def test_load_provider_builds_the_model_on_the_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []
    real_load = load_checkpoint

    def spy(checkpoint: Path, device: torch.device) -> WombatNetwork:
        requested.append(device.type)
        return real_load(checkpoint, torch.device("cpu"))

    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {"model": WombatNetwork(TINY).state_dict(), "model_config": asdict(TINY)}, checkpoint
    )
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.load_checkpoint", spy)
    monkeypatch.setattr("wombat.model.adapter.TorchPolicyProvider.__init__", lambda *a, **k: None)
    load_provider(checkpoint, "cuda")
    load_provider(checkpoint)
    assert requested == ["cuda", "cpu"]


def test_go_reports_the_model_value_so_a_gui_can_apply_maxmoves() -> None:
    board = chess.Board()

    class Valued(FixturePolicy):
        def policy_and_value(
            self, board: chess.Board, reasoning_steps: int
        ) -> tuple[np.ndarray, float]:
            return self.policy_logits(board, reasoning_steps), 0.25

    output = StringIO()
    run_uci(
        StringIO("setoption name ReasoningSteps value 64\nposition startpos\ngo\nquit\n"),
        output,
        Valued({board.fen(): "e2e4"}),
    )
    lines = output.getvalue().splitlines()
    assert lines[-2:] == ["info depth 64 score cp 250", "bestmove e2e4"]


def test_go_without_a_value_sends_only_the_move() -> None:
    board = chess.Board()
    output = StringIO()
    run_uci(
        StringIO("position startpos\ngo\nquit\n"), output, FixturePolicy({board.fen(): "e2e4"})
    )
    assert output.getvalue().splitlines() == ["bestmove e2e4"]
