from pathlib import Path

from wombat.eval.matches import paired_smoke_command


def test_paired_match_command_swaps_reasoning_depths() -> None:
    command = paired_smoke_command(
        Path("cutechess-cli"), Path("python"), Path("project"), Path("match.pgn")
    )
    assert "option.ReasoningSteps=1" in command
    assert "option.ReasoningSteps=8" in command
    assert command.count("-engine") == 2
    assert "-repeat" in command

