import json
from pathlib import Path
import subprocess

import pytest

from wombat.cli import main
from wombat.eval.matches import checkpoint_match_command, paired_smoke_command, summarize_pgn


def test_paired_match_command_swaps_reasoning_depths() -> None:
    command = paired_smoke_command(
        Path("cutechess-cli"), Path("python"), Path("project"), Path("match.pgn")
    )
    assert "option.ReasoningSteps=1" in command
    assert "option.ReasoningSteps=8" in command
    assert command.count("-engine") == 2
    assert "-repeat" in command



def _match_command(**overrides: object) -> list[str]:
    options: dict[str, object] = {
        "first_steps": 1,
        "second_steps": 64,
        "pairs": 3,
        "device": "cuda",
        "max_moves": 150,
    }
    options.update(overrides)
    return checkpoint_match_command(
        Path("cutechess-cli"),
        Path("venv/python"),
        Path("project"),
        Path("model.pt"),
        Path("openings.epd"),
        Path("games.pgn"),
        **options,  # type: ignore[arg-type]
    )


def test_checkpoint_match_runs_the_given_interpreter_not_a_console_script() -> None:
    command = _match_command()
    assert command.count(f"cmd={Path('venv/python')}") == 2
    assert not any("wombat.exe" in token for token in command)
    assert command.count("arg=-m") == 2
    assert command.count("arg=wombat") == 2


def test_checkpoint_match_passes_the_device_and_both_budgets_to_the_engines() -> None:
    command = _match_command(device="cuda", first_steps=1, second_steps=64)
    assert command.count("arg=--device") == 2
    assert command.count("arg=cuda") == 2
    assert command.count("arg=model.pt") == 2
    assert "option.ReasoningSteps=1" in command
    assert "option.ReasoningSteps=64" in command


def test_checkpoint_match_plays_every_opening_once_with_each_colour() -> None:
    command = _match_command(pairs=3, concurrency=2)
    assert command[command.index("-rounds") + 1] == "6"
    assert "-repeat" in command
    assert command[command.index("-concurrency") + 1] == "2"
    assert command[command.index("-maxmoves") + 1] == "150"
    with pytest.raises(ValueError):
        _match_command(pairs=0)


GAME = (
    '[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n[FEN "{fen}"]\n\n1. e4 {result}\n\n'
)
FIRST = "Wombat-H1"
SECOND = "Wombat-H64"
OPENINGS = ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "8/8/8/4k3/8/4K3/3P4/8 w - - 0 1")


def _write_pgn(path: Path, games: list[tuple[str, str, str, str]]) -> None:
    path.write_text(
        "".join(
            GAME.format(white=white, black=black, result=result, fen=fen)
            for white, black, result, fen in games
        ),
        encoding="utf-8",
    )


def test_summary_scores_the_second_engine_by_colour(tmp_path: Path) -> None:
    pgn = tmp_path / "games.pgn"
    _write_pgn(
        pgn,
        [
            (SECOND, FIRST, "1-0", OPENINGS[0]),
            (FIRST, SECOND, "1-0", OPENINGS[0]),
            (FIRST, SECOND, "0-1", OPENINGS[1]),
            (SECOND, FIRST, "1/2-1/2", OPENINGS[1]),
            (SECOND, FIRST, "*", OPENINGS[1]),
        ],
    )
    summary = summarize_pgn(pgn, FIRST, SECOND)
    assert (summary["wins"], summary["draws"], summary["losses"]) == (2, 1, 1)
    assert summary["unfinished"] == 1
    assert summary["games"] == 4
    assert summary["score"] == pytest.approx(0.625)
    pairs = summary["pairs"]
    assert isinstance(pairs, dict)
    assert pairs["pairs"] == 2
    assert pairs["pentanomial"] == [0, 0, 1, 1, 0]


def test_summary_rejects_a_game_between_other_players(tmp_path: Path) -> None:
    pgn = tmp_path / "games.pgn"
    _write_pgn(pgn, [("Other", FIRST, "1-0", OPENINGS[0])])
    with pytest.raises(ValueError, match="unknown player"):
        summarize_pgn(pgn, FIRST, SECOND)


def test_evaluate_match_is_registered_and_runs_cutechess_then_summarises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[list[str]] = []

    def fake_cutechess(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        ran.append(command)
        pgn = Path(command[command.index("-pgnout") + 1])
        _write_pgn(
            pgn,
            [
                (SECOND, FIRST, "1-0", OPENINGS[0]),
                (FIRST, SECOND, "0-1", OPENINGS[0]),
            ],
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("wombat.cli.subprocess.run", fake_cutechess)
    monkeypatch.setattr(
        "wombat.settings.load_executables", lambda: {"cutechess_cli": "cutechess-cli"}
    )
    monkeypatch.setattr("wombat.eval.games.select_openings", lambda *args, **kwargs: list(OPENINGS))
    output = tmp_path / "match"
    assert (
        main(
            [
                "evaluate-match",
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--pairs",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(ran) == 1
    assert ran[0][ran[0].index("-rounds") + 1] == "4"
    assert (output / "openings.epd").read_text(encoding="utf-8").count("\n") == 2
    report = json.loads((output / "match_summary.json").read_text(encoding="utf-8"))
    assert (report["wins"], report["losses"], report["second_steps"]) == (2, 0, 64)
    assert json.loads(capsys.readouterr().out)["schema"] == "wombat.paired-compute-match.v1"


def test_evaluate_match_needs_two_different_budgets(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "evaluate-match",
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--first-steps",
                "8",
                "--second-steps",
                "8",
            ]
        )


def test_cutechess_ends_a_checkpoint_game_at_max_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import asdict

    import torch

    from wombat.model import NetworkConfig, WombatNetwork
    from wombat.settings import load_executables

    if load_executables()["cutechess_cli"] is None:
        pytest.skip("Cute Chess CLI is not configured")
    config = NetworkConfig(
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
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {"model": WombatNetwork(config).state_dict(), "model_config": asdict(config)}, checkpoint
    )
    monkeypatch.setattr(
        "wombat.eval.games.select_openings", lambda *args, **kwargs: [OPENINGS[0]]
    )
    output = tmp_path / "match"
    code = main(
        [
            "evaluate-match",
            "--checkpoint",
            str(checkpoint),
            "--first-steps",
            "1",
            "--second-steps",
            "2",
            "--pairs",
            "1",
            "--device",
            "cpu",
            "--max-moves",
            "3",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    report = json.loads((output / "match_summary.json").read_text(encoding="utf-8"))
    assert report["games"] == 2
    assert report["unfinished"] == 0
    pgn = (output / "games.pgn").read_text(encoding="utf-8")
    plies = [int(line.split('"')[1]) for line in pgn.splitlines() if line.startswith("[PlyCount")]
    assert plies and max(plies) <= 6
