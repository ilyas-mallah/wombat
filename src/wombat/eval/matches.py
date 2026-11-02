"""Cute Chess paired-match orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class MatchRun:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    pgn: str


def paired_smoke_command(
    cutechess_cli: Path,
    python_executable: Path,
    project_dir: Path,
    pgn_path: Path,
) -> list[str]:
    common = [
        f"cmd={python_executable}",
        f"dir={project_dir}",
        "arg=-m",
        "arg=wombat",
        "arg=uci",
        "proto=uci",
    ]
    return [
        str(cutechess_cli),
        "-engine",
        "name=Wombat-T1",
        *common,
        "option.ReasoningSteps=1",
        "-engine",
        "name=Wombat-T8",
        *common,
        "option.ReasoningSteps=8",
        "-each",
        "tc=inf",
        "-rounds",
        "2",
        "-repeat",
        "-maxmoves",
        "20",
        "-recover",
        "-pgnout",
        str(pgn_path),
        "fi",
        "-srand",
        "0",
    ]


def run_paired_smoke(
    cutechess_cli: Path,
    python_executable: Path,
    project_dir: Path,
    pgn_path: Path,
) -> MatchRun:
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    command = paired_smoke_command(cutechess_cli, python_executable, project_dir, pgn_path)
    completed = subprocess.run(
        command,
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = MatchRun(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        pgn=str(pgn_path),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cute Chess failed: {asdict(result)}")
    if not pgn_path.exists():
        raise RuntimeError("Cute Chess completed without producing PGN")
    return result

