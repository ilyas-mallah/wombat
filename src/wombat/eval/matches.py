"""Cute Chess paired-match orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

import chess.pgn

from wombat.eval.games import pair_statistics


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


def checkpoint_match_command(
    cutechess_cli: Path,
    python_executable: Path,
    project_dir: Path,
    checkpoint: Path,
    openings_path: Path,
    pgn_path: Path,
    *,
    first_steps: int,
    second_steps: int,
    pairs: int,
    device: str,
    max_moves: int,
    concurrency: int = 1,
) -> list[str]:
    """Build a same-checkpoint match that plays every opening once with each colour."""

    if pairs < 1:
        raise ValueError("pairs must be positive")
    common = [
        f"cmd={python_executable}",
        f"dir={project_dir}",
        "arg=-m",
        "arg=wombat",
        "arg=uci",
        "arg=--checkpoint",
        f"arg={checkpoint}",
        "arg=--device",
        f"arg={device}",
        "proto=uci",
    ]
    return [
        str(cutechess_cli),
        "-engine",
        f"name=Wombat-H{first_steps}",
        *common,
        f"option.ReasoningSteps={first_steps}",
        "-engine",
        f"name=Wombat-H{second_steps}",
        *common,
        f"option.ReasoningSteps={second_steps}",
        "-each",
        "tc=inf",
        "-openings",
        f"file={openings_path}",
        "format=epd",
        "order=sequential",
        "-rounds",
        str(2 * pairs),
        "-repeat",
        "-concurrency",
        str(concurrency),
        "-maxmoves",
        str(max_moves),
        "-recover",
        "-pgnout",
        str(pgn_path),
        "fi",
        "-srand",
        "0",
    ]


def summarize_pgn(pgn_path: Path, first_name: str, second_name: str) -> dict[str, object]:
    """W/D/L, score and the per-opening pair statistics from the second engine's side.

    Games without a result are counted as ``unfinished`` and left out of the score.
    """

    wins = draws = losses = unfinished = 0
    terminations: dict[str, int] = {}
    by_opening: dict[str, tuple[list[float], list[float]]] = {}
    with pgn_path.open(encoding="utf-8") as source:
        while game := chess.pgn.read_game(source):
            result = game.headers.get("Result")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                unfinished += 1
                continue
            termination = game.headers.get("Termination", "unknown")
            terminations[termination] = terminations.get(termination, 0) + 1
            second_white = game.headers.get("White") == second_name
            if game.headers.get("White") not in (first_name, second_name):
                raise ValueError(f"unknown player in {pgn_path}: {game.headers.get('White')}")
            if result == "1/2-1/2":
                points = 0.5
                draws += 1
            elif (result == "1-0") == second_white:
                points = 1.0
                wins += 1
            else:
                points = 0.0
                losses += 1
            as_white, as_black = by_opening.setdefault(game.headers.get("FEN", ""), ([], []))
            (as_white if second_white else as_black).append(points)
    games = wins + draws + losses
    pair_points = [
        white + black
        for as_white, as_black in by_opening.values()
        for white, black in zip(as_white, as_black, strict=False)
    ]
    summary: dict[str, object] = {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unfinished": unfinished,
        "score": (wins + 0.5 * draws) / games if games else 0.0,
        "terminations": terminations,
    }
    if pair_points:
        summary["pairs"] = asdict(pair_statistics(pair_points))
    return summary
