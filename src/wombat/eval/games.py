"""Paired games against Stockfish that rate Wombat's play in Elo.

Each balanced opening from the held-out game corpus is played twice with the
colours swapped. Stockfish plays at a fixed ``UCI_Elo`` level, so Wombat's rating
is that level plus the Elo difference implied by its score.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import chess
import chess.engine
import chess.pgn
import numpy as np

from wombat.data.game_corpus import GameSplit
from wombat.eval.agreement import DRAW_GUARD_VALUE, select_move
from wombat.policy import PolicyProvider
from wombat.selection import MIX_BETA

MATCH_SCHEMA = "wombat.stockfish-match.v1"
PAIR_POINTS = (0.0, 0.5, 1.0, 1.5, 2.0)
Z95 = 1.959964
# Scores are kept this far from 0 and 1 so a clean sweep maps to a finite Elo (about 1200).
SCORE_FLOOR = 1e-3


@runtime_checkable
class MovePlayer(Protocol):
    """A player that chooses its own moves by its own rule, such as a released external engine."""

    name: str

    def choose_move(self, board: chess.Board) -> chess.Move: ...


def select_openings(
    split_dir: Path,
    count: int,
    *,
    seed: int = 0,
    win_band: tuple[float, float] = (0.40, 0.60),
    min_pieces: int = 26,
) -> list[str]:
    """``count`` seeded balanced positions: best candidate inside ``win_band``, many pieces."""

    split = GameSplit(split_dir)
    if split.fens is None:
        raise ValueError(f"{split_dir} stores no FENs")
    offsets = np.asarray(split.candidate_offsets)
    best = np.asarray(split.candidate_win, dtype=np.float32)[offsets[:-1]]
    low, high = win_band
    balanced = np.flatnonzero((best >= low) & (best <= high))
    openings: list[str] = []
    for index in np.random.default_rng(seed).permutation(balanced):
        board = chess.Board(str(split.fens[index]))
        if (
            len(board.piece_map()) < min_pieces
            or board.status() != chess.STATUS_VALID
            or board.is_game_over(claim_draw=True)
        ):
            continue
        openings.append(board.fen())
        if len(openings) == count:
            return openings
    raise ValueError(f"only {len(openings)} of {count} openings pass the filters")


def elo_difference(score: float, floor: float = SCORE_FLOOR) -> float:
    """Logistic Elo difference for an expected score, clamped away from 0 and 1."""

    clamped = min(max(score, floor), 1 - floor)
    return 400 * math.log10(clamped / (1 - clamped))


def pentanomial(pair_points: Sequence[float]) -> list[int]:
    """Counts of pairs scoring 0, 0.5, 1, 1.5 and 2 points."""

    counts = [0] * len(PAIR_POINTS)
    for points in pair_points:
        counts[PAIR_POINTS.index(points)] += 1
    return counts


@dataclass(frozen=True)
class PairStatistics:
    pairs: int
    pentanomial: list[int]
    pair_score: float
    score_low: float
    score_high: float
    elo_difference: float
    elo_low: float
    elo_high: float


def pair_statistics(pair_points: Sequence[float]) -> PairStatistics:
    """Score, Elo difference and 95% interval from per-pair score variance."""

    if not pair_points:
        raise ValueError("no complete pairs")
    per_game = np.asarray(pair_points, dtype=np.float64) / 2
    score = float(per_game.mean())
    margin = Z95 * math.sqrt(float(per_game.var()) / len(per_game))
    low, high = max(score - margin, 0.0), min(score + margin, 1.0)
    return PairStatistics(
        pairs=len(per_game),
        pentanomial=pentanomial(pair_points),
        pair_score=score,
        score_low=low,
        score_high=high,
        elo_difference=elo_difference(score),
        elo_low=elo_difference(low),
        elo_high=elo_difference(high),
    )


def _round(pair: int, wombat_white: bool) -> str:
    return f"{pair + 1}.{1 if wombat_white else 2}"


@dataclass(frozen=True)
class GameRecord:
    pair: int
    wombat_white: bool
    opening: str
    result: str
    termination: str
    plies: int

    @property
    def round(self) -> str:
        return _round(self.pair, self.wombat_white)

    @property
    def wombat_points(self) -> float:
        if self.result == "1/2-1/2":
            return 0.5
        return float((self.result == "1-0") == self.wombat_white)


def open_opponent(
    command: Sequence[str], elo: int | None
) -> tuple[chess.engine.SimpleEngine, dict[str, object]]:
    """A one-thread engine limited to ``elo`` when it supports ``UCI_Elo``."""

    engine = chess.engine.SimpleEngine.popen_uci(list(command))
    try:
        options: dict[str, chess.engine.ConfigValue] = {}
        if "Threads" in engine.options:
            options["Threads"] = 1
        limited = False
        elo_range = None
        if (
            elo is not None
            and "UCI_LimitStrength" in engine.options
            and "UCI_Elo" in engine.options
        ):
            option = engine.options["UCI_Elo"]
            elo_range = [option.min, option.max]
            low = option.min if option.min is not None else elo
            high = option.max if option.max is not None else elo
            if not low <= elo <= high:
                raise ValueError(f"UCI_Elo {elo} is outside {low}..{high}")
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = elo
            limited = True
        engine.configure(options)
    except BaseException:
        engine.quit()
        raise
    name = str(engine.id.get("name", "engine"))
    return engine, {
        "name": name,
        "label": f"{name} Elo {elo}" if limited else f"{name} full strength",
        "limit_strength": limited,
        "elo": elo if limited else None,
        "elo_range": elo_range,
    }


def adjudicate(board: chess.Board) -> tuple[str, str] | None:
    """Result and termination if the game is over, else None.

    Like Cute Chess, a draw comes once a position has occurred three times or at fifty moves,
    not whenever the side to move merely could claim one with its next move.
    """

    outcome = board.outcome()
    if outcome is not None:
        return outcome.result(), outcome.termination.name.lower()
    if board.is_repetition(3):
        return "1/2-1/2", "threefold_repetition"
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "fifty_moves"
    return None


def play_game(
    provider: PolicyProvider | MovePlayer,
    engine: chess.engine.SimpleEngine,
    opening: str,
    *,
    pair: int,
    wombat_white: bool,
    steps: int,
    limit: chess.engine.Limit,
    max_plies: int,
    draw_guard: float | None = DRAW_GUARD_VALUE,
) -> tuple[GameRecord, chess.Board]:
    """One game from ``opening``; a draw is adjudicated at ``max_plies``.

    ``draw_guard`` is ``select_move``'s value threshold for avoiding stalemate and repetition.
    A ``MovePlayer`` chooses by its own rule and ignores ``steps`` and ``draw_guard``.
    """

    board = chess.Board(opening)
    game_key = object()
    while True:
        ended = adjudicate(board)
        if ended is not None:
            result, termination = ended
            break
        if len(board.move_stack) >= max_plies:
            result, termination = "1/2-1/2", "ply_cap"
            break
        if (board.turn == chess.WHITE) == wombat_white:
            if isinstance(provider, MovePlayer):
                move = provider.choose_move(board)
            else:
                move = select_move(provider, board, steps, draw_guard=draw_guard)
        else:
            played = engine.play(board, limit, game=game_key).move
            if played is None:
                raise RuntimeError(f"engine returned no move in {board.fen()}")
            move = played
        board.push(move)
    record = GameRecord(pair, wombat_white, opening, result, termination, len(board.move_stack))
    return record, board


def _pgn_text(record: GameRecord, board: chess.Board, names: tuple[str, str], event: str) -> str:
    game = chess.pgn.Game.from_board(board)
    wombat, opponent = names
    game.headers["Event"] = event
    game.headers["Site"] = "local"
    game.headers["Round"] = record.round
    game.headers["White"] = wombat if record.wombat_white else opponent
    game.headers["Black"] = opponent if record.wombat_white else wombat
    game.headers["Result"] = record.result
    game.headers["Termination"] = record.termination
    game.headers["PlyCount"] = str(record.plies)
    return str(game) + "\n\n"


def _load_games(path: Path, openings: Sequence[str]) -> dict[str, tuple[GameRecord, str]]:
    """Complete games already in the PGN, keyed by round, with their text."""

    games: dict[str, tuple[GameRecord, str]] = {}
    if not path.exists():
        return games
    with path.open(encoding="utf-8") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            headers = game.headers
            pair_text, _, colour = headers.get("Round", "").partition(".")
            if not pair_text.isdigit() or colour not in ("1", "2"):
                continue
            pair = int(pair_text) - 1
            plies = sum(1 for _ in game.mainline_moves())
            if str(plies) != headers.get("PlyCount") or game.errors:
                continue
            if not 0 <= pair < len(openings) or game.board().fen() != openings[pair]:
                raise ValueError(f"{path} holds games from a different opening suite")
            record = GameRecord(
                pair,
                colour == "1",
                openings[pair],
                headers["Result"],
                headers.get("Termination", ""),
                plies,
            )
            text = io.StringIO()
            print(game, file=text, end="\n\n")
            games[record.round] = (record, text.getvalue())
    return games


def summarize(records: Sequence[GameRecord], elo: int | None) -> dict[str, object]:
    """Wombat's results, plus pair statistics over complete pairs."""

    points = [record.wombat_points for record in records]
    by_pair: dict[int, list[float]] = {}
    for record in records:
        by_pair.setdefault(record.pair, []).append(record.wombat_points)
    pair_points = [sum(items) for _, items in sorted(by_pair.items()) if len(items) == 2]
    summary: dict[str, object] = {
        "games": len(records),
        "wins": points.count(1.0),
        "draws": points.count(0.5),
        "losses": points.count(0.0),
        "score": sum(points) / len(points) if points else None,
        "terminations": dict(Counter(record.termination for record in records)),
    }
    if pair_points:
        statistics = pair_statistics(pair_points)
        summary.update(asdict(statistics))
        if elo is not None:
            summary["wombat_elo"] = elo + statistics.elo_difference
            summary["wombat_elo_low"] = elo + statistics.elo_low
            summary["wombat_elo_high"] = elo + statistics.elo_high
    return summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def play_match(
    provider: PolicyProvider | MovePlayer,
    opponent: Sequence[str],
    openings: Sequence[str],
    output: Path,
    *,
    steps: int = 8,
    elo: int | None = 1320,
    movetime: float | None = 0.1,
    nodes: int | None = None,
    max_plies: int = 300,
    identity: dict[str, object] | None = None,
    draw_guard: float | None = DRAW_GUARD_VALUE,
) -> dict[str, object]:
    """Play every opening with both colours, resuming from games already in ``output``."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if movetime is None and nodes is None:
        raise ValueError("give a movetime or a node limit")
    # A node limit replaces the time limit, so the opponent's strength does not depend on load.
    limit = (
        chess.engine.Limit(nodes=nodes) if nodes is not None else chess.engine.Limit(time=movetime)
    )
    config: dict[str, object] = {
        **(identity or {}),
        "steps": steps,
        "requested_elo": elo,
        "movetime": None if nodes is not None else movetime,
        "nodes": nodes,
        "max_plies": max_plies,
        "draw_guard": draw_guard,
        "openings": list(openings),
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path, pgn_path = output / "match.json", output / "games.pgn"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("config") != config:
            raise ValueError(f"{output} holds a different match; choose another output")
    games = _load_games(pgn_path, openings)
    # Rewriting keeps only complete games, dropping any half-written tail.
    pgn_path.write_text("".join(text for _, text in games.values()), encoding="utf-8")
    records = [record for record, _ in games.values()]

    engine, engine_info = open_opponent(opponent, elo)
    wombat_name = provider.name if isinstance(provider, MovePlayer) else f"Wombat T{steps}"
    names = (wombat_name, str(engine_info["label"]))
    event = f"{wombat_name} vs {engine_info['label']}"
    level = engine_info["elo"] if isinstance(engine_info["elo"], int) else None

    def report(status: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": MATCH_SCHEMA,
            "status": status,
            "config": config,
            "opponent": engine_info,
            **summarize(records, level),
        }
        _write_json(report_path, payload)
        return payload

    try:
        for pair, opening in enumerate(openings):
            for wombat_white in (True, False):
                if _round(pair, wombat_white) in games:
                    continue
                record, board = play_game(
                    provider,
                    engine,
                    opening,
                    pair=pair,
                    wombat_white=wombat_white,
                    steps=steps,
                    limit=limit,
                    max_plies=max_plies,
                    draw_guard=draw_guard,
                )
                with pgn_path.open("a", encoding="utf-8") as handle:
                    handle.write(_pgn_text(record, board, names, event))
                records.append(record)
                report("running")
    finally:
        try:
            engine.quit()
        except chess.engine.EngineError:
            pass
    return report("complete")


def play_stockfish(
    checkpoint: Path | None,
    opponent: Sequence[str],
    *,
    levels: Sequence[int] = (1320,),
    steps: int = 8,
    pairs: int = 50,
    movetime: float | None = 0.1,
    nodes: int | None = None,
    max_plies: int = 300,
    output: Path,
    split_dir: Path = Path("data/processed/games-v1/test"),
    seed: int = 0,
    win_band: tuple[float, float] = (0.40, 0.60),
    low_priority: bool = True,
    draw_guard: float | None = DRAW_GUARD_VALUE,
    select: str = "policy",
    beta: float = MIX_BETA,
    external: str | None = None,
    external_weights: Path | None = None,
    device_name: str = "cpu",
) -> list[dict[str, object]]:
    """One match per Stockfish level, each in ``output/elo-<level>``, the model on ``device_name``.

    ``select`` and ``beta`` choose Wombat's moves as ``wombat.selection.selection_scores`` does.
    ``external`` names a released engine in ``EXTERNAL_PLAYERS`` that plays instead of a
    checkpoint by its own move rule, so ``steps``, ``select`` and ``draw_guard`` do not apply.
    """

    import torch

    from wombat.data.puzzle_corpus import lower_priority

    if (checkpoint is None) == (external is None):
        raise ValueError("give either a checkpoint or an external player")
    if low_priority:
        lower_priority()
    torch.set_num_threads(2)
    device = torch.device(device_name)
    provider: PolicyProvider | MovePlayer
    if external is not None:
        from wombat.external.chessbench import EXTERNAL_PLAYERS

        if external not in EXTERNAL_PLAYERS:
            raise ValueError(f"unknown external player {external}")
        if external_weights is None:
            raise ValueError(f"{external} needs its weights file")
        if select != "policy":
            raise ValueError("move selection modes apply to Wombat checkpoints only")
        provider = EXTERNAL_PLAYERS[external](external_weights, device)
        draw_guard = None
        identity: dict[str, object] = {
            "player": external,
            "weights": str(external_weights),
            "split": str(split_dir),
            "seed": seed,
        }
    else:
        from wombat.eval.puzzle_corpus_eval import load_checkpoint
        from wombat.model.adapter import TorchPolicyProvider

        assert checkpoint is not None
        model = load_checkpoint(checkpoint, device, select)
        provider = TorchPolicyProvider(model, device, select=select, beta=beta)
        identity = {
            "checkpoint": str(checkpoint),
            "split": str(split_dir),
            "seed": seed,
        }
        # Policy matches omit the fields, so existing match directories still resume.
        if select != "policy":
            identity.update(select=select, select_beta=beta if select == "mix" else None)
    openings = select_openings(split_dir, pairs, seed=seed, win_band=win_band)
    return [
        play_match(
            provider,
            opponent,
            openings,
            output / (f"elo-{level}" if level else "full-strength"),
            steps=steps,
            elo=level or None,
            movetime=movetime,
            nodes=nodes,
            max_plies=max_plies,
            identity=identity,
            draw_guard=draw_guard,
        )
        for level in levels
    ]
