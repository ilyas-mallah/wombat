from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import struct

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import move_to_action
from wombat.cli import main
from wombat.data.chessbench import fen_key, import_chessbench
from wombat.eval.game_errors import (
    ErrorSet,
    balance_buckets,
    compare_game_errors,
    depth_report,
    error_summary,
    evaluate_game_errors,
    lichess_accuracy,
    live_mask,
    move_losses,
    non_pawn_pieces,
    paired_errors,
    prepare_error_set,
)
from wombat.eval.strength import move_accuracy
from wombat.external.chessbench import ActionValue9M, ChessBenchPlayer
from wombat.model import NetworkConfig, WombatNetwork

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
OPEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
OPEN_LATER = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 6 5"
ENDING = "8/5k2/8/3p4/8/2K5/4R3/8 b - - 3 60"
MIDDLE = "r2q1rk1/ppp2ppp/2n1bn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 4 10"


def test_move_losses_score_the_chosen_move_against_the_best() -> None:
    offsets = np.array([0, 3, 5, 6])
    actions = np.array([10, 20, 30, 5, 6, 7], dtype=np.int16)
    win = np.array([0.6, 0.9, 0.3, 0.2, 0.15, 0.95], dtype=np.float16)
    loss, best = move_losses(offsets, actions, win, np.array([0, 1, 2]), np.array([30, 6, 7]))
    assert best == pytest.approx([0.9, 0.2, 0.95], abs=1e-3)
    assert loss == pytest.approx([0.6, 0.05, 0.0], abs=1e-3)
    # Rows are looked up by index, so a subset keeps its own labels.
    loss, best = move_losses(offsets, actions, win, np.array([2, 0]), np.array([7, 20]))
    assert loss == pytest.approx([0.0, 0.0]) and best == pytest.approx([0.95, 0.9], abs=1e-3)
    with pytest.raises(ValueError, match="row 1 has no label"):
        move_losses(offsets, actions, win, np.array([0, 1]), np.array([10, 7]))


def test_summary_counts_losses_above_each_threshold() -> None:
    loss = np.array([0.0, 0.03, 0.07, 0.15, 0.25, 0.5, 0.5, 0.0])
    summary = error_summary(loss)
    rates = {name: summary[name]["rate"] for name in ("over_5", "over_10", "over_20", "over_40")}  # type: ignore[index]
    assert rates == pytest.approx(
        {"over_5": 5 / 8, "over_10": 4 / 8, "over_20": 3 / 8, "over_40": 2 / 8}
    )
    over_20 = summary["over_20"]
    assert isinstance(over_20, dict) and over_20["low"] < 3 / 8 < over_20["high"]
    assert summary["mean_loss"]["mean"] == pytest.approx(loss.mean())  # type: ignore[index]
    expected = np.mean([move_accuracy(100 * value) for value in loss])
    assert summary["lichess_accuracy"]["mean"] == pytest.approx(expected)  # type: ignore[index]
    assert lichess_accuracy(np.array([0.0]))[0] == pytest.approx(move_accuracy(0.0))


def test_live_positions_and_balance_buckets() -> None:
    best = np.array([0.02, 0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 1.0])
    assert live_mask(best).tolist() == [False, True, True, True, True, True, False, False]
    buckets = balance_buckets(best)
    assert np.stack(list(buckets.values())).sum(axis=0).tolist() == [1] * len(best)
    live = buckets["worse"] | buckets["level"] | buckets["better"]
    assert live.tolist() == live_mask(best).tolist()
    loss = np.array([0.5, 0.3, 0.0, 0.0, 0.25, 0.0, 0.6, 0.0])
    pieces = np.zeros((len(best), 64), dtype=np.uint8)
    pieces[:, 0] = 6  # a king, which is not counted
    pieces[:4, 1:15] = 2
    report = depth_report(loss, best, non_pawn_pieces(pieces))
    assert report["live"]["positions"] == 5  # type: ignore[index]
    assert report["live"]["over_20"]["count"] == 2  # type: ignore[index]
    assert report["all"]["over_20"]["count"] == 4  # type: ignore[index]
    assert report["phase_live"]["13+"]["positions"] == 3  # type: ignore[index]
    assert report["phase_live"]["0-4"]["positions"] == 2  # type: ignore[index]
    assert report["balance"]["won"]["over_40"]["count"] == 1  # type: ignore[index]


def test_paired_errors_give_second_minus_first() -> None:
    best = np.array([0.5, 0.5, 0.5, 0.5, 0.99])
    first = np.array([0.3, 0.3, 0.0, 0.0, 0.5])
    second = np.array([0.0, 0.3, 0.25, 0.0, 0.0])
    paired = paired_errors(first, second, best)
    live = paired["live"]
    assert live["positions"] == 4
    assert live["over_20"]["difference"] == pytest.approx(0.0)  # type: ignore[index]
    assert (live["over_20"]["gained"], live["over_20"]["lost"]) == (1, 1)  # type: ignore[index]
    assert live["mean_loss"]["mean"] == pytest.approx((-0.3 + 0.25) / 4)  # type: ignore[index]
    overall = paired["all"]
    assert overall["over_40"]["difference"] == pytest.approx(-1 / 5)  # type: ignore[index]
    assert overall["over_40"]["first"] == pytest.approx(1 / 5)  # type: ignore[index]
    same = paired_errors(first, first, best)["all"]
    assert same["over_5"]["difference"] == 0 and same["mean_loss"]["mean"] == 0  # type: ignore[index]
    assert paired_errors(first, second, np.full(5, 0.99))["live"] == {"positions": 0}
    with pytest.raises(ValueError, match="identical positions"):
        paired_errors(first, second[:4], best)


def _record(fen: str, move: str, win: float) -> bytes:
    fen_bytes, move_bytes = fen.encode(), move.encode()
    return (
        bytes([len(fen_bytes)])
        + fen_bytes
        + bytes([len(move_bytes)])
        + move_bytes
        + struct.pack(">d", win)
    )


def _bag(path: Path, records: list[bytes]) -> Path:
    ends = np.cumsum([len(record) for record in records])
    path.write_bytes(b"".join(records) + b"".join(struct.pack("<q", int(end)) for end in ends))
    return path


def _scores(fen: str, seed: int) -> dict[str, float]:
    moves = sorted(move.uci() for move in chess.Board(fen).legal_moves)
    wins = random.Random(seed).sample(range(1, 1000), len(moves))
    return {move: win / 1000 for move, win in zip(moves, wins, strict=True)}


def _actions() -> list[str]:
    """Every from-to pair and pawn promotion, a superset of the 9M's action list."""

    actions = []
    for start in chess.SQUARES:
        for end in chess.SQUARES:
            if start != end:
                actions.append(chess.Move(start, end).uci())
                ranks = (chess.square_rank(start), chess.square_rank(end))
                near = abs(chess.square_file(start) - chess.square_file(end)) <= 1
                if near and ranks in ((6, 7), (1, 0)):
                    actions.extend(chess.Move(start, end, piece).uci() for piece in (2, 3, 4, 5))
    return actions


@pytest.fixture(scope="module")
def error_set(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, dict[str, float]]]:
    """Three imported positions, one repeated with later counters and one in training."""

    root = tmp_path_factory.mktemp("errors")
    scores = {fen: _scores(fen, seed) for seed, fen in enumerate((OPEN, ENDING, MIDDLE))}
    scores[OPEN_LATER] = scores[OPEN]
    records = [
        _record(fen, move, win) for fen, moves in scores.items() for move, win in moves.items()
    ]
    bag = _bag(root / "test.bag", records)
    source = root / "import"
    import_chessbench(
        [("test", str(bag))],
        source,
        work_dir=root / "work",
        fraction=1.0,
        excluded=np.zeros(0, dtype=np.uint64),
        exclusion={},
        partitions=2,
        low_priority=False,
        minimum_free_gb=0,
    )
    training = root / "train"
    training.mkdir()
    # The same ending with another halfmove bucket still counts, and an unrelated board does not.
    boards = [chess.Board(ENDING), chess.Board()]
    boards[0].halfmove_clock = 40
    encoded = [encode_position(board) for board in boards]
    np.save(training / "pieces.npy", np.stack([item.pieces for item in encoded]))
    np.save(
        training / "state.npy", np.stack([state_vector(item) for item in encoded]).astype(np.uint8)
    )
    output = root / "set"
    manifest = prepare_error_set(source, output, [training], low_priority=False)
    assert manifest["positions"] == 3
    assert manifest["training_overlap"] == {"splits": {training.as_posix(): 1}, "positions": 1}
    assert manifest["fens_with_other_counters"] == 1
    return output, scores


def test_prepared_set_keeps_fens_and_leaves_out_training_positions(
    error_set: tuple[Path, dict[str, dict[str, float]]],
) -> None:
    directory, _ = error_set
    loaded = ErrorSet(directory)
    fens = [str(fen) for fen in loaded.fens]
    assert sorted(fens) == sorted(
        [min(OPEN, OPEN_LATER, key=lambda fen: fen_key(fen.encode())), ENDING, MIDDLE]
    )
    assert [fens[row] for row in loaded.overlap] == [ENDING]
    assert [fens[row] for row in loaded.rows()] == [fen for fen in fens if fen != ENDING]
    batch = loaded.batch(np.arange(len(loaded)))
    for row, fen in enumerate(fens):
        assert int(batch["legal_mask"][row].sum()) == chess.Board(fen).legal_moves.count()


def _expected_losses(
    directory: Path, scores: dict[str, dict[str, float]], label: str
) -> tuple[np.ndarray, np.ndarray]:
    loaded = ErrorSet(directory)
    with np.load(directory.parent / f"out-{label}" / "losses.npz") as saved:
        rows, chosen, loss = saved["rows"], saved[f"action_{label}"], saved[f"loss_{label}"]
    expected = []
    for row, action in zip(rows.tolist(), chosen.tolist(), strict=True):
        board = chess.Board(str(loaded.fens[row]))
        wins = {
            move_to_action(board, chess.Move.from_uci(uci)): win
            for uci, win in scores[board.fen()].items()
        }
        expected.append(max(wins.values()) - wins[action])
    return loss, np.array(expected)


def test_checkpoint_and_external_player_are_scored_and_paired(
    error_set: tuple[Path, dict[str, dict[str, float]]], tmp_path: Path
) -> None:
    directory, scores = error_set
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {"model": WombatNetwork(TINY).state_dict(), "model_config": asdict(TINY)}, checkpoint
    )
    first = directory.parent / "out-t1"
    report = evaluate_game_errors(
        directory, first, checkpoint=checkpoint, depths=(2, 1), device_name="cpu", batch_size=2
    )
    assert report["positions"] == 2 and list(report["results"]) == ["t1", "t2"]  # type: ignore[call-overload]
    loss, expected = _expected_losses(directory, scores, "t1")
    assert loss == pytest.approx(expected, abs=1e-3)

    actions = _actions()
    weights = tmp_path / "random.pt"
    torch.save(
        {
            "model": ActionValue9M(vocab=len(actions)).state_dict(),
            "actions": actions,
            "bucket_values": torch.linspace(0.5 / 128, 1 - 0.5 / 128, 128),
        },
        weights,
    )
    second = directory.parent / "out-play"
    evaluate_game_errors(
        directory,
        second,
        external="chessbench9m",
        external_weights=weights,
        device_name="cpu",
        batch_size=16,
    )
    loss, expected = _expected_losses(directory, scores, "play")
    assert loss == pytest.approx(expected, abs=1e-3)
    player = ChessBenchPlayer(weights)
    boards = [chess.Board(str(fen)) for fen in ErrorSet(directory).fens]
    # Small passes split the boards differently but choose the same moves as one at a time.
    assert player.best_moves(boards, batch_size=8) == [
        player.choose_move(board) for board in boards
    ]

    paired = compare_game_errors(first, second, 1)
    with np.load(first / "losses.npz") as left, np.load(second / "losses.npz") as right:
        change = right["loss_play"].astype(float) - left["loss_t1"].astype(float)
    assert paired["all"]["mean_loss"]["mean"] == pytest.approx(change.mean())  # type: ignore[index]
    with pytest.raises(ValueError, match="no depth 3"):
        compare_game_errors(first, second, 3)
    assert (
        main(
            [
                "compare-game-errors",
                "--first",
                str(first),
                "--second",
                str(first),
                "--depth",
                "1",
                "--second-depth",
                "2",
            ]
        )
        == 0
    )
    saved = json.loads((first / "report.json").read_text(encoding="utf-8"))
    assert saved["results"]["t2"]["all"]["positions"] == 2
