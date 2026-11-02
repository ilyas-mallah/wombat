from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat.chess.encode import decode_position
from wombat.chess.moves import action_to_move, move_to_action
from wombat.data.game_corpus import GameSplit, densify, line_actions
from wombat.data.lichess import PuzzleRecord
from wombat.data.move_labels import label_positions
from wombat.data.puzzle_corpus import PuzzleSplit


class FirstMovesEngine:
    """Every legal move scored by its target square; each line continues with first moves."""

    id: ClassVar[dict[str, str]] = {"name": "Fake"}

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(self, board: chess.Board, limit: Any, multipv: int = 1, info: Any = None) -> Any:
        infos = []
        for move in list(board.legal_moves)[:multipv]:
            played, line = board.copy(stack=False), [move]
            played.push(move)
            while len(line) < 5 and (replies := list(played.legal_moves)):
                line.append(replies[0])
                played.push(replies[0])
            score = chess.engine.PovScore(chess.engine.Cp(move.to_square * 4 - 100), board.turn)
            infos.append({"pv": line, "score": score})
        return infos

    def quit(self) -> None:
        pass


@pytest.fixture
def labels(game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    engine = FirstMovesEngine()
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: engine)
    output = tmp_path / "labels"
    label_positions(
        game_corpus / "train",
        output,
        count=6,
        nodes=100,
        engine_command=["stockfish"],
        workers=1,
        chunk_size=3,
        low_priority=False,
    )
    return output


def _line(board: chess.Board, moves: list[str]) -> list[int]:
    played, actions = board.copy(stack=False), []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        actions.append(move_to_action(played, move))
        played.push(move)
    return actions


def test_game_lines_replay_the_stored_principal_variation(game_corpus: Path, labels: Path) -> None:
    split = GameSplit(game_corpus / "train", labels=labels)
    indices = np.sort(split.label_indices)
    lines = split.lookahead_targets(indices)["line_actions"]
    lengths = np.load(labels / "line_length.npy")
    slots = {int(index): slot for slot, index in enumerate(split.label_indices)}
    for row, index in enumerate(indices):
        board = decode_position(split.pieces[index], split.state[index])
        length = int(lengths[slots[int(index)]])
        # The fake engine's line: the deep-best move, then first legal moves.
        for ply in range(length):
            move = action_to_move(board, int(lines[row, ply]))
            if ply:
                assert move == next(iter(board.legal_moves))
            board.push(move)
        assert (lines[row, length:] == -1).all()
    assert (lines[:, 3] >= 0).any()
    dense = densify(split.sparse_batch(indices), 4.0, torch.device("cpu"))
    assert torch.equal(dense["line_actions"], split.batch(indices)["line_actions"])
    assert torch.equal(dense["line_actions"], torch.from_numpy(lines))
    unlabelled = GameSplit(game_corpus / "train").lookahead_targets(indices)["line_actions"]
    assert (unlabelled == -1).all()


def test_a_line_stops_at_its_first_unknown_ply() -> None:
    # Root frame squares: e2e4, then the opponent's e7e5 (stored in the root's frame), missing.
    origin = np.array([[12, 52, 255, 12]])
    target = np.array([[28, 36, 255, 20]])
    actions = line_actions(origin, target)
    board = chess.Board()
    assert actions[0, :2].tolist() == _line(board, ["e2e4", "e7e5"])
    assert actions[0, 2:].tolist() == [-1, -1]


def test_puzzle_lines_replay_the_solution(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], puzzle_corpus: Path
) -> None:
    records = {record.puzzle_id: record for record in puzzle_archive[1]}
    for name in ("train", "test"):
        split = PuzzleSplit(puzzle_corpus / name)
        indices = np.arange(len(split))
        lines = split.batch(indices, lines=True)["line_actions"].numpy()
        for row, index in enumerate(indices):
            record = records[str(split.puzzle_ids[split.puzzle[index]])]
            board = chess.Board(record.fen)
            start = 1 + 2 * int(split.line_index[index])
            for uci in record.moves[:start]:
                board.push(chess.Move.from_uci(uci))
            solution = list(record.moves[start : start + 4])
            assert lines[row].tolist() == _line(board, solution) + [-1] * (4 - len(solution))
        assert (lines[:, 3] >= 0).any()
    assert "line_actions" not in split.batch(indices)
