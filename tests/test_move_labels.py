from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat.chess.encode import decode_position, encode_position, state_vector
from wombat.chess.moves import action_to_move, legal_action_mask, legal_actions, move_to_action
from wombat.data.game_corpus import (
    IGNORE,
    NO_LINE,
    GameSplit,
    JoinedGameSplit,
    densify,
    win_chance,
    with_children,
)
from wombat.data import move_labels
from wombat.data.move_labels import (
    LINE_PLIES,
    finalize_labels,
    label_positions,
    merge_labels,
    principal_line,
    score_all_moves,
)
from wombat import puzzle_training
from wombat.cli import build_parser, main
from wombat.eval.puzzle_corpus_eval import load_checkpoint
from wombat.losses import LossWeights, complete_targets, graded_loss
from wombat.model import NetworkConfig, WombatNetwork
from wombat.puzzle_training import PuzzleTrainingConfig, train_puzzles

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


def fake_line(board: chess.Board, first: chess.Move, plies: int = 5) -> list[chess.Move]:
    """``first`` and then each position's first legal move, up to ``plies`` or the game's end."""

    played = board.copy(stack=False)
    line = [first]
    played.push(first)
    while len(line) < plies and (replies := list(played.legal_moves)):
        line.append(replies[0])
        played.push(replies[0])
    return line


class FakeEngine:
    """Scores every legal move by its target square, as a multi-variation search would."""

    calls = 0

    def __init__(self) -> None:
        self.id = {"name": "Fake"}

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(
        self, board: chess.Board, limit: Any, multipv: int = 1, info: Any = None
    ) -> list[dict]:
        FakeEngine.calls += 1
        moves = list(board.legal_moves)[:multipv]
        return [
            {
                "pv": fake_line(board, move),
                "score": chess.engine.PovScore(
                    chess.engine.Cp(move.to_square * 4 - 100), board.turn
                ),
            }
            for move in moves
        ]

    def quit(self) -> None:
        pass


@pytest.fixture
def labels(game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    FakeEngine.calls = 0
    output = tmp_path / "labels"
    label_positions(
        game_corpus / "train",
        output,
        count=5,
        nodes=100,
        engine_command=["stockfish"],
        workers=1,
        chunk_size=2,
        low_priority=False,
    )
    return output


def test_every_legal_move_of_the_chosen_positions_is_scored(
    game_corpus: Path, labels: Path
) -> None:
    manifest = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["positions"] == 5
    assert len(list((labels / "chunks").glob("chunk-*.npz"))) == 3
    indices = np.load(labels / "label_indices.npy")
    assert np.all(np.diff(indices) > 0)
    split = GameSplit(game_corpus / "train")
    offsets = np.load(labels / "label_offsets.npy")
    actions = np.load(labels / "label_actions.npy")
    masks = split.legal_mask(indices)
    for row in range(len(indices)):
        scored = np.zeros(4672, dtype=bool)
        scored[actions[offsets[row] : offsets[row + 1]]] = True
        assert np.array_equal(scored, masks[row])


def test_labelling_resumes_from_finished_chunks(
    game_corpus: Path, labels: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (labels / "chunks" / "chunk-00001.npz").unlink()
    (labels / "chunks" / "chunk-00002.partial.npz").write_bytes(b"interrupted")
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    FakeEngine.calls = 0
    manifest = label_positions(
        game_corpus / "train",
        labels,
        count=5,
        nodes=100,
        engine_command=["stockfish"],
        workers=1,
        chunk_size=2,
        low_priority=False,
    )
    assert FakeEngine.calls == 2
    assert manifest["positions"] == 5


def _label_rest(
    game_corpus: Path, output: Path, exclude: list[Path], **options: Any
) -> dict[str, object]:
    FakeEngine.calls = 0
    settings: dict[str, Any] = {"nodes": 100, "workers": 1, "chunk_size": 2}
    return label_positions(
        game_corpus / "train",
        output,
        engine_command=["stockfish"],
        low_priority=False,
        exclude=exclude,
        **(settings | options),
    )


def test_exclusion_labels_only_unlabelled_positions_and_resumes(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    size = len(np.load(game_corpus / "train" / "pieces.npy"))
    first = np.load(labels / "label_indices.npy")
    rest = tmp_path / "rest"
    manifest = _label_rest(game_corpus, rest, [labels])
    indices = np.load(rest / "label_indices.npy")
    assert manifest["positions"] == size - 5
    assert not np.isin(indices, first).any()
    assert np.array_equal(np.union1d(first, indices), np.arange(size))
    excluded = manifest["excluded"]
    assert isinstance(excluded, list) and excluded[0]["path"] == labels.as_posix()
    assert excluded[0]["positions"] == 5
    split = manifest["split"]
    assert isinstance(split, dict) and split["positions"] == size

    (rest / "chunks" / "chunk-00000.npz").unlink()
    again = _label_rest(game_corpus, rest, [labels])
    assert FakeEngine.calls == 2
    assert again["sha256"] == manifest["sha256"]
    with pytest.raises(ValueError, match="different selection"):
        _label_rest(game_corpus, rest, [labels], chunk_size=3)

    _label_rest(game_corpus, tmp_path / "two", [labels], count=2)
    order = np.random.default_rng(0).permutation(size)
    expected = np.sort(order[~np.isin(order, first)][:2])
    assert np.array_equal(np.load(tmp_path / "two" / "label_indices.npy"), expected)


def test_reverse_labelling_starts_at_the_last_chunk_and_skips_chunks_that_appear(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    forward = tmp_path / "forward"
    expected = _label_rest(game_corpus, forward, [labels])
    first_chunk = forward / "chunks" / "chunk-00000.npz"
    written: list[str] = []
    write = move_labels._write_chunk

    def record(path: Path, indices: np.ndarray, results: list[Any]) -> None:
        write(path, indices, results)
        written.append(path.name)
        # Another machine finishes the first chunk while this run is going.
        target = path.parent / first_chunk.name
        if len(written) == 1 and not target.exists():
            target.write_bytes(first_chunk.read_bytes())

    monkeypatch.setattr(move_labels, "_write_chunk", record)
    shared = tmp_path / "shared"
    manifest = _label_rest(game_corpus, shared, [labels], reverse=True)
    chunks = sorted(path.name for path in (forward / "chunks").glob("chunk-*.npz"))
    assert written == chunks[::-1][: len(chunks) - 1]
    assert manifest["sha256"] == expected["sha256"]


def test_label_moves_cli_excludes_labelled_positions(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    monkeypatch.setattr(move_labels, "lower_priority", lambda: None)
    args = build_parser().parse_args(["label-moves", "--exclude", "a", "b"])
    assert (args.count, args.exclude) == (None, [Path("a"), Path("b")])
    arguments = ["label-moves", "--split", str(game_corpus / "train"), "--nodes", "100"]
    arguments += ["--output", str(tmp_path / "rest"), "--exclude", str(labels)]
    assert main([*arguments, "--engines", "1", "--stockfish", "stockfish"]) == 0
    _label_rest(game_corpus, tmp_path / "direct", [labels])
    assert np.array_equal(
        np.load(tmp_path / "rest" / "label_indices.npy"),
        np.load(tmp_path / "direct" / "label_indices.npy"),
    )


LABEL_FILES = (
    "label_indices",
    "label_offsets",
    "label_actions",
    "label_win",
    "line_length",
    "line_from",
    "line_to",
    "line_boards",
)


def test_merged_labels_equal_labelling_the_union_directly(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    size = len(np.load(game_corpus / "train" / "pieces.npy"))
    _label_rest(game_corpus, tmp_path / "rest", [labels])
    direct = _label_rest(game_corpus, tmp_path / "all", [])
    # Blocks of two rows split the ragged columns mid source.
    merged = merge_labels([labels, tmp_path / "rest"], tmp_path / "merged", block_rows=2)
    arguments = ["merge-move-labels", "--inputs", str(tmp_path / "rest"), str(labels)]
    assert main([*arguments, "--output", str(tmp_path / "cli")]) == 0
    for output in (tmp_path / "merged", tmp_path / "cli"):
        for name in LABEL_FILES:
            expected = np.load(tmp_path / "all" / f"{name}.npy")
            actual = np.load(output / f"{name}.npy")
            assert actual.dtype == expected.dtype, name
            assert np.array_equal(actual, expected), name
    assert merged["sha256"] == direct["sha256"]
    assert (merged["positions"], merged["unlabelled"]) == (size, 0)
    assert merged["moves"] == direct["moves"]
    assert merged["line_lengths"] == direct["line_lengths"]
    inputs = merged["inputs"]
    assert isinstance(inputs, list) and [entry["positions"] for entry in inputs] == [5, size - 5]
    assert not (tmp_path / "merged.partial").exists()
    split = GameSplit(game_corpus / "train", labels=tmp_path / "merged")
    assert np.array_equal(split.trainable(), np.arange(size))


def test_merging_refuses_overlaps_and_mismatched_labels(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    rest = tmp_path / "rest"
    _label_rest(game_corpus, rest, [labels])
    shutil.copytree(labels, tmp_path / "copy")
    with pytest.raises(ValueError, match="overlap"):
        merge_labels([labels, tmp_path / "copy"], tmp_path / "merged")
    for key, value in (("nodes", 200), ("engine", "Other"), ("split", {"positions": 1})):
        changed = tmp_path / f"changed-{key}"
        shutil.copytree(rest, changed)
        manifest = json.loads((changed / "manifest.json").read_text(encoding="utf-8"))
        manifest[key] = value
        (changed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            merge_labels([labels, changed], tmp_path / "merged")
    with pytest.raises(ValueError, match="different split"):
        _label_rest(game_corpus, tmp_path / "wrong", [tmp_path / "changed-split"])
    (tmp_path / "merged").mkdir()
    with pytest.raises(FileExistsError):
        merge_labels([labels, rest], tmp_path / "merged")
    assert not (tmp_path / "merged.partial").exists()


def test_merging_mixed_node_counts_is_explicit_and_recorded(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    size = len(np.load(game_corpus / "train" / "pieces.npy"))
    rest = tmp_path / "rest"
    _label_rest(game_corpus, rest, [labels], nodes=40)
    _label_rest(game_corpus, tmp_path / "all", [])
    with pytest.raises(ValueError, match="nodes"):
        merge_labels([labels, rest], tmp_path / "strict")
    assert not (tmp_path / "strict").exists()
    arguments = ["merge-move-labels", "--inputs", str(labels), str(rest), "--allow-mixed-nodes"]
    assert main([*arguments, "--output", str(tmp_path / "mixed")]) == 0
    manifest = json.loads((tmp_path / "mixed" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["nodes"] == [40, 100]
    assert [(entry["nodes"], entry["positions"]) for entry in manifest["inputs"]] == [
        (100, 5),
        (40, size - 5),
    ]
    # The fake engine ignores the node count, so the rows match one uniform run.
    for name in LABEL_FILES:
        expected = np.load(tmp_path / "all" / f"{name}.npy")
        assert np.array_equal(np.load(tmp_path / "mixed" / f"{name}.npy"), expected), name


def test_labelled_targets_keep_deep_scores_and_cap_quick_ones(
    game_corpus: Path, labels: Path
) -> None:
    plain = GameSplit(game_corpus / "train")
    labelled = GameSplit(game_corpus / "train", labels=labels)
    indices = np.arange(len(plain))
    chosen = labelled.trainable()
    assert len(chosen) == 5
    targets = labelled.targets(indices)
    baseline = plain.targets(indices)
    legal = plain.legal_mask(indices)
    for row in indices:
        deep = baseline["move_value_mask"][row]
        assert np.allclose(
            targets["move_value_target"][row][deep], baseline["move_value_target"][row][deep]
        )
        assert targets["policy_target"][row].sum() == pytest.approx(1.0, abs=1e-5)
        assert targets["policy_target"][row].argmax() == plain.action[row]
        if row in chosen:
            assert np.array_equal(targets["move_value_mask"][row], legal[row])
            worst = baseline["move_value_target"][row][deep].min()
            assert targets["move_value_target"][row][~deep & legal[row]].max() <= worst + 1e-6
        else:
            assert np.array_equal(targets["move_value_mask"][row], deep)
    assert targets["value"] == pytest.approx(baseline["value"])


def test_quick_scores_use_the_lichess_win_chance() -> None:
    board = chess.Board()
    scored = dict(score_all_moves(FakeEngine(), board, 100))  # type: ignore[arg-type]
    assert len(scored) == board.legal_moves.count()
    assert max(scored.values()) == pytest.approx(
        win_chance(max(m.to_square for m in board.legal_moves) * 4 - 100)
    )


def _deep_best(split: Path, index: int) -> tuple[chess.Board, chess.Move]:
    board = decode_position(
        np.load(split / "pieces.npy")[index], np.load(split / "state.npy")[index]
    )
    offsets = np.load(split / "candidate_offsets.npy")
    actions = np.load(split / "candidate_actions.npy")
    return board, action_to_move(board, int(actions[offsets[index]]))


def _assert_line(labels: Path, row: int, board: chess.Board, pv: list[chess.Move]) -> None:
    """Row ``row`` holds the first four plies of ``pv`` and the boards after plies 2 and 4."""

    length = np.load(labels / "line_length.npy")[row]
    froms = np.load(labels / "line_from.npy")[row]
    tos = np.load(labels / "line_to.npy")[row]
    boards = np.load(labels / "line_boards.npy")[row]
    assert length == min(len(pv), 4)
    # Black-to-move roots are rotated 180 degrees.
    frame = (lambda square: square) if board.turn == chess.WHITE else (lambda square: 63 - square)
    played = board.copy()
    for ply in range(4):
        if ply < length:
            assert (froms[ply], tos[ply]) == (
                frame(pv[ply].from_square),
                frame(pv[ply].to_square),
            )
            played.push(pv[ply])
            if ply % 2 == 1:
                assert played.turn == board.turn
                assert np.array_equal(boards[ply // 2], encode_position(played).pieces)
        else:
            assert (froms[ply], tos[ply]) == (255, 255)
            if ply % 2 == 1:
                assert np.all(boards[ply // 2] == 255)


def test_lines_follow_the_deep_best_move_in_the_roots_frame(
    game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: FakeEngine()
    )
    split = game_corpus / "train"
    size = len(np.load(split / "pieces.npy"))
    manifest = label_positions(
        split,
        tmp_path / "labels",
        count=size,
        nodes=100,
        engine_command=["stockfish"],
        workers=1,
        chunk_size=3,
        low_priority=False,
    )
    indices = np.load(tmp_path / "labels" / "label_indices.npy")
    assert len(indices) == size
    turns = set()
    for row, index in enumerate(indices):
        board, best = _deep_best(split, int(index))
        turns.add(board.turn)
        _assert_line(tmp_path / "labels", row, board, fake_line(board, best))
    assert turns == {chess.WHITE, chess.BLACK}
    assert manifest["line_lengths"] == {0: 0, 1: 0, 2: 0, 3: 0, 4: size}


def test_principal_lines_in_a_black_frame_and_cut_short() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    plies = ["e7e5", "g1f3", "b8c6", "f1b5"]
    line = principal_line(board, [chess.Move.from_uci(ply) for ply in plies])
    assert line.length == 4
    # e7 e5, g1 f3, b8 c6 and f1 b5, each rotated 180 degrees.
    assert line.from_squares.tolist() == [11, 57, 6, 58]
    assert line.to_squares.tolist() == [27, 42, 21, 30]
    # Black's own king on e8 and White's bishop on b5, in Black's frame.
    assert line.boards[1][3] == chess.KING
    assert line.boards[1][30] == chess.BISHOP + 6
    assert line.boards[0][30] == 0

    illegal = principal_line(chess.Board(), [chess.Move.from_uci("e2e4")] * 2)
    assert illegal.length == 1
    assert illegal.from_squares.tolist() == [12, 255, 255, 255]
    assert np.all(illegal.boards == 255)

    fools = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    mate = principal_line(fools, [chess.Move.from_uci("d8h4"), chess.Move.from_uci("e1f2")])
    assert mate.length == 1
    assert principal_line(fools, []).length == 0


def test_chunks_without_lines_finalize_with_empty_lines(labels: Path) -> None:
    before = {
        name: np.load(labels / f"{name}.npy")
        for name in ("label_indices", "line_length", "line_from", "line_to", "line_boards")
    }
    old = labels / "chunks" / "chunk-00001.npz"
    with np.load(old) as part:
        columns = {name: part[name] for name in ("indices", "counts", "actions", "win")}
    np.savez(old, **columns)
    manifest = finalize_labels(labels, nodes=100, engine="Fake", seed=0, requested=5)
    assert np.array_equal(np.load(labels / "label_indices.npy"), before["label_indices"])
    stripped = np.isin(before["label_indices"], columns["indices"])
    assert stripped.sum() == 2
    for name in ("line_length", "line_from", "line_to", "line_boards"):
        after = np.load(labels / f"{name}.npy")
        assert after.dtype == np.uint8
        assert after.shape == before[name].shape
        assert np.array_equal(after[~stripped], before[name][~stripped])
        assert np.all(after[stripped] == (0 if name == "line_length" else 255))
    assert manifest["line_lengths"] == {0: 2, 1: 0, 2: 0, 3: 0, 4: 3}


def test_training_draws_only_labelled_positions(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    config = PuzzleTrainingConfig(
        corpus=str(game_corpus),
        output=str(tmp_path / "run"),
        task="games",
        labels=str(labels),
        move_value_weight=0.5,
        value_weight=0.25,
        steps=2,
        batch_size=2,
        warmup_steps=1,
        max_depth=2,
        max_gradient_steps=1,
        eval_every=2,
        eval_positions=3,
        eval_depths=(0, 1),
        device="cpu",
    )
    report = train_puzzles(config, TINY)
    assert report["train_positions"] == 5
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(row["train_loss"])


class CrashingEngine(FakeEngine):
    """Dies on its first analysis, as Stockfish does on some boards."""

    crashes = 1

    def analyse(
        self, board: chess.Board, limit: Any, multipv: int = 1, info: Any = None
    ) -> list[dict]:
        if CrashingEngine.crashes:
            CrashingEngine.crashes -= 1
            raise chess.engine.EngineTerminatedError("engine process died unexpectedly")
        return super().analyse(board, limit, multipv, info)

    def quit(self) -> None:
        raise chess.engine.EngineTerminatedError("engine event loop dead")


def test_crashed_engines_are_replaced_and_impossible_boards_skipped(
    game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: CrashingEngine()
    )
    impossible = chess.Board("4k3/8/PPPPPPPP/BBBQQNNN/RR1RR1RR/3QQ2K/8/8 w - - 0 1")
    real_decode = move_labels.decode_position
    first = int(np.sort(np.random.default_rng(0).permutation(8)[:4])[0])

    def decode(pieces: np.ndarray, state: np.ndarray) -> chess.Board:
        board = real_decode(pieces, state)
        return impossible if np.array_equal(pieces, split_pieces[first]) else board

    split_pieces = np.load(game_corpus / "train" / "pieces.npy")
    monkeypatch.setattr(move_labels, "decode_position", decode)
    CrashingEngine.crashes = 1
    manifest = label_positions(
        game_corpus / "train",
        tmp_path / "labels",
        count=4,
        nodes=100,
        engine_command=["stockfish"],
        workers=1,
        chunk_size=4,
        low_priority=False,
    )
    assert manifest["positions"] == 3
    assert manifest["skipped"] == 1
    assert first not in np.load(tmp_path / "labels" / "label_indices.npy")


FAKE_UCI = Path(__file__).parent / "fixtures" / "fake_uci.py"


def test_worker_processes_label_with_a_real_uci_engine_and_survive_a_crash(
    game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "crash-once"
    marker.touch()
    monkeypatch.setenv("FAKE_UCI_CRASH_ONCE", str(marker))
    manifest = label_positions(
        game_corpus / "train",
        tmp_path / "labels",
        count=5,
        nodes=100,
        engine_command=[sys.executable, str(FAKE_UCI)],
        workers=2,
        chunk_size=3,
        low_priority=False,
    )
    assert not marker.exists()
    assert manifest["positions"] == 5
    assert manifest["engine"] == "FakeFish"
    indices = np.load(tmp_path / "labels" / "label_indices.npy")
    offsets = np.load(tmp_path / "labels" / "label_offsets.npy")
    actions = np.load(tmp_path / "labels" / "label_actions.npy")
    masks = GameSplit(game_corpus / "train").legal_mask(indices)
    for row in range(len(indices)):
        scored = np.zeros(4672, dtype=bool)
        scored[actions[offsets[row] : offsets[row + 1]]] = True
        assert np.array_equal(scored, masks[row])
    fake_uci = _load_fake_uci()
    for row, index in enumerate(indices):
        board, best = _deep_best(game_corpus / "train", int(index))
        _assert_line(tmp_path / "labels", row, board, fake_uci.principal_variation(board, best))
    lengths = manifest["line_lengths"]
    assert isinstance(lengths, dict) and lengths[3] > 0


def _load_fake_uci() -> Any:
    spec = importlib.util.spec_from_file_location("fake_uci", FAKE_UCI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("with_labels", [False, True])
def test_densified_sparse_batches_match_cpu_targets(
    game_corpus: Path, labels: Path, with_labels: bool
) -> None:
    split = GameSplit(game_corpus / "train", labels=labels if with_labels else None)
    indices = np.arange(len(split))
    dense = split.batch(indices)
    built = densify(split.sparse_batch(indices), split.temperature, torch.device("cpu"))
    assert set(built) == set(dense)
    for key, expected in dense.items():
        if expected.dtype == torch.bool or not expected.is_floating_point():
            assert torch.equal(built[key], expected), key
        else:
            assert torch.allclose(built[key], expected, atol=1e-6), key


def test_joined_splits_serve_each_part_in_the_given_row_order(
    game_corpus: Path, labels: Path
) -> None:
    labelled = GameSplit(game_corpus / "train", labels=labels)
    plain = GameSplit(game_corpus / "train")
    joined = JoinedGameSplit([labelled, plain])
    size = len(labelled)
    assert len(joined) == 2 * size
    assert np.array_equal(
        joined.trainable(), np.concatenate([labelled.label_indices, np.arange(size) + size])
    )
    indices = np.array([size + 1, 0, 2 * size - 1, 3, size, 1])
    built = densify(joined.sparse_batch(indices), 4.0, torch.device("cpu"))
    parts = [
        labelled.batch(np.array([index])) if index < size else plain.batch(np.array([index - size]))
        for index in indices
    ]
    for key, value in built.items():
        expected = torch.cat([part[key] for part in parts])
        if value.is_floating_point():
            assert torch.allclose(value, expected, atol=1e-6, equal_nan=True), key
        else:
            assert torch.equal(value, expected), key
    batches = joined.shuffled_batches(4, np.random.default_rng(0))
    assert len(next(batches)["action"]) == 4
    with pytest.raises(ValueError, match="sparse"):
        joined.shuffled_batches(4, np.random.default_rng(0), sparse=False)


def test_train_games_cli_joins_extra_labelled_corpora(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)
    output = tmp_path / "joined"
    arguments = ["train-games", "--corpus", str(game_corpus), "--labels", str(labels)]
    arguments += ["--extra-train", str(game_corpus), str(labels)]
    arguments += ["--output", str(output), "--device", "cpu", "--steps", "2", "--batch-size", "2"]
    arguments += ["--warmup-steps", "1", "--max-depth", "2", "--max-gradient-steps", "1"]
    arguments += ["--eval-every", "2", "--eval-depths", "0,1", "--d-model", "24", "--heads", "6"]
    arguments += ["--mlp-width", "48", "--latent-tokens", "4", "--recurrent-blocks", "1"]
    assert main(arguments) == 0
    run = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert run["train_positions"] == 10
    assert run["training_config"]["extra_train"] == [[str(game_corpus), str(labels)]]
    row = json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(row["train_loss"])
    assert build_parser().parse_args(["train-games", "--output", "run"]).extra_train is None


def test_lookahead_targets_come_from_the_stored_lines(game_corpus: Path, labels: Path) -> None:
    split = GameSplit(game_corpus / "train", labels=labels)
    indices = np.arange(len(split))
    lookahead = split.lookahead_targets(indices)
    lines = {name: np.load(labels / f"{name}.npy") for name in ("line_from", "line_to")}
    boards = np.load(labels / "line_boards.npy")
    slots = {int(index): slot for slot, index in enumerate(split.label_indices)}

    def expected(values: np.ndarray) -> np.ndarray:
        return np.where(values == NO_LINE, IGNORE, values.astype(np.int64))

    for row in range(len(split)):
        if row not in slots:
            assert (lookahead["reply_from"][row] == IGNORE).all()
            assert (lookahead["board4"][row] == IGNORE).all()
            continue
        slot = slots[row]
        assert lookahead["reply_from"][row] == expected(lines["line_from"][slot, 1])
        assert lookahead["reply_to"][row] == expected(lines["line_to"][slot, 1])
        assert np.array_equal(lookahead["board2"][row], expected(boards[slot, 0]))
        assert np.array_equal(lookahead["board4"][row], expected(boards[slot, 1]))
    assert (lookahead["board4"] != IGNORE).any()
    dense = densify(split.sparse_batch(indices), 4.0, torch.device("cpu"))
    batch = split.batch(indices)
    for key in lookahead:
        assert torch.equal(dense[key], batch[key])


def test_lookahead_training_averages_weights_and_resumes(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    model_config = dataclasses.replace(TINY, lookahead_heads=True, square_attention_bias=True)
    settings: dict[str, Any] = {
        "corpus": str(game_corpus),
        "output": str(tmp_path / "run"),
        "task": "games",
        "labels": str(labels),
        "move_value_weight": 0.5,
        "value_weight": 0.25,
        "lookahead_weight": 0.25,
        "supervise_every_step": True,
        "ema_decay": 0.5,
        "steps": 2,
        "batch_size": 2,
        "warmup_steps": 1,
        "max_depth": 3,
        "max_gradient_steps": 2,
        "eval_every": 2,
        "eval_positions": 3,
        "eval_depths": (0, 1),
        "device": "cpu",
    }
    train_puzzles(PuzzleTrainingConfig(**settings), model_config)
    checkpoint = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    averaged, raw = checkpoint["model"], checkpoint["raw_model"]
    assert not torch.equal(averaged["reply_from_head.weight"], raw["reply_from_head.weight"])
    assert not torch.equal(
        averaged["recurrent_core.0.square_bias"], raw["recurrent_core.0.square_bias"]
    )
    settings["steps"] = 4
    train_puzzles(PuzzleTrainingConfig(**settings), model_config, resume=True)
    rows = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(row)["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(json.loads(row)["train_loss"]) for row in rows)


def test_train_games_cli_trains_the_pair_reply_readout_and_resumes(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)
    output = tmp_path / "pair"
    arguments = ["train-games", "--corpus", str(game_corpus), "--labels", str(labels)]
    arguments += ["--output", str(output), "--device", "cpu", "--steps", "4", "--batch-size", "2"]
    arguments += ["--warmup-steps", "1", "--max-depth", "3", "--max-gradient-steps", "2"]
    arguments += ["--eval-every", "2", "--eval-depths", "0,1", "--snapshot-every", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--latent-tokens", "4"]
    arguments += ["--recurrent-blocks", "1", "--ema-decay", "0.5", "--move-readout", "from_to"]
    arguments += ["--reply-readout", "pair", "--reply-weight", "0.25"]
    assert main(arguments) == 0
    run = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert run["model_config"]["reply_readout"] == "pair"
    assert run["model_config"]["lookahead_heads"] is True
    assert run["training_config"]["reply_weight"] == 0.25
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert "reply_from_head.weight" not in checkpoint["model"]
    key = "reply_query.weight"
    assert not torch.equal(checkpoint["model"][key], checkpoint["raw_model"][key])
    model = load_checkpoint(output / "checkpoint.pt", torch.device("cpu"))
    assert model.config.reply_readout == "pair"
    load_checkpoint(output / "snapshot-step-000004.pt", torch.device("cpu"))
    arguments[arguments.index("--steps") + 1] = "6"
    assert main([*arguments, "--resume"]) == 0
    rows = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 4, 6]
    assert all(np.isfinite(row["train_loss"]) for row in rows)


def test_reply_flags_default_to_the_square_heads(
    game_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    for command in ("train-games", "train-puzzles"):
        args = build_parser().parse_args([command, "--output", "run"])
        assert (args.reply_readout, args.reply_weight) == ("squares", 0.0)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-puzzles", "--output", "run", "--reply-weight", "1"])
    config = PuzzleTrainingConfig(
        corpus=str(game_corpus),
        output=str(tmp_path / "run"),
        task="games",
        reply_weight=0.25,
        device="cpu",
    )
    squares = dataclasses.replace(TINY, lookahead_heads=True)
    with pytest.raises(ValueError, match="pair reply readout"):
        train_puzzles(config, squares)


def _child_board(sparse: dict[str, np.ndarray], row: int, action: int) -> chess.Board:
    board = decode_position(sparse["pieces"][row], sparse["state"][row])
    move = action_to_move(board, action)
    assert move in board.legal_moves
    board.push(move)
    return board


def test_children_follow_the_rules_and_take_one_minus_the_moves_win_chance(
    game_corpus: Path, labels: Path
) -> None:
    split = GameSplit(game_corpus / "train", labels=labels)
    indices = split.label_indices
    sparse = split.sparse_batch(indices)
    parents = split.batch(indices)
    chosen: set[tuple[int, int]] = set()
    for seed in range(8):
        children = with_children(sparse, 1.0, np.random.default_rng(seed))
        dense = densify(children, split.temperature, torch.device("cpu"))
        assert np.array_equal(children["child_rows"], np.arange(len(indices)))
        for row, action in enumerate(children["child_actions"]):
            chosen.add((row, int(action)))
            assert parents["move_value_mask"][row, action]
            board = _child_board(sparse, row, int(action))
            encoded = encode_position(board)
            assert np.array_equal(children["pieces"][row], encoded.pieces)
            assert np.array_equal(children["state"][row], state_vector(encoded))
            expected = -parents["move_value_target"][row, action].item()
            if board.is_checkmate():
                expected = -1.0
            elif board.is_stalemate():
                expected = 0.0
            assert dense["value"][row].item() == pytest.approx(expected, abs=1e-6)
            assert np.array_equal(dense["legal_mask"][row].numpy(), legal_action_mask(board))
        assert not dense["policy_target"].any()
        assert not dense["move_value_mask"].any()
        assert (dense["action"] == -1).all()
        for key in ("reply_from", "reply_to", "board2", "board4"):
            assert (dense[key] == IGNORE).all(), key
        assert (dense["line_actions"] == -1).all()
    # Moves are drawn among all labelled moves, not only the best.
    assert len(chosen) > 2 * len(indices)


def _scored_batch(moves: list[tuple[str, str, float]]) -> dict[str, np.ndarray]:
    """A sparse batch whose rows each score one move, given as (FEN, UCI move, win chance)."""

    boards = [chess.Board(fen) for fen, _, _ in moves]
    encoded = [encode_position(board) for board in boards]
    legal = [legal_actions(board) for board in boards]
    size = len(boards)
    actions = np.array(
        [move_to_action(board, chess.Move.from_uci(move[1])) for board, move in zip(boards, moves)]
    )
    nothing = np.zeros(0, dtype=np.int64)
    return {
        "reply_from": np.full(size, 7, dtype=np.int64),
        "reply_to": np.full(size, 9, dtype=np.int64),
        "board2": np.zeros((size, 64), dtype=np.int64),
        "board4": np.zeros((size, 64), dtype=np.int64),
        "line_actions": np.zeros((size, LINE_PLIES), dtype=np.int64),
        "pieces": np.stack([item.pieces for item in encoded]).astype(np.int64),
        "state": np.stack([state_vector(item) for item in encoded]),
        "action": actions.astype(np.int64),
        "legal_rows": np.repeat(np.arange(size), [len(item) for item in legal]),
        "legal_actions": np.concatenate(legal).astype(np.int64),
        "deep_rows": np.arange(size),
        "deep_actions": actions.astype(np.int64),
        "deep_win": np.array([win for _, _, win in moves], dtype=np.float32),
        "quick_rows": nothing,
        "quick_actions": nothing,
        "quick_win": np.zeros(0, dtype=np.float32),
    }


def test_children_of_special_moves_and_finished_games_match_python_chess() -> None:
    castles = "r3k2r/pppq1ppp/2n2n2/3p4/3Pp3/2N2N2/PPPQPPPP/R3K2R w KQkq - 0 1"
    black = "r3k2r/pppq1ppp/2n2n2/3p4/3PpP2/2N2N2/PPPQP1PP/R3K2R b KQkq f3 0 1"
    promotions = "r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R w KQkq - 0 1"
    moves = [
        (castles, "e1g1", 0.6),
        (castles, "e1c1", 0.55),
        (black, "e8g8", 0.5),
        (black, "e4f3", 0.45),
        (promotions, "b7a8q", 0.9),
        (promotions, "g7g8n", 0.3),
        (promotions, "a1a8", 0.7),
        (chess.STARTING_FEN, "e2e4", 0.52),
        # Mate and stalemate in one, whose scores the finished game overrules.
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", 0.8),
        ("7k/8/5Q2/8/8/8/8/6K1 w - - 0 1", "f6g6", 0.99),
    ]
    children = with_children(_scored_batch(moves), 1.0, np.random.default_rng(0))
    dense = densify(children, 4.0, torch.device("cpu"))
    finished = []
    for row, (fen, uci, win) in enumerate(moves):
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(uci))
        encoded = encode_position(board)
        assert np.array_equal(dense["pieces"][row].numpy(), encoded.pieces), uci
        assert np.array_equal(dense["state"][row].numpy(), state_vector(encoded)), uci
        expected = -1.0 if board.is_checkmate() else 0.0 if board.is_stalemate() else 1 - 2 * win
        assert dense["value"][row].item() == pytest.approx(expected, abs=1e-6), uci
        finished.append(board.is_checkmate() or board.is_stalemate())
    assert finished[-2:] == [True, True] and not any(finished[:-2])
    # A finished game allows every action, so its unused policy stays finite.
    assert dense["legal_mask"][-2:].all()
    assert torch.isfinite(dense["policy_target"]).all()


def test_a_zero_child_share_leaves_batches_and_draws_untouched(
    game_corpus: Path, labels: Path
) -> None:
    split = GameSplit(game_corpus / "train", labels=labels)
    sparse = split.sparse_batch(split.label_indices)
    generator = np.random.default_rng(0)
    before = generator.bit_generator.state
    assert with_children(sparse, 0.0, generator) is sparse
    # A share too small for one row also changes nothing.
    assert with_children(sparse, 0.05, generator) is sparse
    assert generator.bit_generator.state == before


@pytest.mark.parametrize("bins", [0, 8])
def test_child_rows_add_only_a_value_loss(game_corpus: Path, labels: Path, bins: int) -> None:
    split = GameSplit(game_corpus / "train", labels=labels)
    indices = split.label_indices
    children = with_children(split.sparse_batch(indices), 0.4, np.random.default_rng(3))
    rows = children["child_rows"]
    assert len(rows) == 2
    kept = np.setdiff1d(np.arange(len(indices)), rows)
    mixed = densify(children, 4.0, torch.device("cpu"))
    parents = densify(split.sparse_batch(indices[kept]), 4.0, torch.device("cpu"))
    torch.manual_seed(0)
    model = WombatNetwork(dataclasses.replace(TINY, lookahead_heads=True, value_bins=bins))
    cpu = torch.device("cpu")
    no_value = LossWeights(move_value=0.5, lookahead=0.25)
    loss, correct = puzzle_training._loss(model, mixed, 0, 2, cpu, no_value)
    expected, expected_correct = puzzle_training._loss(model, parents, 0, 2, cpu, no_value)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert correct.item() == pytest.approx(expected_correct.item())
    valued = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    loss, _ = puzzle_training._loss(model, mixed, 0, 2, cpu, valued)
    expected, _ = puzzle_training._loss(model, parents, 0, 2, cpu, valued)
    assert loss.item() != pytest.approx(expected.item(), rel=1e-5)


def test_child_rows_get_no_policy_or_move_value_gradient(game_corpus: Path, labels: Path) -> None:
    split = GameSplit(game_corpus / "train", labels=labels)
    sparse = split.sparse_batch(split.label_indices)
    children = with_children(sparse, 0.4, np.random.default_rng(3))
    rows = torch.from_numpy(children["child_rows"])
    targets = complete_targets(densify(children, 4.0, torch.device("cpu")))
    torch.manual_seed(0)
    policy = torch.randn(5, 4672, requires_grad=True)
    value = torch.zeros(5, requires_grad=True)
    move_values = torch.zeros(5, 4672, requires_grad=True)
    loss, _ = graded_loss(policy, value, move_values, targets, LossWeights(0.5, 0.25))
    loss.backward()
    assert policy.grad is not None and move_values.grad is not None and value.grad is not None
    assert not policy.grad[rows].any()
    assert not move_values.grad[rows].any()
    assert value.grad[rows].all()
    others = torch.ones(5, dtype=torch.bool)
    others[rows] = False
    assert policy.grad[others].any(dim=1).all()


def test_train_games_cli_trains_on_children_and_starts_from_a_checkpoint(
    game_corpus: Path, labels: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)
    defaults = build_parser().parse_args(["train-games", "--output", "run"])
    assert (defaults.child_value_share, defaults.init_from) == (0.0, None)
    first = tmp_path / "first"
    arguments = ["train-games", "--corpus", str(game_corpus), "--labels", str(labels)]
    arguments += ["--extra-train", str(game_corpus), str(labels), "--child-value-share", "0.5"]
    arguments += ["--device", "cpu", "--steps", "4", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--max-depth", "3", "--max-gradient-steps", "2", "--eval-every", "2"]
    arguments += ["--eval-depths", "0,1", "--d-model", "24", "--heads", "6", "--mlp-width", "48"]
    arguments += ["--latent-tokens", "4", "--recurrent-blocks", "1", "--value-bins", "8"]
    assert main([*arguments, "--output", str(first)]) == 0
    run = json.loads((first / "config.json").read_text(encoding="utf-8"))
    assert run["training_config"]["child_value_share"] == 0.5
    rows = [json.loads(row) for row in (first / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)

    second = tmp_path / "second"
    start = first / "checkpoint.pt"
    tuned = [*arguments, "--output", str(second), "--init-from", str(start)]
    tuned[tuned.index("--steps") + 1] = "2"
    assert main(tuned) == 0
    rows = [json.loads(row) for row in (second / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2]
    state = torch.load(second / "checkpoint.pt", weights_only=False)
    assert state["step"] == 2
    assert state["training_config"]["init_from"] == str(start)
    # The optimizer starts fresh: it has taken at most this run's two steps.
    assert max(int(item["step"]) for item in state["optimizer"]["state"].values()) == 2
    # Resuming the fine-tuned run continues it instead of starting again.
    tuned[tuned.index("--steps") + 1] = "4"
    assert main([*tuned, "--resume"]) == 0
    rows = [json.loads(row) for row in (second / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 4]


def test_starting_from_a_checkpoint_loads_both_weights_and_checks_the_config(
    game_corpus: Path, labels: Path, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    model, average = WombatNetwork(TINY), WombatNetwork(TINY)
    path = tmp_path / "start.pt"
    saved = {
        "model": average.state_dict(),
        "raw_model": model.state_dict(),
        "model_config": dataclasses.asdict(TINY),
    }
    torch.save(saved, path)
    fresh, fresh_average = WombatNetwork(TINY), WombatNetwork(TINY)
    puzzle_training._start_from(path, fresh, fresh_average, TINY)
    for name, value in fresh.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
        assert torch.equal(fresh_average.state_dict()[name], average.state_dict()[name]), name
    other = dataclasses.replace(TINY, mlp_width=96)
    with pytest.raises(ValueError, match="mlp_width"):
        puzzle_training._start_from(path, WombatNetwork(other), None, other)
    config = PuzzleTrainingConfig(
        corpus=str(game_corpus), output=str(tmp_path / "run"), task="games", device="cpu"
    )
    with pytest.raises(ValueError, match="labelled game positions"):
        train_puzzles(dataclasses.replace(config, child_value_share=0.5), TINY)
    labelled = dataclasses.replace(config, labels=str(labels), child_value_share=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        train_puzzles(labelled, TINY)
