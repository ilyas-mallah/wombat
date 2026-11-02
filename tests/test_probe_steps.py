from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import move_to_action
from wombat.cli import main
from wombat.data.lichess import PuzzleRecord
from wombat.data.move_labels import label_positions
from wombat.data.puzzle_corpus import PuzzleSplit
from wombat.interpretability.labels import move_squares
from wombat.interpretability.probes import (
    auroc,
    cross_fit_binary,
    cross_fit_square,
    folds,
    prior_accuracy,
    square_accuracy,
)
from wombat.interpretability.steps import (
    BINARY_TARGETS,
    SQUARE_TARGETS,
    ProbeScores,
    Sample,
    action_squares,
    blunder_attribution,
    board_features_ablation,
    board_features_off,
    causal_tests,
    collect_steps,
    freeze_mask,
    game_sample,
    improvement,
    king_distance,
    move_losses,
    probe_sample,
    puzzle_sample,
    run_frozen,
    square_mask,
    swap_tokens,
)
from wombat.model import NetworkConfig, WombatNetwork

CPU = torch.device("cpu")
# The long-run recipe: local attention, no prelude or latents, from_to readout, board features.
LOCAL = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=0,
    recurrent_blocks=1,
    latent_tokens=0,
    max_residual_update=None,
    policy_readout="square",
    move_readout="from_to",
    normalize_state=True,
    square_attention_bias=True,
    local_attention=True,
    board_features=True,
    board_features_long_range=True,
    lookahead_heads=True,
)
# A global readout with a prelude, latent tokens and no board features.
GLOBAL = replace(
    LOCAL,
    prelude_blocks=1,
    latent_tokens=4,
    policy_readout="global",
    move_readout="square",
    square_attention_bias=False,
    local_attention=False,
    board_features=False,
    board_features_long_range=False,
    lookahead_heads=False,
)


def _model(config: NetworkConfig, seed: int = 0) -> WombatNetwork:
    torch.manual_seed(seed)
    model = WombatNetwork(config).eval()
    # Nonzero recurrent updates, so every step changes the state.
    for name, parameter in model.recurrent_core.named_parameters():
        if name.endswith(("attention.out_proj.weight", "mlp.output.weight")):
            torch.nn.init.normal_(parameter, std=0.2)
    return model


class _FakeEngine:
    """Scores each legal move by its destination and continues with first legal moves."""

    def __init__(self) -> None:
        self.id = {"name": "Fake"}

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(
        self, board: chess.Board, limit: object, multipv: int = 1, info: object = None
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for move in list(board.legal_moves)[:multipv]:
            played = board.copy(stack=False)
            line = [move]
            played.push(move)
            while len(line) < 4 and (replies := list(played.legal_moves)):
                line.append(replies[0])
                played.push(replies[0])
            score = chess.engine.PovScore(chess.engine.Cp(move.to_square * 4 - 100), board.turn)
            results.append({"pv": line, "score": score})
        return results

    def quit(self) -> None:
        pass


def _checkpoint(path: Path, config: NetworkConfig) -> Path:
    model = _model(config)
    torch.save({"model_config": asdict(config), "model": model.state_dict()}, path)
    return path


def _known(value: float | None) -> float:
    assert value is not None
    return value


def test_auroc_matches_pairwise_counting_with_ties() -> None:
    generator = np.random.default_rng(0)
    scores = generator.integers(0, 5, 60).astype(np.float64)
    labels = generator.random(60) < 0.4
    pairs = [
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive, negative in product(scores[labels], scores[~labels])
    ]
    assert auroc(scores, labels) == pytest.approx(np.mean(pairs))
    assert auroc(scores, np.zeros(60, dtype=bool)) is None


def test_folds_keep_groups_together() -> None:
    groups = np.repeat(np.arange(20), 3)
    fold = folds(groups, seed=1)
    assert set(fold) == {0, 1}
    assert all(len(set(fold[groups == group])) == 1 for group in range(20))
    assert np.array_equal(fold, folds(groups, seed=1))


def test_probes_decode_separable_targets_and_not_shuffled_ones() -> None:
    generator = np.random.default_rng(0)
    tokens = generator.normal(size=(80, 64, 8)).astype(np.float16)
    fold = folds(np.arange(80))
    mask = generator.random((80, 64)) < 0.5
    positions, squares = np.nonzero(mask)
    labels = tokens[positions, squares, 0] + 0.1 * tokens[positions, squares, 1] > 0
    scores = cross_fit_binary(tokens, positions, squares, labels, fold, CPU)
    assert not np.isnan(scores).any()
    assert _known(auroc(scores, labels)) > 0.97
    shuffled = cross_fit_binary(
        tokens, positions, squares, labels, fold, CPU, shuffle=np.random.default_rng(1)
    )
    assert abs(_known(auroc(shuffled, labels)) - 0.5) < 0.1

    targets = generator.integers(0, 64, 80)
    targets[:5] = -1
    tokens[np.arange(80), targets.clip(0), 2] += 10
    logits = cross_fit_square(tokens, targets, fold, CPU)
    assert square_accuracy(logits, targets) == 1.0
    shuffled_logits = cross_fit_square(tokens, targets, fold, CPU, shuffle=np.random.default_rng(1))
    assert _known(square_accuracy(shuffled_logits, targets)) < 0.3
    assert _known(prior_accuracy(targets, fold)) < 0.2


def test_action_squares_and_king_distance() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    for uci in ("e1g1", "a1a8", "h1h5"):
        move = chess.Move.from_uci(uci)
        origin, destination = action_squares(np.array([move_to_action(board, move)]))
        assert (origin[0], destination[0]) == move_squares(board, move)
    distance = king_distance(np.array(chess.E4))
    assert distance[chess.E4] == 0 and distance[chess.F5] == 1 and distance[chess.A8] == 4


def _records_by_id(records: list[PuzzleRecord]) -> dict[str, PuzzleRecord]:
    return {record.puzzle_id: record for record in records}


def test_puzzle_sample_takes_replies_from_the_line(
    puzzle_corpus: Path, puzzle_archive: tuple[Path, list[PuzzleRecord]]
) -> None:
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    split = PuzzleSplit(puzzle_corpus / "validation")
    records = _records_by_id(puzzle_archive[1])
    assert len(sample) == len(split) and sample.puzzles
    for row, index in enumerate(sample.indices):
        record = records[str(split.puzzle_ids[split.puzzle[index]])]
        line = int(split.line_index[index])
        board = sample.boards[row]
        after = 2 * line + 2
        if after < len(record.moves):
            expected = move_squares(board, chess.Move.from_uci(record.moves[after]))
            assert tuple(sample.reply[row]) == expected
        else:
            assert tuple(sample.reply[row]) == (-1, -1)
    assert (sample.reply[:, 0] >= 0).any()


def test_game_samples_decode_boards_without_fens(game_corpus: Path) -> None:
    sample = game_sample(game_corpus / "train", 5)
    assert len(sample) == 5 and not sample.puzzles
    sample.label()
    assert set(sample.binary) == set(BINARY_TARGETS) and set(sample.squares) == set(SQUARE_TARGETS)
    origin = sample.squares["best_move_origin"]
    destination = sample.squares["best_move_destination"]
    for row, board in enumerate(sample.boards):
        assert sample.legal[row].sum() == board.legal_moves.count()
        assert sample.binary["destination_loses_material"][1][row, destination[row]]
        assert 1 <= sample.pieces[row, origin[row]] <= 6
    assert sample.wins is not None and not np.isnan(sample.wins[np.arange(5), sample.target]).any()


@pytest.mark.parametrize("config", [LOCAL, GLOBAL], ids=["local", "global"])
def test_collected_steps_match_the_model(puzzle_corpus: Path, config: NetworkConfig) -> None:
    model = _model(config)
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    steps = collect_steps(model, sample, (1, 3), CPU, batch_size=4)
    assert set(steps.tokens) == {"input", "t0", "t1", "t3"}
    pieces, state = torch.from_numpy(sample.pieces), torch.from_numpy(sample.state)
    legal = torch.from_numpy(sample.legal)
    with torch.inference_mode():
        traces = model.trace_reasoning(pieces, state, 3, capture_attention=False)
        for depth in (0, 1, 3):
            expected = traces[depth].policy_logits.masked_fill(~legal, float("-inf")).argmax(1)
            assert np.array_equal(steps.actions[depth], expected.numpy())
            hidden = traces[depth].hidden[:, :64].numpy()
            assert np.allclose(steps.tokens[f"t{depth}"], hidden, atol=2e-2, rtol=1e-2)
    assert (steps.tokens["t1"] != steps.tokens["t3"]).any()
    assert set(steps.replies) == ({0, 1, 3} if config.lookahead_heads else set())


def test_board_features_off_zeroes_only_the_feature_projection() -> None:
    model = _model(LOCAL)
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3")
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
    state = torch.from_numpy(state_vector(encoded))[None]
    with torch.inference_mode():
        normal = model._tokenize(pieces, state)
        with board_features_off(model):
            plain = model._tokenize(pieces, state)
        again = model._tokenize(pieces, state)
    expected = model.piece_embedding(pieces) + model.square_embedding.weight[None]
    assert torch.allclose(plain[:, :64], expected.detach())
    assert torch.equal(plain[:, 64], normal[:, 64]) and torch.equal(again, normal)
    assert not torch.allclose(plain[:, :64], normal[:, :64])


def test_probes_and_improvement_report_every_target(puzzle_corpus: Path) -> None:
    model = _model(LOCAL)
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    sample.label()
    steps = collect_steps(model, sample, (1, 2, 4), CPU, batch_size=8)
    report, scores = probe_sample(sample, steps, CPU, late="t4")
    assert set(report) == set(BINARY_TARGETS) | set(SQUARE_TARGETS)
    for name in BINARY_TARGETS:
        entry = report[name]
        assert isinstance(entry, dict)
        assert set(entry["by_representation"]) == {"input", "t0", "t1", "t2", "t4"}
        assert "shuffled_labels_t4" in entry and "by_remaining" in entry
    origin = report["best_move_origin"]
    assert isinstance(origin, dict) and set(origin["model_policy"]) == {"t0", "t1", "t2", "t4"}
    reply = report["reply_origin"]
    assert isinstance(reply, dict) and "model_reply_head" in reply

    result = improvement(sample, steps, scores, 1, 4)
    sets = ("gained", "lost", "both_right", "both_wrong")
    assert sum(result[name]["positions"] for name in sets) == len(sample)  # type: ignore[index]
    for name in sets:
        entry = result[name]
        assert isinstance(entry, dict)
        if entry["positions"]:
            assert set(entry["probes"]) == set(BINARY_TARGETS) | set(SQUARE_TARGETS)
            assert set(entry["relative_change"]) == {"origin", "destination", "opponent_king"}


def _states(
    model: WombatNetwork, sample: Sample, early: int, late: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        memory = model.encode_board(torch.from_numpy(sample.pieces), torch.from_numpy(sample.state))
        hidden = model.initial_hidden(memory)
        states = [hidden]
        for _ in range(late):
            hidden = model.recurrent_step(memory, hidden)  # type: ignore[assignment]
            states.append(hidden)
    return memory, states[early], states[late]


def test_swapping_every_square_reproduces_the_later_readout(puzzle_corpus: Path) -> None:
    model = _model(LOCAL)
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    _, early, late = _states(model, sample, 1, 3)
    every = square_mask(np.tile(np.arange(64), (len(sample), 1)), model.hidden_tokens)
    with torch.inference_mode():
        swapped = model.heads(swap_tokens(early, late, every))[0]
        assert torch.equal(swapped, model.heads(late)[0])
        assert not torch.equal(model.heads(early)[0], model.heads(late)[0])
    none = square_mask(np.full((len(sample), 2), -1), model.hidden_tokens)
    assert not none.any() and torch.equal(swap_tokens(early, late, none), early)


def test_freezing_beyond_eight_king_steps_freezes_nothing(puzzle_corpus: Path) -> None:
    model = _model(GLOBAL)
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    memory, early, late = _states(model, sample, 1, 3)
    origin, destination = action_squares(sample.target)
    tokens = model.hidden_tokens
    with torch.inference_mode():
        nothing = freeze_mask(origin, destination, 8, tokens)
        assert not nothing.any()
        assert torch.equal(run_frozen(model, memory, early, 2, nothing), late)
        frozen = freeze_mask(origin, destination, 1, tokens)
        held = run_frozen(model, memory, early, 2, frozen)
    assert not frozen[:, 64:].any()
    distance = np.minimum(king_distance(origin), king_distance(destination))
    assert np.array_equal(frozen[:, :64].numpy(), distance > 1)
    inside = freeze_mask(origin, destination, 1, tokens, inside=True)
    assert torch.equal(inside[:, :64], ~frozen[:, :64])
    kept = frozen[..., None].expand_as(held)
    assert torch.equal(held[kept], early[kept]) and not torch.equal(held[~kept], early[~kept])


def test_causal_tests_report_swaps_and_freezes(puzzle_corpus: Path) -> None:
    model = _model(LOCAL)
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    result = causal_tests(model, sample, 1, 3, CPU, batch_size=4, radii=(1, 8))
    swap, freeze = result["swap"], result["freeze"]
    assert isinstance(swap, dict) and isinstance(freeze, dict)
    assert set(swap) == {"move_squares", "other_squares", "all_squares"}
    assert swap["all_squares"]["agrees_with_late_move"] == 1.0
    assert set(freeze) == {"outside_1", "inside_1", "outside_8", "inside_8"}
    assert freeze["outside_8"]["agrees_with_late_move"] == 1.0
    assert freeze["outside_8"]["frozen_squares"] == 0.0
    assert freeze["outside_8"]["accuracy"] == result["accuracy_late"]
    assert freeze["inside_8"]["frozen_squares"] == 64.0


def test_board_features_ablation(puzzle_corpus: Path) -> None:
    sample = puzzle_sample(puzzle_corpus / "validation", 100)
    model = _model(LOCAL)
    # Large feature weights, so removing the features changes moves.
    with torch.no_grad():
        model.board_feature_projection.weight.mul_(20)
    normal = collect_steps(model, sample, (1, 2), CPU, 8, keep_tokens=False)
    assert not normal.tokens
    result = board_features_ablation(model, sample, normal.actions, CPU, 8)
    assert result is not None and set(result) == {"t0", "t1", "t2"}
    assert 0 <= result["t2"]["same_move"] < 1  # type: ignore[index]
    assert board_features_ablation(_model(GLOBAL), sample, normal.actions, CPU, 8) is None


def test_move_losses_and_blunder_attribution(game_corpus: Path) -> None:
    sample = game_sample(game_corpus / "train", 8)
    sample.label()
    assert sample.wins is not None
    lost, exact = move_losses(sample, sample.target)
    assert exact.all() and np.allclose(lost, 0)
    # A legal move outside the scored candidates is bounded by the worst candidate.
    outside = np.array(
        [
            int(np.flatnonzero(row & np.isnan(wins))[0])
            for row, wins in zip(sample.legal, sample.wins)
        ]
    )
    lost, exact = move_losses(sample, outside)
    bound = (np.nanmax(sample.wins, axis=1) - np.nanmin(sample.wins, axis=1)) * 100
    assert not exact.any() and np.allclose(lost, bound)
    worst = np.nanargmin(sample.wins, axis=1)
    lost, exact = move_losses(sample, worst)
    assert exact.all() and np.allclose(lost, bound)
    blunders = int((bound >= 15).sum())
    assert 0 < blunders < len(sample)

    positions, squares = np.nonzero(sample.binary["destination_loses_material"][1])
    for sign, present in ((1.0, blunders), (-1.0, 0)):
        logits = np.full(len(positions), sign, dtype=np.float32)
        scores = ProbeScores(
            {"destination_loses_material": (positions, squares)},
            {"destination_loses_material": {"t1": logits}},
            {},
        )
        result = blunder_attribution(sample, worst, scores, "t1")
        assert result["blunders"] == blunders and result["knowledge_present"] == present
        assert result["knowledge_absent"] == blunders - present
        by_exchange = result["blunders_losing_material_by_exchange"]
        other = result["other_blunders"]
        assert isinstance(by_exchange, dict) and isinstance(other, dict)
        assert by_exchange["blunders"] + other["blunders"] == blunders
        assert result["unknown"] == 0


REPORT_KEYS = {
    "schema",
    "checkpoint",
    "config",
    "device",
    "batch_size",
    "seed",
    "depths",
    "representations",
    "early",
    "late",
    "best_depth",
    "best_depth_source",
    "samples",
    "probes",
    "improvement",
    "causal",
    "blunders",
    "notes",
    "seconds",
}


def test_probe_steps_command_writes_the_report(
    game_corpus: Path,
    puzzle_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: _FakeEngine())
    labels = tmp_path / "labels"
    label_positions(
        game_corpus / "train",
        labels,
        count=6,
        nodes=100,
        engine_command=["stockfish"],
        workers=1,
        low_priority=False,
    )
    checkpoint = _checkpoint(tmp_path / "model.pt", LOCAL)
    output = tmp_path / "probe"
    arguments = ["probe-steps", "--checkpoint", str(checkpoint), "--output", str(output)]
    arguments += ["--positions", "40", "--depths", "1,2", "--early", "1", "--late", "2"]
    arguments += ["--games", str(game_corpus / "train"), "--labels", str(labels)]
    arguments += ["--puzzles", str(puzzle_corpus / "validation")]
    arguments += ["--labelled-games", str(game_corpus / "train"), "--device", "cpu"]
    assert main(arguments) == 0
    printed = capsys.readouterr().out
    assert "Probes:" in printed and "Blunders at T" in printed

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert set(report) == REPORT_KEYS
    assert report["schema"] == "wombat.probe-steps.v1" and report["depths"] == [0, 1, 2]
    assert report["representations"] == ["input", "t0", "t1", "t2"]
    assert set(report["probes"]) == set(report["samples"]) == {"games", "puzzles", "labelled"}
    for targets in report["probes"].values():
        assert set(targets) == set(BINARY_TARGETS) | set(SQUARE_TARGETS)
    assert report["probes"]["games"]["reply_origin"] is None
    assert report["probes"]["labelled"]["reply_origin"]["positions"] > 0
    assert report["samples"]["labelled"]["positions"] == 6
    assert set(report["improvement"]) >= {"gained", "lost", "both_right", "both_wrong"}
    causal = report["causal"]
    assert set(causal["swap"]) == {"move_squares", "other_squares", "all_squares"}
    assert set(causal["freeze"]) == {"outside_1", "inside_1", "outside_2", "inside_2"}
    assert set(causal["board_features_off"]) == {"games", "puzzles"}
    assert set(report["blunders"]) == {"games", "labelled"}
    assert report["best_depth"] in report["depths"]


def test_probe_steps_takes_positions_and_losses_from_a_strength_report(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _checkpoint(tmp_path / "model.pt", GLOBAL)
    strength = tmp_path / "strength"
    strength.mkdir()
    curve = [
        {"reasoning_steps": 1, "lichess_accuracy": 50.0},
        {"reasoning_steps": 3, "lichess_accuracy": 60.0},
    ]
    report = {"checkpoint": str(checkpoint), "games": str(game_corpus / "test"), "curve": curve}
    (strength / "strength.json").write_text(json.dumps(report), encoding="utf-8")
    np.savez(
        strength / "strength.npz",
        game_indices=np.array([0, 1, 2]),
        lost_t1=np.zeros(3),
        lost_t3=np.array([0.0, 20.0, 40.0]),
    )
    output = tmp_path / "probe"
    arguments = ["probe-steps", "--checkpoint", str(checkpoint), "--output", str(output)]
    arguments += ["--strength", str(strength), "--puzzles", str(puzzle_corpus / "validation")]
    arguments += ["--labels", str(tmp_path / "missing"), "--depths", "1", "--early", "1"]
    arguments += ["--late", "2", "--device", "cpu"]
    assert main(arguments) == 0
    assert "Warning" not in capsys.readouterr().out
    result = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result["best_depth"] == 3 and result["best_depth_source"] == "strength report"
    assert result["depths"] == [0, 1, 2, 3]
    assert result["notes"]["strength"]["same_checkpoint"]
    assert set(result["blunders"]) == {"games"}
    assert result["blunders"]["games"]["blunders"] == 2
    assert result["causal"]["board_features_off"] is None
