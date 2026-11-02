from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
import random
from typing import Any, ClassVar

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat import puzzle_training
from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import action_to_move, legal_action_mask, legal_actions, move_to_action
from wombat.cli import build_parser, main
from wombat.data.lichess import PuzzleRecord
from wombat.data.move_labels import label_positions
from wombat.eval.full_puzzles import export_puzzle_csv
from wombat.eval.puzzle_corpus_eval import load_checkpoint
from wombat.losses import LineWeights, LossWeights, _candidate_loss, complete_targets, line_loss
from wombat.model import NetworkConfig, WombatNetwork
from wombat.puzzle_training import PuzzleTrainingConfig, train_puzzles

WORKSPACE = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=0,
    recurrent_blocks=1,
    latent_tokens=0,
    policy_readout="square",
    move_readout="from_to",
    normalize_state=True,
    max_residual_update=None,
    square_attention_bias=True,
    core="workspace",
    workspace_tokens=4,
    steps_per_ply=2,
    line_plies=2,
)
FULL = dataclasses.replace(
    WORKSPACE, board_features=True, move_consequences=True, value_bins=8, lookahead_heads=True
)
OPENING = ["e7e5", "e2e4", "b8c6"]


class FakeEngine:
    """Scores each legal move by its target square; lines continue with first legal moves."""

    id: ClassVar[dict[str, str]] = {"name": "Fake"}

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(self, board: chess.Board, limit: Any, **kwargs: Any) -> Any:
        if "multipv" not in kwargs:
            cp = 300 if kwargs.get("root_moves") is None else kwargs["root_moves"][0].to_square
            return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}
        infos = []
        for move in list(board.legal_moves)[: kwargs["multipv"]]:
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


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: FakeEngine())


@pytest.fixture
def labels(game_corpus: Path, tmp_path: Path) -> Path:
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


def _tensors(boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_position(board) for board in boards]
    pieces = torch.from_numpy(np.stack([item.pieces for item in encoded]).astype(np.int64))
    state = torch.from_numpy(np.stack([state_vector(item) for item in encoded]))
    return pieces, state


def _targets(boards: list[chess.Board], reference: list[list[int]]) -> dict[str, torch.Tensor]:
    """The reference's first move as the root answer, or the lowest legal action without one."""

    legal = torch.from_numpy(np.stack([legal_action_mask(board) for board in boards]))
    lines = torch.tensor([row + [-1] * (4 - len(row)) for row in reference])
    answer = torch.where(lines[:, 0] >= 0, lines[:, 0], legal.float().argmax(dim=1))
    return complete_targets({"action": answer, "legal_mask": legal, "line_actions": lines})


def _line(board: chess.Board, moves: Sequence[str]) -> list[int]:
    played, actions = board.copy(stack=False), []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        actions.append(move_to_action(played, move))
        played.push(move)
    return actions


def _lowest(board: chess.Board, plies: int, first: int | None = None) -> list[int]:
    """``first`` (by default the lowest legal action), then each position's lowest action."""

    played, actions = board.copy(stack=False), []
    for ply in range(plies):
        action = first if ply == 0 and first is not None else min(legal_actions(played))
        actions.append(action)
        played.push(action_to_move(played, action))
    return actions


def _record(model: WombatNetwork, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make the line policy prefer the lowest action index and record the first row's actions."""

    core = model.workspace_core
    played: list[int] = []
    original = core.play

    def play(memory: torch.Tensor, workspace: torch.Tensor, actions: torch.Tensor) -> Any:
        played.append(int(actions[0]))
        return original(memory, workspace, actions)

    def prefer_lowest(memory: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
        return -torch.arange(4672, dtype=torch.float32).expand(len(memory), -1).clone()

    monkeypatch.setattr(core, "play", play)
    monkeypatch.setattr(model, "line_policy", prefer_lowest)
    return played


def _run(
    model: WombatNetwork, board: chess.Board, reference: list[int], **options: Any
) -> dict[str, torch.Tensor]:
    pieces, state = _tensors([board])
    targets = _targets([board], [reference])
    options = {"lines": 1, "plies": 4, **options}
    return line_loss(model, pieces, state, targets, LossWeights(), LineWeights(), **options)[2]


def test_warm_start_lines_follow_the_reference_after_the_models_own_first_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WombatNetwork(WORKSPACE)
    played = _record(model, monkeypatch)
    board = chess.Board()
    first = min(legal_actions(board))
    reference = [first, *_line(board, [action_to_move(board, first).uci(), *OPENING])[1:]]
    details = _run(model, board, reference, follow_reference=True)
    assert played == reference
    teacher, own = board.copy(stack=False), []
    for action in reference:
        own.append(min(legal_actions(teacher)))
        teacher.push(action_to_move(teacher, action))
    hits = [mine == theirs for mine, theirs in zip(own, reference)]
    assert float(details["line_accuracy"]) == pytest.approx(np.mean(hits))
    assert float(details["first_match"]) == 1.0


def test_lines_play_the_models_own_actions_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    model = WombatNetwork(WORKSPACE)
    played = _record(model, monkeypatch)
    board = chess.Board()
    first = min(legal_actions(board))
    reference = [first, *_line(board, [action_to_move(board, first).uci(), *OPENING])[1:]]
    details = _run(model, board, reference)
    own = _lowest(board, 4)
    assert played == own and own[1] != reference[1]
    # The first move and the reply where the line left the reference were graded.
    assert float(details["line_accuracy"]) == 0.5


def test_a_first_move_off_the_reference_is_graded_against_the_root_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WombatNetwork(WORKSPACE)
    played = _record(model, monkeypatch)
    board = chess.Board()
    reference = _line(board, ["e2e4", "e7e5", "g1f3", "b8c6"])
    details = _run(model, board, reference, follow_reference=True)
    assert played == _lowest(board, 4)
    assert float(details["first_match"]) == 0.0
    assert float(details["line_accuracy"]) == 0.0


def test_every_line_starts_from_the_present_position_and_may_repeat_its_first_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WombatNetwork(WORKSPACE)
    played = _record(model, monkeypatch)
    board = chess.Board()
    details = _run(model, board, [-1], lines=3, plies=2)
    assert played == _lowest(board, 2) * 3
    assert float(details["distinct_first"]) == pytest.approx(1 / 3)
    played.clear()
    with torch.no_grad():
        model.workspace_core.repeat_bias.fill_(-1e4)
    details = _run(model, board, [-1], lines=3, plies=2)
    firsts = sorted(legal_actions(board))[:3]
    assert played == [action for first in firsts for action in _lowest(board, 2, first)]
    assert float(details["distinct_first"]) == 1.0


def test_surplus_steps_start_further_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    model = WombatNetwork(WORKSPACE)
    played = _record(model, monkeypatch)
    board = chess.Board()
    # Two steps per ply: one line of 2 plies and 3 surplus steps start a second line of 2 plies.
    _run(model, board, [-1], lines=1, plies=2, surplus=3)
    assert played == _lowest(board, 2) * 2


def test_later_lines_are_taught_the_best_move_not_yet_tried() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, 5.0]])
    legal = torch.tensor([[True, True, True, False]])
    target = torch.tensor([[0.0, 0.25, 0.75, 0.0]])
    term, hit, count = _candidate_loss(logits, legal, target, torch.tensor([2]))
    log_probs = logits[:, :3].log_softmax(dim=1)[0]
    assert float(term) == pytest.approx(-(0.25 * log_probs[1] + 0.75 * log_probs[2]).item())
    assert (int(hit), count) == (1, 1)
    assert _candidate_loss(logits, legal, torch.zeros(1, 4), torch.tensor([2]))[2] == 0


def test_canonical_loss_is_zero_when_memory_is_the_child_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(FULL)
    core = model.workspace_core
    children: list[tuple[torch.Tensor, torch.Tensor]] = []
    child = core.rules.child

    def recorded(pieces: torch.Tensor, state: torch.Tensor, actions: torch.Tensor) -> Any:
        children.append(child(pieces, state, actions))
        return children[-1]

    def oracle(board: torch.Tensor, memory: torch.Tensor, workspace: torch.Tensor) -> Any:
        # Rotation undoes itself, so the ply's rotation turns this into the child's encoding.
        encoded = model.encode_board(*children[-1], consequences=False)
        return model.rotate_memory(encoded), workspace

    monkeypatch.setattr(core.rules, "child", recorded)
    monkeypatch.setattr(core, "step", oracle)
    boards = [chess.Board(), chess.Board("r3k2r/1P4P1/8/8/8/8/1p4p1/R3K2R w KQkq - 0 1")]
    pieces, state = _tensors(boards)
    targets = _targets(boards, [_line(boards[0], ["e2e4", "e7e5", "g1f3", "b8c6"]), [-1]])
    options: dict[str, Any] = {"lines": 2, "plies": 2}
    _, _, details = line_loss(
        model, pieces, state, targets, LossWeights(), LineWeights(), **options
    )
    assert len(children) == 4
    assert float(details["canonical"]) == pytest.approx(0.0, abs=1e-10)
    model = WombatNetwork(FULL)
    _, _, details = line_loss(
        model, pieces, state, targets, LossWeights(), LineWeights(), **options
    )
    assert float(details["canonical"]) > 0.01


def test_rotation_readdresses_squares_and_undoes_itself() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(WORKSPACE)
    board = chess.Board("r3k2r/pppq1ppp/2n2n2/3p4/3Pp3/2N2N2/PPPQPPPP/R3K2R w KQkq - 0 1")
    pieces, state = _tensors([board])
    swapped = model.workspace_core.rules.moves.swap[pieces]
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    child_pieces, child_state = _tensors([passed])
    with torch.no_grad():
        rotated = model.rotate_memory(model.encode_board(swapped, state))
        expected = model.encode_board(child_pieces, child_state)
        memory = torch.randn(1, 65, 24)
    assert torch.allclose(rotated[:, :64], expected[:, :64], atol=1e-6)
    assert torch.allclose(model.rotate_memory(model.rotate_memory(memory)), memory, atol=1e-6)


def test_inference_runs_lines_of_the_configured_plies_at_any_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(FULL).eval()
    played = _record(model, monkeypatch)
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    pieces, state = _tensors([board, chess.Board()])
    with torch.no_grad(), torch.inference_mode():
        outputs = model(pieces, state, 10, intermediate_steps=frozenset(range(11)))
        final = model(pieces, state, 10, include_move_values=True)
    assert isinstance(outputs, dict) and sorted(outputs) == list(range(11))
    assert isinstance(final, tuple) and len(final) == 3
    assert torch.equal(final[0], outputs[10][0])
    assert all(torch.isfinite(output[0]).all() for output in outputs.values())
    # Ten steps at two per ply and two plies per line: two lines and one ply of a third.
    assert played[:5] == [*_lowest(board, 2), *_lowest(board, 2), _lowest(board, 1)[0]]


def test_budgets_are_drawn_per_batch_ramp_up_and_the_reference_fades(tmp_path: Path) -> None:
    config = PuzzleTrainingConfig(
        corpus=".",
        output=str(tmp_path),
        steps=1_000,
        lines_max_final=6,
        line_plies_max_final=4,
        budget_ramp_from=100,
        budget_ramp_until=300,
        reference_line_until=200,
        random_prefix=3,
        random_prefix_from=500,
    )
    randomizer = random.Random(0)
    early = [puzzle_training._draw_line(config, randomizer, 50) for _ in range(400)]
    late = [puzzle_training._draw_line(config, randomizer, 600) for _ in range(400)]
    assert {draw[1] for draw in early} == {1, 2, 3} == {draw[2] for draw in early}
    assert {draw[1] for draw in late} == set(range(1, 7))
    assert {draw[2] for draw in late} == {1, 2, 3, 4}
    assert {draw[3] for draw in early} == {0} and {draw[3] for draw in late} == {0, 1, 2, 3}
    assert 0.6 < np.mean([draw[4] for draw in early]) < 0.9
    assert not any(draw[4] for draw in late)
    assert puzzle_training._ramped(config, 3, 6, 200) == 4
    assert puzzle_training._reference_share(config, 100) == pytest.approx(0.5)
    default = dataclasses.replace(config, reference_line_until=None)
    assert puzzle_training._reference_share(default, 150) == pytest.approx(0.5)


def test_workspace_options_are_checked() -> None:
    for change in (
        {"move_readout": "square"},
        {"latent_tokens": 2},
        {"local_attention": True},
        {"qk_norm": True},
        {"workspace_tokens": 0},
        {"line_plies": 0},
        {"core": "other"},
    ):
        with pytest.raises(ValueError):
            WombatNetwork(dataclasses.replace(WORKSPACE, **change))
    recurrent = WombatNetwork(dataclasses.replace(WORKSPACE, core="recurrent", latent_tokens=2))
    assert not any(name.startswith("workspace_core") for name, _ in recurrent.named_parameters())
    args = build_parser().parse_args(["train-games", "--output", "run"])
    assert (args.core, args.workspace_tokens, args.steps_per_ply, args.line_plies) == (
        "recurrent",
        12,
        2,
        2,
    )
    assert (args.lines_min, args.lines_max, args.line_plies_min, args.line_plies_max) == (
        1,
        3,
        1,
        3,
    )
    assert (args.line_policy_weight, args.reference_line_share) == (0.25, 1.0)
    assert build_parser().parse_args(["train-puzzles", "--output", "run"]).core == "recurrent"


@pytest.mark.parametrize(
    "overrides",
    [
        {"labels": None},
        {"supervise_every_step": True},
        {"fixed_depth": 8},
        {"lines_min": 3, "lines_max": 2},
        {"line_plies_max_final": 2, "line_plies_max": 3},
        {"reference_line_share": 1.5},
        {"depth_distribution": "lognormal", "max_gradient_steps": 2},
    ],
)
def test_workspace_training_is_checked(
    overrides: dict[str, object], game_corpus: Path, tmp_path: Path
) -> None:
    settings: dict[str, object] = {
        "corpus": str(game_corpus),
        "output": str(tmp_path / "run"),
        "task": "games",
        "labels": str(tmp_path),
        "steps": 2,
        "batch_size": 2,
        "supervise_every_step": False,
        "device": "cpu",
        **overrides,
    }
    with pytest.raises(ValueError):
        train_puzzles(PuzzleTrainingConfig(**settings), WORKSPACE)  # type: ignore[arg-type]


def test_train_games_cli_trains_the_workspace_core_and_evaluations_sweep_it(
    game_corpus: Path,
    puzzle_corpus: Path,
    puzzle_archive: tuple[Path, list[PuzzleRecord]],
    labels: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    output = tmp_path / "v34"
    arguments = ["train-games", "--corpus", str(game_corpus), "--labels", str(labels)]
    arguments += ["--output", str(output), "--device", "cpu", "--steps", "4", "--batch-size", "2"]
    arguments += ["--warmup-steps", "1", "--eval-every", "2", "--eval-depths", "0,2,5"]
    arguments += ["--snapshot-every", "2", "--no-supervise-every-step", "--ema-decay", "0.9"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--prelude-blocks", "0"]
    arguments += ["--recurrent-blocks", "1", "--latent-tokens", "0", "--move-readout", "from_to"]
    arguments += ["--board-features", "--move-consequences", "--value-bins", "8"]
    arguments += ["--core", "workspace", "--workspace-tokens", "4", "--steps-per-ply", "1"]
    arguments += ["--line-plies", "2", "--lines-max", "2", "--line-plies-max", "2"]
    arguments += [
        "--reference-line-until",
        "2",
        "--random-prefix",
        "2",
        "--random-prefix-from",
        "2",
    ]
    arguments += ["--puzzle-corpus", str(puzzle_corpus), "--puzzle-share", "0.5"]
    arguments += ["--recompute-steps"]
    assert main(arguments) == 0
    run = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert (run["model_config"]["core"], run["model_config"]["line_plies"]) == ("workspace", 2)
    assert run["training_config"]["lines_max"] == 2
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 4]
    for row in rows:
        assert np.isfinite(row["train_loss"])
        assert 1 <= row["mean_tail"] <= 4
        assert set(row["line"]) == {
            "lines",
            "plies",
            "follow_reference",
            "canonical",
            "line_policy",
            "line_accuracy",
            "first_match",
            "distinct_first",
        }
        assert set(row["validation"]) == {"t0", "t2", "t5"}
    assert rows[0]["mean_prefix"] == 0
    assert rows[1]["line"]["follow_reference"] == 0
    arguments[arguments.index("--steps") + 1] = "6"
    assert main([*arguments, "--resume"]) == 0
    checkpoint = output / "checkpoint.pt"
    assert load_checkpoint(checkpoint, torch.device("cpu")).config.line_plies == 2
    wider = load_checkpoint(checkpoint, torch.device("cpu"), line_plies=5)
    assert wider.config.line_plies == 5
    load_checkpoint(output / "snapshot-step-000002.pt", torch.device("cpu"))
    puzzles = tmp_path / "puzzles.csv"
    export_puzzle_csv(puzzle_corpus / "test", puzzle_archive[0], puzzles, count=50)
    # Budgets far past the trained ones, with longer lines than any batch drew.
    full = ["evaluate-full-puzzles", "--puzzles", str(puzzles), "--checkpoint", str(checkpoint)]
    full += ["--depths", "0,4,24", "--batch-size", "4", "--output", str(tmp_path / "full")]
    assert main([*full, "--line-plies", "3"]) == 0
    report = json.loads((tmp_path / "full" / "report.json").read_text(encoding="utf-8"))
    assert list(report["results"]) == ["t0", "t4", "t24"]
    assert report["line_plies"] == 3
    monkeypatch.chdir(tmp_path)
    strength = ["evaluate-strength", "--checkpoint", str(checkpoint)]
    strength += ["--games", str(game_corpus / "test"), "--puzzles", str(puzzle_corpus / "test")]
    strength += ["--depths", "2,16", "--positions", "2", "--puzzle-count", "1", "--engines", "1"]
    strength += ["--stockfish", "stockfish", "--output", str(tmp_path / "strength")]
    assert main(strength) == 0
    report = json.loads((tmp_path / "strength" / "strength.json").read_text(encoding="utf-8"))
    assert [row["reasoning_steps"] for row in report["curve"]] == [1, 2, 16]
    assert report["line_plies"] == 2
