"""Choosing moves by policy, per-move value, or their mix, in evaluation and play."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat.chess.moves import legal_action_mask, move_to_action
from wombat.cli import build_parser, main
from wombat.cuda_graphs import GraphedPredictor
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit
from wombat.engine.uci import run_uci
from wombat.eval import games
from wombat.eval.agreement import select_move
from wombat.eval.full_puzzles import EncodedPositions
from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
from wombat.model import NetworkConfig, WombatNetwork
from wombat.model.adapter import TorchPolicyProvider
from wombat.selection import (
    CHILD_MIX_BETA,
    CHILD_MODES,
    MIX_BETA,
    SELECT_MODES,
    child_choice,
    selection_scores,
)

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
GLOBAL = dataclasses.replace(TINY, policy_readout="global")
CPU = torch.device("cpu")
DEPTHS = (0, 1, 3)


@pytest.fixture(autouse=True)
def _normal_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.strength.lower_priority", lambda: None)


def _save(
    path: Path, config: NetworkConfig = TINY, training: dict[str, object] | None = None
) -> WombatNetwork:
    torch.manual_seed(0)
    model = WombatNetwork(config)
    payload: dict[str, object] = {"model": model.state_dict(), "model_config": asdict(config)}
    if training is not None:
        payload["training_config"] = training
    torch.save(payload, path)
    return model.eval()


def _direct_scores(
    policy: np.ndarray, values: np.ndarray, legal: np.ndarray, select: str, beta: float
) -> np.ndarray:
    """The selection rule written out in numpy, illegal actions at -inf."""

    logits = np.where(legal, policy, -np.inf)
    if select == "policy":
        return logits
    if select == "value":
        return np.where(legal, values, -np.inf)
    top = logits.max(axis=1, keepdims=True)
    log_probs = logits - top - np.log(np.exp(logits - top).sum(axis=1, keepdims=True))
    return np.where(legal, log_probs + beta * values, -np.inf)


def test_selection_scores_follow_each_rule() -> None:
    generator = torch.Generator().manual_seed(0)
    policy = torch.randn((5, 40), generator=generator)
    values = torch.tanh(torch.randn((5, 40), generator=generator))
    legal = torch.rand((5, 40), generator=generator) < 0.3
    legal[:, 0] = True
    for select in SELECT_MODES:
        for beta in (0.0, 2.0, MIX_BETA):
            scores = selection_scores(policy, values, legal, select, beta)
            expected = _direct_scores(policy.numpy(), values.numpy(), legal.numpy(), select, beta)
            assert np.allclose(scores.numpy(), expected, atol=1e-5)
            assert torch.isneginf(scores[~legal]).all() and torch.isfinite(scores[legal]).all()
            assert legal[torch.arange(5), scores.argmax(dim=1)].all()
    # With beta 0 the mix ranks moves exactly as the policy does.
    mixed = selection_scores(policy, values, legal, "mix", 0.0).argmax(dim=1)
    assert torch.equal(mixed, selection_scores(policy, None, legal, "policy").argmax(dim=1))
    with pytest.raises(ValueError, match="per-move value"):
        selection_scores(policy, None, legal, "value")
    with pytest.raises(ValueError, match="unknown move selection"):
        selection_scores(policy, values, legal, "greedy")


@pytest.mark.parametrize("config", [TINY, GLOBAL], ids=["square", "global"])
@pytest.mark.parametrize("corpus", ["games", "puzzles"])
def test_predict_depths_agrees_with_the_raw_outputs(
    config: NetworkConfig, corpus: str, game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    model = _save(tmp_path / "model.pt", config)
    split = (
        GameSplit(game_corpus / "test")
        if corpus == "games"
        else PuzzleSplit(puzzle_corpus / "test")
    )
    indices = np.arange(len(split))
    batch = split.batch(indices)
    legal = batch["legal_mask"]
    with torch.no_grad():
        outputs = model(
            batch["pieces"],
            batch["state"],
            max(DEPTHS),
            intermediate_steps=frozenset(DEPTHS),
            include_move_values=True,
        )
    assert isinstance(outputs, dict)
    chosen = {
        select: predict_depths(
            model, split, indices, DEPTHS, batch_size=3, device=CPU, select=select
        )
        for select in SELECT_MODES
    }
    rows = np.arange(len(indices))
    for depth in DEPTHS:
        policy, _, values = (item.numpy() for item in outputs[depth])
        # Policy selection repeats the masked-logit argmax exactly.
        before = outputs[depth][0].float().masked_fill(~legal, float("-inf")).argmax(dim=1)
        assert np.array_equal(chosen["policy"][depth], before.numpy())
        for select in SELECT_MODES:
            actions = chosen[select][depth]
            assert legal.numpy()[rows, actions].all()
            direct = _direct_scores(policy, values, legal.numpy(), select, MIX_BETA)
            assert np.allclose(direct[rows, actions], direct.max(axis=1), atol=1e-5)
    differs = any(
        not np.array_equal(chosen["policy"][depth], chosen["value"][depth]) for depth in DEPTHS
    )
    assert differs, "a random network's values should disagree with its policy somewhere"


def test_mix_beta_moves_between_policy_and_value(puzzle_corpus: Path, tmp_path: Path) -> None:
    model = _save(tmp_path / "model.pt")
    split = PuzzleSplit(puzzle_corpus / "test")
    indices = np.arange(len(split))

    def pick(select: str, beta: float = MIX_BETA) -> np.ndarray:
        return predict_depths(model, split, indices, (1,), device=CPU, select=select, beta=beta)[1]

    assert np.array_equal(pick("mix", 0.0), pick("policy"))
    assert np.array_equal(pick("mix", 1e6), pick("value"))


def test_child_choice_follows_each_rule() -> None:
    log_probs = torch.tensor([[-0.1, -2.0, float("-inf")], [-0.5, -1.0, -3.0]])
    child_scores = torch.tensor([[0.0, 1.0, 5.0], [0.2, 0.2, 0.1]])
    # Padding never wins, and equal child scores go to the likelier move.
    assert child_choice(log_probs, child_scores, "child", 0.0).tolist() == [1, 0]
    assert child_choice(log_probs, child_scores, "child-mix", 0.0).tolist() == [0, 0]
    assert child_choice(log_probs, child_scores, "child-mix", 4.0).tolist() == [1, 0]
    with pytest.raises(ValueError, match="unknown child selection"):
        child_choice(log_probs, child_scores, "mix", 1.0)


class _NoFens:
    """A split that serves batches but keeps no FENs, so boards are decoded."""

    def __init__(self, split: GameSplit | PuzzleSplit) -> None:
        self.split = split

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        return self.split.batch(indices)


@pytest.mark.parametrize("corpus", ["games", "puzzles", "decoded"])
def test_child_modes_reduce_to_the_policy(
    corpus: str, game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    model = _save(tmp_path / "model.pt")
    base = (
        GameSplit(game_corpus / "test")
        if corpus == "games"
        else PuzzleSplit(puzzle_corpus / "test")
    )
    split = _NoFens(base) if corpus == "decoded" else base
    indices = np.arange(min(12, len(base)))

    def pick(select: str, **options: Any) -> dict[int, np.ndarray]:
        return predict_depths(
            model, split, indices, DEPTHS, batch_size=5, device=CPU, select=select, **options
        )

    policy = pick("policy")
    single = pick("child", top_k=1)
    blended = pick("child-mix", beta=0.0, top_k=3)
    chosen = pick("child", top_k=3)
    for depth in DEPTHS:
        assert np.array_equal(single[depth], policy[depth])
        assert np.array_equal(blended[depth], policy[depth])
    differs = any(not np.array_equal(chosen[depth], policy[depth]) for depth in DEPTHS)
    assert differs, "a random network's child values should disagree with its policy somewhere"


def test_child_selection_plays_a_mate_in_the_top_k(tmp_path: Path) -> None:
    model = _save(tmp_path / "model.pt")
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    mate = chess.Move.from_uci("a1a8")
    positions = EncodedPositions([board])
    batch = positions.batch(np.arange(1))
    with torch.no_grad():
        outputs = model(
            batch["pieces"], batch["state"], max(DEPTHS), intermediate_steps=frozenset(DEPTHS)
        )
    assert isinstance(outputs, dict)
    logits = {
        depth: outputs[depth][0][0].masked_fill(~batch["legal_mask"][0], -1e9) for depth in DEPTHS
    }
    target = move_to_action(board, mate)
    ranks = [int((logits[depth] > logits[depth][target]).sum()) for depth in DEPTHS]
    assert max(ranks) > 0, "the policy alone should miss the mate somewhere"
    chosen = predict_depths(
        model, positions, np.arange(1), DEPTHS, device=CPU, select="child", top_k=max(ranks) + 1
    )
    assert all(chosen[depth][0] == target for depth in DEPTHS)


def test_a_predictor_captured_for_another_selection_is_refused(
    puzzle_corpus: Path, tmp_path: Path
) -> None:
    model = _save(tmp_path / "model.pt")
    split = PuzzleSplit(puzzle_corpus / "test")
    predictor = GraphedPredictor(model, 4, CPU)
    with pytest.raises(ValueError, match="different move selection"):
        predict_depths(
            model, split, np.arange(2), (1,), device=CPU, predictor=predictor, select="mix"
        )
    with pytest.raises(ValueError, match="unknown move selection"):
        GraphedPredictor(model, 4, CPU, select="greedy")


def test_value_selection_needs_trained_move_values(tmp_path: Path) -> None:
    untrained = tmp_path / "untrained.pt"
    _save(untrained, training={"move_value_weight": 0.0})
    load_checkpoint(untrained, CPU)
    for select in ("value", "mix"):
        with pytest.raises(ValueError, match="move_value_weight 0"):
            load_checkpoint(untrained, CPU, select)
    # A config from before move values existed trained none.
    _save(tmp_path / "old.pt", training={"steps": 10})
    with pytest.raises(ValueError, match="move_value_weight 0"):
        load_checkpoint(tmp_path / "old.pt", CPU, "value")
    trained = tmp_path / "trained.pt"
    _save(trained, training={"move_value_weight": 0.5})
    load_checkpoint(trained, CPU, "value")
    # Snapshots carry no training config; their run's config.json stands in.
    run = tmp_path / "run"
    run.mkdir()
    snapshot = run / "snapshot-step-000005.pt"
    _save(snapshot)
    load_checkpoint(snapshot, CPU, "mix")
    (run / "config.json").write_text(
        json.dumps({"training_config": {"move_value_weight": 0.0}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="move_value_weight 0"):
        load_checkpoint(snapshot, CPU, "mix")
    headless = tmp_path / "headless.pt"
    model = _save(headless)
    state = {key: value for key, value in model.state_dict().items() if "move_value" not in key}
    torch.save({"model": state, "model_config": asdict(TINY)}, headless)
    with pytest.raises(ValueError, match="needs a move-value head"):
        load_checkpoint(headless, CPU, "value")
    with pytest.raises(ValueError, match="unknown move selection"):
        load_checkpoint(trained, CPU, "greedy")


def test_selection_flags_parse() -> None:
    parser = build_parser()
    for command in (
        ["evaluate-strength", "--checkpoint", "x.pt", "--output", "out"],
        ["evaluate-puzzle-depth", "--checkpoint", "x.pt", "--output", "out"],
    ):
        defaults = parser.parse_args(command)
        assert (defaults.select, defaults.select_beta) == ("policy", MIX_BETA)
        chosen = parser.parse_args([*command, "--select", "mix", "--select-beta", "4"])
        assert (chosen.select, chosen.select_beta) == ("mix", 4.0)
        with pytest.raises(SystemExit):
            parser.parse_args([*command, "--select", "greedy"])


class _FakeEngine:
    """Scores the best move +300 cp and any restricted move by its target square."""

    id = {"name": "Fake"}

    def configure(self, options: dict[str, object]) -> None:
        pass

    def analyse(self, board: chess.Board, limit: Any, root_moves: Any = None) -> dict:
        cp = 300 if root_moves is None else root_moves[0].to_square * 5
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}

    def quit(self) -> None:
        pass


def test_reports_record_the_selection(
    game_corpus: Path,
    puzzle_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: _FakeEngine())
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "model.pt"
    _save(checkpoint, training={"move_value_weight": 0.5})
    strength = [
        "evaluate-strength",
        "--checkpoint",
        str(checkpoint),
        "--games",
        str(game_corpus / "test"),
        "--puzzles",
        str(puzzle_corpus / "test"),
        "--depths",
        "1,2",
        "--positions",
        "2",
        "--puzzle-count",
        "1",
        "--engines",
        "1",
        "--stockfish",
        "stockfish",
    ]
    for select, beta in (("policy", None), ("value", None), ("mix", 3.0)):
        output = tmp_path / f"strength-{select}"
        flags = ["--select", select] + ([] if beta is None else ["--select-beta", str(beta)])
        assert main([*strength, "--output", str(output), *flags]) == 0
        report = json.loads((output / "strength.json").read_text(encoding="utf-8"))
        assert (report["select"], report["select_beta"]) == (select, beta)
    depth = [
        "evaluate-puzzle-depth",
        "--checkpoint",
        str(checkpoint),
        "--split",
        str(puzzle_corpus / "test"),
        "--depths",
        "1,2",
        "--per-group",
        "2",
        "--device",
        "cpu",
    ]
    output = tmp_path / "depth"
    assert main([*depth, "--output", str(output), "--select", "value"]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert (report["select"], report["select_beta"]) == ("value", None)
    untrained = tmp_path / "untrained.pt"
    _save(untrained, training={"move_value_weight": 0.0})
    depth[depth.index(str(checkpoint))] = str(untrained)
    with pytest.raises(ValueError, match="needs trained move values"):
        main([*depth, "--output", str(tmp_path / "refused"), "--select", "mix"])


def test_strength_reports_record_the_child_selection(
    game_corpus: Path,
    puzzle_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: _FakeEngine())
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.chdir(tmp_path)
    # Child selection reads only the position value, so no trained move values are needed.
    checkpoint = tmp_path / "model.pt"
    _save(checkpoint, training={"move_value_weight": 0.0})
    strength = [
        "evaluate-strength",
        "--checkpoint",
        str(checkpoint),
        "--games",
        str(game_corpus / "test"),
        "--puzzles",
        str(puzzle_corpus / "test"),
        "--depths",
        "1,2",
        "--positions",
        "2",
        "--puzzle-count",
        "1",
        "--engines",
        "1",
        "--stockfish",
        "stockfish",
    ]
    cases = (
        (["--select", "policy"], ("policy", None, None)),
        (["--select", "child", "--select-top-k", "2"], ("child", None, 2)),
        (["--select", "child-mix"], ("child-mix", CHILD_MIX_BETA, 3)),
        (["--select-beta", "2", "--select", "child-mix"], ("child-mix", 2.0, 3)),
        (["--select", "mix"], ("mix", MIX_BETA, None)),
    )
    for number, (flags, expected) in enumerate(cases):
        output = tmp_path / f"strength-{number}"
        if flags[1] == "mix":
            _save(checkpoint, training={"move_value_weight": 0.5})
        assert main([*strength, "--output", str(output), *flags]) == 0
        report = json.loads((output / "strength.json").read_text(encoding="utf-8"))
        assert (report["select"], report["select_beta"], report["select_top_k"]) == expected
    parser = build_parser()
    for mode in CHILD_MODES:
        parser.parse_args(
            ["evaluate-full-puzzles", "--puzzles", "p", "--output", "o", "--select", mode]
        )
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["evaluate-puzzle-depth", "--checkpoint", "x.pt", "--output", "o", "--select", mode]
            )


def test_provider_plays_the_selected_move(tmp_path: Path) -> None:
    model = _save(tmp_path / "model.pt")
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    provider = TorchPolicyProvider(model)
    with torch.no_grad():
        policy, value, values = model(*provider.inputs(board), 2, include_move_values=True)
    logits, reported = provider.policy_and_value(board, 2)
    assert np.array_equal(logits, policy[0].numpy()) and reported == float(value[0])
    legal = legal_action_mask(board)
    for select in SELECT_MODES:
        provider.set_selection(select, MIX_BETA)
        scores, reported = provider.policy_and_value(board, 2)
        assert np.isfinite(scores).all() and reported == float(value[0])
        direct = _direct_scores(policy.numpy(), values.numpy(), legal[None], select, MIX_BETA)[0]
        move = select_move(provider, board, 2)
        assert move in board.legal_moves
        action = int(np.argmax(np.where(legal, scores, -np.inf)))
        assert direct[action] == pytest.approx(direct.max(), abs=1e-5)
        # A guard below every value ranks all legal moves by the same scores.
        assert move == select_move(provider, board, 2, draw_guard=-1.0)
    with pytest.raises(ValueError, match="unknown move selection"):
        provider.set_selection("greedy", 1.0)


def _uci(provider: TorchPolicyProvider, *options: str) -> list[str]:
    commands = "".join(f"setoption name {option}\n" for option in options)
    output = StringIO()
    run_uci(StringIO(f"uci\n{commands}position startpos\ngo\nquit\n"), output, provider)
    return output.getvalue().splitlines()


def test_uci_move_selection_option(tmp_path: Path) -> None:
    trained = tmp_path / "trained.pt"
    model = _save(trained, training={"move_value_weight": 0.5})
    board = chess.Board()
    reference = TorchPolicyProvider(model)
    expected = {}
    for select in SELECT_MODES:
        reference.set_selection(select, MIX_BETA)
        expected[select] = select_move(reference, board, 1).uci()
    assert expected["policy"] != expected["value"]
    lines = _uci(TorchPolicyProvider(model, checkpoint=trained))
    combo = "option name MoveSelection type combo default policy var policy var value var mix"
    assert combo in lines
    assert f"option name SelectBeta type string default {MIX_BETA:g}" in lines
    assert f"bestmove {expected['policy']}" in lines
    lines = _uci(TorchPolicyProvider(model, checkpoint=trained), "MoveSelection value value")
    assert f"bestmove {expected['value']}" in lines
    mixed = _uci(
        TorchPolicyProvider(model, checkpoint=trained),
        "MoveSelection value mix",
        "SelectBeta value 1e6",
    )
    assert f"bestmove {expected['value']}" in mixed
    untrained = tmp_path / "untrained.pt"
    _save(untrained, training={"move_value_weight": 0.0})
    refused = _uci(TorchPolicyProvider(model, checkpoint=untrained), "MoveSelection value value")
    assert any(line.startswith("info string error value selection needs") for line in refused)
    assert f"bestmove {expected['policy']}" in refused


def test_stockfish_matches_play_and_record_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = ["play-stockfish", "--checkpoint", "x.pt"]
    defaults = build_parser().parse_args(command)
    assert (defaults.select, defaults.select_beta) == ("policy", MIX_BETA)
    chosen = build_parser().parse_args([*command, "--select", "mix", "--select-beta", "4"])
    assert (chosen.select, chosen.select_beta) == ("mix", 4.0)
    matches: list[dict[str, Any]] = []
    monkeypatch.setattr(games, "select_openings", lambda *args, **kwargs: [chess.STARTING_FEN])
    monkeypatch.setattr(
        games,
        "play_match",
        lambda provider, *args, **kwargs: matches.append({"provider": provider, **kwargs}),
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    checkpoint = tmp_path / "model.pt"
    _save(checkpoint, training={"move_value_weight": 0.5})
    for select in SELECT_MODES:
        games.play_stockfish(
            checkpoint, ["stockfish"], output=tmp_path, low_priority=False, select=select, beta=3.0
        )
    policy, value, mix = (match["identity"] for match in matches)
    # Policy matches keep their identity, so existing match directories resume.
    assert "select" not in policy and "select_beta" not in policy
    assert (value["select"], value["select_beta"]) == ("value", None)
    assert (mix["select"], mix["select_beta"]) == ("mix", 3.0)
    assert [match["provider"].select for match in matches] == list(SELECT_MODES)
    assert matches[2]["provider"].beta == 3.0
    untrained = tmp_path / "untrained.pt"
    _save(untrained, training={"move_value_weight": 0.0})
    with pytest.raises(ValueError, match="needs trained move values"):
        games.play_stockfish(
            untrained, ["stockfish"], output=tmp_path, low_priority=False, select="value"
        )
