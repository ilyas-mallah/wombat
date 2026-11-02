import json
from pathlib import Path

import chess
import numpy as np
import zstandard
import torch

from wombat.chess.moves import move_to_action
from wombat.data.torch_dataset import EvaluationDataset, _value_target, collate_examples
from wombat.training import _continued_improvement_loss, _recurrent_contraction_loss
from wombat.eval.quality_curve import classify_quality_trajectory


def test_value_targets_use_canonical_side_to_move_scores() -> None:
    assert _value_target({"score_cp": 0, "score_mate": None}) == 0.0
    assert _value_target({"score_cp": None, "score_mate": 3}) == 1.0
    assert _value_target({"score_cp": None, "score_mate": -2}) == -1.0


def test_training_dataset_loads_compact_shard(tmp_path: Path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    path = tmp_path / "train.jsonl.zst"
    payload = {
        "fen": " ".join(board.fen().split()[:4]),
        "best_move": move.uci(),
        "action": move_to_action(board, move),
        "score_cp": 20,
        "score_mate": None,
    }
    with zstandard.open(path, mode="wt", encoding="utf-8") as output:
        output.write(json.dumps(payload) + "\n")
    dataset = EvaluationDataset(path)
    assert len(dataset) == 1
    assert dataset[0].legal_mask[dataset[0].action]


def test_multipv_candidates_produce_soft_policy_and_move_values(tmp_path: Path) -> None:
    board = chess.Board()
    path = tmp_path / "train.jsonl.zst"
    payload = {
        "fen": " ".join(board.fen().split()[:4]),
        "best_move": "e2e4",
        "action": move_to_action(board, chess.Move.from_uci("e2e4")),
        "score_cp": 30,
        "score_mate": None,
        "alternatives": [
            {"move": "e2e4", "score_cp": 30, "score_mate": None},
            {"move": "d2d4", "score_cp": 20, "score_mate": None},
        ],
    }
    with zstandard.open(path, mode="wt", encoding="utf-8") as output:
        output.write(json.dumps(payload) + "\n")
    example = EvaluationDataset(path)[0]
    assert len(example.candidate_actions) == 2
    assert np.isclose(example.candidate_weights.sum(), 1.0)
    assert np.all(example.candidate_weights > 0)
    assert example.candidate_weights[0] > example.candidate_weights[1]


def test_continuation_lines_are_encoded_in_each_side_to_move_frame(tmp_path: Path) -> None:
    board = chess.Board()
    path = tmp_path / "train.jsonl.zst"
    payload = {
        "fen": " ".join(board.fen().split()[:4]),
        "best_move": "e2e4",
        "action": move_to_action(board, chess.Move.from_uci("e2e4")),
        "score_cp": 20,
        "score_mate": None,
        "continuations": [
            {
                "move": "e2e4",
                "score_cp": 20,
                "score_mate": None,
                "pv": ["e2e4", "e7e5", "g1f3"],
            }
        ],
    }
    with zstandard.open(path, mode="wt", encoding="utf-8") as output:
        output.write(json.dumps(payload) + "\n")
    example = EvaluationDataset(path)[0]
    assert len(example.continuation_actions) == 1
    assert len(example.continuation_actions[0]) == 3
    assert example.continuation_pieces[0].shape == (3, 64)
    assert example.continuation_states[0].shape == (3, 26)
    batch = collate_examples([example])
    # The first resulting board is after e2e4 and is canonicalized for black.
    assert batch["continuation_pieces"].shape == (1, 1, 3, 64)
    assert batch["continuation_states"].shape == (1, 1, 3, 26)


def test_paired_depth_annotations_keep_shallow_and_deep_targets_separate(
    tmp_path: Path,
) -> None:
    board = chess.Board()
    path = tmp_path / "paired.jsonl.zst"
    payload = {
        "fen": " ".join(board.fen().split()[:4]),
        "best_move": "e2e4",
        "action": move_to_action(board, chess.Move.from_uci("e2e4")),
        "score_cp": 20,
        "score_mate": None,
        "continuations": [
            {
                "move": "e2e4",
                "score_cp": 20,
                "score_mate": None,
                "pv": ["e2e4", "e7e5"],
            }
        ],
        "deep_continuations": [
            {
                "move": "d2d4",
                "score_cp": 35,
                "score_mate": None,
                "pv": ["d2d4", "d7d5", "c2c4"],
            }
        ],
    }
    with zstandard.open(path, mode="wt", encoding="utf-8") as output:
        output.write(json.dumps(payload) + "\n")

    example = EvaluationDataset(path)[0]
    assert example.shallow_actions[0] == move_to_action(
        board, chess.Move.from_uci("e2e4")
    )
    assert example.action == move_to_action(board, chess.Move.from_uci("d2d4"))
    assert example.shallow_deep_disagree
    assert len(example.shallow_continuation_actions[0]) == 2
    assert len(example.continuation_actions[0]) == 3


def test_improvement_loss_is_per_example_and_does_not_push_early_target() -> None:
    # The batch means are equal, but the second position regresses. A
    # batch-mean hinge would incorrectly report no regression.
    early = torch.tensor([2.0, 1.0], requires_grad=True)
    final = torch.tensor([1.0, 2.0], requires_grad=True)
    loss, stats = _continued_improvement_loss(
        final,
        early,
        margin=0.1,
        fraction=0.1,
        eligibility_epsilon=0.0,
    )
    assert loss.item() > 0
    loss.backward()
    assert early.grad is None
    assert final.grad is not None
    assert final.grad[0] == 0
    assert final.grad[1] > 0
    assert int(stats["improved"]) == 1


def test_improvement_margin_adapts_for_nearly_solved_positions() -> None:
    early = torch.tensor([0.001])
    final = torch.tensor([0.00095], requires_grad=True)
    loss, stats = _continued_improvement_loss(
        final,
        early,
        margin=0.1,
        fraction=0.1,
        eligibility_epsilon=0.0,
    )
    # Desired gain is 0.0001 rather than the impossible fixed 0.1.
    assert 0 < loss.item() < 0.001
    assert int(stats["eligible"]) == 1


def test_contraction_loss_does_not_collapse_the_earlier_update() -> None:
    early_previous = torch.zeros(1, 2, 3, requires_grad=True)
    early_hidden = torch.ones(1, 2, 3, requires_grad=True)
    final_previous = torch.zeros(1, 2, 3, requires_grad=True)
    final_hidden = torch.ones(1, 2, 3, requires_grad=True)
    loss = _recurrent_contraction_loss(
        early_previous,
        early_hidden,
        final_previous,
        final_hidden,
        steps_apart=2,
        ratio=0.9,
    ).mean()
    loss.backward()
    assert early_previous.grad is None
    assert early_hidden.grad is None
    assert final_previous.grad is not None
    assert final_hidden.grad is not None


def test_quality_trajectory_classification() -> None:
    assert classify_quality_trajectory(np.array([3.0, 2.0, 1.0])) == "monotonic_improvement"
    assert classify_quality_trajectory(np.array([1.0, 2.0, 3.0])) == "monotonic_degradation"
    assert classify_quality_trajectory(np.array([2.0, 1.0, 1.5])) == "oscillation"
