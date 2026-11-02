"""CUDA-graph equivalence checks; they use the GPU, so they run only on request."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from wombat import puzzle_training
from wombat.cuda_graphs import WARMUP_ITERATIONS, GraphedPredictor, GraphedTrainer
from wombat.data.game_corpus import IGNORE, GameSplit
from wombat.data.puzzle_corpus import PuzzleSplit
from wombat.eval.puzzle_corpus_eval import predict_depths
from wombat.losses import LossWeights
from wombat.model import NetworkConfig, WombatNetwork
from wombat.model.board_features import BOARD_FEATURES
from wombat.puzzle_training import PuzzleTrainingConfig, train_puzzles
from wombat.selection import MIX_BETA

pytestmark = pytest.mark.skipif(
    os.environ.get("WOMBAT_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set WOMBAT_GPU_TESTS=1 to run GPU tests",
)

SMALL = NetworkConfig(
    d_model=32,
    attention_heads=4,
    mlp_width=64,
    prelude_blocks=1,
    recurrent_blocks=2,
    latent_tokens=4,
    policy_readout="square",
    normalize_state=True,
    max_residual_update=None,
)
DEVICE = torch.device("cuda")


def _optimizer(model: WombatNetwork, learning_rate: torch.Tensor) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05, fused=True, capturable=True
    )


def test_graphed_iterations_match_eager_loss(puzzle_corpus: Path) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(SMALL).to(DEVICE).train()
    learning_rate = torch.tensor(1e-3, device=DEVICE)
    trainer = GraphedTrainer(
        model, _optimizer(model, learning_rate), batch_size=4, max_tail=3, device=DEVICE
    )
    split = PuzzleSplit(puzzle_corpus / "train")
    batch = split.batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 2), (2, 1), (1, 3), (0, 1)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
        changed = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.parameters(), strict=True)
        )
        assert changed
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert trainer.graphs


def test_graphed_predictor_matches_eager(puzzle_corpus: Path) -> None:
    torch.manual_seed(1)
    model = WombatNetwork(SMALL).to(DEVICE)
    split = PuzzleSplit(puzzle_corpus / "test")
    indices = np.arange(len(split))
    depths = (0, 1, 3, 5)
    eager = predict_depths(model, split, indices, depths, device=DEVICE, graphs=False)
    predictor = GraphedPredictor(model, 4, DEVICE)
    graphed = predict_depths(
        model, split, indices, depths, batch_size=4, device=DEVICE, predictor=predictor
    )
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])


@pytest.mark.parametrize(("select", "beta"), [("value", MIX_BETA), ("mix", MIX_BETA), ("mix", 2.0)])
def test_graphed_move_selection_matches_eager(
    puzzle_corpus: Path, select: str, beta: float
) -> None:
    torch.manual_seed(1)
    model = WombatNetwork(SMALL).to(DEVICE)
    split = PuzzleSplit(puzzle_corpus / "test")
    indices = np.arange(len(split))
    depths = (0, 1, 3, 5)
    eager = predict_depths(
        model, split, indices, depths, device=DEVICE, graphs=False, select=select, beta=beta
    )
    predictor = GraphedPredictor(model, 4, DEVICE, select=select, beta=beta)
    graphed = predict_depths(
        model,
        split,
        indices,
        depths,
        batch_size=4,
        device=DEVICE,
        predictor=predictor,
        select=select,
        beta=beta,
    )
    policy = predict_depths(model, split, indices, depths, device=DEVICE, graphs=False)
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])
    assert any(not np.array_equal(eager[depth], policy[depth]) for depth in depths)


def test_graphed_training_run_end_to_end(puzzle_corpus: Path, tmp_path: Path) -> None:
    config = PuzzleTrainingConfig(
        corpus=str(puzzle_corpus),
        output=str(tmp_path / "run"),
        steps=8,
        batch_size=4,
        warmup_steps=1,
        max_depth=4,
        max_gradient_steps=2,
        eval_every=4,
        eval_positions=8,
        eval_depths=(0, 1, 4),
        device="cuda",
    )
    train_puzzles(config, SMALL)
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert [row["step"] for row in rows] == [4, 8]
    assert all(np.isfinite(row["train_loss"]) for row in rows)


def test_graphed_graded_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(2)
    model = WombatNetwork(SMALL).to(DEVICE).train()
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=2,
        device=DEVICE,
        weights=weights,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 1), (3, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)


V7 = dataclasses.replace(SMALL, lookahead_heads=True, square_attention_bias=True)


def test_graphed_v7_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(3)
    model = WombatNetwork(V7).to(DEVICE).train()
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
        every_step=True,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    generator = torch.Generator().manual_seed(0)
    batch["reply_from"] = torch.randint(0, 64, (4,), generator=generator)
    batch["reply_to"] = torch.tensor([10, IGNORE, 20, 30])
    batch["board2"] = batch["pieces"].clone()
    batch["board4"][1:] = batch["pieces"][1:]
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (1, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(
            reference, batch, prefix, tail, DEVICE, weights, every_step=True
        )
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert torch.count_nonzero(model.recurrent_core[0].square_bias) > 0


V8 = dataclasses.replace(V7, scratch_boards=2)


def test_graphed_v8_recomputed_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(4)
    model = WombatNetwork(V8).to(DEVICE).train()
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
        every_step=True,
        recompute=True,
    )
    assert trainer.buffers.hidden.shape == (4, 65 + 4 + 128, 32)
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    generator = torch.Generator().manual_seed(0)
    batch["reply_from"] = torch.randint(0, 64, (4,), generator=generator)
    batch["reply_to"] = torch.tensor([10, IGNORE, 20, 30])
    batch["board2"] = batch["pieces"].clone()
    batch["board4"][1:] = batch["pieces"][1:]
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (1, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(
            reference, batch, prefix, tail, DEVICE, weights, every_step=True
        )
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


LOCAL = dataclasses.replace(V7, prelude_blocks=0, latent_tokens=0, local_attention=True)


def test_graphed_local_attention_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(6)
    model = WombatNetwork(LOCAL).to(DEVICE).train()
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (4, 3)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs


LOCAL_FROM_TO = dataclasses.replace(
    LOCAL, local_neighbourhood="king_knight", move_readout="from_to"
)


def test_graphed_king_knight_from_to_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(7)
    model = WombatNetwork(LOCAL_FROM_TO).to(DEVICE).train()
    assert model.move_destinations.device.type == "cuda"
    initial_key = model.policy_key.weight.detach().clone()
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (4, 3)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert not torch.equal(model.policy_key.weight, initial_key)


LOCAL_FEATURES = dataclasses.replace(LOCAL, board_features=True)


def test_graphed_board_feature_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(8)
    model = WombatNetwork(LOCAL_FEATURES).to(DEVICE).train()
    assert {buffer.device.type for buffer in model.board_features.buffers()} == {"cuda"}
    initial_projection = model.board_feature_projection.weight.detach().clone()
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (4, 3)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert not torch.equal(model.board_feature_projection.weight, initial_projection)


# The v15d model: from_to readout, king and knight neighbourhood, both board feature sets.
LOCAL_LONG_RANGE = dataclasses.replace(
    LOCAL_FROM_TO, board_features=True, board_features_long_range=True
)


def test_graphed_long_range_board_feature_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(9)
    model = WombatNetwork(LOCAL_LONG_RANGE).to(DEVICE).train()
    assert {buffer.device.type for buffer in model.board_features.buffers()} == {"cuda"}
    assert model.board_features.material_scale.device.type == "cuda"
    initial_projection = model.board_feature_projection.weight.detach().clone()
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (4, 3)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    updated = model.board_feature_projection.weight
    assert not torch.equal(updated[:, BOARD_FEATURES:], initial_projection[:, BOARD_FEATURES:])


# The v16 model with the joint reply head over square pairs.
PAIR_REPLY = dataclasses.replace(LOCAL_LONG_RANGE, reply_readout="pair")


def test_graphed_pair_reply_iterations_match_eager_loss(game_corpus: Path) -> None:
    torch.manual_seed(10)
    model = WombatNetwork(PAIR_REPLY).to(DEVICE).train()
    initial_key = model.reply_key.weight.detach().clone()
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25, reply=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
        every_step=True,
    )
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    generator = torch.Generator().manual_seed(0)
    batch["reply_from"] = torch.tensor([12, IGNORE, 52, 3])
    batch["reply_to"] = torch.tensor([28, 5, IGNORE, 63])
    batch["board2"] = batch["pieces"].clone()
    batch["board4"][1:] = torch.randint(0, 13, (3, 64), generator=generator)
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(batch)
        trainer.run(1, 2)
    for prefix, tail in ((0, 3), (2, 1), (1, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(
            reference, batch, prefix, tail, DEVICE, weights, every_step=True
        )
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert not torch.equal(model.reply_key.weight, initial_key)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("neighbourhood", ["move_graph", "move_lines"])
def test_graphed_move_graph_iterations_and_predictions_match_eager(
    neighbourhood: str, game_corpus: Path
) -> None:
    torch.manual_seed(11)
    # The v16 model with move-graph local attention, whose mask depends on each position.
    config = dataclasses.replace(LOCAL_LONG_RANGE, local_neighbourhood=neighbourhood)
    model = WombatNetwork(config).to(DEVICE).train()
    assert {buffer.device.type for buffer in model.move_graph.buffers()} == {"cuda"}
    lines = neighbourhood == "move_lines"
    blockers = [block.blocker_bias.detach().clone() for block in model.recurrent_core if lines]
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
        recompute=True,
    )
    split = GameSplit(game_corpus / "train")
    first, second = split.batch(np.arange(4)), split.batch(np.arange(4, 8))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(first)
        trainer.run(1, 2)
    # Prefix steps replay the advance graph, which rebuilds the mask from the loaded pieces.
    for batch, prefix, tail in ((first, 0, 3), (second, 2, 1), (first, 4, 3), (second, 3, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    # The blocker bias learns inside the captured graphs.
    for block, old in zip(model.recurrent_core if lines else [], blockers, strict=True):
        assert not torch.equal(block.blocker_bias, old)
    model.eval()
    # Every position of the split, so the last batch is a partial one.
    indices = np.arange(len(split))
    depths = (0, 1, 3, 6)
    eager = predict_depths(model, split, indices, depths, batch_size=4, device=DEVICE, graphs=False)
    predictor = GraphedPredictor(model, 4, DEVICE)
    graphed = predict_depths(
        model, split, indices, depths, batch_size=4, device=DEVICE, predictor=predictor
    )
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])


# The v16 model with move consequences, whose features the graphs recompute from the pieces.
CONSEQUENCES = dataclasses.replace(LOCAL_LONG_RANGE, move_consequences=True)


@pytest.mark.parametrize("select", ["policy", "mix"])
def test_graphed_move_consequence_iterations_and_predictions_match_eager(
    select: str, game_corpus: Path
) -> None:
    torch.manual_seed(12)
    model = WombatNetwork(CONSEQUENCES).to(DEVICE).train()
    assert {buffer.device.type for buffer in model.move_consequences.buffers()} == {"cuda"}
    # Nonzero maps, so the features reach the loss from the first iteration.
    maps = [
        model.policy_consequence_origin,
        model.policy_consequence_destination,
        model.value_consequence_origin,
        model.value_consequence_destination,
    ]
    for layer in maps:
        torch.nn.init.normal_(layer.weight, std=0.1)
    initial = [layer.weight.detach().clone() for layer in maps]
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    split = GameSplit(game_corpus / "train")
    first, second = split.batch(np.arange(4)), split.batch(np.arange(4, 8))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(first)
        trainer.run(1, 2)
    # Eager runs size every stage exactly; the graphs use the fixed caps.
    for batch, prefix, tail in ((first, 0, 3), (second, 2, 1), (first, 4, 3), (second, 3, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for layer, old in zip(maps, initial, strict=True):
        assert not torch.equal(layer.weight, old)
    model.eval()
    indices = np.arange(len(split))
    depths = (0, 1, 3, 6)
    eager = predict_depths(
        model, split, indices, depths, batch_size=4, device=DEVICE, graphs=False, select=select
    )
    predictor = GraphedPredictor(model, 4, DEVICE, select=select)
    graphed = predict_depths(
        model,
        split,
        indices,
        depths,
        batch_size=4,
        device=DEVICE,
        predictor=predictor,
        select=select,
    )
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])
    # The predictor's static features equal the eager ones for its last batch.
    buffers = predictor.buffers
    assert buffers.consequences is not None
    with torch.no_grad():
        features = model.consequences(buffers.pieces, buffers.state)
    assert features is not None and torch.equal(buffers.consequences, features)


# The v22 model with relative key and value vectors per king offset.
RELATIVE = dataclasses.replace(
    LOCAL_LONG_RANGE, local_neighbourhood="king", relative_attention=True
)


def test_graphed_relative_attention_iterations_and_predictions_match_eager(
    game_corpus: Path,
) -> None:
    torch.manual_seed(13)
    model = WombatNetwork(RELATIVE).to(DEVICE).train()
    blocks = list(model.recurrent_core)
    assert {buffer.device.type for block in blocks for buffer in block.buffers()} == {"cuda"}
    # Nonzero vectors, so the relative terms reach the loss from the first iteration.
    names = ("relative_keys", "relative_values")
    vectors = [getattr(block, name) for block in blocks for name in names]
    with torch.no_grad():
        for vector in vectors:
            vector.normal_(std=0.3)
    initial = [vector.detach().clone() for vector in vectors]
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    split = GameSplit(game_corpus / "train")
    first, second = split.batch(np.arange(4)), split.batch(np.arange(4, 8))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(first)
        trainer.run(1, 2)
    for batch, prefix, tail in ((first, 0, 3), (second, 2, 1), (first, 4, 3), (second, 3, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for vector, old in zip(vectors, initial, strict=True):
        assert not torch.equal(vector, old)
    model.eval()
    indices = np.arange(len(split))
    depths = (0, 1, 3, 6)
    eager = predict_depths(model, split, indices, depths, batch_size=4, device=DEVICE, graphs=False)
    predictor = GraphedPredictor(model, 4, DEVICE)
    graphed = predict_depths(
        model, split, indices, depths, batch_size=4, device=DEVICE, predictor=predictor
    )
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])


# The v22 model with square pair logits generated from H at every step.
STATE_BIAS = dataclasses.replace(
    LOCAL_LONG_RANGE,
    local_neighbourhood="king",
    state_attention_bias=True,
    state_attention_bias_width=4,
)


def test_graphed_state_attention_bias_iterations_and_predictions_match_eager(
    game_corpus: Path,
) -> None:
    torch.manual_seed(14)
    model = WombatNetwork(STATE_BIAS).to(DEVICE).train()
    # Nonzero templates, so the generated bias reaches the loss from the first iteration.
    templates = model.state_bias_templates.weight
    with torch.no_grad():
        templates.normal_()
    parameters = [p for name, p in model.named_parameters() if "state_bias" in name]
    initial = [parameter.detach().clone() for parameter in parameters]
    weights = LossWeights(move_value=0.5, value=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    split = GameSplit(game_corpus / "train")
    first, second = split.batch(np.arange(4)), split.batch(np.arange(4, 8))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(first)
        trainer.run(1, 2)
    for batch, prefix, tail in ((first, 0, 3), (second, 2, 1), (first, 4, 3), (second, 3, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for parameter, old in zip(parameters, initial, strict=True):
        assert not torch.equal(parameter, old)
    model.eval()
    indices = np.arange(len(split))
    depths = (0, 1, 3, 6)
    eager = predict_depths(model, split, indices, depths, batch_size=4, device=DEVICE, graphs=False)
    predictor = GraphedPredictor(model, 4, DEVICE)
    graphed = predict_depths(
        model, split, indices, depths, batch_size=4, device=DEVICE, predictor=predictor
    )
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])


# The v26 model with every value read from 8 win-chance bins.
VALUE_BINS = dataclasses.replace(
    CONSEQUENCES, local_neighbourhood="king", relative_attention=True, value_bins=8
)


@pytest.mark.parametrize("select", ["policy", "value", "mix"])
def test_graphed_value_bin_iterations_and_predictions_match_eager(
    select: str, game_corpus: Path
) -> None:
    torch.manual_seed(15)
    model = WombatNetwork(VALUE_BINS).to(DEVICE).train()
    assert model.value_bin_values.device.type == "cuda"
    # Nonzero consequence maps, so they reach the bins through the tilted scores.
    for name, parameter in model.named_parameters():
        if "consequence" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    parameters = [p for name, p in model.named_parameters() if "bin" in name]
    parameters.append(model.value_head[2].weight)
    initial = [parameter.detach().clone() for parameter in parameters]
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=3,
        device=DEVICE,
        weights=weights,
    )
    split = GameSplit(game_corpus / "train")
    first, second = split.batch(np.arange(4)), split.batch(np.arange(4, 8))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(first)
        trainer.run(1, 2)
    for batch, prefix, tail in ((first, 0, 3), (second, 2, 1), (first, 4, 3), (second, 3, 2)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for parameter, old in zip(parameters, initial, strict=True):
        assert not torch.equal(parameter, old)
    model.eval()
    indices = np.arange(len(split))
    depths = (0, 1, 3, 6)
    eager = predict_depths(
        model, split, indices, depths, batch_size=4, device=DEVICE, graphs=False, select=select
    )
    predictor = GraphedPredictor(model, 4, DEVICE, select=select)
    graphed = predict_depths(
        model,
        split,
        indices,
        depths,
        batch_size=4,
        device=DEVICE,
        predictor=predictor,
        select=select,
    )
    for depth in depths:
        assert np.array_equal(eager[depth], graphed[depth])


def test_graphed_mixed_batches_match_eager_loss(game_corpus: Path, puzzle_corpus: Path) -> None:
    torch.manual_seed(5)
    model = WombatNetwork(V7).to(DEVICE).train()
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=4,
        device=DEVICE,
        weights=weights,
    )
    games = GameSplit(game_corpus / "train").batch(np.arange(4))
    split = PuzzleSplit(puzzle_corpus / "train")
    puzzles = {key: value.to(DEVICE) for key, value in split.batch(np.arange(4)).items()}
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(games)
        trainer.run(0, 4)
    for batch, tail in ((puzzles, 2), (games, 4), (puzzles, 4), (games, 4)):
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, 0, tail, DEVICE, weights)
        if batch is puzzles:
            policy_only, _ = puzzle_training._loss(copy.deepcopy(model), batch, 0, tail, DEVICE)
            assert float(expected) == pytest.approx(float(policy_only), rel=1e-3, abs=1e-3)
        trainer.load(batch)
        trainer.run(0, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_graphed_lognormal_prefixes_match_eager_loss(
    game_corpus: Path, puzzle_corpus: Path
) -> None:
    torch.manual_seed(6)
    model = WombatNetwork(V7).to(DEVICE).train()
    weights = LossWeights(move_value=0.5, value=0.25, lookahead=0.25)
    config = PuzzleTrainingConfig(
        corpus="games",
        output="run",
        task="games",
        depth_distribution="lognormal",
        max_gradient_steps=4,
        puzzle_corpus="puzzles",
        puzzle_share=0.5,
        puzzle_steps_per_move=2,
    )
    puzzle_training._check_depth_distribution(config)
    puzzle_training._check_puzzle_mix(config)
    assert puzzle_training._max_tail(config) == 4
    trainer = GraphedTrainer(
        model,
        _optimizer(model, torch.tensor(1e-3, device=DEVICE)),
        batch_size=4,
        max_tail=4,
        device=DEVICE,
        weights=weights,
    )
    games = GameSplit(game_corpus / "train").batch(np.arange(4))
    puzzles = PuzzleSplit(puzzle_corpus / "train").batch(np.arange(4))
    for _ in range(WARMUP_ITERATIONS):
        trainer.load(games)
        trainer.run(3, 4)
    randomizer = random.Random(0)
    draws = [puzzle_training._draw(config, randomizer) for _ in range(6)]
    # A prefix near depth_max replays the advance graph far past max_depth.
    for remaining, prefix, tail in [*draws, (None, 44, 4), (None, 0, 3)]:
        batch = games if remaining is None else puzzles
        reference = copy.deepcopy(model)
        expected, _ = puzzle_training._loss(reference, batch, prefix, tail, DEVICE, weights)
        trainer.load(batch)
        trainer.run(prefix, tail)
        torch.cuda.synchronize()
        assert float(trainer.loss) == pytest.approx(float(expected.detach()), rel=1e-3, abs=1e-3)
    assert trainer.graphs
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_graphed_initial_noise_perturbs_every_replay_of_the_training_graphs(
    game_corpus: Path,
) -> None:
    torch.manual_seed(7)
    model = WombatNetwork(SMALL).to(DEVICE).train()
    batch = GameSplit(game_corpus / "train").batch(np.arange(4))
    expected = {
        prefix: float(puzzle_training._loss(copy.deepcopy(model), batch, prefix, 2, DEVICE)[0])
        for prefix in (0, 3)
    }
    losses: dict[float, list[tuple[int, float]]] = {}
    for noise in (1e-4, 5.0):
        copied = copy.deepcopy(model)
        # A zero learning rate keeps the weights, so every iteration sees the same model.
        trainer = GraphedTrainer(
            copied,
            _optimizer(copied, torch.tensor(0.0, device=DEVICE)),
            batch_size=4,
            max_tail=2,
            device=DEVICE,
            initial_noise=noise,
        )
        assert trainer.buffers.initial_noise == noise
        losses[noise] = []
        for prefix in (0, 3) * (WARMUP_ITERATIONS + 2):
            trainer.load(batch)
            trainer.run(prefix, 2)
            torch.cuda.synchronize()
            losses[noise].append((prefix, float(trainer.loss)))
        assert trainer.graphs
    for prefix, loss in losses[1e-4]:
        assert loss == pytest.approx(expected[prefix], rel=1e-3, abs=1e-3)
    graphed = losses[5.0][WARMUP_ITERATIONS:]
    for prefix in (0, 3):
        replays = [loss for drawn, loss in graphed if drawn == prefix]
        assert all(loss != pytest.approx(expected[prefix], rel=1e-3) for loss in replays)
        assert replays[0] != replays[1]
    predictor = GraphedPredictor(model, 4, DEVICE)
    assert predictor.buffers.initial_noise == 0.0


def test_graphed_lognormal_noisy_run_trains_with_puzzles(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    config = PuzzleTrainingConfig(
        corpus=str(game_corpus),
        output=str(tmp_path / "run"),
        task="games",
        move_value_weight=0.5,
        value_weight=0.25,
        lookahead_weight=0.25,
        ema_decay=0.9,
        depth_distribution="lognormal",
        depth_mean=6.0,
        depth_max=20,
        max_gradient_steps=4,
        initial_noise=0.1,
        puzzle_corpus=str(puzzle_corpus),
        puzzle_share=0.5,
        puzzle_steps_per_move=2,
        steps=8,
        cooldown_steps=4,
        batch_size=4,
        warmup_steps=1,
        eval_every=4,
        eval_positions=8,
        eval_depths=(0, 1, 4, 8),
        device="cuda",
    )
    train_puzzles(config, V7)
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert [row["step"] for row in rows] == [4, 8]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    assert rows[1]["learning_rate"] < rows[0]["learning_rate"]


def test_graphed_mixed_games_run_logs_puzzle_validation(
    game_corpus: Path, puzzle_corpus: Path, tmp_path: Path
) -> None:
    config = PuzzleTrainingConfig(
        corpus=str(game_corpus),
        output=str(tmp_path / "run"),
        task="games",
        move_value_weight=0.5,
        value_weight=0.25,
        lookahead_weight=0.25,
        ema_decay=0.9,
        fixed_depth=4,
        puzzle_corpus=str(puzzle_corpus),
        puzzle_share=0.5,
        puzzle_steps_per_move=2,
        steps=8,
        batch_size=4,
        warmup_steps=1,
        eval_every=4,
        eval_positions=8,
        eval_depths=(0, 1, 4, 8),
        device="cuda",
    )
    train_puzzles(config, V7)
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert [row["step"] for row in rows] == [4, 8]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    assert all(row["puzzle_validation"] for row in rows)


def test_graphed_v7_games_run_serves_averaged_weights(game_corpus: Path, tmp_path: Path) -> None:
    config = PuzzleTrainingConfig(
        corpus=str(game_corpus),
        output=str(tmp_path / "run"),
        task="games",
        move_value_weight=0.5,
        value_weight=0.25,
        lookahead_weight=0.25,
        supervise_every_step=True,
        ema_decay=0.9,
        steps=8,
        batch_size=4,
        warmup_steps=1,
        max_depth=4,
        max_gradient_steps=2,
        eval_every=4,
        eval_positions=8,
        eval_depths=(0, 1, 4),
        device="cuda",
    )
    train_puzzles(config, V7)
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    assert [row["step"] for row in rows] == [4, 8]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    key = "recurrent_core.0.square_bias"
    assert not torch.equal(checkpoint["model"][key], checkpoint["raw_model"][key])
