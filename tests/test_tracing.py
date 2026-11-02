from pathlib import Path

import chess
import torch

from wombat.eval.tracing import trace_positions
from wombat.model import NetworkConfig, WombatNetwork


def test_trace_artifact_contains_every_state(tmp_path: Path) -> None:
    model = WombatNetwork(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            prelude_blocks=0,
            recurrent_blocks=1,
            mlp_width=48,
            latent_tokens=4,
        )
    ).eval()
    summary = trace_positions(model, [chess.Board()], 2, tmp_path, top_k=3)
    assert len(summary["steps"]) == 3
    assert len(summary["steps"][0]["positions"][0]["top_moves"]) == 3
    payload = torch.load(tmp_path / "trajectory.pt", weights_only=False)
    assert [step["step"] for step in payload["steps"]] == [0, 1, 2]
    assert payload["steps"][1]["attention_maps"][0].shape == (1, 6, 69, 134)
    assert payload["steps"][1]["scratchpad"].shape == (1, 4, 24)
    assert "scratch_boards" not in payload["steps"][1]


def test_trace_splits_scratch_boards_from_the_latent_tokens(tmp_path: Path) -> None:
    model = WombatNetwork(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            prelude_blocks=0,
            recurrent_blocks=1,
            mlp_width=48,
            latent_tokens=4,
            policy_readout="square",
            square_attention_bias=True,
            scratch_boards=2,
        )
    ).eval()
    trace_positions(model, [chess.Board()], 1, tmp_path)
    step = torch.load(tmp_path / "trajectory.pt", weights_only=False)["steps"][1]
    assert step["hidden"].shape == (1, 197, 24)
    assert step["scratchpad"].shape == (1, 4, 24)
    assert step["scratch_boards"].shape == (1, 2, 64, 24)
    assert torch.equal(step["scratch_boards"][0, 1, 5], step["hidden"][0, 69 + 64 + 5])
    assert step["attention_maps"][0].shape == (1, 6, 197, 262)

