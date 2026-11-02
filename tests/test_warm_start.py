from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from wombat.data.lichess import PuzzleRecord
from wombat.data.puzzle_corpus import solver_positions
from wombat.eval import warm_start
from wombat.eval.full_puzzles import evaluate_full_puzzles, export_puzzle_csv
from wombat.eval.warm_start import (
    WarmStart,
    evaluate_warm_start,
    solve_lines,
    warm_initial_hidden,
    warm_variants,
)
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


def _memory(model: WombatNetwork, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    pieces = torch.randint(0, 13, (2, 64), generator=generator)
    return model.encode_board(pieces, torch.rand((2, 26), generator=generator))


@torch.inference_mode()
def test_warm_starts_change_only_their_tokens() -> None:
    torch.manual_seed(0)
    model = WombatNetwork(TINY).eval()
    memory, previous_memory = _memory(model, 1), _memory(model, 2)
    previous_hidden = torch.randn(2, model.hidden_tokens, TINY.d_model)
    fresh = model.initial_hidden(memory)
    for warm in (WarmStart("fresh"), WarmStart("squares", 0.5), WarmStart("state")):
        assert torch.equal(warm_initial_hidden(model, memory, None, None, warm), fresh)
    assert torch.equal(
        warm_initial_hidden(model, memory, previous_memory, previous_hidden, WarmStart("fresh")),
        fresh,
    )
    squares = warm_initial_hidden(
        model, memory, previous_memory, previous_hidden, WarmStart("squares", 0.5)
    )
    change = previous_hidden[:, :64] - previous_memory[:, :64]
    torch.testing.assert_close(squares[:, :64], fresh[:, :64] + 0.5 * change)
    assert torch.equal(squares[:, 64:], fresh[:, 64:])
    state = warm_initial_hidden(model, memory, previous_memory, previous_hidden, WarmStart("state"))
    assert torch.equal(state[:, 64], previous_hidden[:, 64])
    assert torch.equal(state[:, :64], fresh[:, :64]) and torch.equal(state[:, 65:], fresh[:, 65:])
    with pytest.raises(ValueError):
        warm_initial_hidden(model, memory, None, previous_hidden, WarmStart("squares", 1.0))


def test_variant_names() -> None:
    names = [warm.name for warm in warm_variants((0.25, 1.0))]
    assert names == ["fresh", "squares 0.25", "squares 1", "state"]


def test_each_later_solver_move_receives_the_previous_final_state(
    monkeypatch: pytest.MonkeyPatch, puzzle_archive: tuple[Path, list[PuzzleRecord]]
) -> None:
    torch.manual_seed(0)
    model = WombatNetwork(TINY).eval()
    lines = [solver_positions(record) for record in puzzle_archive[1]]
    calls: list[tuple[int, torch.Tensor | None]] = []
    finals: list[torch.Tensor] = []
    original_step = model.recurrent_step

    def spy(
        model: WombatNetwork,
        memory: torch.Tensor,
        previous_memory: torch.Tensor | None,
        previous_hidden: torch.Tensor | None,
        warm: WarmStart,
    ) -> torch.Tensor:
        calls.append((len(memory), previous_hidden))
        return warm_initial_hidden(model, memory, previous_memory, previous_hidden, warm)

    def step(memory: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        result = original_step(memory, hidden)
        assert isinstance(result, torch.Tensor)
        finals.append(result)
        return result

    monkeypatch.setattr(warm_start, "warm_initial_hidden", spy)
    monkeypatch.setattr(model, "recurrent_step", step)
    solve_lines(
        model, lines, 2, WarmStart("state"), batch_size=len(lines), device=torch.device("cpu")
    )
    lengths = np.array([len(line) for line in lines])
    assert [count for count, _ in calls] == [
        int((lengths > move).sum()) for move in range(lengths.max())
    ]
    assert len(calls) > 1 and calls[0][1] is None
    for move in range(1, len(calls)):
        carried = calls[move][1]
        assert carried is not None
        # Two steps per solver move; the carried rows are the lines still going.
        rows = np.searchsorted(np.flatnonzero(lengths > move - 1), np.flatnonzero(lengths > move))
        assert torch.equal(carried, finals[2 * move - 1][rows])


def test_fresh_start_reproduces_full_puzzle_evaluation(
    puzzle_archive: tuple[Path, list[PuzzleRecord]], puzzle_corpus: Path, tmp_path: Path
) -> None:
    csv_path = tmp_path / "puzzles.csv"
    export_puzzle_csv(puzzle_corpus / "test", puzzle_archive[0], csv_path)
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {"model": WombatNetwork(TINY).state_dict(), "model_config": asdict(TINY)}, checkpoint
    )
    full = evaluate_full_puzzles(
        checkpoint, csv_path, (1, 3), output=tmp_path / "full", batch_size=2
    )
    report = evaluate_warm_start(
        checkpoint, csv_path, (1, 3), (0.5,), output=tmp_path / "warm", batch_size=2
    )
    results = report["results"]
    assert isinstance(results, dict) and isinstance(full["results"], dict)
    for depth in ("t1", "t3"):
        rows = results[depth]
        assert list(rows) == ["fresh", "squares 0.5", "state"]
        assert rows["fresh"]["solved"] == full["results"][depth]["solved"]
        assert rows["fresh"]["versus_fresh"]["difference"] == 0
    with (
        np.load(tmp_path / "warm" / "solved.npz") as saved,
        np.load(tmp_path / "full" / "solved.npz") as reference,
    ):
        assert np.array_equal(saved["t3_fresh"], reference["t3"])
        assert len(saved["t3_squares_0.5"]) == report["puzzles"]
