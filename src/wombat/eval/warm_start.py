"""Carrying a solver move's thinking into the next solver move, zero shot (backlog item 16 in docs/plan.md).

Whole-puzzle solving as ``evaluate-full-puzzles`` scores it, except that H0 of every solver move
after the first can start from the previous move's final state. The solver keeps the side to
move, so both boards share one canonical frame and square i means the same square in both.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

import chess
import numpy as np
import torch
from torch import Tensor

from wombat.chess.moves import action_to_move
from wombat.data.puzzle_corpus import SolverPosition, lower_priority, solver_positions
from wombat.eval.full_puzzles import EncodedPositions, line_verdict, read_puzzle_csv, solving_report
from wombat.eval.puzzle_corpus_eval import load_checkpoint, paired_difference
from wombat.gpu_share import share_gpu
from wombat.model import WombatNetwork
from wombat.model.network import board_consequences, board_phase

WARM_ALPHAS = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class WarmStart:
    """How H0 of the next solver move uses the previous move's final state H and memory B.

    ``fresh`` ignores them. ``squares`` adds ``alpha`` times (H - B) on the 64 square tokens.
    ``state`` replaces the non-square state token with the previous final one.
    """

    kind: str
    alpha: float = 0.0

    @property
    def name(self) -> str:
        return f"squares {self.alpha:g}" if self.kind == "squares" else self.kind


def warm_variants(alphas: Iterable[float] = WARM_ALPHAS) -> list[WarmStart]:
    return [
        WarmStart("fresh"),
        *(WarmStart("squares", alpha) for alpha in alphas),
        WarmStart("state"),
    ]


def warm_initial_hidden(
    model: WombatNetwork,
    board_memory: Tensor,
    previous_memory: Tensor | None,
    previous_hidden: Tensor | None,
    warm: WarmStart,
) -> Tensor:
    """H0 for ``board_memory`` given the previous solver move's B and final H, when there is one."""

    hidden = model.initial_hidden(board_memory)
    if warm.kind == "fresh" or previous_hidden is None:
        return hidden
    if warm.kind == "squares":
        if previous_memory is None:
            raise ValueError("the squares warm start needs the previous board memory")
        change = previous_hidden[:, :64].float() - previous_memory[:, :64].float()
        squares = (hidden[:, :64].float() + warm.alpha * change).to(hidden.dtype)
        return torch.cat((squares, hidden[:, 64:]), dim=1)
    if warm.kind == "state":
        state = previous_hidden[:, 64:65].to(hidden.dtype)
        return torch.cat((hidden[:, :64], state, hidden[:, 65:]), dim=1)
    raise ValueError(f"unknown warm start {warm.kind!r}")


@torch.inference_mode()
def solve_lines(
    model: WombatNetwork,
    lines: Sequence[Sequence[SolverPosition]],
    depth: int,
    warm: WarmStart,
    *,
    batch_size: int = 512,
    device: torch.device,
) -> list[list[chess.Move]]:
    """The argmax after ``depth`` steps at every solver position, lines walked move by move.

    Every line's next position follows the puzzle's own line, whatever was chosen before.
    """

    model.eval()
    lengths = np.array([len(line) for line in lines])
    starts = np.cumsum(lengths) - lengths
    flat = [position for line in lines for position in line]
    encoded = EncodedPositions([position.board for position in flat])
    actions = np.zeros(len(flat), dtype=np.int64)
    for first in range(0, len(lines), batch_size):
        chunk = np.arange(first, min(first + batch_size, len(lines)))
        previous_memory: Tensor | None = None
        previous_hidden: Tensor | None = None
        for move in range(int(lengths[chunk].max())):
            active = np.flatnonzero(lengths[chunk] > move)
            indices = starts[chunk[active]] + move
            batch = encoded.batch(indices)
            legal = batch["legal_mask"].to(device)
            rows = torch.from_numpy(active).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                memory = model.encode_board(batch["pieces"].to(device), batch["state"].to(device))
                hidden = warm_initial_hidden(
                    model,
                    memory,
                    None if previous_memory is None else previous_memory[rows],
                    None if previous_hidden is None else previous_hidden[rows],
                    warm,
                )
                for _ in range(depth):
                    hidden = cast(Tensor, model.recurrent_step(memory, hidden))
                policy = model.heads(
                    hidden,
                    memory.shape[1],
                    consequences=board_consequences(memory),
                    phase=board_phase(memory),
                )[0]
            scores = policy.float().masked_fill(~legal, float("-inf"))
            actions[indices] = scores.argmax(dim=1).cpu().numpy()
            if previous_hidden is None:
                previous_memory = memory.new_zeros((len(chunk), *memory.shape[1:]))
                previous_hidden = hidden.new_zeros((len(chunk), *hidden.shape[1:]))
            assert previous_memory is not None
            previous_memory[rows] = memory.to(previous_memory.dtype)
            previous_hidden[rows] = hidden.to(previous_hidden.dtype)
    moves = [action_to_move(item.board, int(action)) for item, action in zip(flat, actions)]
    return [moves[start : start + length] for start, length in zip(starts, lengths)]


def evaluate_warm_start(
    checkpoint: Path,
    puzzles: Path,
    depths: Iterable[int] = (8, 16),
    alphas: Iterable[float] = WARM_ALPHAS,
    *,
    output: Path,
    batch_size: int = 512,
    device_name: str = "cpu",
    max_vram_gb: float | None = 1.0,
) -> dict[str, object]:
    """Whole-puzzle accuracy per warm start and depth, each paired with the fresh start."""

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    model = load_checkpoint(checkpoint, device)
    records = read_puzzle_csv(puzzles)
    lines = [solver_positions(record) for record in records]
    ratings = np.array([record.rating for record in records])
    several = np.array([len(line) > 1 for line in lines])
    variants = warm_variants(alphas)
    results: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {"rating": ratings}
    for depth in sorted(set(depths)):
        solved_by: dict[str, np.ndarray] = {}
        rows: dict[str, object] = {}
        for warm in variants:
            chosen = solve_lines(model, lines, depth, warm, batch_size=batch_size, device=device)
            verdicts = np.array([line_verdict(line, moves) for line, moves in zip(lines, chosen)])
            solved = verdicts[:, 0]
            solved_by[warm.name] = solved
            fresh = solved_by["fresh"]
            report = solving_report(ratings, solved, verdicts[:, 1])
            del report["by_rating"]
            rows[warm.name] = {
                **report,
                "versus_fresh": paired_difference(fresh, solved),
                "versus_fresh_several_moves": paired_difference(fresh[several], solved[several]),
            }
            arrays[f"t{depth}_{warm.name.replace(' ', '_')}"] = solved
        results[f"t{depth}"] = rows
    report_out: dict[str, object] = {
        "schema": "wombat.warm-start.v1",
        "checkpoint": str(checkpoint),
        "puzzles_csv": str(puzzles),
        "puzzles": len(records),
        "puzzles_with_several_moves": int(several.sum()),
        "positions": sum(len(line) for line in lines),
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report_out, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "solved.npz", **arrays)  # type: ignore[arg-type]
    return report_out
