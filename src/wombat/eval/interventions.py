"""Mechanistic controls for testing whether useful history is carried in H."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from wombat.model.network import WombatNetwork, board_consequences, board_phase


@dataclass(frozen=True)
class InterventionOutputs:
    normal: tuple[Tensor, Tensor]
    reset: tuple[Tensor, Tensor]
    corrupted: tuple[Tensor, Tensor]
    earlier_state: tuple[Tensor, Tensor]
    swapped: tuple[Tensor, Tensor] | None


def advance(model: WombatNetwork, board_memory: Tensor, hidden: Tensor, steps: int) -> Tensor:
    if steps < 0:
        raise ValueError("steps cannot be negative")
    for _ in range(steps):
        hidden = model.recurrent_step(board_memory, hidden)
    return hidden


def evaluate_hidden_interventions(
    model: WombatNetwork,
    pieces: Tensor,
    state: Tensor,
    *,
    total_steps: int,
    midpoint: int,
    noise_std: float = 0.25,
    seed: int = 0,
) -> InterventionOutputs:
    """Return matched outputs for continuity-breaking intervention controls.

    A batch of at least two positions enables the cross-position swap control.
    This function deliberately reports raw outputs; chess-quality metrics are
    computed by the frozen evaluator so no intervention receives a bespoke
    success definition.
    """

    if not 0 < midpoint < total_steps:
        raise ValueError("midpoint must lie strictly inside total_steps")
    board_memory = model.encode_board(pieces, state)
    initial = model.initial_hidden(board_memory)
    at_midpoint = advance(model, board_memory, initial, midpoint)
    remaining = total_steps - midpoint

    normal_hidden = advance(model, board_memory, at_midpoint, remaining)
    reset_hidden = advance(model, board_memory, initial, remaining)

    generator = torch.Generator(device=at_midpoint.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        at_midpoint.shape,
        dtype=at_midpoint.dtype,
        device=at_midpoint.device,
        generator=generator,
    )
    corrupted_hidden = advance(
        model, board_memory, at_midpoint + noise * noise_std, remaining
    )

    earlier_step = max(0, midpoint // 2)
    earlier_hidden = advance(model, board_memory, initial, earlier_step)
    earlier_hidden = advance(model, board_memory, earlier_hidden, remaining)

    consequences, phase = board_consequences(board_memory), board_phase(board_memory)

    def read(hidden: Tensor) -> tuple[Tensor, Tensor]:
        tokens = board_memory.shape[1]
        output = model.heads(hidden, tokens, consequences=consequences, phase=phase)
        return output[0], output[1]

    swapped_output = None
    if pieces.shape[0] >= 2:
        swapped_hidden = torch.roll(at_midpoint, shifts=1, dims=0)
        swapped_hidden = advance(model, board_memory, swapped_hidden, remaining)
        swapped_output = read(swapped_hidden)

    return InterventionOutputs(
        normal=read(normal_hidden),
        reset=read(reset_hidden),
        corrupted=read(corrupted_hidden),
        earlier_state=read(earlier_hidden),
        swapped=swapped_output,
    )

