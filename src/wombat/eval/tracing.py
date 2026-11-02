"""Persist recurrent trajectories for policy, value, probe, and attention analysis."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch

from wombat.chess.encode import encode_position
from wombat.chess.moves import action_to_move, legal_action_mask
from wombat.model.adapter import state_vector
from wombat.model.network import WombatNetwork


def trace_positions(
    model: WombatNetwork,
    boards: list[chess.Board],
    reasoning_steps: int,
    destination: Path,
    *,
    top_k: int = 5,
) -> dict[str, object]:
    if not boards:
        raise ValueError("at least one board is required")
    device = next(model.parameters()).device
    encoded = [encode_position(board) for board in boards]
    pieces = torch.from_numpy(np.stack([item.pieces for item in encoded]).astype(np.int64)).to(device)
    states = torch.from_numpy(np.stack([state_vector(item) for item in encoded])).to(device)
    traces = model.trace_reasoning(pieces, states, reasoning_steps, capture_attention=True)

    summary_steps: list[dict[str, object]] = []
    tensor_steps: list[dict[str, object]] = []
    for trace in traces:
        positions: list[dict[str, object]] = []
        for index, board in enumerate(boards):
            logits = trace.policy_logits[index].numpy()
            mask = legal_action_mask(board)
            legal_logits = np.where(mask, logits, -np.inf)
            legal_indices = np.flatnonzero(mask)
            shifted = legal_logits[legal_indices] - np.max(legal_logits[legal_indices])
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            order = np.argsort(probabilities)[::-1][:top_k]
            positions.append(
                {
                    "fen": board.fen(en_passant="fen"),
                    "value": float(trace.value[index]),
                    "top_moves": [
                        {
                            "move": action_to_move(board, int(legal_indices[item])).uci(),
                            "probability": float(probabilities[item]),
                        }
                        for item in order
                    ],
                }
            )
        summary_steps.append({"step": trace.step, "positions": positions})
        latent_end = 65 + model.config.latent_tokens
        step_tensors: dict[str, object] = {
            "step": trace.step,
            "hidden": trace.hidden,
            "board_aligned": trace.hidden[:, :65],
            "scratchpad": trace.hidden[:, 65:latent_end],
            "policy_logits": trace.policy_logits,
            "value": trace.value,
            "attention_maps": trace.attention_maps,
        }
        if model.config.scratch_boards:
            # (positions, boards, 64 squares, width)
            step_tensors["scratch_boards"] = trace.hidden[:, latent_end:].unflatten(
                1, (model.config.scratch_boards, 64)
            )
        tensor_steps.append(step_tensors)

    destination.mkdir(parents=True, exist_ok=True)
    tensor_path = destination / "trajectory.pt"
    torch.save(
        {
            "schema": "wombat.reasoning-trajectory.v1",
            "fens": [board.fen(en_passant="fen") for board in boards],
            "steps": tensor_steps,
        },
        tensor_path,
    )
    summary: dict[str, object] = {
        "schema": "wombat.reasoning-trace-summary.v1",
        "reasoning_steps": reasoning_steps,
        "positions": len(boards),
        "tensor_file": tensor_path.name,
        "steps": summary_steps,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary

