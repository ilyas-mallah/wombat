"""Per-position held-out quality curves across recurrent compute budgets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wombat.data.torch_dataset import EvaluationDataset, collate_examples
from wombat.model import NetworkConfig, WombatNetwork
from wombat.training import _candidate_value_loss, _soft_policy_loss


def classify_quality_trajectory(quality: np.ndarray, tolerance: float = 1e-3) -> str:
    """Classify one lower-is-better quality sequence without chess assumptions."""

    deltas = quality[:-1] - quality[1:]
    positive = deltas > tolerance
    negative = deltas < -tolerance
    net = quality[0] - quality[-1]
    if positive.any() and negative.any():
        return "oscillation"
    if net > tolerance:
        return "monotonic_improvement"
    if net < -tolerance:
        return "monotonic_degradation"
    return "flat"


@torch.inference_mode()
def evaluate_quality_curve(
    checkpoint_path: Path,
    dataset_path: Path,
    depths: tuple[int, ...],
    *,
    value_weight: float = 0.25,
    move_value_weight: float = 0.25,
    continuation_weight: float = 0.25,
    policy_temperature: float = 0.75,
    batch_size: int = 128,
    device_name: str = "cuda",
    limit: int | None = None,
) -> dict[str, object]:
    if not depths or any(depth < 0 for depth in depths):
        raise ValueError("depths must contain non-negative integers")
    depths = tuple(sorted(set(depths)))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WombatNetwork(NetworkConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    device = torch.device(device_name)
    model.to(device).eval()
    dataset = EvaluationDataset(
        dataset_path, limit=limit, policy_temperature=policy_temperature
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples
    )
    quality_parts: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    policy_parts: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    value_parts: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    correct_parts: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    continuation_parts: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    move_value_parts: dict[int, list[np.ndarray]] = {depth: [] for depth in depths}
    disagreement_parts: list[np.ndarray] = []

    for batch in loader:
        pieces = batch["pieces"].to(device)
        state = batch["state"].to(device)
        actions = batch["action"].to(device)
        values = batch["value"].to(device)
        legal = batch["legal_mask"].to(device)
        maximum = depths[-1]
        requested = frozenset(depth for depth in depths if depth != maximum)
        policy_target = batch["policy_target"].to(device)
        move_value_target = batch["move_value_target"].to(device)
        move_value_mask = batch["move_value_mask"].to(device)
        continuation_actions = batch["continuation_actions"].to(device)
        continuation_mask = batch["continuation_mask"].to(device)
        disagreement_parts.append(batch["shallow_deep_disagree"].numpy())
        outputs = model(
            pieces,
            state,
            maximum,
            intermediate_steps=requested,
            include_move_values=True,
            include_hidden=True,
        )
        if not isinstance(outputs, dict):
            outputs = {maximum: outputs}
        for depth in depths:
            policy, predicted_value, predicted_move_values, hidden = outputs[depth]
            masked = policy.masked_fill(~legal, torch.finfo(policy.dtype).min)
            policy_loss = _soft_policy_loss(masked, policy_target)
            value_loss = nn.functional.mse_loss(predicted_value, values, reduction="none")
            move_value_loss = _candidate_value_loss(
                predicted_move_values, move_value_target, move_value_mask
            )
            continuation_loss = model.continuation_loss(
                hidden, continuation_actions, continuation_mask
            )
            quality = (
                policy_loss
                + value_weight * value_loss
                + move_value_weight * move_value_loss
                + continuation_weight * continuation_loss
            )
            quality_parts[depth].append(quality.float().cpu().numpy())
            policy_parts[depth].append(policy_loss.float().cpu().numpy())
            value_parts[depth].append(value_loss.float().cpu().numpy())
            correct_parts[depth].append((masked.argmax(1) == actions).cpu().numpy())
            continuation_parts[depth].append(continuation_loss.float().cpu().numpy())
            move_value_parts[depth].append(move_value_loss.float().cpu().numpy())

    quality = np.stack(
        [np.concatenate(quality_parts[depth]) for depth in depths], axis=1
    )
    curve = []
    for column, depth in enumerate(depths):
        curve.append(
            {
                "reasoning_steps": depth,
                "quality_loss": float(quality[:, column].mean()),
                "policy_loss": float(np.concatenate(policy_parts[depth]).mean()),
                "value_mse": float(np.concatenate(value_parts[depth]).mean()),
                "continuation_loss": float(
                    np.concatenate(continuation_parts[depth]).mean()
                ),
                "move_value_loss": float(np.concatenate(move_value_parts[depth]).mean()),
                "top1": float(np.concatenate(correct_parts[depth]).mean()),
            }
        )
    transitions = []
    for column in range(1, len(depths)):
        gain = quality[:, column - 1] - quality[:, column]
        transitions.append(
            {
                "from_steps": depths[column - 1],
                "to_steps": depths[column],
                "mean_quality_gain": float(gain.mean()),
                "fraction_improved": float((gain > 1e-3).mean()),
                "fraction_regressed": float((gain < -1e-3).mean()),
                "fraction_flat": float((np.abs(gain) <= 1e-3).mean()),
            }
        )
    category_counts: dict[str, int] = {}
    for row in quality:
        category = classify_quality_trajectory(row)
        category_counts[category] = category_counts.get(category, 0) + 1

    disagreement = np.concatenate(disagreement_parts).astype(bool)
    groups = {}
    for name, selected in (
        ("shallow_deep_agree", ~disagreement),
        ("shallow_deep_disagree", disagreement),
    ):
        groups[name] = {
            "positions": int(selected.sum()),
            "curve": [
                {
                    "reasoning_steps": depth,
                    "quality_loss": float(quality[selected, column].mean()),
                    "policy_loss": float(
                        np.concatenate(policy_parts[depth])[selected].mean()
                    ),
                    "top1": float(np.concatenate(correct_parts[depth])[selected].mean()),
                }
                for column, depth in enumerate(depths)
            ],
        }

    # Initial-loss quartiles provide model-relative difficulty buckets and do
    # not inject a handcrafted chess concept into training.
    quartile_edges = np.quantile(quality[:, 0], [0.25, 0.5, 0.75])
    difficulty = {}
    bucket_ids = np.digitize(quality[:, 0], quartile_edges)
    for bucket in range(4):
        selected = bucket_ids == bucket
        difficulty[f"q{bucket + 1}"] = {
            "positions": int(selected.sum()),
            "initial_quality_loss": float(quality[selected, 0].mean()),
            "final_quality_loss": float(quality[selected, -1].mean()),
            "mean_quality_gain": float(
                (quality[selected, 0] - quality[selected, -1]).mean()
            ),
        }
    return {
        "schema": "wombat.quality-curve.v1",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "positions": len(dataset),
        "depths": list(depths),
        "curve": curve,
        "transitions": transitions,
        "trajectory_categories": category_counts,
        "teacher_disagreement_groups": groups,
        "initial_quality_quartiles": difficulty,
    }


def write_quality_curve(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
