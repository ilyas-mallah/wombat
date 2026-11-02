"""Reproducible supervised overfit and baseline training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import time
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wombat.data.torch_dataset import EvaluationDataset, collate_examples
from wombat.model import NetworkConfig, WombatNetwork, count_parameters


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str
    output: str
    examples: int = 256
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    value_weight: float = 0.25
    move_value_weight: float = 0.25
    continuation_weight: float = 0.25
    policy_temperature: float = 0.75
    target_kind: str = "deep"
    reasoning_steps: int = 0
    seed: int = 0
    device: str = "cuda"


@dataclass(frozen=True)
class RecurrentTrainingConfig:
    dataset: str
    output: str
    initial_checkpoint: str
    examples: int = 512
    epochs: int = 2
    batch_size: int = 16
    effective_batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    value_weight: float = 0.25
    move_value_weight: float = 0.25
    continuation_weight: float = 0.25
    policy_temperature: float = 0.75
    initial_max_steps: int = 4
    final_max_steps: int = 8
    minimum_steps: int = 1
    proposal_auxiliary_weight: float = 0.0
    disagreement_weight: float = 0.0
    early_auxiliary_weight: float = 0.1
    progressive_weight: float = 0.5
    max_gradient_steps: int = 8
    improvement_weight: float = 0.1
    improvement_margin: float = 0.01
    improvement_fraction: float = 0.1
    improvement_eligibility_epsilon: float = 1e-3
    contraction_weight: float = 0.0
    contraction_ratio: float = 0.9
    use_scratchpad_backup: bool = False
    use_future_workspaces: bool = False
    future_workspace_weight: float = 1.0
    future_board_weight: float = 0.25
    future_value_weight: float = 1.0
    max_residual_update: float = 0.25
    reset_recurrent_to_identity: bool = True
    seed: int = 0
    device: str = "cuda"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _soft_policy_loss(
    masked_policy: torch.Tensor, policy_target: torch.Tensor
) -> torch.Tensor:
    return -(policy_target * nn.functional.log_softmax(masked_policy, dim=1)).sum(dim=1)


def _candidate_value_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    squared = (predicted - target).square() * mask
    return squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def train_supervised(config: TrainingConfig, model_config: NetworkConfig = NetworkConfig()) -> dict[str, object]:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    dataset = EvaluationDataset(
        Path(config.dataset), limit=config.examples, policy_temperature=config.policy_temperature
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = WombatNetwork(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "move_value": 0.0,
            "continuation": 0.0,
            "correct": 0,
            "seen": 0,
        }
        started = time.perf_counter()
        for batch in loader:
            pieces = batch["pieces"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)
            values = batch["value"].to(device, non_blocking=True)
            legal_mask = batch["legal_mask"].to(device, non_blocking=True)
            policy_target = batch["policy_target"].to(device, non_blocking=True)
            move_value_target = batch["move_value_target"].to(device, non_blocking=True)
            move_value_mask = batch["move_value_mask"].to(device, non_blocking=True)
            if config.target_kind == "shallow":
                policy_target = batch["shallow_policy_target"].to(device, non_blocking=True)
                move_value_target = batch["shallow_move_value_target"].to(
                    device, non_blocking=True
                )
                move_value_mask = batch["shallow_move_value_mask"].to(
                    device, non_blocking=True
                )
            elif config.target_kind != "deep":
                raise ValueError("target_kind must be 'deep' or 'shallow'")
            continuation_actions = batch["continuation_actions"].to(device, non_blocking=True)
            continuation_mask = batch["continuation_mask"].to(device, non_blocking=True)
            if config.target_kind == "shallow":
                continuation_actions = batch["shallow_continuation_actions"].to(
                    device, non_blocking=True
                )
                continuation_mask = batch["shallow_continuation_mask"].to(
                    device, non_blocking=True
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                policy, predicted_values, predicted_move_values, hidden = model(
                    pieces,
                    state,
                    config.reasoning_steps,
                    include_move_values=True,
                    include_hidden=True,
                )
                masked_policy = policy.masked_fill(~legal_mask, torch.finfo(policy.dtype).min)
                policy_loss = _soft_policy_loss(masked_policy, policy_target).mean()
                value_loss = nn.functional.mse_loss(predicted_values, values)
                move_value_loss = _candidate_value_loss(
                    predicted_move_values, move_value_target, move_value_mask
                ).mean()
                continuation_loss = model.continuation_loss(
                    hidden, continuation_actions, continuation_mask
                ).mean()
                loss = (
                    policy_loss
                    + config.value_weight * value_loss
                    + config.move_value_weight * move_value_loss
                    + config.continuation_weight * continuation_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            size = pieces.shape[0]
            totals["loss"] += float(loss.detach()) * size
            totals["policy"] += float(policy_loss.detach()) * size
            totals["value"] += float(value_loss.detach()) * size
            totals["move_value"] += float(move_value_loss.detach()) * size
            totals["continuation"] += float(continuation_loss.detach()) * size
            totals["correct"] += int((masked_policy.argmax(dim=1) == actions).sum())
            totals["seen"] += size

        seen = int(totals["seen"])
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / seen,
            "policy_loss": totals["policy"] / seen,
            "value_loss": totals["value"] / seen,
            "move_value_loss": totals["move_value"] / seen,
            "continuation_loss": totals["continuation"] / seen,
            "top1": totals["correct"] / seen,
            "seconds": time.perf_counter() - started,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": asdict(model_config),
                "training_config": asdict(config),
                "metrics": list(metrics),
            },
            output / f"checkpoint-epoch-{epoch}.pt",
        )

    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "metrics": metrics,
        },
        checkpoint,
    )
    report: dict[str, object] = {
        "schema": "wombat.training-report.v1",
        "parameters": count_parameters(model),
        "examples": len(dataset),
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "training_config": asdict(config),
        "model_config": asdict(model_config),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


@torch.inference_mode()
def evaluate_supervised(
    model: WombatNetwork,
    dataset: EvaluationDataset,
    *,
    reasoning_steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
        num_workers=0,
    )
    model.to(device).eval()
    total_policy = 0.0
    total_value = 0.0
    correct = 0
    seen = 0
    for batch in loader:
        pieces = batch["pieces"].to(device)
        state = batch["state"].to(device)
        actions = batch["action"].to(device)
        values = batch["value"].to(device)
        mask = batch["legal_mask"].to(device)
        policy, predicted_values = model(pieces, state, reasoning_steps)
        masked = policy.masked_fill(~mask, torch.finfo(policy.dtype).min)
        policy_loss = nn.functional.cross_entropy(masked, actions, reduction="sum")
        value_loss = nn.functional.mse_loss(predicted_values, values, reduction="sum")
        total_policy += float(policy_loss)
        total_value += float(value_loss)
        correct += int((masked.argmax(dim=1) == actions).sum())
        seen += pieces.shape[0]
    return {
        "positions": seen,
        "reasoning_steps": reasoning_steps,
        "policy_loss": total_policy / seen,
        "value_mse": total_value / seen,
        "top1": correct / seen,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    reasoning_steps: int,
    batch_size: int = 128,
    device_name: str = "cuda",
) -> dict[str, float | int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WombatNetwork(NetworkConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    dataset = EvaluationDataset(dataset_path)
    result = evaluate_supervised(
        model,
        dataset,
        reasoning_steps=reasoning_steps,
        batch_size=batch_size,
        device=torch.device(device_name),
    )
    result["checkpoint"] = str(checkpoint_path)
    result["dataset"] = str(dataset_path)
    return result


def _sample_log_uniform_depth(
    randomizer: random.Random, maximum: int, minimum: int = 1
) -> int:
    if minimum < 1 or maximum < minimum:
        raise ValueError("recurrent depth bounds must satisfy 1 <= minimum <= maximum")
    # Every integer is reachable; mass is approximately even per doubling.
    return min(
        maximum,
        max(
            minimum,
            int(
                math.exp(
                    randomizer.uniform(math.log(minimum), math.log(maximum + 1))
                )
            ),
        ),
    )


def _sample_progressive_segment(
    randomizer: random.Random, maximum: int, maximum_tail: int | None = None
) -> tuple[int, int]:
    """Sample an unobserved warm-up prefix and a gradient-carrying tail."""

    if maximum < 1:
        raise ValueError("maximum progressive depth must be positive")
    if maximum_tail is not None and maximum_tail < 1:
        raise ValueError("maximum gradient tail must be positive")
    prefix = randomizer.randrange(0, maximum)
    available = maximum - prefix
    tail = randomizer.randrange(1, min(available, maximum_tail or available) + 1)
    return prefix, tail


def _quality_from_hidden(
    model: WombatNetwork,
    hidden: torch.Tensor,
    board_tokens: int,
    *,
    legal_mask: torch.Tensor,
    policy_target: torch.Tensor,
    values: torch.Tensor,
    move_value_target: torch.Tensor,
    move_value_mask: torch.Tensor,
    continuation_actions: torch.Tensor,
    continuation_mask: torch.Tensor,
    value_weight: float,
    move_value_weight: float,
    continuation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the normal chess objective from an arbitrary recurrent state."""

    policy, predicted_values, predicted_move_values = model.heads(
        hidden, board_tokens, include_move_values=True
    )
    masked_policy = policy.masked_fill(~legal_mask, torch.finfo(policy.dtype).min)
    quality = (
        _soft_policy_loss(masked_policy, policy_target)
        + value_weight
        * nn.functional.mse_loss(predicted_values, values, reduction="none")
        + move_value_weight
        * _candidate_value_loss(predicted_move_values, move_value_target, move_value_mask)
        + continuation_weight
        * model.continuation_loss(hidden, continuation_actions, continuation_mask)
    )
    return quality, masked_policy


def _future_workspace_loss(
    model: WombatNetwork,
    hidden: torch.Tensor,
    actions: torch.Tensor,
    pieces: torch.Tensor,
    states: torch.Tensor,
    mask: torch.Tensor,
    move_value_target: torch.Tensor,
    *,
    board_weight: float,
    value_weight: float,
    board_tokens: int = 65,
) -> torch.Tensor:
    """Permutation-invariant supervision for compact imagined futures.

    Candidate slots are matched to teacher MultiPV lines using detached root
    action costs. The matched recurrent tokens must predict each action and its
    exact resulting board, making action/world consistency part of the policy
    computation rather than an independent continuation decoder.
    """

    (
        predicted_actions,
        predicted_pieces,
        predicted_states,
        predicted_values,
    ) = model.future_predictions(hidden, board_tokens)
    batch, slots, model_horizon, action_count = predicted_actions.shape
    lines = min(actions.shape[1], slots)
    horizon = min(actions.shape[2], model_horizon)
    target_actions = actions[:, :lines, :horizon]
    target_mask = mask[:, :lines, :horizon]

    # Pairwise trajectory losses for every predicted-slot/teacher-line pair.
    # Everything stays on the GPU; the former CPU greedy matching left the GPU
    # idle between many small synchronizations.
    action_log_probs = nn.functional.log_softmax(
        predicted_actions[:, :, :horizon].float(), dim=-1
    )
    action_loss = -action_log_probs[:, :, None].expand(
        -1, -1, lines, -1, -1
    ).gather(
        -1,
        target_actions[:, None, :, :, None].expand(-1, slots, -1, -1, -1),
    ).squeeze(-1)

    piece_log_probs = nn.functional.log_softmax(
        predicted_pieces[:, :, :horizon].float(), dim=-1
    )
    target_pieces = pieces[:, :lines, :horizon]
    piece_loss = -piece_log_probs[:, :, None].expand(
        -1, -1, lines, -1, -1, -1
    ).gather(
        -1,
        target_pieces.long()[:, None, :, :, :, None].expand(
            -1, slots, -1, -1, -1, -1
        ),
    ).squeeze(-1).mean(dim=-1)
    state_loss = nn.functional.binary_cross_entropy_with_logits(
        predicted_states[:, :, None, :horizon].float().expand(
            -1, -1, lines, -1, -1
        ),
        states[:, None, :lines, :horizon].float().expand(-1, slots, -1, -1, -1),
        reduction="none",
    ).mean(dim=-1)
    # Every root workspace also estimates the value of its own candidate. This
    # is the scalar used by the differentiable root backup, so tie it to the
    # corresponding MultiPV score instead of only a position-wide mean.
    target_root_values = move_value_target.gather(1, target_actions[:, :, 0])
    root_value_loss = (
        predicted_values[:, :, 0, None] - target_root_values[:, None, :]
    ).square()
    root_only = torch.zeros(horizon, device=hidden.device, dtype=torch.float32)
    root_only[0] = 1.0
    combined = (
        action_loss
        + board_weight * (piece_loss + state_loss)
        + value_weight * root_value_loss[:, :, :, None] * root_only
    )

    # A detached Sinkhorn transport provides a soft permutation-invariant
    # one-to-one assignment. Missing teacher lines become zero-cost dummy
    # columns, allowing all candidate slots to remain interchangeable.
    square_logits = torch.zeros(
        batch, slots, slots, device=hidden.device, dtype=torch.float32
    )
    square_logits[:, :, :lines] = -combined[:, :, :, 0].detach()
    assignment = torch.exp(square_logits - square_logits.amax(dim=(1, 2), keepdim=True))
    for _ in range(8):
        assignment = assignment / assignment.sum(dim=2, keepdim=True).clamp_min(1e-8)
        assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-8)
    assignment = assignment[:, :, :lines].detach()
    valid = target_mask[:, None].float()
    weighted = combined * assignment[:, :, :, None] * valid
    return weighted.sum(dim=(1, 2, 3)) / target_mask.float().sum(
        dim=(1, 2)
    ).clamp_min(1)


def _continued_improvement_loss(
    final_quality: torch.Tensor,
    early_quality: torch.Tensor,
    *,
    margin: float,
    fraction: float,
    eligibility_epsilon: float,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize per-position regressions while treating the earlier state as a target.

    The desired gain shrinks for already-solved examples, so saturated easy
    positions are not asked to achieve an impossible fixed improvement.
    """

    if final_quality.shape != early_quality.shape or final_quality.ndim != 1:
        raise ValueError("quality tensors must be matching one-dimensional batches")
    early_target = early_quality.detach()
    desired = torch.minimum(
        torch.full_like(early_target, margin), early_target.clamp_min(0) * fraction
    )
    eligible = early_target > eligibility_epsilon
    penalties = torch.relu(final_quality - early_target + desired)
    eligible_float = eligible.to(penalties.dtype)
    weights = torch.ones_like(penalties) if sample_weight is None else sample_weight
    loss = (penalties * eligible_float * weights).sum() / (
        eligible_float * weights
    ).sum().clamp_min(1)
    return loss, {
        "eligible": eligible.sum(),
        "improved": ((final_quality.detach() < early_target) & eligible).sum(),
        "satisfied": ((final_quality.detach() + desired <= early_target) & eligible).sum(),
        "quality_delta_sum": ((early_target - final_quality.detach()) * eligible_float).sum(),
    }


def _recurrent_contraction_loss(
    early_previous: torch.Tensor,
    early_hidden: torch.Tensor,
    final_previous: torch.Tensor,
    final_hidden: torch.Tensor,
    *,
    steps_apart: int,
    ratio: float,
) -> torch.Tensor:
    """Make successive updates decay without suppressing the earlier update.

    The early update norm is detached, so this objective cannot be satisfied by
    collapsing useful early computation. It only closes later updates that fail
    to contract along the same recurrent trajectory.
    """

    if not 0 < ratio <= 1:
        raise ValueError("contraction ratio must be in (0, 1]")
    if steps_apart < 1:
        raise ValueError("steps_apart must be positive")
    early_update = (early_hidden - early_previous).float().square().mean(
        dim=(1, 2)
    ).sqrt()
    final_update = (final_hidden - final_previous).float().square().mean(
        dim=(1, 2)
    ).sqrt()
    allowed = early_update.detach() * (ratio**steps_apart)
    return torch.relu(final_update - allowed)


def train_recurrent(config: RecurrentTrainingConfig) -> dict[str, object]:
    _seed_everything(config.seed)
    if config.effective_batch_size % config.batch_size:
        raise ValueError("effective_batch_size must be divisible by batch_size")
    if not 0 <= config.progressive_weight <= 1:
        raise ValueError("progressive_weight must be in [0, 1]")
    if config.max_gradient_steps < 1:
        raise ValueError("max_gradient_steps must be positive")
    device = torch.device(config.device)
    initial = torch.load(config.initial_checkpoint, map_location="cpu", weights_only=False)
    # Residual bounding changes dynamics but has no parameters, so an H0
    # checkpoint can safely opt into the current recurrent-training limit.
    model_config = replace(
        NetworkConfig(**initial["model_config"]),
        max_residual_update=config.max_residual_update,
        use_scratchpad_backup=config.use_scratchpad_backup,
        use_future_workspaces=config.use_future_workspaces,
    )
    model = WombatNetwork(model_config)
    model.load_state_dict(initial["model"], strict=False)
    if config.reset_recurrent_to_identity:
        # H0 baseline training never exercises these branches; retaining their
        # random checkpoint values would make "more compute" harmful at step 1.
        model.reset_recurrent_to_identity()
    model.to(device)
    dataset = EvaluationDataset(
        Path(config.dataset), limit=config.examples, policy_temperature=config.policy_temperature
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    accumulation = config.effective_batch_size // config.batch_size
    randomizer = random.Random(config.seed)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        progress = (epoch - 1) / max(1, config.epochs - 1)
        maximum = max(
            config.initial_max_steps,
            round(
                config.initial_max_steps
                * (config.final_max_steps / config.initial_max_steps) ** progress
            ),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "correct": 0,
            "seen": 0,
            "depth_sum": 0,
            "gap_sum": 0,
            "compared": 0,
            "eligible": 0,
            "improved": 0,
            "satisfied": 0,
            "quality_delta_sum": 0.0,
            "contraction_loss": 0.0,
            "progressive_loss": 0.0,
            "progressive_prefix_sum": 0,
            "progressive_tail_sum": 0,
        }
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            depth = _sample_log_uniform_depth(randomizer, maximum, config.minimum_steps)
            progressive_prefix, progressive_tail = _sample_progressive_segment(
                randomizer, maximum, config.max_gradient_steps
            )
            pieces = batch["pieces"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)
            values = batch["value"].to(device, non_blocking=True)
            legal_mask = batch["legal_mask"].to(device, non_blocking=True)
            policy_target = batch["policy_target"].to(device, non_blocking=True)
            move_value_target = batch["move_value_target"].to(device, non_blocking=True)
            move_value_mask = batch["move_value_mask"].to(device, non_blocking=True)
            shallow_policy_target = batch["shallow_policy_target"].to(
                device, non_blocking=True
            )
            shallow_move_value_target = batch["shallow_move_value_target"].to(
                device, non_blocking=True
            )
            shallow_move_value_mask = batch["shallow_move_value_mask"].to(
                device, non_blocking=True
            )
            deep_weight = 1.0 + config.disagreement_weight * batch[
                "shallow_deep_disagree"
            ].to(device, non_blocking=True).float()
            continuation_actions = batch["continuation_actions"].to(device, non_blocking=True)
            continuation_mask = batch["continuation_mask"].to(device, non_blocking=True)
            continuation_pieces = batch["continuation_pieces"].to(
                device, non_blocking=True
            )
            continuation_states = batch["continuation_states"].to(
                device, non_blocking=True
            )
            shallow_continuation_actions = batch["shallow_continuation_actions"].to(
                device, non_blocking=True
            )
            shallow_continuation_mask = batch["shallow_continuation_mask"].to(
                device, non_blocking=True
            )
            # The comparison location is randomized too: no midpoint becomes a
            # privileged implicit phase of the recurrent computation.
            early_step = randomizer.randint(1, depth - 1) if depth > 1 else 0
            requested_steps = {early_step}
            if config.contraction_weight > 0 and depth > 1:
                requested_steps.update((early_step - 1, depth - 1))
            if config.proposal_auxiliary_weight > 0:
                requested_steps.add(1)
            intermediate = frozenset(requested_steps)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                # In progressive-only mode this trajectory is diagnostic only.
                # Its graph would otherwise grow with the maximum horizon even
                # though its objective has zero weight.
                with torch.set_grad_enabled(config.progressive_weight < 1.0):
                    outputs = model(
                        pieces,
                        state,
                        depth,
                        intermediate_steps=intermediate,
                        include_move_values=True,
                        include_hidden=True,
                    )
                early_policy, early_values, early_move_values, early_hidden = outputs[early_step]
                policy, predicted_values, predicted_move_values, hidden = outputs[depth]
                early_masked = early_policy.masked_fill(
                    ~legal_mask, torch.finfo(early_policy.dtype).min
                )
                early_policy_loss = _soft_policy_loss(early_masked, policy_target)
                early_value_loss = nn.functional.mse_loss(
                    early_values, values, reduction="none"
                )
                early_move_value_loss = _candidate_value_loss(
                    early_move_values, move_value_target, move_value_mask
                )
                early_continuation_loss = model.continuation_loss(
                    early_hidden, continuation_actions, continuation_mask
                )
                masked_policy = policy.masked_fill(~legal_mask, torch.finfo(policy.dtype).min)
                policy_loss = _soft_policy_loss(masked_policy, policy_target)
                value_loss = nn.functional.mse_loss(predicted_values, values, reduction="none")
                move_value_loss = _candidate_value_loss(
                    predicted_move_values, move_value_target, move_value_mask
                )
                continuation_loss = model.continuation_loss(
                    hidden, continuation_actions, continuation_mask
                )
                final_quality = (
                    policy_loss
                    + config.value_weight * value_loss
                    + config.move_value_weight * move_value_loss
                    + config.continuation_weight * continuation_loss
                )
                if config.proposal_auxiliary_weight > 0:
                    proposal_policy, _, proposal_move_values, proposal_hidden = outputs[1]
                    proposal_masked = proposal_policy.masked_fill(
                        ~legal_mask, torch.finfo(proposal_policy.dtype).min
                    )
                    proposal_loss = _soft_policy_loss(
                        proposal_masked, shallow_policy_target
                    ) + config.move_value_weight * _candidate_value_loss(
                        proposal_move_values,
                        shallow_move_value_target,
                        shallow_move_value_mask,
                    )
                    proposal_loss = proposal_loss + config.continuation_weight * (
                        model.continuation_loss(
                            proposal_hidden,
                            shallow_continuation_actions,
                            shallow_continuation_mask,
                        )
                    )
                else:
                    proposal_loss = final_quality * 0
                early_quality = (
                    early_policy_loss
                    + config.value_weight * early_value_loss
                    + config.move_value_weight * early_move_value_loss
                    + config.continuation_weight * early_continuation_loss
                )
                improvement, improvement_stats = _continued_improvement_loss(
                    final_quality,
                    early_quality,
                    margin=config.improvement_margin,
                    fraction=config.improvement_fraction,
                    eligibility_epsilon=config.improvement_eligibility_epsilon,
                    sample_weight=deep_weight,
                )
                if config.contraction_weight > 0 and depth > 1:
                    contraction = _recurrent_contraction_loss(
                        outputs[early_step - 1][-1],
                        early_hidden,
                        outputs[depth - 1][-1],
                        hidden,
                        steps_apart=depth - early_step,
                        ratio=config.contraction_ratio,
                    )
                else:
                    contraction = final_quality * 0
                # Incremental-progress training from a random detached state.
                # Prefix/tail counts are deliberately never exposed to the model.
                progressive_memory = model.encode_board(pieces, state)
                with torch.no_grad():
                    progressive_hidden = model.initial_hidden(progressive_memory.detach())
                    for _ in range(progressive_prefix):
                        progressive_hidden = model.recurrent_step(
                            progressive_memory.detach(), progressive_hidden
                        )
                progressive_hidden = progressive_hidden.detach()
                progressive_start_hidden = progressive_hidden
                for _ in range(progressive_tail):
                    progressive_hidden = model.recurrent_step(
                        progressive_memory, progressive_hidden
                    )
                progressive_quality, _ = _quality_from_hidden(
                    model,
                    progressive_hidden,
                    progressive_memory.shape[1],
                    legal_mask=legal_mask,
                    policy_target=policy_target,
                    values=values,
                    move_value_target=move_value_target,
                    move_value_mask=move_value_mask,
                    continuation_actions=continuation_actions,
                    continuation_mask=continuation_mask,
                    value_weight=config.value_weight,
                    move_value_weight=config.move_value_weight,
                    continuation_weight=config.continuation_weight,
                )
                if config.use_future_workspaces:
                    future_loss = _future_workspace_loss(
                        model,
                        hidden,
                        continuation_actions,
                        continuation_pieces,
                        continuation_states,
                        continuation_mask,
                        move_value_target,
                        board_weight=config.future_board_weight,
                        value_weight=config.future_value_weight,
                    )
                    early_future_loss = _future_workspace_loss(
                        model,
                        early_hidden,
                        continuation_actions,
                        continuation_pieces,
                        continuation_states,
                        continuation_mask,
                        move_value_target,
                        board_weight=config.future_board_weight,
                        value_weight=config.future_value_weight,
                    )
                    progressive_future_loss = _future_workspace_loss(
                        model,
                        progressive_hidden,
                        continuation_actions,
                        continuation_pieces,
                        continuation_states,
                        continuation_mask,
                        move_value_target,
                        board_weight=config.future_board_weight,
                        value_weight=config.future_value_weight,
                    )
                    progressive_start_quality, _ = _quality_from_hidden(
                        model,
                        progressive_start_hidden,
                        progressive_memory.shape[1],
                        legal_mask=legal_mask,
                        policy_target=policy_target,
                        values=values,
                        move_value_target=move_value_target,
                        move_value_mask=move_value_mask,
                        continuation_actions=continuation_actions,
                        continuation_mask=continuation_mask,
                        value_weight=config.value_weight,
                        move_value_weight=config.move_value_weight,
                        continuation_weight=config.continuation_weight,
                    )
                    progressive_start_future_loss = _future_workspace_loss(
                        model,
                        progressive_start_hidden,
                        continuation_actions,
                        continuation_pieces,
                        continuation_states,
                        continuation_mask,
                        move_value_target,
                        board_weight=config.future_board_weight,
                        value_weight=config.future_value_weight,
                    )
                    mixed_future_loss = (
                        (1.0 - config.progressive_weight) * future_loss.mean()
                        + config.progressive_weight * progressive_future_loss.mean()
                    )
                    full_improvement, full_improvement_stats = _continued_improvement_loss(
                        final_quality + config.future_workspace_weight * future_loss,
                        early_quality + config.future_workspace_weight * early_future_loss,
                        margin=config.improvement_margin,
                        fraction=config.improvement_fraction,
                        eligibility_epsilon=config.improvement_eligibility_epsilon,
                        sample_weight=deep_weight,
                    )
                    progressive_improvement, _ = _continued_improvement_loss(
                        progressive_quality
                        + config.future_workspace_weight * progressive_future_loss,
                        progressive_start_quality
                        + config.future_workspace_weight * progressive_start_future_loss,
                        margin=config.improvement_margin,
                        fraction=config.improvement_fraction,
                        eligibility_epsilon=config.improvement_eligibility_epsilon,
                    )
                    improvement = (
                        progressive_improvement
                        if config.progressive_weight == 1.0
                        else 0.5 * (full_improvement + progressive_improvement)
                    )
                    improvement_stats = full_improvement_stats
                else:
                    mixed_future_loss = final_quality.mean() * 0
                total_loss = (
                    (1.0 - config.progressive_weight)
                    * (final_quality * deep_weight).sum()
                    / deep_weight.sum()
                    + config.early_auxiliary_weight * early_quality.mean()
                    + config.progressive_weight * progressive_quality.mean()
                    + config.future_workspace_weight * mixed_future_loss
                    + config.improvement_weight * improvement
                    + config.proposal_auxiliary_weight * proposal_loss.mean()
                    + config.contraction_weight * contraction.mean()
                )
                loss = total_loss / accumulation
            scaler.scale(loss).backward()
            if batch_index % accumulation == 0 or batch_index == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            size = pieces.shape[0]
            totals["loss"] += float(total_loss.detach()) * size
            totals["correct"] += int((masked_policy.argmax(dim=1) == actions).sum())
            totals["seen"] += size
            totals["depth_sum"] += depth * size
            totals["compared"] += size
            totals["gap_sum"] += (depth - early_step) * size
            for name in ("eligible", "improved", "satisfied"):
                totals[name] += int(improvement_stats[name])
            totals["quality_delta_sum"] += float(improvement_stats["quality_delta_sum"])
            totals["contraction_loss"] += float(contraction.detach().sum())
            totals["progressive_loss"] += float(progressive_quality.detach().sum())
            totals["progressive_prefix_sum"] += progressive_prefix * size
            totals["progressive_tail_sum"] += progressive_tail * size
        seen = int(totals["seen"])
        row = {
            "epoch": epoch,
            "maximum_depth": maximum,
            "mean_sampled_depth": totals["depth_sum"] / seen,
            "mean_comparison_gap": totals["gap_sum"] / max(1, totals["compared"]),
            "loss": totals["loss"] / seen,
            "top1": totals["correct"] / seen,
            "improvement_eligible": totals["eligible"],
            "fraction_improved": totals["improved"] / max(1, totals["eligible"]),
            "fraction_margin_satisfied": totals["satisfied"] / max(1, totals["eligible"]),
            "mean_quality_gain": totals["quality_delta_sum"] / max(1, totals["eligible"]),
            "mean_contraction_loss": totals["contraction_loss"] / seen,
            "mean_progressive_loss": totals["progressive_loss"] / seen,
            "mean_progressive_prefix": totals["progressive_prefix_sum"] / seen,
            "mean_progressive_tail": totals["progressive_tail_sum"] / seen,
            "seconds": time.perf_counter() - started,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": asdict(model_config),
                "training_config": asdict(config),
                "metrics": list(metrics),
            },
            output / f"checkpoint-epoch-{epoch}.pt",
        )

    checkpoint = output / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "metrics": metrics,
        },
        checkpoint,
    )
    report: dict[str, object] = {
        "schema": "wombat.recurrent-training-report.v1",
        "parameters": count_parameters(model),
        "examples": len(dataset),
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "training_config": asdict(config),
        "model_config": asdict(model_config),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
