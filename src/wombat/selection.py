"""Choosing a move from the policy readout, the per-move value readout, or a mix of both.

Move values are tanh outputs, or with value bins the expected value of their bins, trained on
``2 * win - 1``, where ``win`` is the move's 0..1 win chance. The policy target weights scored
moves by ``exp(100 * win / T)`` (T = 4), so its log probabilities are ``(50 / T) * value``
plus a constant. ``MIX_BETA`` = 50 / 4 therefore counts the value head once more at the scale
the policy already uses: in ``mix`` a value 0.08 higher (4 win points) offsets a policy log
probability 1 nat lower.

The child modes play each of the policy's top k moves and read the model's value of the
resulting position at the same depth. ``child`` picks the move whose child value is worst for
the opponent; ``child-mix`` adds ``beta`` times that negated child value to the policy log
probability. ``CHILD_MIX_BETA`` = 4 weighs the child value at about a third of the scale the
policy already uses: a child value 0.25 better (12.5 win points) offsets 1 nat.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

SELECT_MODES = ("policy", "value", "mix")
MIX_BETA = 12.5
CHILD_MODES = ("child", "child-mix")
CHILD_MIX_BETA = 4.0
CHILD_TOP_K = 3


def default_beta(select: str) -> float:
    return CHILD_MIX_BETA if select == "child-mix" else MIX_BETA


def recorded_beta(select: str, beta: float) -> float | None:
    """The beta a report records: only the blending modes use one."""

    return beta if select in ("mix", "child-mix") else None


def child_choice(log_probs: Tensor, child_scores: Tensor, select: str, beta: float) -> Tensor:
    """Column of the chosen candidate per row, over (batch, k) top-k candidates.

    ``log_probs`` are the candidates' policy log probabilities, -inf for padding past the legal
    moves; ``child_scores`` are their negated child values. Ties go to the likelier move.
    """

    if select not in CHILD_MODES:
        raise ValueError(f"unknown child selection {select!r}; choose one of {CHILD_MODES}")
    scores = child_scores.float() if select == "child" else log_probs + beta * child_scores
    return scores.masked_fill(log_probs.isneginf(), float("-inf")).argmax(dim=1)


def selection_scores(
    policy: Tensor,
    move_values: Tensor | None,
    legal: Tensor,
    select: str = "policy",
    beta: float = MIX_BETA,
) -> Tensor:
    """Per-action float scores whose argmax is the chosen move; illegal actions score -inf.

    ``policy`` scores the raw logits, ``value`` the move values, and ``mix`` the log softmax of
    the policy over legal moves plus ``beta`` times the move values.
    """

    logits = policy.float().masked_fill(~legal, float("-inf"))
    if select == "policy":
        return logits
    if select not in SELECT_MODES:
        raise ValueError(f"unknown move selection {select!r}; choose one of {SELECT_MODES}")
    if move_values is None:
        raise ValueError(f"{select} selection needs the per-move value readout")
    if select == "value":
        return move_values.float().masked_fill(~legal, float("-inf"))
    mixed = logits.log_softmax(dim=-1) + beta * move_values.float()
    return mixed.masked_fill(~legal, float("-inf"))


def recorded_move_value_weight(checkpoint: Mapping[str, Any], path: Path) -> float | None:
    """The run's ``move_value_weight``, from the checkpoint or its run's config.json.

    None means no training config was found. A config without the field trained no move values.
    """

    config = checkpoint.get("training_config")
    if config is None:
        run_config = path.parent / "config.json"
        if run_config.exists():
            config = json.loads(run_config.read_text(encoding="utf-8")).get("training_config")
    if config is None:
        return None
    return float(config.get("move_value_weight", 0.0))


def require_move_values(
    checkpoint: Mapping[str, Any], path: Path, parameters: Iterable[str], select: str
) -> None:
    """Raise unless ``checkpoint`` holds move-value weights that were trained.

    ``parameters`` are the state dict keys the network expects. Policy and child selection
    read no move values, so they pass.
    """

    if select == "policy" or select in CHILD_MODES:
        return
    if select not in SELECT_MODES:
        raise ValueError(f"unknown move selection {select!r}; choose one of {SELECT_MODES}")
    missing = [key for key in parameters if "move_value" in key and key not in checkpoint["model"]]
    if missing:
        raise ValueError(f"{select} selection needs a move-value head, and {path} has none")
    weight = recorded_move_value_weight(checkpoint, path)
    if weight is not None and weight <= 0:
        raise ValueError(
            f"{select} selection needs trained move values, and {path} was trained with "
            f"move_value_weight {weight:g}"
        )
