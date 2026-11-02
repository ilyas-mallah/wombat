"""Linear probes on square tokens, each position scored by a probe fitted on other positions.

Positions fall into two folds by group (a puzzle or a game position), so a probe never sees the
positions it scores. Binary probes read one square token and are scored by AUROC; square probes
score every square token and pick the highest, scored by accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

L2 = 1e-3
ITERATIONS = 100


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve with tied scores averaged, or None without both classes."""

    labels = np.asarray(labels, dtype=np.bool_)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    _, first, counts = np.unique(np.asarray(scores)[order], return_index=True, return_counts=True)
    ranks = np.empty(len(labels))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def folds(groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fold 0 or 1 per position, with every position of a group in the same fold."""

    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    chosen = np.random.default_rng(seed).permutation(len(unique)) < len(unique) // 2
    return chosen[inverse].astype(np.int64)


@dataclass(frozen=True)
class LinearProbe:
    """A linear score of standardized features."""

    mean: Tensor
    scale: Tensor
    weight: Tensor
    bias: Tensor

    def __call__(self, features: Tensor) -> Tensor:
        return ((features - self.mean) * self.scale) @ self.weight + self.bias


def _standardize(features: Tensor) -> tuple[Tensor, Tensor]:
    flat = features.reshape(-1, features.shape[-1])
    return flat.mean(0), 1 / (flat.std(0) + 1e-6)


def _fit(features: Tensor, loss: Callable[[Tensor], Tensor], bias: bool) -> LinearProbe:
    # Plain tensors even when the caller runs under inference mode, which autograd rejects.
    with torch.inference_mode(False), torch.enable_grad():
        features = features.clone()
        mean, scale = _standardize(features)
        standardized = (features - mean) * scale
        weight = torch.zeros(features.shape[-1], device=features.device, requires_grad=True)
        offset = torch.zeros((), device=features.device, requires_grad=bias)
        parameters = [weight, offset] if bias else [weight]
        optimizer = torch.optim.LBFGS(
            parameters, lr=1, max_iter=ITERATIONS, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            value = loss(standardized @ weight + offset) + L2 * weight.square().sum()
            value.backward()
            return value

        optimizer.step(closure)
    return LinearProbe(mean, scale, weight.detach(), offset.detach())


def fit_binary(features: Tensor, labels: Tensor) -> LinearProbe:
    """Logistic regression of (rows, width) features on boolean labels."""

    targets = labels.float()

    def loss(logits: Tensor) -> Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    return _fit(features, loss, bias=True)


def fit_square(features: Tensor, targets: Tensor) -> LinearProbe:
    """Softmax over each position's 64 squares of a shared linear score, (positions, 64, width)."""

    def loss(logits: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(logits, targets)

    return _fit(features, loss, bias=False)


def cross_fit_binary(
    tokens: np.ndarray,
    positions: np.ndarray,
    squares: np.ndarray,
    labels: np.ndarray,
    fold: np.ndarray,
    device: torch.device,
    *,
    shuffle: np.random.Generator | None = None,
) -> np.ndarray:
    """Out-of-fold logits for the rows (position, square) of (positions, 64, width) tokens.

    ``shuffle`` permutes the fitted labels within each fold, the shuffled-label control.
    """

    scores = np.full(len(labels), np.nan, dtype=np.float32)
    row_fold = fold[positions]
    for held in (0, 1):
        fitted, scored = row_fold != held, row_fold == held
        fitted_labels = labels[fitted]
        if not scored.any() or fitted_labels.all() or not fitted_labels.any():
            continue
        if shuffle is not None:
            fitted_labels = shuffle.permutation(fitted_labels)
        features = torch.from_numpy(tokens[positions[fitted], squares[fitted]]).to(device).float()
        probe = fit_binary(features, torch.from_numpy(fitted_labels).to(device))
        del features
        held_features = tokens[positions[scored], squares[scored]]
        with torch.inference_mode():
            logits = probe(torch.from_numpy(held_features).to(device).float())
        scores[scored] = logits.cpu().numpy()
    return scores


def cross_fit_square(
    tokens: np.ndarray,
    targets: np.ndarray,
    fold: np.ndarray,
    device: torch.device,
    *,
    shuffle: np.random.Generator | None = None,
) -> np.ndarray:
    """Out-of-fold (positions, 64) logits for target squares; a target of -1 is unknown."""

    logits = np.full((len(targets), 64), np.nan, dtype=np.float32)
    known = targets >= 0
    for held in (0, 1):
        fitted = known & (fold != held)
        scored = fold == held
        if not fitted.any() or not scored.any():
            continue
        fitted_targets = targets[fitted]
        if shuffle is not None:
            fitted_targets = shuffle.permutation(fitted_targets)
        features = torch.from_numpy(tokens[fitted]).to(device).float()
        probe = fit_square(features, torch.from_numpy(fitted_targets).to(device))
        del features
        with torch.inference_mode():
            scored_logits = probe(torch.from_numpy(tokens[scored]).to(device).float())
        logits[scored] = scored_logits.cpu().numpy()
    return logits


def square_accuracy(logits: np.ndarray, targets: np.ndarray) -> float | None:
    """Top-1 accuracy over positions with a known target and a score, or None without any."""

    scored = (targets >= 0) & ~np.isnan(logits).any(axis=1)
    if not scored.any():
        return None
    return float((logits[scored].argmax(axis=1) == targets[scored]).mean())


def prior_accuracy(targets: np.ndarray, fold: np.ndarray) -> float | None:
    """Accuracy of always naming the other fold's most common target square."""

    correct = total = 0
    for held in (0, 1):
        fitted = (targets >= 0) & (fold != held)
        scored = (targets >= 0) & (fold == held)
        if not fitted.any() or not scored.any():
            continue
        common = np.bincount(targets[fitted], minlength=64).argmax()
        correct += int((targets[scored] == common).sum())
        total += int(scored.sum())
    return correct / total if total else None
