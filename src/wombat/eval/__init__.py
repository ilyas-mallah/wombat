"""Frozen-suite evaluation infrastructure."""

from wombat.eval.agreement import AgreementResult, evaluate_agreement, select_move
from wombat.eval.manifest import PositionCase, PositionManifest

__all__ = [
    "AgreementResult",
    "PositionCase",
    "PositionManifest",
    "evaluate_agreement",
    "select_move",
]

