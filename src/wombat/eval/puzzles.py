"""Batched first-critical-move evaluation on frozen Lichess puzzles."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wombat.chess.encode import encode_position
from wombat.chess.moves import legal_action_mask, move_to_action
from wombat.eval.manifest import PositionManifest
from wombat.model import NetworkConfig, WombatNetwork
from wombat.model.adapter import state_vector


class PuzzleDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest: PositionManifest, limit: int | None = None) -> None:
        self.examples: list[dict[str, object]] = []
        for case in manifest.positions[:limit]:
            board = chess.Board(case.fen)
            encoded = encode_position(board)
            rating_tag = next((tag for tag in case.tags if tag.startswith("rating:")), "rating:0")
            self.examples.append(
                {
                    "pieces": encoded.pieces.astype(np.int64),
                    "state": state_vector(encoded),
                    "action": move_to_action(board, chess.Move.from_uci(case.best_move)),
                    "legal_mask": legal_action_mask(board),
                    "rating": int(rating_tag.split(":", 1)[1]),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


def _collate(examples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    return {
        "pieces": torch.from_numpy(np.stack([item["pieces"] for item in examples])),
        "state": torch.from_numpy(np.stack([item["state"] for item in examples])),
        "action": torch.tensor([item["action"] for item in examples], dtype=torch.long),
        "legal_mask": torch.from_numpy(np.stack([item["legal_mask"] for item in examples])),
        "rating": torch.tensor([item["rating"] for item in examples], dtype=torch.long),
    }


def _bucket(rating: int) -> str:
    if rating < 1500:
        return "<1500"
    if rating < 2000:
        return "1500-1999"
    if rating < 2500:
        return "2000-2499"
    if rating < 3000:
        return "2500-2999"
    return "3000+"


@torch.inference_mode()
def evaluate_puzzle_curve(
    checkpoint_path: Path,
    manifest_path: Path,
    depths: tuple[int, ...],
    *,
    batch_size: int = 128,
    device_name: str = "cuda",
    limit: int | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WombatNetwork(NetworkConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    device = torch.device(device_name)
    model.to(device).eval()
    manifest = PositionManifest.read(manifest_path)
    dataset = PuzzleDataset(manifest, limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    curve: list[dict[str, object]] = []
    for depth in depths:
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for batch in loader:
            pieces = batch["pieces"].to(device)
            state = batch["state"].to(device)
            actions = batch["action"].to(device)
            mask = batch["legal_mask"].to(device)
            policy, _ = model(pieces, state, depth)
            predicted = policy.masked_fill(~mask, torch.finfo(policy.dtype).min).argmax(dim=1)
            correct = (predicted == actions).cpu()
            for rating, success in zip(batch["rating"].tolist(), correct.tolist(), strict=True):
                totals[_bucket(rating)][1] += 1
                totals[_bucket(rating)][0] += int(success)
        correct_total = sum(item[0] for item in totals.values())
        count_total = sum(item[1] for item in totals.values())
        curve.append(
            {
                "reasoning_steps": depth,
                "positions": count_total,
                "correct": correct_total,
                "accuracy": correct_total / count_total,
                "rating_buckets": {
                    name: {"correct": item[0], "positions": item[1], "accuracy": item[0] / item[1]}
                    for name, item in sorted(totals.items())
                },
            }
        )
    return {
        "schema": "wombat.puzzle-curve.v1",
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.sha256,
        "curve": curve,
    }


def write_puzzle_curve(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
