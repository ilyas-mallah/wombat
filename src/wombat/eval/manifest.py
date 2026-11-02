"""Versioned, content-addressed position manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import chess


MANIFEST_SCHEMA = "wombat.position-manifest.v1"


@dataclass(frozen=True)
class PositionCase:
    id: str
    fen: str
    best_move: str
    tags: tuple[str, ...] = ()

    def validate(self) -> None:
        board = chess.Board(self.fen)
        move = chess.Move.from_uci(self.best_move)
        if move not in board.legal_moves:
            raise ValueError(f"case {self.id}: best move {self.best_move} is illegal")


@dataclass(frozen=True)
class PositionManifest:
    name: str
    positions: tuple[PositionCase, ...]
    schema: str = MANIFEST_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "positions": [asdict(case) for case in self.positions],
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported manifest schema: {self.schema}")
        ids: set[str] = set()
        for case in self.positions:
            if case.id in ids:
                raise ValueError(f"duplicate case id: {case.id}")
            ids.add(case.id)
            case.validate()

    def write(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "PositionManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            name=payload["name"],
            positions=tuple(
                PositionCase(
                    id=item["id"],
                    fen=item["fen"],
                    best_move=item["best_move"],
                    tags=tuple(item.get("tags", ())),
                )
                for item in payload["positions"]
            ),
            schema=payload["schema"],
        )
        manifest.validate()
        return manifest

