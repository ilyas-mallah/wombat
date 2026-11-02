"""Machine-local executable discovery."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import yaml


def load_executables(config_path: Path = Path("configs/local.yaml")) -> dict[str, str | None]:
    configured: dict[str, str | None] = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        configured = payload.get("executables", {})
    return {
        "stockfish": os.environ.get("WOMBAT_STOCKFISH")
        or configured.get("stockfish")
        or shutil.which("stockfish"),
        "cutechess_cli": os.environ.get("WOMBAT_CUTECHESS_CLI")
        or configured.get("cutechess_cli")
        or shutil.which("cutechess-cli"),
        "ordo": os.environ.get("WOMBAT_ORDO") or configured.get("ordo") or shutil.which("ordo"),
    }

