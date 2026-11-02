from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import chess
import chess.engine
import numpy as np
import pytest
import torch

from wombat.chess.encode import play_encoding, state_vector
from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.eval import reply as reply_eval
from wombat.interpretability.explain import _legal_replies
from wombat.model import NetworkConfig, WombatNetwork

FAKE_UCI = Path(__file__).parent / "fixtures" / "fake_uci.py"
TINY = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=1,
    recurrent_blocks=1,
    latent_tokens=4,
    policy_readout="square",
    normalize_state=True,
    max_residual_update=None,
    lookahead_heads=True,
)


def _fake_reply(board: chess.Board) -> chess.Move:
    """The fake engine's reply to its own best move: the legal move landing furthest up."""

    after = board.copy()
    after.push(max(board.legal_moves, key=lambda move: move.to_square))
    return max(after.legal_moves, key=lambda move: move.to_square)


def _legal_hits(model: WombatNetwork, fens: list[str], steps: int) -> int:
    """Legal top-1 matches counted through the explainer's own reply ranking."""

    hits = 0
    for fen in fens:
        board = chess.Board(fen)
        encoded = play_encoding(board)
        pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
        state = torch.from_numpy(state_vector(encoded))[None]
        with torch.inference_mode():
            lookahead = model.lookahead(model(pieces, state, steps, include_hidden=True)[-1])
        best = max(board.legal_moves, key=lambda move: move.to_square)
        top = chess.Move.from_uci(_legal_replies(board, best, lookahead, count=1)[0]["uci"])
        expected = _fake_reply(board)
        hits += (top.from_square, top.to_square) == (expected.from_square, expected.to_square)
    return hits


@pytest.mark.parametrize("readout", ["squares", "pair"])
def test_evaluate_reply_reports_agreement_with_the_engine_reply(
    readout: str,
    game_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(reply_eval, "lower_priority", lambda: None)
    started: list[Any] = []
    popen = chess.engine.SimpleEngine.popen_uci

    def fake_engine(command: Any, **kwargs: Any) -> chess.engine.SimpleEngine:
        started.append(command)
        return popen([sys.executable, str(FAKE_UCI)], **kwargs)

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", fake_engine)
    config = replace(TINY, reply_readout=readout)
    torch.manual_seed(0)
    model = WombatNetwork(config).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(config)}, checkpoint)
    output = tmp_path / "reply.json"
    arguments = ["evaluate-reply", "--checkpoint", str(checkpoint)]
    arguments += ["--split", str(game_corpus / "validation"), "--positions", "10", "--steps", "2"]
    arguments += ["--engines", "2", "--stockfish", "stockfish", "--output", str(output)]
    arguments += ["--cache-dir", str(tmp_path / "cache")]
    assert main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert started == [["stockfish"], ["stockfish"]]
    stored = GameSplit(game_corpus / "validation").fens
    assert stored is not None
    fens = [str(fen) for fen in stored]
    assert report["schema"] == "wombat.reply-agreement.v1"
    assert report["reply_readout"] == readout and report["engine"] == "FakeFish"
    assert report["positions"] == report["evaluated"] == len(fens) == 3
    assert report["skipped"] == 0
    for key in ("reply_legal_top1", "reply_origin_top1", "reply_destination_top1", "policy_top1"):
        assert report[key]["positions"] == 3 and 0 <= report[key]["accuracy"] <= 1
    assert report["reply_legal_top1"]["correct"] == _legal_hits(model, fens, 2)
    assert set(report["reply_minus_policy"]) >= {"difference", "low", "high"}
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 3
    # A second run reads every line from the cache.
    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_evaluate_reply_defaults() -> None:
    args = build_parser().parse_args(["evaluate-reply", "--checkpoint", "run.pt"])
    assert (args.positions, args.nodes, args.steps) == (1_000, 200_000, 8)
    assert (args.device, args.max_vram_gb) == ("cpu", 1.0)
    assert args.split == Path("data/processed/games-v1/validation")
