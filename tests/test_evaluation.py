from dataclasses import asdict

import chess

from wombat.eval import PositionCase, PositionManifest, evaluate_agreement
from wombat.policy import FixturePolicy


def test_fixture_policy_scores_exactly() -> None:
    board = chess.Board()
    manifest = PositionManifest(
        name="fixture",
        positions=(PositionCase(id="start", fen=board.fen(), best_move="e2e4"),),
    )
    result = evaluate_agreement(FixturePolicy({board.fen(): "e2e4"}), manifest, 8)
    assert result.accuracy == 1.0
    assert result.correct == 1
    assert asdict(result)["manifest_sha256"] == manifest.sha256


def test_manifest_round_trip(tmp_path) -> None:
    manifest = PositionManifest(
        name="fixture",
        positions=(PositionCase(id="start", fen=chess.STARTING_FEN, best_move="e2e4"),),
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    restored = PositionManifest.read(path)
    assert restored == manifest
    assert restored.sha256 == manifest.sha256

