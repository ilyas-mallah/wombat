import chess

from wombat.eval.manifest import PositionCase, PositionManifest
from wombat.eval.puzzles import PuzzleDataset


def test_puzzle_dataset_preserves_rating_and_legal_target() -> None:
    manifest = PositionManifest(
        "fixture",
        (PositionCase("one", chess.STARTING_FEN, "e2e4", ("rating:1777", "opening")),),
    )
    dataset = PuzzleDataset(manifest)
    assert dataset[0]["rating"] == 1777
    assert dataset[0]["legal_mask"][dataset[0]["action"]]

