from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import canonical_square, encode_position
from wombat.cli import build_parser, main
from wombat.data.game_corpus import GameSplit
from wombat.eval.puzzle_corpus_eval import load_checkpoint, predict_depths
from wombat.interpretability.explain import explain_move
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.network import KING_OFFSETS, king_offsets

BASE = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=1,
    latent_tokens=2,
    policy_readout="square",
    move_readout="from_to",
    normalize_state=True,
    max_residual_update=None,
    square_attention_bias=True,
    local_attention=True,
    lookahead_heads=True,
    board_features=True,
)
WITH = dataclasses.replace(BASE, king_relative=True)
FENS = (
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3",
    "8/8/3k4/8/8/8/6K1/8 w - - 0 1",
    "8/8/3k4/8/8/8/6K1/8 b - - 0 1",
    "7k/8/8/8/8/8/8/K7 w - - 0 1",
)


def _encoded(fens: tuple[str, ...]) -> torch.Tensor:
    boards = [chess.Board(fen) for fen in fens]
    return torch.from_numpy(np.stack([encode_position(board).pieces for board in boards])).long()


def _positions(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pieces = torch.randint(0, 13, (count, 64), generator=generator)
    return pieces, torch.rand((count, 26), generator=generator)


def test_the_option_is_off_by_default_and_parses() -> None:
    assert not NetworkConfig().king_relative
    for command in ("train-games", "train-puzzles"):
        assert not build_parser().parse_args([command, "--output", "run"]).king_relative
        parsed = build_parser().parse_args([command, "--output", "run", "--king-relative"])
        assert parsed.king_relative
    # Two tables of one vector per (file, rank) offset.
    added = count_parameters(WombatNetwork(WITH)) - count_parameters(WombatNetwork(BASE))
    assert added == 2 * KING_OFFSETS * BASE.d_model == 2 * 225 * 24


def test_offsets_match_python_chess_from_either_side_to_move() -> None:
    offsets = king_offsets(_encoded(FENS))
    assert offsets.shape == (len(FENS), 2, 64)
    assert int(offsets.min()) >= 0 and int(offsets.max()) < KING_OFFSETS
    for board, rows in zip((chess.Board(fen) for fen in FENS), offsets, strict=True):
        for row, color in zip(rows, (board.turn, not board.turn), strict=True):
            king = canonical_square(board.king(color), board.turn)
            for square in chess.SQUARES:
                mine = canonical_square(square, board.turn)
                file_step = chess.square_file(mine) - chess.square_file(king)
                rank_step = chess.square_rank(mine) - chess.square_rank(king)
                assert -7 <= file_step <= 7 and -7 <= rank_step <= 7
                assert int(row[mine]) == 15 * (rank_step + 7) + file_step + 7, board.fen()
    # In the side-to-move frame the pawn before each side's own king is one rank ahead of it:
    # e2 for White to move, e7 for Black to move (d2 once the board is turned).
    ahead = 15 * (1 + 7) + 0 + 7
    assert int(offsets[0, 0, chess.E2]) == ahead and int(offsets[1, 0, chess.D2]) == ahead
    # From the enemy king, on its back rank in either frame, the same pawns are six ranks back.
    assert int(offsets[1, 1, chess.D2]) == int(offsets[0, 1, chess.E2]) == 15 * (-6 + 7) + 7
    # Opposite corners are the extreme offsets.
    assert int(offsets[4, 0, chess.H8]) == KING_OFFSETS - 1
    assert int(offsets[4, 1, chess.A1]) == 0


def test_zero_tables_leave_every_output_bit_identical() -> None:
    torch.manual_seed(0)
    plain = WombatNetwork(BASE).eval()
    model = WombatNetwork(WITH).eval()
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {"own_king_offset.weight", "enemy_king_offset.weight"}
    assert float(model.own_king_offset.weight.abs().sum()) == 0
    assert float(model.enemy_king_offset.weight.abs().sum()) == 0
    pieces, state = _positions(5, seed=1)
    steps = frozenset({0, 2})
    with torch.no_grad():
        expected = plain(pieces, state, 3, intermediate_steps=steps, include_move_values=True)
        outputs = model(pieces, state, 3, intermediate_steps=steps, include_move_values=True)
    assert isinstance(expected, dict) and isinstance(outputs, dict)
    for step, tensors in expected.items():
        for reference, tensor in zip(tensors, outputs[step], strict=True):
            assert torch.equal(reference, tensor)


def test_tokens_add_the_vectors_of_both_offsets() -> None:
    torch.manual_seed(2)
    model = WombatNetwork(WITH).eval()
    pieces = _encoded(FENS)
    state = torch.rand(len(FENS), 26)
    with torch.no_grad():
        before = model._tokenize(pieces, state)
        model.own_king_offset.weight.normal_()
        model.enemy_king_offset.weight.normal_()
        after = model._tokenize(pieces, state)
    offsets = king_offsets(pieces)
    own = model.own_king_offset.weight[offsets[:, 0]]
    enemy = model.enemy_king_offset.weight[offsets[:, 1]]
    assert torch.allclose(after[:, :64] - before[:, :64], own + enemy, atol=1e-5)
    assert torch.equal(after[:, 64:], before[:, 64:])


def test_king_relative_model_trains_resumes_and_loads_everywhere(
    game_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "king-relative"
    arguments = ["train-games", "--corpus", str(game_corpus), "--output", str(output)]
    arguments += ["--device", "cpu", "--steps", "2", "--batch-size", "2", "--warmup-steps", "1"]
    arguments += ["--fixed-depth", "2", "--eval-every", "2", "--eval-depths", "0,2"]
    arguments += ["--snapshot-every", "2", "--random-prefix", "2"]
    arguments += ["--d-model", "24", "--heads", "6", "--mlp-width", "48", "--recurrent-blocks", "2"]
    arguments += ["--prelude-blocks", "0", "--latent-tokens", "0", "--local-attention"]
    arguments += ["--move-readout", "from_to", "--relative-attention", "--state-attention-bias"]
    arguments += ["--board-features", "--move-consequences", "--value-bins", "8"]
    arguments += ["--king-relative"]
    assert main(arguments) == 0
    arguments[arguments.index("--steps") + 1] = "4"
    assert main([*arguments, "--resume"]) == 0
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["step"] for row in rows] == [2, 4]
    assert all(np.isfinite(row["train_loss"]) for row in rows)
    checkpoint = output / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["king_relative"]
    # The tables learned something from zero.
    assert float(saved["model"]["own_king_offset.weight"].abs().sum()) > 0
    assert float(saved["model"]["enemy_king_offset.weight"].abs().sum()) > 0
    model = load_checkpoint(checkpoint, torch.device("cpu"))
    assert model.config == NetworkConfig(**saved["model_config"])
    board = chess.Board(FENS[0])
    report = explain_move(model, board, chess.Move.from_uci("f1b5"), steps=2, depths=(0, 2))
    assert report["move"]["san"] == "Bb5"
    split = GameSplit(game_corpus / "validation")
    indices = np.arange(min(6, len(split)))
    cpu = torch.device("cpu")
    for select in ("policy", "mix"):
        predicted = predict_depths(
            model, split, indices, (0, 1, 3), batch_size=4, device=cpu, select=select
        )
        assert all(len(actions) == len(indices) for actions in predicted.values())
    capsys.readouterr()
