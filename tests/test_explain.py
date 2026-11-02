from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import encode_position, state_vector
from wombat.chess.moves import legal_actions
from wombat.cli import main
from wombat.interpretability.explain import DEPTHS, explain_move, format_explanation
from wombat.model import NetworkConfig, WombatNetwork

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
    square_attention_bias=True,
    lookahead_heads=True,
)
PLAIN = replace(TINY, square_attention_bias=False, lookahead_heads=False)
BLACK_TO_MOVE = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3"
# Without the e5 knight Black is in check with White to move; without c3, d2c3 is illegal.
OCCLUSION = "4k3/8/8/4n3/8/2p5/3P4/4RK2 w - - 0 1"


def _model(config: NetworkConfig, seed: int = 0) -> WombatNetwork:
    torch.manual_seed(seed)
    model = WombatNetwork(config).eval()
    # Nonzero recurrent updates, so reasoning steps change the readout.
    for name, parameter in model.recurrent_core.named_parameters():
        if name.endswith(("attention.out_proj.weight", "mlp.output.weight")):
            torch.nn.init.normal_(parameter, std=0.2)
    return model


def _probabilities(model: WombatNetwork, board: chess.Board, steps: int) -> np.ndarray:
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
    state = torch.from_numpy(state_vector(encoded))[None]
    with torch.inference_mode():
        logits = model(pieces, state, steps)[0][0]
    return torch.softmax(logits[legal_actions(board)], dim=0).numpy()


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, BLACK_TO_MOVE], ids=["white", "black"])
def test_explanation_is_well_formed_for_either_side(fen: str) -> None:
    model = _model(TINY)
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    move, compare = moves[3], moves[7]
    result = explain_move(model, board, move, compare=compare, steps=3)
    assert json.loads(json.dumps(result)) == result
    assert board.fen() == fen
    assert result["turn"] == ("white" if board.turn else "black") and result["steps"] == 3
    assert (result["move"]["uci"], result["compare"]["uci"]) == (move.uci(), compare.uci())

    legal = {item.uci() for item in moves}
    assert [depth["steps"] for depth in result["depths"]] == list(DEPTHS)
    for depth in result["depths"]:
        probabilities = _probabilities(model, board, depth["steps"])
        assert depth["move_probability"] == pytest.approx(
            probabilities[moves.index(move)], abs=1e-5
        )
        assert len(depth["top"]) == 4 and all(item["uci"] in legal for item in depth["top"])
        assert depth["top"][0]["probability"] == pytest.approx(probabilities.max(), abs=1e-5)
    assert result["depths"][0]["top"] != result["depths"][-1]["top"]

    roles = [item["role"] for item in result["moves"]]
    assert roles[:2] == ["played", "compare"]
    assert all(-1 <= item["value"] <= 1 for item in result["moves"]) and -1 <= result["value"] <= 1
    if "best" in roles:
        assert result["moves"][-1]["uci"] == result["best"]["uci"]
        assert result["best"]["uci"] not in (move.uci(), compare.uci())

    reply = result["reply"]
    assert [len(reply["from"]), len(reply["to"])] == [3, 3]
    assert all(0 < item["probability"] <= 1 for item in (*reply["from"], *reply["to"]))
    assert reply["after"]["uci"] == result["best"]["uci"]
    after = board.copy()
    after.push_uci(reply["after"]["uci"])
    assert 0 < len(reply["legal"]) <= 3
    assert all(chess.Move.from_uci(item["uci"]) in after.legal_moves for item in reply["legal"])
    shares = [item["probability"] for item in reply["legal"]]
    assert shares == sorted(shares, reverse=True) and sum(shares) <= 1 + 1e-6

    attention = result["attention"]
    assert attention["square"] == chess.square_name(move.from_square)
    assert len(attention["weights"]) == 64 and min(attention["weights"]) >= 0
    assert sum(attention["weights"]) + attention["elsewhere"] == pytest.approx(1, abs=1e-4)
    top = attention["top"]
    assert [item["weight"] for item in top] == sorted(attention["weights"], reverse=True)[:8]
    for item in top:
        piece = board.piece_at(chess.parse_square(item["square"]))
        assert item["piece"] == (piece.symbol() if piece else None)

    occlusion = result["occlusion"]
    assert occlusion["probability"] == pytest.approx(result["move"]["probability"])
    base = _probabilities(model, board, 3)[moves.index(move)]
    assert occlusion["probability"] == pytest.approx(base, abs=1e-5)
    changes = [abs(row["change"]) for row in occlusion["pieces"]]
    assert changes == sorted(changes, reverse=True)
    removed = {row["square"] for row in occlusion["pieces"]}
    skipped = {item["square"]: item["reason"] for item in occlusion["skipped"]}
    assert removed.isdisjoint(skipped) and len(removed) + len(skipped) == len(board.piece_map())
    assert skipped[chess.square_name(move.from_square)] == "moving piece"
    assert skipped["e1"] == skipped["e8"] == "king"
    # A rook that loses its castling rights with it is still a valid removal.
    assert {"a1", "h1", "a8", "h8"} <= removed


def test_occlusion_matches_one_board_at_a_time_and_skips_impossible_removals() -> None:
    model = _model(TINY, seed=1)
    board = chess.Board(OCCLUSION)
    move = chess.Move.from_uci("d2c3")
    result = explain_move(model, board, move, steps=2)
    skipped = {item["square"]: item["reason"] for item in result["occlusion"]["skipped"]}
    assert skipped == {
        "c3": "move illegal",
        "d2": "moving piece",
        "e5": "invalid position",
        "e8": "king",
        "f1": "king",
    }
    (row,) = result["occlusion"]["pieces"]
    assert (row["square"], row["piece"], row["own"]) == ("e1", "R", True)
    without = board.copy()
    without.remove_piece_at(chess.E1)
    probabilities = _probabilities(model, without, 2)
    moves = list(without.legal_moves)
    assert row["probability"] == pytest.approx(probabilities[moves.index(move)], abs=1e-5)
    assert row["best"]["uci"] == moves[int(np.argmax(probabilities))].uci()
    base = _probabilities(model, board, 2)[list(board.legal_moves).index(move)]
    assert row["change"] == pytest.approx(row["probability"] - base, abs=1e-5)


def test_black_squares_are_rotated_back_to_their_real_names() -> None:
    model = _model(TINY, seed=2)
    board = chess.Board(BLACK_TO_MOVE)
    move = chess.Move.from_uci("g8f6")
    result = explain_move(model, board, move, steps=2, depths=(2,))
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
    state = torch.from_numpy(state_vector(encoded))[None]
    traces = model.trace_reasoning(pieces, state, 2)
    maps = traces[-1].attention_maps[-1][0].mean(0)
    # g8 is canonical square 63 - 62 = 1; its row, keyed by canonical square.
    row = maps[1]
    tokens = maps.shape[1] - maps.shape[0]
    for square in (chess.A1, chess.E5, chess.H8):
        canonical = 63 - square
        expected = float(row[canonical] + row[tokens + canonical])
        assert result["attention"]["weights"][square] == pytest.approx(expected, abs=1e-6)
    with torch.inference_mode():
        lookahead = model.lookahead(traces[-1].hidden)
    best_from = int(lookahead["reply_from"][0].argmax())
    assert result["reply"]["from"][0]["square"] == chess.square_name(63 - best_from)
    # The legal reply is White's move scored in Black's rotated frame.
    after = board.copy()
    after.push_uci(result["reply"]["after"]["uci"])
    origin = torch.softmax(lookahead["reply_from"][0], dim=0)
    target = torch.softmax(lookahead["reply_to"][0], dim=0)
    scores = {
        reply.uci(): float(origin[63 - reply.from_square] * target[63 - reply.to_square])
        for reply in after.legal_moves
    }
    assert result["reply"]["legal"][0]["uci"] == max(scores, key=scores.__getitem__)


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, BLACK_TO_MOVE], ids=["white", "black"])
def test_pair_readout_scores_legal_replies_by_their_square_pair(fen: str) -> None:
    model = _model(replace(TINY, reply_readout="pair"), seed=3)
    board = chess.Board(fen)
    result = explain_move(model, board, next(iter(board.legal_moves)), steps=2, depths=(2,))
    encoded = encode_position(board)
    pieces = torch.from_numpy(encoded.pieces.astype(np.int64))[None]
    state = torch.from_numpy(state_vector(encoded))[None]
    with torch.inference_mode():
        pair = model.lookahead(model.trace_reasoning(pieces, state, 2)[-1].hidden)["reply_pair"]
    probabilities = torch.softmax(pair[0].flatten(), dim=0).view(64, 64)

    def canonical(square: int) -> int:
        return square if board.turn == chess.WHITE else 63 - square

    after = board.copy()
    after.push_uci(result["reply"]["after"]["uci"])
    scores = {
        reply.uci(): float(probabilities[canonical(reply.from_square), canonical(reply.to_square)])
        for reply in after.legal_moves
    }
    legal = result["reply"]["legal"]
    assert 0 < len(legal) <= 3 and legal[0]["uci"] == max(scores, key=scores.__getitem__)
    total = sum(scores.values())
    for item in legal:
        assert item["probability"] == pytest.approx(scores[item["uci"]] / total, rel=1e-4)
    # The square lists are the pair's marginals, named by their real squares.
    for key, dim in (("from", 1), ("to", 0)):
        marginal = probabilities.sum(dim=dim)
        top = result["reply"][key][0]
        assert top["square"] == chess.square_name(canonical(int(marginal.argmax())))
        assert top["probability"] == pytest.approx(float(marginal.max()), rel=1e-4)
    assert "Expected opponent reply after its top move" in format_explanation(result)


def test_without_lookahead_heads_or_reasoning_steps() -> None:
    model = _model(PLAIN)
    board = chess.Board()
    result = explain_move(model, board, chess.Move.from_uci("e2e4"), steps=0, depths=(0, 1))
    assert result["reply"] is None and result["attention"] is None and result["compare"] is None
    assert result["occlusion"]["steps"] == 0
    text = format_explanation(result)
    assert "no lookahead heads" in text and "No attention at T0" in text
    with pytest.raises(ValueError, match="illegal move"):
        explain_move(model, board, chess.Move.from_uci("e2e5"))
    with pytest.raises(ValueError, match="illegal move"):
        explain_move(model, board, chess.Move.from_uci("e2e4"), compare=chess.Move.from_uci("a1a8"))


def test_explain_move_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": _model(TINY).state_dict(), "model_config": asdict(TINY)}, checkpoint)
    arguments = ["explain-move", "--checkpoint", str(checkpoint), "--fen", BLACK_TO_MOVE]
    arguments += ["--move", "f8c5", "--compare", "g8f6", "--steps", "2"]
    assert main(arguments) == 0
    text = capsys.readouterr().out
    assert "played Bc5, compared with Nf6" in text
    assert "Remove one piece and re-ask at T2: change in P(Bc5)" in text
    assert main([*arguments, "--depths", "0,2", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [depth["steps"] for depth in result["depths"]] == [0, 2]
    assert result["move"]["san"] == "Bc5" and result["reply"] is not None
