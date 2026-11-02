"""Why the network plays a move: depth strip, values, expected reply, attention and occlusion.

Squares are reported by their absolute names; the network itself works in the side-to-move
frame, where Black to move is rotated 180 degrees.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import chess
import numpy as np
import torch

from wombat.chess.encode import canonical_square, original_square, play_encoding, state_vector
from wombat.chess.moves import legal_actions
from wombat.model import ReasoningStepTrace, WombatNetwork

DEPTHS = (0, 1, 2, 4, 8, 16)


def _inputs(
    boards: Sequence[chess.Board], device: torch.device, zero_halfmove: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [play_encoding(board, zero_halfmove=zero_halfmove) for board in boards]
    pieces = np.stack([item.pieces for item in encoded]).astype(np.int64)
    state = np.stack([state_vector(item) for item in encoded])
    return torch.from_numpy(pieces).to(device), torch.from_numpy(state).to(device)


def _legal(board: chess.Board, move: chess.Move) -> chess.Move:
    """The legal move ``move`` names, with castling spelled the way ``legal_moves`` spells it."""

    try:
        return board.parse_uci(move.uci())
    except ValueError as error:
        raise ValueError(f"illegal move for position: {move.uci()}") from error


class _Policy:
    """A policy readout restricted to one board's legal moves."""

    def __init__(self, board: chess.Board, logits: torch.Tensor) -> None:
        self.board = board
        self.moves = list(board.legal_moves)
        actions = torch.tensor(legal_actions(board), dtype=torch.long, device=logits.device)
        self.probabilities = torch.softmax(logits.float()[actions], dim=0).cpu().numpy()
        self.actions = actions.cpu().numpy()
        self.order = np.argsort(-self.probabilities, kind="stable")

    def probability(self, move: chess.Move) -> float:
        return float(self.probabilities[self.moves.index(move)])

    def action(self, move: chess.Move) -> int:
        return int(self.actions[self.moves.index(move)])

    def ranked(self, count: int) -> list[dict[str, Any]]:
        return [
            {
                "uci": self.moves[index].uci(),
                "san": self.board.san(self.moves[index]),
                "probability": float(self.probabilities[index]),
            }
            for index in self.order[:count]
        ]


def _piece(board: chess.Board, square: chess.Square) -> str | None:
    piece = board.piece_at(square)
    return piece.symbol() if piece is not None else None


def _squares(board: chess.Board, logits: torch.Tensor, count: int = 3) -> list[dict[str, Any]]:
    """The likeliest absolute squares under 64 canonical square logits."""

    probabilities = torch.softmax(logits.float(), dim=0).cpu()
    ranked = torch.argsort(probabilities, descending=True)[:count]
    return [
        {
            "square": chess.square_name(original_square(int(index), board.turn)),
            "probability": float(probabilities[index]),
        }
        for index in ranked
    ]


def _reply_squares(lookahead: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Origin and destination logits over 64 squares; for a pair readout, its marginals."""

    if "reply_pair" not in lookahead:
        return lookahead["reply_from"][0], lookahead["reply_to"][0]
    pair = lookahead["reply_pair"][0].float()
    return pair.logsumexp(dim=1), pair.logsumexp(dim=0)


def _legal_replies(
    board: chess.Board, move: chess.Move, lookahead: dict[str, torch.Tensor], count: int = 3
) -> list[dict[str, Any]]:
    """The opponent's legal replies to ``move``, scored and renormalized.

    A pair readout scores each reply's (from, to) pair; square heads score P(from) x P(to), and
    being independent their separate favourites can name an illegal move.
    """

    after = board.copy(stack=False)
    after.push(move)
    replies = list(after.legal_moves)
    if not replies:
        return []
    # The heads speak in the frame of ``board``'s side to move.
    origins = [canonical_square(reply.from_square, board.turn) for reply in replies]
    targets = [canonical_square(reply.to_square, board.turn) for reply in replies]
    if "reply_pair" in lookahead:
        pair = torch.softmax(lookahead["reply_pair"][0].float().flatten(), dim=0).view(64, 64)
        scores = pair.cpu().numpy()[origins, targets]
    else:
        origin = torch.softmax(lookahead["reply_from"][0].float(), dim=0).cpu().numpy()
        target = torch.softmax(lookahead["reply_to"][0].float(), dim=0).cpu().numpy()
        scores = origin[origins] * target[targets]
    scores = scores / scores.sum()
    ranked = np.argsort(-scores, kind="stable")[:count]
    return [
        {
            "uci": replies[index].uci(),
            "san": after.san(replies[index]),
            "probability": float(scores[index]),
        }
        for index in ranked
    ]


def _attention(
    board: chess.Board, move: chess.Move, trace: ReasoningStepTrace, count: int
) -> dict[str, Any] | None:
    """Where the moving piece's square token looked in the last block of this step."""

    if not trace.attention_maps:
        return None
    # (heads, queries, keys); keys are the board memory followed by the hidden tokens.
    maps = trace.attention_maps[-1][0].float().mean(0).cpu()
    board_tokens = maps.shape[1] - maps.shape[0]
    row = maps[canonical_square(move.from_square, board.turn)]
    canonical = (row[:64] + row[board_tokens : board_tokens + 64]).numpy()
    weights = [float(canonical[canonical_square(square, board.turn)]) for square in chess.SQUARES]
    ranked = sorted(chess.SQUARES, key=lambda square: -weights[square])[:count]
    return {
        "steps": trace.step,
        "square": chess.square_name(move.from_square),
        "weights": weights,
        "elsewhere": max(0.0, 1.0 - sum(weights)),
        "top": [
            {
                "square": chess.square_name(square),
                "piece": _piece(board, square),
                "weight": weights[square],
            }
            for square in ranked
        ],
    }


def _occlusion(
    model: WombatNetwork,
    board: chess.Board,
    move: chess.Move,
    steps: int,
    base: float,
    device: torch.device,
    zero_halfmove: bool = False,
) -> dict[str, Any]:
    """Remove each other non-king piece and re-read the policy, all boards in one batch."""

    kept: list[tuple[chess.Square, chess.Piece, chess.Board]] = []
    skipped: list[dict[str, Any]] = []
    for square, piece in sorted(board.piece_map().items()):
        reason = None
        if piece.piece_type == chess.KING:
            reason = "king"
        elif square == move.from_square:
            reason = "moving piece"
        else:
            removed = board.copy(stack=False)
            removed.remove_piece_at(square)
            removed.castling_rights = removed.clean_castling_rights()
            if not removed.is_valid():
                reason = "invalid position"
            elif move not in removed.legal_moves:
                reason = "move illegal"
            else:
                kept.append((square, piece, removed))
        if reason is not None:
            skipped.append(
                {"square": chess.square_name(square), "piece": piece.symbol(), "reason": reason}
            )
    rows: list[dict[str, Any]] = []
    if kept:
        pieces, state = _inputs([removed for _, _, removed in kept], device, zero_halfmove)
        policy = model(pieces, state, steps)[0]
        for (square, piece, removed), logits in zip(kept, policy):
            readout = _Policy(removed, logits)
            probability = readout.probability(move)
            best = readout.moves[int(readout.order[0])]
            rows.append(
                {
                    "square": chess.square_name(square),
                    "piece": piece.symbol(),
                    "own": piece.color == board.turn,
                    "probability": probability,
                    "change": probability - base,
                    "best": {"uci": best.uci(), "san": removed.san(best)},
                }
            )
    rows.sort(key=lambda row: -abs(row["change"]))
    return {"steps": steps, "probability": base, "pieces": rows, "skipped": skipped}


@torch.inference_mode()
def explain_move(
    model: WombatNetwork,
    board: chess.Board,
    move: chess.Move,
    *,
    compare: chess.Move | None = None,
    steps: int = 8,
    depths: Sequence[int] = DEPTHS,
    top: int = 4,
    attention_top: int = 8,
    zero_halfmove: bool = False,
) -> dict[str, Any]:
    """A JSON-serialisable account of why ``model`` would play ``move`` on ``board``.

    Values are ``2W - 1`` for the side to move (per move, for the mover). Attention and
    occlusion are read after ``steps`` reasoning steps. ``zero_halfmove`` encodes as for play.
    """

    if steps < 0 or any(depth < 0 for depth in depths):
        raise ValueError("reasoning steps cannot be negative")
    move = _legal(board, move)
    compare = _legal(board, compare) if compare is not None else None
    device = next(model.parameters()).device
    pieces, state = _inputs([board], device, zero_halfmove)
    # One traced pass serves the depth strip, the readout at ``steps`` and its attention.
    traces = model.trace_reasoning(
        pieces, state, max((*depths, steps)), capture_attention=True, move_to_cpu=False
    )
    strip = []
    for depth in depths:
        readout = _Policy(board, traces[depth].policy_logits[0])
        strip.append(
            {
                "steps": depth,
                "top": readout.ranked(top),
                "move_probability": readout.probability(move),
            }
        )
    hidden = traces[steps].hidden
    logits, value, move_values = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model.heads(
            hidden,
            include_move_values=True,
            consequences=model.consequences(pieces, state),
            phase=model.phase(pieces),
        ),
    )
    policy = _Policy(board, logits[0])
    best = policy.moves[int(policy.order[0])]

    def described(role: str, item: chess.Move) -> dict[str, Any]:
        return {
            "role": role,
            "uci": item.uci(),
            "san": board.san(item),
            "probability": policy.probability(item),
            "value": float(move_values[0, policy.action(item)]),
        }

    candidates = [described("played", move)]
    if compare is not None and compare != move:
        candidates.append(described("compare", compare))
    if best not in (move, compare):
        candidates.append(described("best", best))

    reply = None
    if model.config.lookahead_heads:
        lookahead = model.lookahead(hidden)
        origin, target = _reply_squares(lookahead)
        reply = {
            "from": _squares(board, origin),
            "to": _squares(board, target),
            "after": {"uci": best.uci(), "san": board.san(best)},
            "legal": _legal_replies(board, best, lookahead),
        }
    base = policy.probability(move)
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "steps": steps,
        "move": candidates[0],
        "compare": candidates[1] if compare is not None and compare != move else None,
        "best": {
            "uci": best.uci(),
            "san": board.san(best),
            "probability": policy.probability(best),
        },
        "depths": strip,
        "value": float(value[0]),
        "moves": candidates,
        "reply": reply,
        "attention": _attention(board, move, traces[steps], attention_top),
        "occlusion": _occlusion(model, board, move, steps, base, device, zero_halfmove),
    }


def format_explanation(result: dict[str, Any], rows: int = 10) -> str:
    """A compact plain-text report of :func:`explain_move`'s result."""

    played = result["move"]["san"]
    steps = result["steps"]
    compare = result["compare"]
    lines = [f"Position {result['fen']}"]
    lines.append(f"played {played}" + (f", compared with {compare['san']}" if compare else ""))
    lines += ["", "Top moves by reasoning depth (probability):"]
    for depth in result["depths"]:
        moves = ", ".join(f"{item['san']} {item['probability']:.0%}" for item in depth["top"])
        lines.append(f"  T{depth['steps']:<2} {moves}   ({played} {depth['move_probability']:.0%})")
    lines.append("")
    lines.append(
        f"Its own evaluation at T{steps}: {result['value']:+.2f}"
        " (-1 lost .. +1 won for the side to move)"
    )
    for item in result["moves"]:
        note = " (its top move)" if item["role"] == "best" else ""
        lines.append(f"  its value for {item['san']}: {item['value']:+.2f}{note}")
    reply = result["reply"]
    if reply is None:
        lines.append("Expected opponent reply: this checkpoint has no lookahead heads")
    else:
        after = reply["after"]["san"]
        if reply["legal"]:
            likely = reply["legal"][0]
            lines.append(
                f"Expected opponent reply after its top move {after}: {likely['san']}"
                f" ({likely['probability']:.0%} of the legal replies)"
            )
        else:
            lines.append(f"Its top move {after} leaves the opponent no legal reply")
    attention = result["attention"]
    lines.append("")
    if attention is None:
        lines.append("No attention at T0: the reasoning block never ran.")
    else:
        lines.append(f"Squares the {played} piece's token attended to most (last block, T{steps}):")
        lines.append(
            "  "
            + ", ".join(
                f"{item['square']}{'(' + item['piece'] + ')' if item['piece'] else ''}"
                f" {item['weight']:.0%}"
                for item in attention["top"]
            )
        )
    occlusion = result["occlusion"]
    lines += [
        "",
        f"Remove one piece and re-ask at T{steps}: change in P({played})"
        f" (it was {occlusion['probability']:.0%})",
    ]
    for row in occlusion["pieces"][:rows]:
        label = f"{row['piece']}{row['square']}"
        lines.append(
            f"  without {label:4s} {row['change']:+.0%}   (it would then play {row['best']['san']})"
        )
    if not occlusion["pieces"]:
        lines.append("  no piece can be removed with the move still legal")
    return "\n".join(lines)
