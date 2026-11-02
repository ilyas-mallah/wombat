"""How well a checkpoint predicts Stockfish's reply to Stockfish's best move.

The reply heads, read at the root, are scored over the legal replies to Stockfish's best move.
The model's own policy on the position after that move, playing the opponent, is the reference.
A prediction matches when its origin and destination squares do, so promotions fold into their
squares as in the pair reply readout.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import queue
from typing import Self, cast

import chess
import chess.engine
import numpy as np
import torch

from wombat.chess.encode import canonical_square, play_encoding, state_vector
from wombat.chess.moves import legal_actions
from wombat.data.game_corpus import GameSplit
from wombat.data.puzzle_corpus import lower_priority
from wombat.eval.puzzle_corpus_eval import _accuracy, load_checkpoint, paired_difference
from wombat.eval.strength import valid_sample
from wombat.gpu_share import share_gpu
from wombat.model import WombatNetwork

REPLY_SCHEMA = "wombat.reply-agreement.v1"
# Per-position matches: the policy after the best move, then the reply heads at the root.
HITS = ("policy", "legal", "origin", "destination")


class StockfishLines:
    """Stockfish's first two principal-variation plies per FEN from a pool of engines.

    Lines are cached on disk by engine, node count and FEN.
    """

    def __init__(
        self, command: Sequence[str], nodes: int, cache_dir: Path, *, engines: int = 4
    ) -> None:
        self.nodes = nodes
        self.cache_dir = cache_dir
        self.pool: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        for _ in range(engines):
            engine = chess.engine.SimpleEngine.popen_uci(list(command))
            self.pool.put(engine)
            engine.configure({"Threads": 1})
        engine = self.pool.get()
        self.label = str(engine.id.get("name", "stockfish"))
        self.pool.put(engine)

    def close(self) -> None:
        while not self.pool.empty():
            self.pool.get().quit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _path(self, fen: str) -> Path:
        payload = json.dumps(
            {
                "schema": "wombat.reply-line.v1",
                "engine": self.label,
                "nodes": self.nodes,
                "fen": fen,
            },
            sort_keys=True,
        )
        key = sha256(payload.encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def line(self, fen: str) -> list[str]:
        """Up to two plies in UCI: Stockfish's best move and its expected reply."""

        path = self._path(fen)
        if path.exists():
            return cast(list[str], json.loads(path.read_text(encoding="utf-8"))["pv"])
        engine = self.pool.get()
        try:
            info = engine.analyse(
                chess.Board(fen), chess.engine.Limit(nodes=self.nodes), info=chess.engine.INFO_PV
            )
        finally:
            self.pool.put(engine)
        pv = [move.uci() for move in info.get("pv", [])[:2]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pv": pv}), encoding="utf-8")
        return pv


def _inputs(boards: Sequence[chess.Board], device: torch.device) -> tuple[torch.Tensor, ...]:
    encoded = [play_encoding(board) for board in boards]
    pieces = np.stack([item.pieces for item in encoded]).astype(np.int64)
    state = np.stack([state_vector(item) for item in encoded])
    return torch.from_numpy(pieces).to(device), torch.from_numpy(state).to(device)


def _pair_scores(lookahead: dict[str, torch.Tensor]) -> torch.Tensor:
    """(batch, 64, 64) log-scores of each (origin, destination) reply, either readout."""

    if "reply_pair" in lookahead:
        return lookahead["reply_pair"].float()
    origin = lookahead["reply_from"].float().log_softmax(dim=1)
    destination = lookahead["reply_to"].float().log_softmax(dim=1)
    return origin[:, :, None] + destination[:, None, :]


@torch.inference_mode()
def _hits(
    model: WombatNetwork,
    roots: Sequence[chess.Board],
    afters: Sequence[chess.Board],
    replies: Sequence[chess.Move],
    steps: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Per position: whether each readout's top prediction is the reply."""

    heads = model.config.lookahead_heads
    hits: dict[str, list[bool]] = {key: [] for key in HITS if heads or key == "policy"}
    policy = model(*_inputs(afters, device), steps)[0].float().cpu()
    if heads:
        hidden = model(*_inputs(roots, device), steps, include_hidden=True)[-1]
        pair = _pair_scores(model.lookahead(hidden)).cpu()
    for row, (root, after, reply) in enumerate(zip(roots, afters, replies, strict=True)):
        moves = list(after.legal_moves)
        chosen = moves[int(policy[row, legal_actions(after)].argmax())]
        squares = (reply.from_square, reply.to_square)
        hits["policy"].append((chosen.from_square, chosen.to_square) == squares)
        if not heads:
            continue
        # The reply heads speak in the frame of the root's side to move.
        origins = [canonical_square(move.from_square, root.turn) for move in moves]
        targets = [canonical_square(move.to_square, root.turn) for move in moves]
        best = int(pair[row, origins, targets].argmax())
        expected = tuple(canonical_square(square, root.turn) for square in squares)
        hits["legal"].append((origins[best], targets[best]) == expected)
        scores = pair[row].flatten().log_softmax(dim=0).view(64, 64)
        hits["origin"].append(int(scores.logsumexp(dim=1).argmax()) == expected[0])
        hits["destination"].append(int(scores.logsumexp(dim=0).argmax()) == expected[1])
    return {key: np.array(values, dtype=bool) for key, values in hits.items()}


def evaluate_reply(
    checkpoint: Path,
    *,
    split: Path,
    engine_command: Sequence[str],
    positions: int = 1_000,
    nodes: int = 200_000,
    steps: int = 8,
    engines: int = 4,
    cache_dir: Path = Path("runs/stockfish-cache/reply-lines"),
    device_name: str = "cpu",
    max_vram_gb: float | None = 1.0,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, object]:
    """Reply-head and policy agreement with Stockfish's reply to its own best move.

    Positions whose line stops before a reply (mate, or a one-move line) are skipped.
    """

    lower_priority()
    device = torch.device(device_name)
    share_gpu(device, max_vram_gb)
    model = load_checkpoint(checkpoint, device)
    game_split = GameSplit(split)
    assert game_split.fens is not None
    fens = [str(game_split.fens[index]) for index in valid_sample(game_split, positions, seed)]
    # Stockfish children inherit this process's idle priority.
    with StockfishLines(engine_command, nodes, cache_dir, engines=engines) as lines:
        with ThreadPoolExecutor(engines) as executor:
            pvs = list(executor.map(lines.line, fens))
        engine = lines.label
    roots: list[chess.Board] = []
    afters: list[chess.Board] = []
    replies: list[chess.Move] = []
    for fen, pv in zip(fens, pvs, strict=True):
        if len(pv) < 2:
            continue
        root = chess.Board(fen)
        best, reply = chess.Move.from_uci(pv[0]), chess.Move.from_uci(pv[1])
        if not root.is_legal(best):
            continue
        after = root.copy(stack=False)
        after.push(best)
        if not after.is_legal(reply):
            continue
        roots.append(root)
        afters.append(after)
        replies.append(reply)
    if not roots:
        raise ValueError("no sampled position has a Stockfish reply")
    chunks = [
        _hits(
            model,
            roots[start : start + batch_size],
            afters[start : start + batch_size],
            replies[start : start + batch_size],
            steps,
            device,
        )
        for start in range(0, len(roots), batch_size)
    ]
    hits = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    reply_heads = model.config.lookahead_heads
    return {
        "schema": REPLY_SCHEMA,
        "checkpoint": str(checkpoint),
        "split": str(split),
        "engine": engine,
        "nodes": nodes,
        "steps": steps,
        "reply_readout": model.config.reply_readout if reply_heads else None,
        "positions": len(fens),
        "evaluated": len(roots),
        "skipped": len(fens) - len(roots),
        "reply_legal_top1": _accuracy(hits["legal"]) if reply_heads else None,
        # With the pair readout these are its origin and destination marginals.
        "reply_origin_top1": _accuracy(hits["origin"]) if reply_heads else None,
        "reply_destination_top1": _accuracy(hits["destination"]) if reply_heads else None,
        "policy_top1": _accuracy(hits["policy"]),
        "reply_minus_policy": (
            paired_difference(hits["policy"], hits["legal"]) if reply_heads else None
        ),
    }
