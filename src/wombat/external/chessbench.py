"""DeepMind's ChessBench 9M action-value transformer (searchless_chess), ported to PyTorch.

The weights file is the converted release: ``{"model": state_dict, "actions": [uci, ...],
"bucket_values": tensor}``. Moves follow their ``ActionValueEngine.play`` with no temperature.

Derived from google-deepmind/searchless_chess, Copyright DeepMind Technologies Limited,
Apache License 2.0 (https://github.com/google-deepmind/searchless_chess). Changed: JAX to
PyTorch. The released weights are CC BY 4.0 and are not part of this repository.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LAYERS, DIM, HEADS, BUCKETS, SEQUENCE = 8, 256, 8, 128, 79
NAME = "ChessBench 9M"

_CHARACTERS = "0123456789abcdefghpnrkqPBNRQKw."
_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}


def tokenize(fen: str) -> np.ndarray:
    """Their ``tokenizer.tokenize``: 77 tokens from the FEN string."""

    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")
    indices: list[int] = []
    for char in side + board.replace("/", ""):
        if char in "12345678":
            indices.extend(int(char) * [_INDEX["."]])
        else:
            indices.append(_INDEX[char])
    castling = "" if castling == "-" else castling
    indices.extend(_INDEX[char] for char in castling)
    indices.extend((4 - len(castling)) * [_INDEX["."]])
    en_passant = ".." if en_passant == "-" else en_passant
    indices.extend(_INDEX[char] for char in en_passant)
    indices.extend(_INDEX[char] for char in halfmoves + "." * (3 - len(halfmoves)))
    indices.extend(_INDEX[char] for char in fullmoves + "." * (3 - len(fullmoves)))
    if len(indices) != 77:
        raise ValueError(f"cannot tokenize {fen}")
    return np.asarray(indices, dtype=np.int64)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-5)
        self.q = nn.Linear(DIM, DIM, bias=False)
        self.k = nn.Linear(DIM, DIM, bias=False)
        self.v = nn.Linear(DIM, DIM, bias=False)
        self.out = nn.Linear(DIM, DIM, bias=False)
        self.norm2 = nn.LayerNorm(DIM, eps=1e-5)
        self.gate = nn.Linear(DIM, 4 * DIM, bias=False)
        self.up = nn.Linear(DIM, 4 * DIM, bias=False)
        self.down = nn.Linear(4 * DIM, DIM, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, _ = h.shape
        x = self.norm1(h)
        q, k, v = (
            layer(x).view(batch, length, HEADS, DIM // HEADS).transpose(1, 2)
            for layer in (self.q, self.k, self.v)
        )
        attention = F.scaled_dot_product_attention(q, k, v)
        h = h + self.out(attention.transpose(1, 2).reshape(batch, length, DIM))
        x = self.norm2(h)
        return h + self.down(F.silu(self.gate(x)) * self.up(x))


class ActionValue9M(nn.Module):
    """Their ``transformer_decoder`` with the 9M config: 8 layers, 256 wide, 8 heads, 128 bins."""

    def __init__(self, vocab: int = 1968) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, DIM)
        self.position = nn.Embedding(SEQUENCE, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(DIM, eps=1e-5)
        self.head = nn.Linear(DIM, BUCKETS)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return-bucket log probabilities for (batch, 79) sequences ending in a dummy bucket."""

        inputs = F.pad(sequences, (1, 0))[:, :-1]
        h = self.embed(inputs) * math.sqrt(DIM) + self.position.weight[: inputs.shape[1]]
        for block in self.blocks:
            h = block(h)
        return F.log_softmax(self.head(self.norm(h[:, -1])), dim=-1)


class ChessBenchPlayer:
    """Plays the 9M as their ``ActionValueEngine.play`` does, one forward pass per legal move."""

    name = NAME

    def __init__(self, weights: Path, device: torch.device | None = None) -> None:
        saved = torch.load(weights, map_location="cpu", weights_only=True)
        self.device = device or torch.device("cpu")
        self.model = ActionValue9M(vocab=len(saved["actions"])).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.actions = {uci: index for index, uci in enumerate(saved["actions"])}
        self.bucket_values = np.asarray(saved["bucket_values"], dtype=np.float32)

    def ordered_moves(self, board: chess.Board) -> list[chess.Move]:
        """Their ``get_ordered_legal_moves``: legal moves sorted by action index."""

        return sorted(board.legal_moves, key=lambda move: self.actions[move.uci()])

    def win_probabilities(self, board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
        """Ordered legal moves and each move's expected bucket value."""

        return self.score_boards([board])[0]

    @torch.inference_mode()
    def score_boards(
        self, boards: Sequence[chess.Board], batch_size: int = 2048
    ) -> list[tuple[list[chess.Move], np.ndarray]]:
        """``win_probabilities`` of each board, passes holding up to ``batch_size`` moves."""

        scored: list[tuple[list[chess.Move], np.ndarray]] = []
        pending: list[tuple[list[chess.Move], np.ndarray]] = []

        def run() -> None:
            rows = np.concatenate([sequences for _, sequences in pending])
            log_probs = self.model(torch.from_numpy(rows).to(self.device)).float().cpu().numpy()
            values = np.exp(log_probs) @ self.bucket_values
            ends = np.cumsum([len(moves) for moves, _ in pending])
            for (moves, _), part in zip(pending, np.split(values, ends[:-1]), strict=True):
                scored.append((moves, part))
            pending.clear()

        size = 0
        for board in boards:
            moves = self.ordered_moves(board)
            if not moves:
                raise ValueError("cannot select a move in a terminal position")
            sequences = np.zeros((len(moves), SEQUENCE), dtype=np.int64)
            sequences[:, :77] = tokenize(board.fen())
            sequences[:, 77] = [self.actions[move.uci()] for move in moves]
            if pending and size + len(moves) > batch_size:
                run()
                size = 0
            pending.append((moves, sequences))
            size += len(moves)
        if pending:
            run()
        return scored

    def best_moves(self, boards: Sequence[chess.Board], batch_size: int = 2048) -> list[chess.Move]:
        """Their play rule without game history: the highest win probability per board."""

        return [
            moves[int(np.argmax(scores))] for moves, scores in self.score_boards(boards, batch_size)
        ]

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Highest win probability, with 0.5 for moves that allow a repetition draw."""

        moves, scores = self.win_probabilities(board)
        for index, move in enumerate(moves):
            board.push(move)
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                scores[index] = 0.5
            board.pop()
        return moves[int(np.argmax(scores))]


EXTERNAL_PLAYERS = {"chessbench9m": ChessBenchPlayer}
