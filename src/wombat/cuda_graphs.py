"""CUDA-graph replay of training iterations and depth-curve inference.

Graphs record one fixed batch shape. A no-gradient prefix of any length replays
a one-step graph; each gradient-tail length has a whole-iteration graph covering
forward, backward, clipping and the optimizer step. At batch 128 this measured
about 5% faster than eager launches.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

from wombat.chess.moves import ACTION_SIZE, PLANES_PER_SQUARE
from wombat.losses import POLICY_ONLY, LossWeights, complete_targets, readouts, unrolled_loss
from wombat.model import WombatNetwork
from wombat.model.move_consequences import CONSEQUENCES
from wombat.model.network import board_consequences, board_phase, noisy_squares
from wombat.selection import MIX_BETA, SELECT_MODES, selection_scores

# Eager iterations run before capture so lazy CUDA and optimizer state exist.
WARMUP_ITERATIONS = 3
# Iterations the CPU may queue ahead of the GPU.
LOOKAHEAD = 2
# Opaque handle from torch.cuda.graph_pool_handle(); torch types it only for checkers.
if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE as GraphPool


def _autocast() -> torch.autocast:
    # The autocast weight cache is incompatible with graph capture.
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)


def _step(
    model: WombatNetwork,
    memory: torch.Tensor,
    hidden: torch.Tensor,
    local_edges: torch.Tensor | None = None,
) -> torch.Tensor:
    return cast(torch.Tensor, model.recurrent_step(memory, hidden, local_edges=local_edges))


def _capture_graphs(
    works: dict[str, Callable[[], None]], pool: GraphPool | None
) -> dict[str, torch.cuda.CUDAGraph]:
    """Capture each callable into its own graph, all sharing one memory pool.

    Sharing is safe in any replay order because graphs exchange data only through
    tensors allocated outside the pool.
    """

    torch.cuda.synchronize()
    # Blocks cached by eager warmup would otherwise sit beside the graph pool.
    torch.cuda.empty_cache()
    if pool is None:
        pool = torch.cuda.graph_pool_handle()
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, work in works.items():
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name], pool=pool):
            work()
    return graphs


class _BoardBuffers:
    """Static device inputs and recurrent state shared by every captured graph."""

    def __init__(
        self,
        model: WombatNetwork,
        batch_size: int,
        device: torch.device,
        initial_noise: float = 0.0,
        *,
        reads: bool = False,
    ) -> None:
        self.model = model
        # Only the trainer's buffers perturb H0; the predictor's never do.
        self.initial_noise = initial_noise
        self.pieces = torch.zeros((batch_size, 64), dtype=torch.long, device=device)
        self.state = torch.zeros((batch_size, 26), device=device)
        tokens = model.hidden_tokens
        self.memory = torch.zeros((batch_size, 65, model.config.d_model), device=device)
        self.hidden = torch.zeros((batch_size, tokens, model.config.d_model), device=device)
        # The static memory carries no square pair codes, so ``start`` keeps them here.
        edges = model.local_edges(self.pieces)
        self.edges = None if edges is None else torch.zeros_like(edges)
        # Readouts from these buffers (``reads``) need the move consequences ``start`` computes.
        self.consequences: torch.Tensor | None = None
        if reads and model.config.move_consequences:
            shape = (batch_size, 64, PLANES_PER_SQUARE, CONSEQUENCES)
            self.consequences = torch.zeros(shape, device=device)
        # Likewise the phase buckets of phase heads.
        self.phase: torch.Tensor | None = None
        if reads and model.config.phase_heads:
            self.phase = torch.zeros(batch_size, dtype=torch.long, device=device)

    def initial(self, memory: torch.Tensor) -> torch.Tensor:
        hidden = self.model.initial_hidden(memory)
        return noisy_squares(hidden, self.initial_noise) if self.initial_noise else hidden

    def start(self) -> None:
        with torch.no_grad(), _autocast():
            reads = self.consequences is not None
            memory = self.model.encode_board(self.pieces, self.state, consequences=reads)
            self.memory.copy_(memory)
            self.hidden.copy_(self.initial(memory))
            if self.edges is not None:
                self.edges.copy_(memory.local_edges)  # type: ignore[attr-defined]
            if self.consequences is not None:
                self.consequences.copy_(cast(torch.Tensor, board_consequences(memory)))
            if self.phase is not None:
                self.phase.copy_(cast(torch.Tensor, board_phase(memory)))

    def advance(self) -> None:
        model = self.model
        with torch.no_grad(), _autocast():
            self.hidden.copy_(_step(model, self.memory, self.hidden, self.edges))


class GraphedTrainer:
    """Run training iterations eagerly for warmup, then as captured graphs."""

    def __init__(
        self,
        model: WombatNetwork,
        optimizer: torch.optim.Optimizer,
        *,
        batch_size: int,
        max_tail: int,
        device: torch.device,
        weights: LossWeights = POLICY_ONLY,
        every_step: bool = False,
        recompute: bool = False,
        initial_noise: float = 0.0,
        pool: GraphPool | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.max_tail = max_tail
        self.weights = weights
        self.every_step = every_step
        self.recompute = recompute
        self.pool = pool
        self.buffers = _BoardBuffers(model, batch_size, device, initial_noise)
        shape = (batch_size, ACTION_SIZE)
        self.targets = {
            "action": torch.zeros(batch_size, dtype=torch.long, device=device),
            "legal_mask": torch.ones(shape, dtype=torch.bool, device=device),
            "policy_target": torch.zeros(shape, device=device),
            "move_value_target": torch.zeros(shape, device=device),
            "move_value_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "value": torch.zeros(batch_size, device=device),
            "value_mask": torch.ones(batch_size, dtype=torch.bool, device=device),
            "reply_from": torch.zeros(batch_size, dtype=torch.long, device=device),
            "reply_to": torch.zeros(batch_size, dtype=torch.long, device=device),
            "board2": torch.zeros((batch_size, 64), dtype=torch.long, device=device),
            "board4": torch.zeros((batch_size, 64), dtype=torch.long, device=device),
        }
        self.from_start = torch.ones((), device=device)
        self.loss = torch.zeros((), device=device)
        self.correct = torch.zeros((), device=device)
        self.loss_total = torch.zeros((), device=device)
        self.correct_total = torch.zeros((), device=device)
        self.eager_iterations = 0
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self.pending: deque[torch.cuda.Event] = deque()

    def load(self, batch: dict[str, torch.Tensor]) -> None:
        """Copy a batch into the static buffers; it may already be on the device."""

        batch = complete_targets(batch)

        def staged(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.is_cuda else tensor.pin_memory()

        self.buffers.pieces.copy_(staged(batch["pieces"]), non_blocking=True)
        self.buffers.state.copy_(staged(batch["state"]), non_blocking=True)
        for key, buffer in self.targets.items():
            buffer.copy_(staged(batch[key]), non_blocking=True)

    def _iteration(self, tail: int) -> None:
        """Gradient tail from H0 or the prefix state, then backward and the optimizer step."""

        model, buffers = self.model, self.buffers
        with _autocast():
            memory = model.encode_board(buffers.pieces, buffers.state)
            # Multiplying by 0/1 keeps one graph valid for both starting points.
            hidden = (
                self.from_start * buffers.initial(memory) + (1 - self.from_start) * buffers.hidden
            )
            loss, correct = unrolled_loss(
                model,
                memory,
                hidden,
                tail,
                self.targets,
                self.weights,
                every_step=self.every_step,
                recompute=self.recompute,
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        self.optimizer.step()
        self.loss.copy_(loss.detach())
        self.correct.copy_(correct)

    def _run_eager(self, prefix: int, tail: int) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.optimizer.zero_grad(set_to_none=True)
            if prefix:
                self.buffers.start()
                for _ in range(prefix):
                    self.buffers.advance()
            self._iteration(tail)
        torch.cuda.current_stream().wait_stream(side)

    def _tail_work(self, tail: int) -> Callable[[], None]:
        def work() -> None:
            # Gradients are then allocated inside the graph's recorded pool.
            self.optimizer.zero_grad(set_to_none=True)
            self._iteration(tail)

        return work

    def _capture_all(self) -> None:
        works: dict[str, Callable[[], None]] = {
            "start": self.buffers.start,
            "advance": self.buffers.advance,
        }
        for tail in range(1, self.max_tail + 1):
            works[f"tail{tail}"] = self._tail_work(tail)
        self.graphs = _capture_graphs(works, self.pool)

    def run(self, prefix: int, tail: int) -> None:
        if not 1 <= tail <= self.max_tail:
            raise ValueError(f"tail must be in 1..{self.max_tail}")
        self.from_start.fill_(0.0 if prefix else 1.0)
        if self.eager_iterations < WARMUP_ITERATIONS:
            self._run_eager(prefix, tail)
            self.eager_iterations += 1
        else:
            if not self.graphs:
                torch.cuda.synchronize()
                self._capture_all()
            if prefix:
                self.graphs["start"].replay()
                for _ in range(prefix):
                    self.graphs["advance"].replay()
            self.graphs[f"tail{tail}"].replay()
        self.loss_total += self.loss
        self.correct_total += self.correct
        event = torch.cuda.Event()
        event.record()
        self.pending.append(event)
        if len(self.pending) > LOOKAHEAD:
            self.pending.popleft().synchronize()

    def drain_totals(self) -> tuple[float, float]:
        """Summed loss and accuracy since the last drain; synchronizes once."""

        totals = (float(self.loss_total), float(self.correct_total))
        self.loss_total.zero_()
        self.correct_total.zero_()
        return totals


class GraphedPredictor:
    """Encode, step and read out a fixed-size batch by replaying captured graphs."""

    def __init__(
        self,
        model: WombatNetwork,
        batch_size: int,
        device: torch.device,
        pool: GraphPool | None = None,
        *,
        select: str = "policy",
        beta: float = MIX_BETA,
    ) -> None:
        if select not in SELECT_MODES:
            raise ValueError(f"unknown move selection {select!r}; choose one of {SELECT_MODES}")
        self.model = model
        self.batch_size = batch_size
        self.pool = pool
        self.select = select
        self.beta = beta
        self.buffers = _BoardBuffers(model, batch_size, device, reads=True)
        self.policy = torch.zeros((batch_size, ACTION_SIZE), device=device)
        # Only value and mix selection read the per-move values.
        self.move_values = (
            None if select == "policy" else torch.zeros((batch_size, ACTION_SIZE), device=device)
        )
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}

    def _read(self) -> None:
        consequences, phase = self.buffers.consequences, self.buffers.phase
        hidden = self.buffers.hidden
        with torch.no_grad(), _autocast():
            if self.move_values is None:
                policy = self.model.heads(hidden, 65, consequences=consequences, phase=phase)[0]
                self.policy.copy_(policy)
                return
            policy, _, move_values = readouts(self.model, hidden, 65, consequences, phase)
            self.policy.copy_(policy)
            self.move_values.copy_(move_values)

    def _capture(self) -> None:
        work = {"start": self.buffers.start, "advance": self.buffers.advance, "read": self._read}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_ITERATIONS):
                for step in work.values():
                    step()
        torch.cuda.current_stream().wait_stream(side)
        self.graphs = _capture_graphs(work, self.pool)

    @torch.no_grad()
    def predict(
        self, batch: dict[str, torch.Tensor], depths: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Legal argmax actions under ``select`` for up to ``batch_size`` positions per depth."""

        count = len(batch["pieces"])
        if count > self.batch_size:
            raise ValueError("batch exceeds the captured size")
        device = self.buffers.pieces.device
        # Rows past ``count`` keep stale but finite inputs and are discarded.
        self.buffers.pieces[:count].copy_(batch["pieces"].to(device))
        self.buffers.state[:count].copy_(batch["state"].to(device))
        mask = batch["legal_mask"].to(device)
        if not self.graphs:
            self._capture()
        self.graphs["start"].replay()
        wanted = set(depths)
        results: dict[int, torch.Tensor] = {}
        for depth in range(max(depths) + 1):
            if depth:
                self.graphs["advance"].replay()
            if depth in wanted:
                self.graphs["read"].replay()
                values = None if self.move_values is None else self.move_values[:count]
                scores = selection_scores(self.policy[:count], values, mask, self.select, self.beta)
                results[depth] = scores.argmax(dim=1)
        return {depth: actions.cpu().numpy() for depth, actions in results.items()}
