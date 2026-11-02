"""Training loss shared by eager and CUDA-graph iterations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from wombat.data.game_corpus import IGNORE
from wombat.data.move_labels import LINE_PLIES
from wombat.model import WombatNetwork
from wombat.model.lines import one_hot_actions, own_moves
from wombat.model.network import MoveValueBins, board_consequences, board_phase
from wombat.model.workspace import WorkspaceCore


@dataclass(frozen=True)
class LossWeights:
    move_value: float = 0.0
    value: float = 0.0
    # Total weight of the reply-square and future-board predictions.
    lookahead: float = 0.0
    # Weight of the joint reply (origin, destination) prediction of the pair reply readout.
    reply: float = 0.0


POLICY_ONLY = LossWeights()
# Shares of the lookahead weight: reply from and to squares, then boards after plies 2 and 4.
# The pair reply readout has no square heads, so only the board shares count.
LOOKAHEAD_SHARES = {"reply_from": 0.2, "reply_to": 0.2, "board2": 0.3, "board4": 0.3}
# A chess position has at most 218 legal moves, so every labelled move fits in this many slots.
MOVE_VALUE_SLOTS = 218
# HL-Gauss smoothing std in bin widths (Farebrother et al. 2024, as ChessBench).
HL_GAUSS_SIGMA_BINS = 0.75


def hl_gauss_targets(values: torch.Tensor, bins: int) -> torch.Tensor:
    """(..., bins) HL-Gauss targets of -1..1 ``values`` over equal win-chance bins on [0, 1].

    Each bin gets the mass a Gaussian of std 0.75 / bins centred on the win chance
    (value + 1) / 2 puts on it; the mass outside [0, 1] is dropped and the rest renormalised.
    """

    win = ((values.float() + 1) / 2).clamp(0, 1).unsqueeze(-1)
    edges = torch.linspace(0, 1, bins + 1, device=values.device)
    cdf = torch.special.erf((edges - win) / (HL_GAUSS_SIGMA_BINS / bins * math.sqrt(2)))
    return (cdf[..., 1:] - cdf[..., :-1]) / (cdf[..., -1:] - cdf[..., :1])


def hl_gauss_loss(logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of (..., bins) logits against the HL-Gauss targets of ``values``."""

    targets = hl_gauss_targets(values, logits.shape[-1])
    return -(targets * nn.functional.log_softmax(logits.float(), dim=-1)).sum(dim=-1)


def binned_move_value_loss(
    move_values: MoveValueBins, targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """HL-Gauss loss of the labelled moves, averaged per position as the squared errors are.

    Only the labelled moves' logits are formed, in ``MOVE_VALUE_SLOTS`` fixed slots per position.
    """

    mask = targets["move_value_mask"]
    labelled, actions = mask.float().topk(min(MOVE_VALUE_SLOTS, mask.shape[1]), dim=1)
    values = targets["move_value_target"].gather(1, actions)
    errors = hl_gauss_loss(move_values.logits(actions), values) * labelled
    return _row_mean(errors.sum(dim=1) / labelled.sum(dim=1).clamp_min(1), mask.any(dim=1))


def _row_mean(per_row: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_row`` over the rows ``rows`` marks; rows without a target add nothing."""

    counted = rows.float()
    return (per_row * counted).sum() / counted.sum().clamp_min(1)


def readouts(
    model: WombatNetwork,
    hidden: torch.Tensor,
    board_tokens: int,
    consequences: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy logits, position value and per-move values from one hidden state."""

    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model.heads(
            hidden,
            board_tokens,
            include_move_values=True,
            consequences=consequences,
            phase=phase,
        ),
    )


def graded_loss(
    policy: torch.Tensor,
    value: torch.Tensor,
    move_values: torch.Tensor | MoveValueBins,
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft policy cross-entropy plus candidate and position value errors.

    Value errors count only where ``value_mask`` holds, averaged over those positions. With
    value bins ``value`` is (batch, bins) logits, ``move_values`` the ``MoveValueBins``, and
    both errors are HL-Gauss cross-entropies in place of squared errors. The policy loss and
    accuracy average over positions with a policy target, the move-value loss over positions
    with a labelled move, so value-only child positions leave both unchanged.
    Returns the loss and the fraction of positions whose argmax is the reference move.
    """

    logits = policy.float().masked_fill(~targets["legal_mask"], float("-inf"))
    log_probs = nn.functional.log_softmax(logits, dim=1)
    graded = targets["policy_target"].sum(dim=1) > 0
    # Illegal logits are -inf; zero targets there must not produce 0 * -inf.
    policy_loss = _row_mean(
        -(targets["policy_target"] * log_probs.clamp_min(-1e4)).sum(dim=1), graded
    )
    loss = policy_loss
    if weights.move_value:
        if isinstance(move_values, MoveValueBins):
            move_loss = binned_move_value_loss(move_values, targets)
        else:
            mask = targets["move_value_mask"].float()
            errors = (move_values.float() - targets["move_value_target"]).square() * mask
            move_loss = _row_mean(errors.sum(dim=1) / mask.sum(dim=1).clamp_min(1), mask.any(1))
        loss = loss + weights.move_value * move_loss
    if weights.value:
        valued = targets["value_mask"].float()
        if value.ndim == 2:
            errors = hl_gauss_loss(value, targets["value"]) * valued
        else:
            errors = (value.float() - targets["value"]).square() * valued
        loss = loss + weights.value * errors.sum() / valued.sum().clamp_min(1)
    correct = _row_mean((logits.argmax(dim=1) == targets["action"]).float(), graded)
    return loss, correct


def lookahead_loss(
    outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Cross-entropy of reply squares and future boards, ignoring unlabelled entries.

    Heads the outputs lack contribute nothing.
    """

    total = torch.zeros((), device=outputs["board2"].device)
    for key, share in LOOKAHEAD_SHARES.items():
        if key not in outputs:
            continue
        logits = outputs[key].float()
        target = targets[key]
        if logits.dim() == 3:
            logits = logits.flatten(0, 1)
            target = target.flatten()
        summed = nn.functional.cross_entropy(logits, target, ignore_index=IGNORE, reduction="sum")
        total = total + share * summed / (target != IGNORE).sum().clamp_min(1)
    return total


def reply_pair_loss(pair: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy over all 4,096 (origin, destination) pairs, averaged over labelled rows.

    A row is unlabelled when either reply square is ``IGNORE``.
    """

    origin, destination = targets["reply_from"], targets["reply_to"]
    labelled = (origin != IGNORE) & (destination != IGNORE)
    target = torch.where(labelled, origin * 64 + destination, IGNORE)
    summed = nn.functional.cross_entropy(
        pair.float().flatten(1), target, ignore_index=IGNORE, reduction="sum"
    )
    return summed / labelled.sum().clamp_min(1)


def step_loss(
    model: WombatNetwork,
    hidden: torch.Tensor,
    board_tokens: int,
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    consequences: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graded, lookahead and joint reply loss read from one recurrent state."""

    if model.config.value_bins:
        bins = model.value_bin_heads(hidden, board_tokens, consequences=consequences, phase=phase)
        loss, correct = graded_loss(*bins, targets, weights)
    else:
        policy, value, move_values = readouts(model, hidden, board_tokens, consequences, phase)
        loss, correct = graded_loss(policy, value, move_values, targets, weights)
    if weights.lookahead or weights.reply:
        outputs = model.lookahead(hidden)
        if weights.lookahead:
            loss = loss + weights.lookahead * lookahead_loss(outputs, targets)
        if weights.reply:
            loss = loss + weights.reply * reply_pair_loss(outputs["reply_pair"], targets)
    return loss, correct


def recurrent_step(
    model: WombatNetwork, memory: torch.Tensor, hidden: torch.Tensor, *, recompute: bool = False
) -> torch.Tensor:
    """One recurrent step; ``recompute`` keeps only its inputs and reruns it in backward."""

    if not recompute:
        return cast(torch.Tensor, model.recurrent_step(memory, hidden))
    # The step draws no random numbers without dropout, so no RNG state needs restoring,
    # which also keeps the recomputation inside CUDA graph capture free of RNG calls.
    return cast(
        torch.Tensor,
        checkpoint(
            model.recurrent_step,
            memory,
            hidden,
            use_reentrant=False,
            preserve_rng_state=model.config.dropout > 0,
        ),
    )


def unrolled_loss(
    model: WombatNetwork,
    memory: torch.Tensor,
    hidden: torch.Tensor,
    tail: int,
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    *,
    every_step: bool,
    recompute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``tail`` recurrent steps and grade them.

    With ``every_step`` each step is graded, weighted in proportion to its index so
    the last step counts most; otherwise only the final state is. A tail of 0
    grades the initial state, which is how the feedforward baseline trains.
    ``recompute`` trades a second forward pass per step for its activation memory.
    Move consequences and phase buckets come from ``memory`` as ``encode_board`` attached them.
    """

    consequences, phase = board_consequences(memory), board_phase(memory)
    if tail == 0:
        return step_loss(model, hidden, memory.shape[1], targets, weights, consequences, phase)
    loss = torch.zeros((), device=hidden.device)
    normalizer = tail * (tail + 1) / 2
    correct = torch.zeros((), device=hidden.device)
    for step in range(1, tail + 1):
        hidden = recurrent_step(model, memory, hidden, recompute=recompute)
        if every_step or step == tail:
            step_value, correct = step_loss(
                model, hidden, memory.shape[1], targets, weights, consequences, phase
            )
            loss = loss + (step / normalizer if every_step else 1.0) * step_value
    return loss, correct


@dataclass(frozen=True)
class LineWeights:
    """Loss weights of the workspace core's imagined lines."""

    # Mean squared distance of M after each ply from the stop-gradient encoding of the position
    # its chosen action produced.
    canonical: float = 1.0
    # Line policy cross-entropy: at a line's start against the graded root policy over the moves
    # no earlier line tried, later against the reference line while the line has matched it.
    policy: float = 0.25


def line_loss(
    model: WombatNetwork,
    pieces: torch.Tensor,
    state: torch.Tensor,
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    line: LineWeights,
    *,
    lines: int,
    plies: int,
    surplus: int = 0,
    follow_reference: bool = False,
    recompute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Root loss after ``lines`` lines of ``plies`` plies plus ``surplus`` steps, and line losses.

    Every line starts from the present position and plays the line policy's own best legal
    actions; its first move is always its own, since the reference's first move is the root
    answer. With ``follow_reference`` (a warm start) a line whose moves so far match
    ``targets["line_actions"]`` plays the reference's next move. Every completed ply pulls M
    toward the encoding of the position its action produced. Returns the loss, the root
    accuracy, and the canonical error, line policy loss and accuracy, the share of first lines
    starting with the reference move and the share of distinct first moves among the lines.
    """

    core = model.workspace_core
    board_memory = model.encode_board(pieces, state)
    consequences = board_consequences(board_memory)
    workspace = core.start(board_memory)
    memory = board_memory
    reference = targets["line_actions"]
    root = board = child = (pieces, state)
    with torch.no_grad():
        root_legal = core.rules.legal(pieces, state)
    tried = torch.zeros_like(root_legal)
    following = torch.ones(len(pieces), dtype=torch.bool, device=pieces.device)
    # Plies past the reference's end have no reference action.
    missing = torch.full((len(pieces),), -1, dtype=torch.long, device=pieces.device)
    canonical: list[torch.Tensor] = []
    policy_terms: list[torch.Tensor] = []
    hits: list[torch.Tensor] = []
    graded = 0
    firsts: list[torch.Tensor] = []
    for step in range(lines * plies * core.steps_per_ply + surplus):
        starts_line, ply, starts_ply, ends_ply = core.phase(step, plies)
        if starts_line:
            memory, board = board_memory, root
        if starts_ply:
            logits = _line_logits(model, memory, workspace, recompute).float()
            wanted = reference[:, ply] if ply < reference.shape[1] else missing
            if ply == 0:
                logits = core.first_logits(logits, tried)
                actions = own_moves(logits.detach(), root_legal)
                # The best moves no earlier line tried, as the root policy grades them.
                untried = targets["policy_target"].masked_fill(tried, 0)
                term, hit, count = _candidate_loss(logits, root_legal, untried, actions)
                following = (actions == wanted) & (wanted >= 0)
                tried = tried | one_hot_actions(actions, tried.shape[1])
                firsts.append(actions)
            else:
                with torch.no_grad():
                    legal = core.rules.legal(*board)
                own = own_moves(logits.detach(), legal)
                known = following & (wanted >= 0)
                term, hit, count = _reference_loss(logits, legal, wanted, known, own)
                actions = torch.where(known & follow_reference, wanted, own)
                following = known & (actions == wanted)
            policy_terms.append(term)
            hits.append(hit)
            graded += count
            memory, workspace = core.play(memory, workspace, actions)
            with torch.no_grad():
                child = core.rules.child(*board, actions)
        memory, workspace = _workspace_step(core, board_memory, memory, workspace, recompute)
        if ends_ply:
            memory, board = model.rotate_memory(memory), child
            with torch.no_grad():
                encoded = model.encode_board(*board, consequences=False)
            canonical.append(nn.functional.mse_loss(memory.float(), encoded.float()))
    tokens = model.workspace_readout(board_memory, workspace)
    loss, correct = step_loss(model, tokens, 65, targets, weights, consequences)
    zero = torch.zeros((), device=pieces.device)
    canonical_error = torch.stack(canonical).mean() if canonical else zero
    policy_loss = torch.stack(policy_terms).sum() / max(graded, 1) if policy_terms else zero
    accuracy = torch.stack(hits).sum() / max(graded, 1) if hits else zero
    loss = loss + line.canonical * canonical_error + line.policy * policy_loss
    details = {
        "canonical": canonical_error.detach(),
        "line_policy": policy_loss.detach(),
        "line_accuracy": accuracy.detach(),
        "first_match": zero,
        "distinct_first": zero,
    }
    if firsts:
        details["first_match"] = (firsts[0] == reference[:, 0]).float().mean()
        chosen = torch.stack(firsts, dim=1).sort(dim=1).values
        distinct = 1 + (chosen[:, 1:] != chosen[:, :-1]).sum(dim=1)
        details["distinct_first"] = distinct.float().mean() / len(firsts)
    return loss, correct, details


def _line_logits(
    model: WombatNetwork, memory: torch.Tensor, workspace: torch.Tensor, recompute: bool
) -> torch.Tensor:
    if not recompute:
        return model.line_policy(memory, workspace)
    return cast(torch.Tensor, checkpoint(model.line_policy, memory, workspace, use_reentrant=False))


def _workspace_step(
    core: WorkspaceCore,
    board_memory: torch.Tensor,
    memory: torch.Tensor,
    workspace: torch.Tensor,
    recompute: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step; ``recompute`` keeps only its inputs and reruns it in backward."""

    if not recompute:
        return core.step(board_memory, memory, workspace)
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        checkpoint(core.step, board_memory, memory, workspace, use_reentrant=False),
    )


def _candidate_loss(
    logits: torch.Tensor, legal: torch.Tensor, target: torch.Tensor, chosen: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Summed cross-entropy against ``target`` renormalised per row, hits of its best move, and
    the rows graded: those where ``target`` keeps any mass."""

    mass = target.sum(dim=1)
    rows = (mass > 1e-6).nonzero().squeeze(1)
    if not len(rows):
        return logits.new_zeros(()), logits.new_zeros(()), 0
    wanted = target[rows] / mass[rows, None]
    log_probs = logits[rows].masked_fill(~legal[rows], float("-inf")).log_softmax(dim=1)
    # Illegal logits are -inf; zero targets there must not produce 0 * -inf.
    term = -(wanted * log_probs.clamp_min(-1e4)).sum()
    return term, (chosen[rows] == wanted.argmax(dim=1)).sum(), len(rows)


def _reference_loss(
    logits: torch.Tensor,
    legal: torch.Tensor,
    wanted: torch.Tensor,
    known: torch.Tensor,
    chosen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Summed cross-entropy of the ``known`` rows' reference actions, their hits and count."""

    rows = known.nonzero().squeeze(1)
    if not len(rows):
        return logits.new_zeros(()), logits.new_zeros(()), 0
    # Only rows with a reference action, which is always allowed, so none is empty.
    allowed = legal[rows].clone()
    allowed[torch.arange(len(rows), device=rows.device), wanted[rows]] = True
    log_probs = logits[rows].masked_fill(~allowed, float("-inf")).log_softmax(dim=1)
    picked = log_probs.gather(1, wanted[rows, None]).squeeze(1)
    return -picked.clamp_min(-1e4).sum(), (chosen[rows] == wanted[rows]).sum(), len(rows)


def complete_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fill graded-target and lookahead keys a batch lacks.

    Single-answer batches get a one-hot policy and no value or move-value targets;
    missing lookahead labels are ignored.
    """

    size, actions = batch["legal_mask"].shape
    device = batch["legal_mask"].device
    completed = dict(batch)
    if "policy_target" not in batch:
        policy = torch.zeros((size, actions), device=device)
        policy[torch.arange(size, device=device), batch["action"]] = 1.0
        completed.update(
            policy_target=policy,
            move_value_target=torch.zeros((size, actions), device=device),
            move_value_mask=torch.zeros((size, actions), dtype=torch.bool, device=device),
            value=torch.zeros(size, device=device),
            value_mask=torch.zeros(size, dtype=torch.bool, device=device),
        )
    completed.setdefault("value_mask", torch.ones(size, dtype=torch.bool, device=device))
    for key in ("reply_from", "reply_to"):
        completed.setdefault(key, torch.full((size,), IGNORE, dtype=torch.long, device=device))
    for key in ("board2", "board4"):
        completed.setdefault(key, torch.full((size, 64), IGNORE, dtype=torch.long, device=device))
    line = torch.full((size, LINE_PLIES), -1, dtype=torch.long, device=device)
    completed.setdefault("line_actions", line)
    return completed
