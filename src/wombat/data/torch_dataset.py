"""Torch dataset backed by compact Wombat JSONL/Zstandard shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch.utils.data import Dataset
import zstandard

from wombat.chess.encode import encode_position
from wombat.chess.moves import ACTION_SIZE, legal_action_mask, move_to_action
from wombat.model.adapter import state_vector


@dataclass(frozen=True)
class TrainingExample:
    pieces: np.ndarray
    state: np.ndarray
    action: int
    value: float
    legal_mask: np.ndarray
    candidate_actions: np.ndarray
    candidate_weights: np.ndarray
    candidate_values: np.ndarray
    shallow_actions: np.ndarray
    shallow_weights: np.ndarray
    shallow_values: np.ndarray
    shallow_deep_disagree: bool
    shallow_continuation_actions: tuple[np.ndarray, ...]
    continuation_actions: tuple[np.ndarray, ...]
    continuation_pieces: tuple[np.ndarray, ...]
    continuation_states: tuple[np.ndarray, ...]


def _value_target(payload: dict[str, object]) -> float:
    mate = payload.get("score_mate")
    if mate is not None:
        return 1.0 if int(mate) > 0 else -1.0
    cp = payload.get("score_cp")
    if cp is None:
        raise ValueError("record has neither centipawn nor mate score")
    return float(np.tanh(int(cp) / 600.0))


def _score_utility(payload: dict[str, object]) -> float:
    """Map an engine score to ordered pawn-like utility for policy softmax."""

    mate = payload.get("score_mate")
    if mate is not None:
        mate = int(mate)
        return (15.0 if mate > 0 else -15.0) - 0.05 * mate
    cp = payload.get("score_cp")
    if cp is None:
        raise ValueError("candidate has neither centipawn nor mate score")
    return float(np.clip(int(cp) / 100.0, -12.0, 12.0))


class EvaluationDataset(Dataset[TrainingExample]):
    def __init__(
        self, path: Path, limit: int | None = None, *, policy_temperature: float = 0.75
    ) -> None:
        if policy_temperature <= 0:
            raise ValueError("policy_temperature must be positive")
        self.examples: list[TrainingExample] = []
        stat = path.stat()
        cache_path = path.with_name(
            f"{path.name}.tensor-cache-v5-t{policy_temperature:g}-n{limit or 'all'}.pt"
        )
        cache_key = {
            "schema": "wombat.tensor-cache.v5",
            "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "policy_temperature": policy_temperature,
            "limit": limit,
        }
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("key") == cache_key:
                self.examples = cached["examples"]
                return
            del cached
        with zstandard.open(path, mode="rt", encoding="utf-8") as source:
            for line in source:
                payload = json.loads(line)
                board = chess.Board(payload["fen"])
                encoded = encode_position(board)
                stored_action = int(payload["action"])
                mask = legal_action_mask(board)
                if not mask[stored_action]:
                    raise ValueError(f"stored target action is illegal for {payload['fen']}")
                candidates: dict[int, tuple[float, float]] = {}
                # Deep Lichess evaluations are the final target. Low-node
                # continuations remain a separate proposal target so Wombat
                # can learn to refine shallow analysis rather than averaging it
                # into the answer it is supposed to improve upon.
                deep_annotations = payload.get("deep_continuations", [])
                candidate_source = deep_annotations or payload.get("alternatives", [])
                for candidate in candidate_source:
                    candidate_action = move_to_action(
                        board, chess.Move.from_uci(str(candidate["move"]))
                    )
                    utility = _score_utility(candidate)
                    candidates[candidate_action] = (
                        utility,
                        float(np.tanh(utility / 6.0)),
                    )
                if not deep_annotations and stored_action not in candidates:
                    utility = _score_utility(payload)
                    candidates[stored_action] = (utility, _value_target(payload))
                action = next(iter(candidates))
                value = candidates[action][1]
                candidate_actions = np.fromiter(candidates, dtype=np.int64)
                utilities = np.array(
                    [candidates[item][0] for item in candidate_actions], dtype=np.float32
                )
                weights = np.exp(
                    (utilities - utilities.max()) / np.float32(policy_temperature)
                )
                weights /= weights.sum()
                candidate_values = np.array(
                    [candidates[item][1] for item in candidate_actions], dtype=np.float32
                )
                shallow_candidates: dict[int, tuple[float, float]] = {}
                for candidate in payload.get("continuations", []):
                    shallow_action = move_to_action(
                        board, chess.Move.from_uci(str(candidate["move"]))
                    )
                    shallow_utility = _score_utility(candidate)
                    shallow_candidates[shallow_action] = (
                        shallow_utility,
                        float(np.tanh(shallow_utility / 6.0)),
                    )
                if not shallow_candidates:
                    shallow_candidates = dict(candidates)
                shallow_actions = np.fromiter(shallow_candidates, dtype=np.int64)
                shallow_utilities = np.array(
                    [shallow_candidates[item][0] for item in shallow_actions], dtype=np.float32
                )
                shallow_weights = np.exp(
                    (shallow_utilities - shallow_utilities.max())
                    / np.float32(policy_temperature)
                )
                shallow_weights /= shallow_weights.sum()
                shallow_values = np.array(
                    [shallow_candidates[item][1] for item in shallow_actions], dtype=np.float32
                )
                def encode_lines(
                    field: str,
                ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
                    encoded_lines: list[np.ndarray] = []
                    piece_lines: list[np.ndarray] = []
                    state_lines: list[np.ndarray] = []
                    for continuation in payload.get(field, []):
                        line_board = board.copy(stack=False)
                        encoded_line: list[int] = []
                        piece_line: list[np.ndarray] = []
                        state_line: list[np.ndarray] = []
                        try:
                            for uci in continuation.get("pv", []):
                                move = line_board.parse_uci(str(uci))
                                encoded_line.append(move_to_action(line_board, move))
                                line_board.push(move)
                                future = encode_position(line_board)
                                piece_line.append(future.pieces.astype(np.int8))
                                state_line.append(state_vector(future).astype(np.float16))
                        except ValueError:
                            continue
                        if len(encoded_line) >= 2:
                            encoded_lines.append(np.array(encoded_line, dtype=np.int64))
                            piece_lines.append(np.stack(piece_line))
                            state_lines.append(np.stack(state_line))
                    return tuple(encoded_lines), tuple(piece_lines), tuple(state_lines)

                shallow_continuation_actions, _, _ = encode_lines("continuations")
                deep_lines = (
                    encode_lines("deep_continuations")
                    if deep_annotations
                    else encode_lines("continuations")
                )
                continuation_actions, continuation_pieces, continuation_states = deep_lines
                self.examples.append(
                    TrainingExample(
                        pieces=encoded.pieces.astype(np.int64),
                        state=state_vector(encoded),
                        action=action,
                        value=value,
                        legal_mask=mask,
                        candidate_actions=candidate_actions,
                        candidate_weights=weights,
                        candidate_values=candidate_values,
                        shallow_actions=shallow_actions,
                        shallow_weights=shallow_weights,
                        shallow_values=shallow_values,
                        shallow_deep_disagree=bool(shallow_actions[0] != action),
                        shallow_continuation_actions=shallow_continuation_actions,
                        continuation_actions=continuation_actions,
                        continuation_pieces=continuation_pieces,
                        continuation_states=continuation_states,
                    )
                )
                if limit is not None and len(self.examples) >= limit:
                    break
        temporary_cache = cache_path.with_suffix(cache_path.suffix + ".partial")
        torch.save({"key": cache_key, "examples": self.examples}, temporary_cache)
        temporary_cache.replace(cache_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def collate_examples(examples: list[TrainingExample]) -> dict[str, torch.Tensor]:
    policy_target = np.zeros((len(examples), ACTION_SIZE), dtype=np.float32)
    move_value_target = np.zeros((len(examples), ACTION_SIZE), dtype=np.float32)
    move_value_mask = np.zeros((len(examples), ACTION_SIZE), dtype=np.bool_)
    shallow_policy_target = np.zeros((len(examples), ACTION_SIZE), dtype=np.float32)
    shallow_move_value_target = np.zeros((len(examples), ACTION_SIZE), dtype=np.float32)
    shallow_move_value_mask = np.zeros((len(examples), ACTION_SIZE), dtype=np.bool_)
    for row, example in enumerate(examples):
        policy_target[row, example.candidate_actions] = example.candidate_weights
        move_value_target[row, example.candidate_actions] = example.candidate_values
        move_value_mask[row, example.candidate_actions] = True
        shallow_policy_target[row, example.shallow_actions] = example.shallow_weights
        shallow_move_value_target[row, example.shallow_actions] = example.shallow_values
        shallow_move_value_mask[row, example.shallow_actions] = True
    def pad_lines(attribute: str) -> tuple[np.ndarray, np.ndarray]:
        all_lines = [getattr(example, attribute) for example in examples]
        max_lines = max((len(lines) for lines in all_lines), default=0)
        max_plies = max(
            (len(line) for lines in all_lines for line in lines), default=0
        )
        # Keep a non-empty shape so batches without annotations follow the same
        # model path and simply receive a zero continuation loss.
        actions = np.zeros(
            (len(examples), max(1, max_lines), max(2, max_plies)), dtype=np.int64
        )
        mask = np.zeros_like(actions, dtype=np.bool_)
        for row, lines in enumerate(all_lines):
            for line_index, line in enumerate(lines):
                actions[row, line_index, : len(line)] = line
                mask[row, line_index, : len(line)] = True
        return actions, mask

    shallow_continuation_actions, shallow_continuation_mask = pad_lines(
        "shallow_continuation_actions"
    )
    continuation_actions, continuation_mask = pad_lines("continuation_actions")
    max_lines = continuation_actions.shape[1]
    max_plies = continuation_actions.shape[2]
    continuation_pieces = np.zeros(
        (len(examples), max_lines, max_plies, 64), dtype=np.int8
    )
    continuation_states = np.zeros(
        (len(examples), max_lines, max_plies, 26), dtype=np.float16
    )
    for row, example in enumerate(examples):
        for line_index, (pieces_line, states_line) in enumerate(
            zip(example.continuation_pieces, example.continuation_states, strict=True)
        ):
            plies = min(max_plies, len(pieces_line))
            continuation_pieces[row, line_index, :plies] = pieces_line[:plies]
            continuation_states[row, line_index, :plies] = states_line[:plies]
    return {
        "pieces": torch.from_numpy(np.stack([example.pieces for example in examples])),
        "state": torch.from_numpy(np.stack([example.state for example in examples])),
        "action": torch.tensor([example.action for example in examples], dtype=torch.long),
        "value": torch.tensor([example.value for example in examples], dtype=torch.float32),
        "legal_mask": torch.from_numpy(np.stack([example.legal_mask for example in examples])),
        "policy_target": torch.from_numpy(policy_target),
        "move_value_target": torch.from_numpy(move_value_target),
        "move_value_mask": torch.from_numpy(move_value_mask),
        "shallow_policy_target": torch.from_numpy(shallow_policy_target),
        "shallow_move_value_target": torch.from_numpy(shallow_move_value_target),
        "shallow_move_value_mask": torch.from_numpy(shallow_move_value_mask),
        "shallow_deep_disagree": torch.tensor(
            [example.shallow_deep_disagree for example in examples], dtype=torch.bool
        ),
        "shallow_continuation_actions": torch.from_numpy(shallow_continuation_actions),
        "shallow_continuation_mask": torch.from_numpy(shallow_continuation_mask),
        "continuation_actions": torch.from_numpy(continuation_actions),
        "continuation_mask": torch.from_numpy(continuation_mask),
        "continuation_pieces": torch.from_numpy(continuation_pieces),
        "continuation_states": torch.from_numpy(continuation_states),
    }
