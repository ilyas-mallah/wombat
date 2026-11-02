import dataclasses
import random

import chess
import numpy as np
import pytest
import torch

from wombat.chess.encode import canonical_move, canonical_square, encode_position, state_vector
from wombat.chess.moves import action_to_move, move_to_action
from wombat.model import NetworkConfig, WombatNetwork, count_parameters
from wombat.model.board_features import (
    BOARD_FEATURES,
    LONG_RANGE_FEATURES,
    BoardFeatures,
    MoveGraph,
)
from wombat.model.network import RecurrentBlock, _move_destinations


def test_network_shapes_and_finite_outputs() -> None:
    config = NetworkConfig(d_model=48, attention_heads=6, mlp_width=96, prelude_blocks=1)
    model = WombatNetwork(config)
    pieces = torch.zeros((2, 64), dtype=torch.long)
    state = torch.zeros((2, 26))
    policy, value = model(pieces, state, reasoning_steps=2)
    assert policy.shape == (2, 4672)
    assert value.shape == (2,)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


def test_same_parameters_are_used_at_every_depth() -> None:
    config = NetworkConfig(d_model=24, attention_heads=6, mlp_width=48, prelude_blocks=1)
    model = WombatNetwork(config)
    before = count_parameters(model)
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    outputs = model(
        pieces,
        state,
        reasoning_steps=8,
        intermediate_steps=frozenset({1, 2, 4, 8}),
    )
    assert set(outputs) == {1, 2, 4, 8}
    assert count_parameters(model) == before


def test_recurrent_function_preserves_full_hidden_shape() -> None:
    config = NetworkConfig(
        d_model=24,
        attention_heads=6,
        mlp_width=48,
        prelude_blocks=0,
        recurrent_blocks=1,
        latent_tokens=8,
    )
    model = WombatNetwork(config)
    pieces = torch.zeros((2, 64), dtype=torch.long)
    state = torch.zeros((2, 26))
    board_memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(board_memory)
    updated = model.recurrent_step(board_memory, hidden)
    assert hidden.shape == (2, 73, 24)
    assert updated.shape == hidden.shape
    assert torch.equal(updated, hidden)


def test_recurrent_core_can_leave_identity_after_learning_signal() -> None:
    config = NetworkConfig(
        d_model=24, attention_heads=6, mlp_width=48, prelude_blocks=0, recurrent_blocks=1
    )
    model = WombatNetwork(config)
    with torch.no_grad():
        model.recurrent_core[0].attention.out_proj.weight.fill_(0.01)
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory)
    assert not torch.equal(model.recurrent_step(memory, hidden), hidden)


def test_reasoning_depth_is_not_architecturally_capped() -> None:
    config = NetworkConfig(
        d_model=12,
        attention_heads=3,
        mlp_width=24,
        prelude_blocks=0,
        recurrent_blocks=1,
        latent_tokens=4,
    )
    model = WombatNetwork(config)
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    policy, value = model(pieces, state, reasoning_steps=129)
    assert policy.shape == (1, 4672)
    assert value.shape == (1,)


def test_v1_model_stays_inside_parameter_budget() -> None:
    parameters = count_parameters(WombatNetwork())
    assert 2_000_000 <= parameters <= 5_000_000


def test_reasoning_trace_records_h0_through_hn_and_attention() -> None:
    config = NetworkConfig(
        d_model=24,
        attention_heads=6,
        mlp_width=48,
        prelude_blocks=0,
        recurrent_blocks=2,
        latent_tokens=4,
    )
    model = WombatNetwork(config).eval()
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    traces = model.trace_reasoning(pieces, state, reasoning_steps=3)
    assert [trace.step for trace in traces] == [0, 1, 2, 3]
    assert traces[0].attention_maps == ()
    for trace in traces:
        assert trace.hidden.shape == (1, 69, 24)
        assert trace.policy_logits.shape == (1, 4672)
        assert trace.value.shape == (1,)
    for trace in traces[1:]:
        assert len(trace.attention_maps) == 2
        assert trace.attention_maps[0].shape == (1, 6, 69, 134)


def test_h0_is_a_position_aware_non_recurrent_baseline() -> None:
    model = WombatNetwork(
        NetworkConfig(d_model=24, attention_heads=6, mlp_width=48, prelude_blocks=1)
    ).eval()
    pieces = torch.zeros((2, 64), dtype=torch.long)
    pieces[1, 0] = 6
    state = torch.zeros((2, 26))
    policy, value = model(pieces, state, reasoning_steps=0)
    assert not torch.equal(policy[0], policy[1])
    assert not torch.equal(value[0], value[1])


def test_recurrent_residual_bound_caps_each_token_rms() -> None:
    limit = 0.2
    block = RecurrentBlock(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            mlp_width=48,
            prelude_blocks=0,
            recurrent_blocks=1,
            max_residual_update=limit,
        )
    )
    update = torch.randn(3, 7, 24) * 100
    bounded = block._bound_update(update)
    rms = bounded.float().square().mean(dim=-1).sqrt()
    assert torch.all(rms <= limit + 1e-6)
    cosine = torch.nn.functional.cosine_similarity(update, bounded, dim=-1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5)


def test_zero_residual_update_has_finite_gradient() -> None:
    block = RecurrentBlock(
        NetworkConfig(d_model=24, attention_heads=6, mlp_width=48, prelude_blocks=0)
    )
    update = torch.zeros(2, 3, 24, requires_grad=True)
    block._bound_update(update).sum().backward()
    assert update.grad is not None
    assert torch.isfinite(update.grad).all()


def test_continuation_decoder_trains_on_candidate_lines() -> None:
    model = WombatNetwork(
        NetworkConfig(d_model=24, attention_heads=6, mlp_width=48, prelude_blocks=0)
    )
    pieces = torch.zeros((2, 64), dtype=torch.long)
    state = torch.zeros((2, 26))
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory)
    actions = torch.tensor([[[10, 20, 30]], [[40, 50, 0]]])
    mask = torch.tensor([[[True, True, True]], [[True, True, False]]])
    loss = model.continuation_loss(hidden, actions, mask)
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.continuation_cell.weight_hh.grad is not None
    assert model.initial_scratchpad.grad is not None
    assert model.initial_scratchpad.grad.abs().sum() > 0


def test_scratchpad_backup_connects_latent_branches_to_policy() -> None:
    model = WombatNetwork(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            mlp_width=48,
            prelude_blocks=0,
            latent_tokens=4,
            use_scratchpad_backup=True,
        )
    )
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory).detach().requires_grad_(True)
    policy, _, move_values = model.heads(hidden, include_move_values=True)
    (policy[0, 0] + move_values[0, 0]).backward()
    assert hidden.grad is not None
    assert hidden.grad[:, 65:].abs().sum() > 0


def test_future_workspaces_directly_drive_policy_and_predict_resulting_boards() -> None:
    model = WombatNetwork(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            mlp_width=48,
            prelude_blocks=0,
            latent_tokens=6,
            use_future_workspaces=True,
            future_lines=2,
            future_horizon=3,
        )
    )
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory).detach().requires_grad_(True)
    policy, value = model.heads(hidden)
    actions, future_pieces, future_states, future_values = model.future_predictions(hidden)
    assert actions.shape == (1, 2, 3, 4672)
    assert future_pieces.shape == (1, 2, 3, 64, 13)
    assert future_states.shape == (1, 2, 3, 26)
    assert future_values.shape == (1, 2, 3)
    (policy[0, 0] + value[0]).backward()
    assert hidden.grad is not None
    assert hidden.grad[:, 65:].abs().sum() > 0


def _square_model(**overrides: object) -> WombatNetwork:
    settings: dict[str, object] = {
        "d_model": 24,
        "attention_heads": 6,
        "mlp_width": 48,
        "prelude_blocks": 0,
        "recurrent_blocks": 1,
        "latent_tokens": 4,
        "policy_readout": "square",
    }
    settings.update(overrides)
    return WombatNetwork(NetworkConfig(**settings)).eval()  # type: ignore[arg-type]


def test_square_readout_keeps_each_squares_moves_on_its_own_token() -> None:
    model = _square_model()
    pieces = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 26))
    hidden = model.initial_hidden(model.encode_board(pieces, state))
    policy, value, move_values = model.heads(hidden, include_move_values=True)
    assert policy.shape == (1, 4672)
    assert move_values.shape == (1, 4672)
    assert value.shape == (1,)
    changed = hidden.clone()
    changed[:, 12] += torch.randn(24)
    moved, _ = model.heads(changed)
    difference = (moved - policy).abs()[0].reshape(64, 73).sum(dim=1)
    assert difference[12] > 0
    assert torch.count_nonzero(difference) == 1


def test_square_readout_drops_global_action_heads() -> None:
    square = _square_model()
    global_readout = _square_model(policy_readout="global")
    assert count_parameters(square) < count_parameters(global_readout) - 64 * 73 * 24
    actions = torch.zeros((1, 1, 2), dtype=torch.long)
    mask = torch.ones_like(actions, dtype=torch.bool)
    hidden = torch.zeros((1, 69, 24))
    with pytest.raises(RuntimeError):
        square.continuation_loss(hidden, actions, mask)
    with pytest.raises(ValueError):
        _square_model(use_future_workspaces=True, latent_tokens=32)


def test_normalized_state_stays_on_unit_rms_at_any_depth() -> None:
    model = _square_model(normalize_state=True, max_residual_update=None)
    with torch.no_grad():
        for block in model.recurrent_core:
            block.attention.out_proj.weight.normal_(std=0.5)
            block.mlp.output.weight.normal_(std=0.5)
    pieces = torch.randint(0, 13, (2, 64))
    state = torch.zeros((2, 26))
    outputs = model(
        pieces,
        state,
        reasoning_steps=200,
        intermediate_steps=frozenset({1, 200}),
        include_hidden=True,
    )
    for step in (1, 200):
        hidden = outputs[step][-1]
        rms = hidden.square().mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
        assert torch.isfinite(outputs[step][0]).all()


def test_unbounded_residual_update_is_passed_through() -> None:
    block = RecurrentBlock(
        NetworkConfig(d_model=24, attention_heads=6, mlp_width=48, max_residual_update=None)
    )
    update = torch.randn(2, 3, 24) * 100
    assert torch.equal(block._bound_update(update), update)


def test_zero_square_bias_matches_standard_attention_and_learns() -> None:
    torch.manual_seed(0)
    plain = RecurrentBlock(
        NetworkConfig(d_model=24, attention_heads=6, mlp_width=48, max_residual_update=None)
    )
    biased = RecurrentBlock(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            mlp_width=48,
            max_residual_update=None,
            square_attention_bias=True,
        )
    )
    with torch.no_grad():
        plain.attention.out_proj.weight.normal_(std=0.2)
    biased.load_state_dict(plain.state_dict(), strict=False)
    memory = torch.randn(2, 65, 24)
    hidden = torch.randn(2, 69, 24)
    assert torch.allclose(plain(memory, hidden), biased(memory, hidden), atol=1e-5)
    traced, attention = biased(memory, hidden, capture_attention=True)
    assert attention.shape == (2, 6, 69, 134)
    assert torch.allclose(traced, biased(memory, hidden), atol=1e-5)
    with torch.no_grad():
        biased.square_bias[:, 0, 5] = 3.0
    changed = biased(memory, hidden)
    assert not torch.allclose(changed, plain(memory, hidden), atol=1e-4)
    changed.sum().backward()
    assert biased.square_bias.grad is not None
    assert biased.square_bias.grad.abs().sum() > 0


def test_lookahead_heads_score_future_squares_with_the_piece_embeddings() -> None:
    model = _square_model(lookahead_heads=True)
    pieces = torch.randint(0, 13, (2, 64))
    state = torch.zeros((2, 26))
    hidden = model.initial_hidden(model.encode_board(pieces, state))
    outputs = model.lookahead(hidden)
    assert outputs["reply_from"].shape == (2, 64)
    assert outputs["reply_to"].shape == (2, 64)
    assert outputs["board2"].shape == (2, 64, 13)
    assert outputs["board4"].shape == (2, 64, 13)
    outputs["board2"].sum().backward()
    assert model.piece_embedding.weight.grad is not None
    with pytest.raises(ValueError):
        _square_model(lookahead_heads=True, policy_readout="global")


V7_KEYS = [
    "initial_scratchpad",
    "piece_embedding.weight",
    "square_embedding.weight",
    "state_projection.0.weight",
    "state_projection.0.bias",
    "state_projection.2.weight",
    "state_projection.2.bias",
    "prelude.0.self_attn.in_proj_weight",
    "prelude.0.self_attn.in_proj_bias",
    "prelude.0.self_attn.out_proj.weight",
    "prelude.0.self_attn.out_proj.bias",
    "prelude.0.linear1.weight",
    "prelude.0.linear1.bias",
    "prelude.0.linear2.weight",
    "prelude.0.linear2.bias",
    "prelude.0.norm1.weight",
    "prelude.0.norm1.bias",
    "prelude.0.norm2.weight",
    "prelude.0.norm2.bias",
    "recurrent_core.0.square_bias",
    "recurrent_core.0.hidden_norm.weight",
    "recurrent_core.0.memory_norm.weight",
    "recurrent_core.0.attention.in_proj_weight",
    "recurrent_core.0.attention.in_proj_bias",
    "recurrent_core.0.attention.out_proj.weight",
    "recurrent_core.0.attention.out_proj.bias",
    "recurrent_core.0.mlp_norm.weight",
    "recurrent_core.0.mlp.input.weight",
    "recurrent_core.0.mlp.output.weight",
    "recurrent_core.0.update_gate.weight",
    "recurrent_core.0.update_gate.bias",
    "state_norm.weight",
    "final_norm.weight",
    "final_norm.bias",
    "square_policy_head.weight",
    "square_policy_head.bias",
    "square_move_value_head.0.weight",
    "square_move_value_head.0.bias",
    "reply_from_head.weight",
    "reply_from_head.bias",
    "reply_to_head.weight",
    "reply_to_head.bias",
    "ply2_projection.weight",
    "ply2_projection.bias",
    "ply4_head.weight",
    "ply4_head.bias",
    "value_head.0.weight",
    "value_head.0.bias",
    "value_head.2.weight",
    "value_head.2.bias",
]


def _v7_model(**overrides: object) -> WombatNetwork:
    """A tiny v7-shaped model with its recurrent core moved off the identity."""

    torch.manual_seed(0)
    model = _square_model(
        prelude_blocks=1,
        normalize_state=True,
        max_residual_update=None,
        square_attention_bias=True,
        lookahead_heads=True,
        **overrides,
    )
    with torch.no_grad():
        for block in model.recurrent_core:
            assert isinstance(block, RecurrentBlock)
            block.attention.out_proj.weight.normal_(std=0.3)
            block.mlp.output.weight.normal_(std=0.3)
            block.square_bias.normal_()
    return model


def _first_block(model: WombatNetwork) -> RecurrentBlock:
    block = model.recurrent_core[0]
    assert isinstance(block, RecurrentBlock)
    return block


def _random_positions(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    pieces = torch.randint(0, 13, (count, 64), generator=generator)
    return pieces, torch.rand((count, 26), generator=generator)


def test_zero_scratch_boards_reproduce_the_v7_model() -> None:
    model = _v7_model(scratch_boards=0)
    assert list(model.state_dict()) == V7_KEYS
    assert model.hidden_tokens == 69
    with torch.no_grad():
        policy, value, move_values = model(*_random_positions(2), 3, include_move_values=True)
    # Recorded from the network before scratch boards existed, same seed and inputs.
    expected_policy = torch.tensor(
        [[0.028516, 0.509502, -0.017759], [0.154749, -0.692079, -0.264188]]
    )
    assert torch.allclose(policy[:, :3], expected_policy, atol=1e-5)
    assert policy.sum().item() == pytest.approx(118.543259, abs=1e-3)
    assert torch.allclose(value, torch.tensor([-0.136581, -0.098831]), atol=1e-5)
    assert move_values.sum().item() == pytest.approx(-2.367568, abs=1e-4)


def test_scratch_boards_extend_the_state_but_not_the_readouts() -> None:
    model = _v7_model(scratch_boards=2)
    assert model.hidden_tokens == 69 + 128
    assert model.scratch_board_embedding.shape == (2, 24)
    assert model.scratch_content.shape == (24,)
    assert not hasattr(model, "flat_scratch_embedding")
    extra = set(model.state_dict()) - set(V7_KEYS)
    assert extra == {"scratch_board_embedding", "scratch_content"}
    pieces, state = _random_positions(2)
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory)
    assert hidden.shape == (2, 197, 24)
    # Scratch token (board b, square s) is square s's embedding plus board b's plus the content.
    boards = hidden[:, 69:].unflatten(1, (2, 64))
    expected = (
        model.square_embedding.weight[None]
        + model.scratch_board_embedding[:, None]
        + model.scratch_content
    )
    assert torch.allclose(boards[0], expected)
    assert torch.equal(boards[0], boards[1])
    outputs = model(pieces, state, 2, include_move_values=True, include_hidden=True)
    policy, value, move_values, final = outputs
    assert policy.shape == move_values.shape == (2, 4672)
    assert value.shape == (2,)
    assert final.shape == (2, 197, 24)
    lookahead = model.lookahead(final)
    assert lookahead["board2"].shape == (2, 64, 13)
    # Heads read only the squares and the state token: rewriting a scratch board changes nothing.
    changed = final.clone()
    changed[:, 69:] = torch.randn(2, 128, 24)
    assert torch.equal(model.heads(changed)[0], model.heads(final)[0])
    assert torch.equal(model.lookahead(changed)["board4"], lookahead["board4"])
    (policy.sum() + value.sum() + move_values.sum() + lookahead["board2"].sum()).backward()
    for parameter in (model.scratch_board_embedding, model.scratch_content):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model.recurrent_step(memory, hidden[:, :69])


def test_refreshed_board_carries_memory_only_in_scratch_tokens() -> None:
    model = _v7_model(scratch_boards=1, refresh_board=True)
    pieces, state = _random_positions(2)
    with torch.no_grad():
        memory = model.encode_board(pieces, state)
        hidden = model.recurrent_step(memory, model.initial_hidden(memory))
        assert isinstance(hidden, torch.Tensor)
        overwritten = hidden.clone()
        overwritten[:, :65] = torch.randn(2, 65, 24)
        # The step starts from the board again, so old square and state tokens are ignored.
        assert torch.equal(
            model.recurrent_step(memory, overwritten), model.recurrent_step(memory, hidden)
        )
        scrambled = hidden.clone()
        scrambled[:, 65:] = torch.randn(2, 4 + 64, 24)
        assert not torch.allclose(
            model.recurrent_step(memory, scrambled), model.recurrent_step(memory, hidden)
        )


def test_refreshed_board_without_scratch_memory_gains_nothing_from_depth() -> None:
    model = _v7_model(latent_tokens=0, refresh_board=True)
    pieces, state = _random_positions(2)
    with torch.no_grad():
        once = model(pieces, state, 1)[0]
        assert torch.allclose(model(pieces, state, 5)[0], once)


def _changed_by_a1(model: WombatNetwork) -> list[int]:
    """Square tokens one block changes when a1's board and H tokens change."""

    pieces, state = _random_positions(2)
    block = _first_block(model)
    with torch.no_grad():
        memory = model.encode_board(pieces, state)
        hidden = model.initial_hidden(memory)
        before = block(memory, hidden)
        moved_memory, moved_hidden = memory.clone(), hidden.clone()
        moved_memory[:, 0] += 1.0
        moved_hidden[:, 0] += 1.0
        after = block(moved_memory, moved_hidden)
    assert isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor)
    changed = (after - before).abs().amax(dim=(0, 2)) > 1e-6
    return changed[:64].nonzero().flatten().tolist()


def test_local_attention_can_also_reach_a_knight_jump() -> None:
    assert _changed_by_a1(_v7_model(local_attention=True)) == [0, 1, 8, 9]
    king = _v7_model(local_attention=True, local_neighbourhood="king")
    assert _changed_by_a1(king) == [0, 1, 8, 9]
    knight = _v7_model(local_attention=True, local_neighbourhood="king_knight")
    # a1, b1, a2, b2, then the knight jumps c2 and b3.
    assert _changed_by_a1(knight) == [0, 1, 8, 9, 10, 17]
    assert list(knight.state_dict()) == list(king.state_dict())
    # Each square reaches its king neighbours and knight jumps, and nothing else.
    mask = _first_block(knight).king_mask
    assert mask.device == knight.square_embedding.weight.device
    assert torch.equal((mask == 0).sum(dim=1)[[0, 27]], torch.tensor([6, 17]))
    with pytest.raises(ValueError):
        _v7_model(local_attention=True, local_neighbourhood="queen")
    with pytest.raises(ValueError):
        _v7_model(local_neighbourhood="king_knight")


def test_local_attention_moves_information_one_king_step_per_block() -> None:
    model = _v7_model(local_attention=True)
    pieces, state = _random_positions(2)
    block = _first_block(model)
    with torch.no_grad():
        memory = model.encode_board(pieces, state)
        hidden = model.initial_hidden(memory)
        before = block(memory, hidden)
        assert isinstance(before, torch.Tensor)
        moved_memory, moved_hidden = memory.clone(), hidden.clone()
        moved_memory[:, 0] += 1.0  # a1 in the canonical frame
        moved_hidden[:, 0] += 1.0
        after = block(moved_memory, moved_hidden)
        assert isinstance(after, torch.Tensor)
        changed = (after - before).abs().amax(dim=(0, 2)) > 1e-6
        # Within one king step of a1: a1, b1, a2, b2.
        assert changed[:64].nonzero().flatten().tolist() == [0, 1, 8, 9]
        # H's state and latent tokens see the whole board.
        assert changed[64:].all()
        rewritten = hidden.clone()
        rewritten[:, 64:] = torch.randn_like(rewritten[:, 64:])
        rewritten_after = block(memory, rewritten)
        assert isinstance(rewritten_after, torch.Tensor)
        assert torch.equal(rewritten_after[:, :64], before[:, :64])


def _attached_edges(memory: torch.Tensor) -> torch.Tensor:
    edges = getattr(memory, "local_edges")
    assert isinstance(edges, torch.Tensor)
    return edges


def test_move_graph_moves_information_along_a_rooks_open_file_in_one_block() -> None:
    model = _v7_model(local_attention=True, local_neighbourhood="move_graph")
    assert list(model.state_dict()) == list(_v7_model(local_attention=True).state_dict())
    board = chess.Board("7k/8/8/8/8/8/8/R3K3 w - - 0 1")
    pieces, state = _encoded([board])
    block = _first_block(model)
    with torch.no_grad():
        memory = model.encode_board(pieces, state)
        edges = _attached_edges(memory)
        assert torch.equal(edges == 1, MoveGraph()(pieces))
        hidden = model.initial_hidden(memory)
        before = block(memory, hidden, local_edges=edges)
        moved_memory, moved_hidden = memory.clone(), hidden.clone()
        moved_memory[:, chess.A1] += 1.0
        moved_hidden[:, chess.A1] += 1.0
        after = block(moved_memory, moved_hidden, local_edges=edges)
    assert isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor)
    changed = (after - before).abs().amax(dim=(0, 2)) > 1e-6
    # Exactly the squares the rook reaches or that reach it: a2 to a8 and b1 to e1, and b2.
    reached = [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.B2]
    reached += [chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7, chess.A8]
    assert changed[:64].nonzero().flatten().tolist() == sorted(reached)
    assert changed[chess.A8] and not changed[chess.C5] and not changed[chess.H8]
    assert changed[64:].all()


def test_move_graph_edges_follow_each_position_through_every_step_path() -> None:
    model = _v7_model(local_attention=True, local_neighbourhood="move_graph").eval()
    boards = _positions_for_both_colours()
    pieces, state = _encoded(boards)
    memory = model.encode_board(pieces, state)
    edges = _attached_edges(memory)
    assert edges.shape == (len(boards), 64, 64)
    assert not torch.equal(edges[0], edges[1])
    hidden = model.initial_hidden(memory)
    with torch.no_grad():
        stepped = model.recurrent_step(memory, hidden)
        explicit = model.recurrent_step(memory.clone(), hidden, local_edges=edges)
        assert isinstance(stepped, torch.Tensor) and isinstance(explicit, torch.Tensor)
        assert torch.equal(stepped, explicit)
        with pytest.raises(ValueError, match="local edges"):
            model.recurrent_step(memory.clone(), hidden)
        # Another position's edges change the result.
        swapped = model.recurrent_step(memory, hidden, local_edges=edges.flip(0))
        assert isinstance(swapped, torch.Tensor)
        assert not torch.allclose(swapped, stepped)
        policy = model(pieces, state, 3)[0]
    traces = model.trace_reasoning(pieces, state, 3, move_to_cpu=False)
    assert torch.allclose(traces[-1].policy_logits, policy, atol=1e-5)
    assert traces[-1].attention_maps[0].shape == (len(boards), 6, 69, 134)
    with pytest.raises(ValueError, match="only to local attention"):
        _v7_model(local_neighbourhood="move_graph")


def test_move_lines_add_a_blocker_bias_on_line_edges_only() -> None:
    graph = _v7_model(local_attention=True, local_neighbourhood="move_graph", recurrent_blocks=2)
    model = _v7_model(local_attention=True, local_neighbourhood="move_lines", recurrent_blocks=2)
    extra = set(model.state_dict()) - set(graph.state_dict())
    assert extra == {"recurrent_core.0.blocker_bias", "recurrent_core.1.blocker_bias"}
    # The bishop on b5 pins the knight on c6; the rook on a1 sees a8 past the pawn on a4.
    board = chess.Board("4k3/8/2n5/1B6/p7/8/8/R3K3 w - - 0 1")
    pieces, _ = _encoded([board])
    edges = model.local_edges(pieces)
    assert edges is not None
    block = _first_block(model)
    with torch.no_grad():
        block.blocker_bias.copy_(torch.arange(18.0).view(6, 3))
    blocker = block.blocker_bias.detach()
    bias = block.attention_bias(65, model.hidden_tokens, local_edges=edges).detach()[0]
    table = block.square_bias.detach()
    a1, a8, b5, c6, e8 = chess.A1, chess.A8, chess.B5, chess.C6, chess.E8
    assert torch.equal(bias[:, b5, e8], table[:, b5, e8] + blocker[:, 1])
    assert torch.equal(bias[:, a8, 65 + a1], table[:, a8, a1] + blocker[:, 1])
    assert torch.equal(bias[:, b5, 65 + c6], table[:, b5, c6] + blocker[:, 0])
    # A knight's jump and a king step off every line get no blocker bias; other pairs are blocked.
    assert torch.equal(bias[:, c6, chess.E5], table[:, c6, chess.E5])
    assert torch.equal(bias[:, chess.E1, chess.F2], table[:, chess.E1, chess.F2])
    assert torch.isinf(bias[:, b5, chess.H8]).all()


def test_move_lines_blocker_bias_learns_in_a_training_step() -> None:
    model = _v7_model(local_attention=True, local_neighbourhood="move_lines", recurrent_blocks=2)
    model.train()
    boards = _positions_for_both_colours()
    pieces, state = _encoded(boards)
    first_moves = [next(iter(board.legal_moves)) for board in boards]
    actions = torch.tensor([move_to_action(*pair) for pair in zip(boards, first_moves)])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    policy, value = model(pieces, state, 2)
    loss = torch.nn.functional.cross_entropy(policy, actions) + value.square().mean()
    loss.backward()
    before = [block.blocker_bias.detach().clone() for block in model.recurrent_core]
    for block in model.recurrent_core:
        gradient = block.blocker_bias.grad
        assert gradient is not None and bool((gradient != 0).all())
    optimizer.step()
    for block, old in zip(model.recurrent_core, before):
        assert not torch.equal(block.blocker_bias, old)


def test_local_attention_needs_the_square_bias() -> None:
    with pytest.raises(ValueError):
        _square_model(local_attention=True)


def _square_blocks(bias: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """(heads, row squares, 64, column squares, 64) view of a bias whose rows and columns are
    whole square spans."""

    return bias.reshape(bias.shape[0], rows, 64, columns, 64)


def test_square_bias_tiles_over_real_and_scratch_squares() -> None:
    model = _v7_model(scratch_boards=2)
    block = _first_block(model)
    table = block.square_bias.detach()
    bias = block.attention_bias(65, model.hidden_tokens).detach()
    assert bias.shape == (6, 197, 65 + 197)
    # H rows: squares 0..63, state 64, latent 65..68, boards 69..196. Memory columns: board
    # squares 0..63, board state 64, then H's layout shifted by 65.
    square_rows = torch.cat((torch.arange(64), torch.arange(69, 197)))
    square_columns = torch.cat((torch.arange(64), 65 + square_rows))
    blocks = _square_blocks(bias[:, square_rows][:, :, square_columns], 3, 4)
    for row in range(3):
        for column in range(4):
            assert torch.equal(blocks[:, row, :, column], table)
    other_rows = torch.arange(64, 69)
    other_columns = torch.tensor([64, 129, 130, 131, 132, 133])
    assert torch.count_nonzero(bias[:, other_rows]) == 0
    assert torch.count_nonzero(bias[:, :, other_columns]) == 0
    # Scratch board 1 square 10 against real square 20, and board 0 square 3 against board 1
    # square 7.
    assert torch.equal(bias[:, 69 + 64 + 10, 20], table[:, 10, 20])
    assert torch.equal(bias[:, 69 + 3, 65 + 69 + 64 + 7], table[:, 3, 7])


def test_biased_attention_matches_its_traced_form_with_scratch_boards() -> None:
    model = _v7_model(scratch_boards=2)
    pieces, state = _random_positions(2)
    memory = model.encode_board(pieces, state)
    hidden = model.initial_hidden(memory)
    block = _first_block(model)
    traced, attention = block(memory, hidden, capture_attention=True)
    assert attention.shape == (2, 6, 197, 262)
    assert torch.allclose(traced, block(memory, hidden), atol=1e-5)


def test_flat_scratch_has_no_square_embedding_or_square_bias() -> None:
    model = _v7_model(scratch_boards=2, flat_scratch=True)
    assert model.hidden_tokens == 197
    assert model.flat_scratch_embedding.shape == (128, 24)
    tokens = model.scratch_board_tokens()
    expected = (
        model.flat_scratch_embedding.unflatten(0, (2, 64))
        + model.scratch_board_embedding[:, None]
        + model.scratch_content
    )
    assert torch.allclose(tokens, expected.flatten(0, 1))
    with torch.no_grad():
        model.square_embedding.weight.add_(1.0)
    assert torch.equal(model.scratch_board_tokens(), tokens)
    block = _first_block(model)
    bias = block.attention_bias(65, 197).detach()
    table = block.square_bias.detach()
    assert torch.count_nonzero(bias[:, 69:]) == 0
    assert torch.count_nonzero(bias[:, :, 65 + 69 :]) == 0
    assert torch.equal(bias[:, :64, :64], table)
    assert torch.equal(bias[:, :64, 65:129], table)
    policy, value = model(*_random_positions(2), 2)
    assert policy.shape == (2, 4672)
    board_count = count_parameters(_v7_model(scratch_boards=2))
    assert count_parameters(model) == board_count + 128 * 24
    with pytest.raises(ValueError):
        _square_model(flat_scratch=True)


def test_feedforward_baseline_reads_out_without_recurrence() -> None:
    model = _square_model(prelude_blocks=2, recurrent_blocks=0)
    pieces = torch.randint(0, 13, (2, 64))
    policy, value = model(pieces, torch.zeros((2, 26)), reasoning_steps=0)
    assert policy.shape == (2, 4672)
    assert torch.isfinite(value).all()


FROM_TO_KEYS = {
    f"{name}.{part}"
    for name in ("policy_query", "policy_key", "move_value_query", "move_value_key")
    for part in ("weight", "bias")
}


def test_from_to_readout_adds_only_its_projections() -> None:
    assert list(_v7_model(move_readout="square").state_dict()) == V7_KEYS
    model = _v7_model(move_readout="from_to")
    assert set(model.state_dict()) - set(V7_KEYS) == FROM_TO_KEYS
    assert set(V7_KEYS) <= set(model.state_dict())
    assert model.policy_query.weight.shape == (64, 24)
    assert model.move_destinations.shape == (64, 73)
    with pytest.raises(ValueError):
        _square_model(move_readout="to")
    with pytest.raises(ValueError):
        _square_model(move_readout="from_to", policy_readout="global")


def test_from_to_readout_adds_the_origin_destination_score() -> None:
    square = _v7_model()
    model = _v7_model(move_readout="from_to")
    model.load_state_dict(square.state_dict(), strict=False)
    pieces, state = _random_positions(2)
    with torch.no_grad():
        expected = square(pieces, state, 2, include_move_values=True, include_hidden=True)
        policy, value, move_values, hidden = model(
            pieces, state, 2, include_move_values=True, include_hidden=True
        )
        assert torch.equal(hidden, expected[3])
        assert torch.equal(value, expected[1])
        tokens = model.final_norm(hidden[:, :64])
        destinations = tokens[:, model.move_destinations]

        def score(query: torch.nn.Linear, key: torch.nn.Linear) -> torch.Tensor:
            return (query(tokens).unsqueeze(2) * key(destinations)).sum(-1) / 8

        logits = square.square_policy_head(tokens) + score(model.policy_query, model.policy_key)
        assert torch.allclose(policy, logits.flatten(1), atol=1e-5)
        linear = square.square_move_value_head[0]
        values = torch.tanh(linear(tokens) + score(model.move_value_query, model.move_value_key))
        assert torch.allclose(move_values, values.flatten(1), atol=1e-5)
        # Without the query projections the readout is the square readout exactly.
        model.policy_query.weight.zero_()
        model.policy_query.bias.zero_()
        model.move_value_query.weight.zero_()
        model.move_value_query.bias.zero_()
        unscored = model(pieces, state, 2, include_move_values=True)
        assert torch.equal(unscored[0], expected[0])
        assert torch.equal(unscored[2], expected[2])


def _positions_for_both_colours() -> list[chess.Board]:
    randomizer = random.Random(3)
    boards: list[chess.Board] = []
    while len(boards) < 12:
        board = chess.Board()
        for _ in range(randomizer.randrange(1, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        if any(board.legal_moves):
            boards.append(board)
    # Castling both ways, and promotions straight ahead and capturing either way, for each side.
    for position in ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R", "rn2k2n/1P4P1/8/8/8/8/1p4p1/RN2K2N"):
        castling = "KQkq" if position.startswith("r3") else "-"
        for turn in "wb":
            boards.append(chess.Board(f"{position} {turn} {castling} - 0 1"))
    return boards


def test_move_destinations_match_python_chess() -> None:
    table = _move_destinations()
    boards = _positions_for_both_colours()
    assert {board.turn for board in boards} == {chess.WHITE, chess.BLACK}
    checked = set()
    for board in boards:
        for move in board.legal_moves:
            action = move_to_action(board, move)
            assert action_to_move(board, action) == move
            origin, plane = divmod(action, 73)
            canonical = canonical_move(move, board.turn)
            assert origin == canonical.from_square
            assert table[origin, plane] == canonical.to_square
            assert canonical.to_square == canonical_square(move.to_square, board.turn)
            checked.add(plane)
    # Every knight and underpromotion plane, and castling in both directions.
    assert set(range(56, 73)) <= checked
    assert {15, 43} <= checked
    assert ((table >= 0) & (table < 64)).all()


def test_from_to_logit_reads_the_destination_square() -> None:
    model = _v7_model(move_readout="from_to")
    pieces, state = _random_positions(2)
    with torch.no_grad():
        hidden = model(pieces, state, 1, include_hidden=True)[-1]
        readouts = model.heads(hidden, include_move_values=True)
        changed = hidden.clone()
        changed[:, 28] += torch.randn(24)  # e4, reached from e2 (square 12) by a two-square push
        moved = model.heads(changed, include_move_values=True)
    table = model.move_destinations
    # Policy logits, then move values.
    for before, after in ((readouts[0], moved[0]), (readouts[-1], moved[-1])):
        difference = (after - before).abs().amax(dim=0).reshape(64, 73)
        # From e2 only the move to e4 changes; a move from e2 to e3 does not.
        assert (difference[12] > 0).tolist() == (table[12] == 28).tolist()
        assert torch.count_nonzero(table[12] == 28) == 1
        # Each other square changes exactly its moves that land on e4.
        others = torch.arange(64) != 28
        assert torch.equal((difference > 0)[others], (table == 28)[others])


def test_from_to_projections_receive_gradients() -> None:
    model = _v7_model(move_readout="from_to")
    policy, _, move_values = model(*_random_positions(2), 2, include_move_values=True)
    (policy.sum() + move_values.sum()).backward()
    for layer in (
        model.policy_query,
        model.policy_key,
        model.move_value_query,
        model.move_value_key,
    ):
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0


PAIR_KEYS = {
    "reply_query.weight",
    "reply_query.bias",
    "reply_key.weight",
    "reply_key.bias",
    "reply_origin_bias.weight",
    "reply_origin_bias.bias",
}


def test_pair_reply_readout_replaces_only_the_square_reply_heads() -> None:
    assert list(_v7_model(reply_readout="squares").state_dict()) == V7_KEYS
    model = _v7_model(reply_readout="pair")
    square_heads = {key for key in V7_KEYS if key.startswith(("reply_from", "reply_to"))}
    assert set(model.state_dict()) == set(V7_KEYS) - square_heads | PAIR_KEYS
    assert model.reply_query.weight.shape == model.reply_key.weight.shape == (64, 24)
    wide = NetworkConfig(d_model=256, attention_heads=8, mlp_width=1024, policy_readout="square")
    wide = dataclasses.replace(wide, lookahead_heads=True)
    pair = dataclasses.replace(wide, reply_readout="pair")
    added = count_parameters(WombatNetwork(pair)) - count_parameters(WombatNetwork(wide))
    assert added == 2 * (256 * 64 + 64) + 257 - 2 * 257 == 32_639
    with pytest.raises(ValueError, match="unknown reply_readout"):
        _square_model(lookahead_heads=True, reply_readout="moves")
    with pytest.raises(ValueError, match="pair reply readout"):
        _square_model(reply_readout="pair")
    with pytest.raises(ValueError, match="pair reply readout"):
        _square_model(lookahead_heads=True, reply_readout="pair", policy_readout="global")


def test_pair_reply_logits_score_origin_queries_against_destination_keys() -> None:
    model = _v7_model(reply_readout="pair")
    pieces, state = _random_positions(2)
    with torch.no_grad():
        hidden = model(pieces, state, 2, include_hidden=True)[-1]
        outputs = model.lookahead(hidden)
        assert set(outputs) == {"reply_pair", "board2", "board4"}
        pair = outputs["reply_pair"]
        assert pair.shape == (2, 64, 64)
        tokens = model.final_norm(hidden[:, :64])
        scores = model.reply_query(tokens) @ model.reply_key(tokens).transpose(1, 2) / 8
        assert torch.allclose(pair, scores + model.reply_origin_bias(tokens), atol=1e-5)
        changed = hidden.clone()
        changed[:, 28] += torch.randn(24)
        difference = (model.lookahead(changed)["reply_pair"] - pair).abs().amax(dim=0)
    # Square 28's token is every pair's destination in column 28 and the origin of row 28.
    assert (difference[:, 28] > 0).all()
    others = torch.arange(64) != 28
    assert not difference[others][:, others].any()


def test_pair_reply_projections_receive_gradients() -> None:
    model = _v7_model(reply_readout="pair")
    hidden = model(*_random_positions(2), 2, include_hidden=True)[-1]
    pair = model.lookahead(hidden)["reply_pair"]
    loss = torch.nn.functional.cross_entropy(pair.flatten(1), torch.tensor([12 * 64 + 28, 700]))
    loss.backward()
    for layer in (model.reply_query, model.reply_key, model.reply_origin_bias):
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0


def test_board_features_add_only_their_projection() -> None:
    plain = _v7_model()
    assert list(plain.state_dict()) == V7_KEYS
    assert not hasattr(plain, "board_features")
    model = _v7_model(board_features=True)
    assert set(model.state_dict()) - set(V7_KEYS) == {"board_feature_projection.weight"}
    assert set(V7_KEYS) <= set(model.state_dict())
    assert count_parameters(model) - count_parameters(plain) == BOARD_FEATURES * 24
    wide = NetworkConfig(d_model=256, attention_heads=8, mlp_width=1024)
    featured = dataclasses.replace(wide, board_features=True)
    added = count_parameters(WombatNetwork(featured)) - count_parameters(WombatNetwork(wide))
    assert added == BOARD_FEATURES * 256 == 7936
    # The feature tables move with the model.
    model.to(torch.device("meta"))
    assert {buffer.device.type for buffer in model.board_features.buffers()} == {"meta"}


def _encoded(boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_position(board) for board in boards]
    pieces = torch.tensor(np.stack([position.pieces for position in encoded])).long()
    state = torch.tensor(np.stack([state_vector(position) for position in encoded]))
    return pieces, state


def test_board_features_are_projected_onto_the_square_tokens() -> None:
    plain = _v7_model()
    model = _v7_model(board_features=True)
    model.load_state_dict(plain.state_dict(), strict=False)
    pieces, state = _encoded(_positions_for_both_colours())
    projection = model.board_feature_projection.weight
    with torch.no_grad():
        tokens = model._tokenize(pieces, state)
        expected = plain._tokenize(pieces, state)
        features = BoardFeatures()(pieces)
        assert features.abs().sum() > 0
        assert torch.allclose(tokens[:, :64], expected[:, :64] + features @ projection.T, atol=1e-6)
        assert torch.equal(tokens[:, 64], expected[:, 64])
        # Without the projection the model is the plain model exactly.
        projection.zero_()
        outputs = model(pieces, state, 2, include_move_values=True)
        reference = plain(pieces, state, 2, include_move_values=True)
    for tensor, target in zip(outputs, reference):
        assert torch.equal(tensor, target)


def test_board_feature_projection_receives_gradients() -> None:
    model = _v7_model(board_features=True)
    policy, value = model(*_encoded(_positions_for_both_colours()), 2)
    (policy.sum() + value.sum()).backward()
    gradient = model.board_feature_projection.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_long_range_board_features_widen_only_the_projection() -> None:
    with pytest.raises(ValueError, match="needs board_features"):
        _v7_model(board_features_long_range=True)
    featured = _v7_model(board_features=True)
    model = _v7_model(board_features=True, board_features_long_range=True)
    assert list(model.state_dict()) == list(featured.state_dict())
    assert model.board_feature_projection.weight.shape == (24, 61)
    assert count_parameters(model) - count_parameters(featured) == LONG_RANGE_FEATURES * 24
    wide = NetworkConfig(d_model=256, attention_heads=8, mlp_width=1024, board_features=True)
    plain = dataclasses.replace(wide, board_features=False)
    long_range = dataclasses.replace(wide, board_features_long_range=True)
    added = count_parameters(WombatNetwork(long_range)) - count_parameters(WombatNetwork(wide))
    assert added == LONG_RANGE_FEATURES * 256 == 7680
    total = count_parameters(WombatNetwork(long_range)) - count_parameters(WombatNetwork(plain))
    assert total == (BOARD_FEATURES + LONG_RANGE_FEATURES) * 256 == 15616
    model.to(torch.device("meta"))
    assert {buffer.device.type for buffer in model.board_features.buffers()} == {"meta"}


def test_long_range_board_features_are_projected_onto_the_square_tokens() -> None:
    featured = _v7_model(board_features=True)
    model = _v7_model(board_features=True, board_features_long_range=True)
    state = featured.state_dict()
    state["board_feature_projection.weight"] = torch.randn(24, 61)
    model.load_state_dict(state)
    pieces, board_state = _encoded(_positions_for_both_colours())
    projection = model.board_feature_projection.weight
    with torch.no_grad():
        tokens = model._tokenize(pieces, board_state)
        features = BoardFeatures(long_range=True)(pieces)
        assert features[..., BOARD_FEATURES:].abs().sum() > 0
        embedded = model.piece_embedding(pieces) + model.square_embedding.weight
        assert torch.allclose(tokens[:, :64], embedded + features @ projection.T, atol=1e-5)
        assert torch.equal(tokens[:, 64], featured._tokenize(pieces, board_state)[:, 64])
        # Zero long-range columns reproduce the model with only the first feature set.
        projection[:, BOARD_FEATURES:] = 0
        featured.board_feature_projection.weight.copy_(projection[:, :BOARD_FEATURES])
        outputs = model(pieces, board_state, 2, include_move_values=True)
        reference = featured(pieces, board_state, 2, include_move_values=True)
    for tensor, target in zip(outputs, reference):
        assert torch.allclose(tensor, target, atol=1e-6)


def test_long_range_projection_receives_gradients() -> None:
    model = _v7_model(board_features=True, board_features_long_range=True)
    policy, value = model(*_encoded(_positions_for_both_colours()), 2)
    (policy.sum() + value.sum()).backward()
    gradient = model.board_feature_projection.weight.grad
    assert gradient is not None and gradient[:, BOARD_FEATURES:].abs().sum() > 0
