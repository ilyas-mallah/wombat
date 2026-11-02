from __future__ import annotations

import dataclasses

import pytest
import torch

from wombat.cli import build_parser
from wombat.model import NetworkConfig, WombatNetwork
from wombat.model.network import RecurrentBlock

BASE = NetworkConfig(
    d_model=24,
    attention_heads=6,
    mlp_width=48,
    prelude_blocks=0,
    latent_tokens=0,
    policy_readout="square",
    move_readout="from_to",
    square_attention_bias=True,
    local_attention=True,
    relative_attention=True,
    lookahead_heads=True,
)
WITH = dataclasses.replace(BASE, qk_norm=True)


def _positions(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    pieces = torch.randint(0, 13, (count, 64), generator=generator)
    return pieces, torch.rand((count, 26), generator=generator)


def _largest_logit(model: WombatNetwork, monkeypatch: pytest.MonkeyPatch) -> float:
    seen: list[float] = []
    original = torch.nn.functional.scaled_dot_product_attention

    def watched(q, k, v, *args, **kwargs):  # type: ignore[no-untyped-def]
        scale = kwargs.get("scale")
        factor = scale if scale is not None else q.shape[-1] ** -0.5
        seen.append(float((q @ k.transpose(-2, -1) * factor).abs().max()))
        return original(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", watched)
    with torch.no_grad():
        model(*_positions(4), 3)
    return max(seen)


def test_the_option_is_off_by_default_and_needs_the_square_bias() -> None:
    assert not NetworkConfig().qk_norm
    assert not build_parser().parse_args(["train-games", "--output", "run"]).qk_norm
    assert build_parser().parse_args(["train-games", "--output", "run", "--qk-norm"]).qk_norm
    plain = dataclasses.replace(
        WITH, local_attention=False, square_attention_bias=False, relative_attention=False
    )
    with pytest.raises(ValueError, match="square bias"):
        WombatNetwork(plain)


def test_normalized_queries_and_keys_bound_logits_that_large_weights_blow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    plain, normed = WombatNetwork(BASE), WombatNetwork(WITH)
    normed.load_state_dict(plain.state_dict(), strict=False)
    for model in (plain, normed):
        with torch.no_grad():
            for block in model.recurrent_core:
                assert isinstance(block, RecurrentBlock)
                block.attention.in_proj_weight.mul_(50)
    head_width = BASE.d_model // BASE.attention_heads
    # Unit-RMS queries and keys give logits of at most sqrt(head width) before learned gains.
    assert _largest_logit(normed, monkeypatch) <= head_width**0.5 + 1e-4
    assert _largest_logit(plain, monkeypatch) > 10 * head_width**0.5
