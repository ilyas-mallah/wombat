import torch

from wombat.eval.interventions import evaluate_hidden_interventions
from wombat.model import NetworkConfig, WombatNetwork


def test_all_hidden_state_controls_preserve_output_contract() -> None:
    model = WombatNetwork(
        NetworkConfig(
            d_model=24,
            attention_heads=6,
            prelude_blocks=0,
            recurrent_blocks=1,
            mlp_width=48,
            latent_tokens=4,
        )
    ).eval()
    # New recurrent cores intentionally start as identity maps; perturb the
    # output projection to emulate a trained recurrent checkpoint.
    with torch.no_grad():
        model.recurrent_core[0].attention.out_proj.weight.fill_(0.01)
    pieces = torch.zeros((2, 64), dtype=torch.long)
    pieces[1, 0] = 6
    state = torch.zeros((2, 26))
    outputs = evaluate_hidden_interventions(
        model, pieces, state, total_steps=8, midpoint=4, seed=7
    )
    for policy, value in (
        outputs.normal,
        outputs.reset,
        outputs.corrupted,
        outputs.earlier_state,
        outputs.swapped,
    ):
        assert policy.shape == (2, 4672)
        assert value.shape == (2,)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()
    assert not torch.equal(outputs.normal[0], outputs.reset[0])
