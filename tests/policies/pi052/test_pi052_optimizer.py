"""PI052 backend options must not require changes to the shared optimizer config."""

import pytest
import torch

from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy


@pytest.mark.parametrize("backbone_scale", [1.0, 0.5])
def test_adamw_backend_options_stay_in_policy_groups(backbone_scale):
    policy = PI052Policy.__new__(PI052Policy)
    torch.nn.Module.__init__(policy)
    policy.weight = torch.nn.Parameter(torch.ones(2))
    policy.config = PI052Config(
        device="cpu",
        backbone_lr_scale=backbone_scale,
        optimizer_foreach=False,
        optimizer_fused=False,
    )
    preset = policy.config.get_optimizer_preset()
    assert not hasattr(preset, "fused")
    optimizer = preset.build(policy.get_optim_params())
    assert all(group["fused"] is False and group["foreach"] is False for group in optimizer.param_groups)
    policy.weight.sum().backward()
    optimizer.step()
    assert torch.isfinite(policy.weight).all()
    assert (policy.weight < 1).all()
