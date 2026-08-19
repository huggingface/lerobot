import pytest
import torch
from conftest import make_batch, make_policy

from lerobot.policies.cig_vla.flow_matching import compute_flow_loss, make_flow_training_sample


def gradient_norm(module):
    values = [p.grad.square().sum() for p in module.parameters() if p.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)


@pytest.mark.parametrize(("detach_main", "expect_stage_a"), [(True, False), (False, True)])
def test_main_detach_gradient_boundary(detach_main, expect_stage_a):
    policy = make_policy(detach_main, True)
    batch = make_batch()
    prediction = policy._predict(batch)
    target = policy.target_builder.build(
        batch["action"], batch["observation.state"], policy.dataset_stats, batch["action_is_pad"]
    )
    geometry_loss, _ = policy.compute_geometry_loss(prediction, target)
    policy.zero_grad(set_to_none=True)
    geometry_loss.backward()
    assert gradient_norm(policy.grounding_head) > 1e-8
    policy.zero_grad(set_to_none=True)
    prediction = policy._predict(batch)
    bottleneck = prediction.detached() if detach_main else prediction
    flow = make_flow_training_sample(
        batch["action"],
        batch["action_is_pad"],
        noise=torch.randn_like(batch["action"]),
        timestep=torch.full((2,), 0.5),
    )
    velocity = policy.controller(bottleneck, batch["observation.state"], flow.noisy_actions, flow.timestep)
    action_loss = compute_flow_loss(velocity, flow.target_velocity, flow.action_is_pad)
    action_loss.backward()
    assert bool(gradient_norm(policy.grounding_head) > 1e-8) is expect_stage_a
    assert gradient_norm(policy.controller) > 1e-8
