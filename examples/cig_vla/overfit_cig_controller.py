#!/usr/bin/env python
import torch

from lerobot.policies.cig_vla.flow_controller import FlowMatchingController
from lerobot.policies.cig_vla.flow_matching import compute_flow_loss, make_flow_training_sample
from lerobot.policies.cig_vla.interaction_bottleneck import InteractionGeometryBottleneck


def main():
    torch.manual_seed(0)
    controller = FlowMatchingController(5, 7, 16, 1, 4)
    batch, chunk = 8, 4
    source = torch.randn(batch, 3)
    bottleneck = InteractionGeometryBottleneck(
        translation_goal=source,
        approach_direction=torch.nn.functional.normalize(source, dim=-1),
        translation_magnitude=source.norm(dim=-1, keepdim=True),
        rotation_goal=torch.zeros_like(source),
        gripper_transition=torch.zeros(batch, 1),
        confidence_logit=torch.zeros(batch, 1),
        valid_mask=torch.ones(batch, 1, dtype=torch.bool),
    )
    state = torch.randn(batch, 5)
    clean = (source[:, None, :1] + state[:, None, :1]).expand(-1, chunk, 7).clone()
    noise, timestep = torch.randn_like(clean), torch.full((batch,), 0.5)
    sample = make_flow_training_sample(clean, noise=noise, timestep=timestep)
    optimizer = torch.optim.Adam(controller.parameters(), lr=3e-3)

    def loss():
        output = controller(bottleneck, state, sample.noisy_actions, sample.timestep)
        return compute_flow_loss(output, sample.target_velocity)

    initial = loss().item()
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
    final = loss().item()
    changed = bottleneck.with_translation_offset(torch.full_like(source, 0.05))
    response = (
        (
            controller(changed, state, sample.noisy_actions, sample.timestep)
            - controller(bottleneck, state, sample.noisy_actions, sample.timestep)
        )
        .norm()
        .item()
    )
    print(f"Stage B overfit: initial={initial:.6f} final={final:.6f} intervention_response={response:.6f}")
    if not final < initial * 0.5 or response <= 0:
        raise SystemExit("Stage B overfit/response threshold failed")


if __name__ == "__main__":
    main()
