from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FlowTrainingSample:
    clean_actions: Tensor
    noise: Tensor
    timestep: Tensor
    noisy_actions: Tensor
    target_velocity: Tensor
    action_is_pad: Tensor | None = None


def _time_like(timestep: Tensor, actions: Tensor) -> Tensor:
    return timestep.to(device=actions.device, dtype=actions.dtype).reshape(-1, *([1] * (actions.ndim - 1)))


def interpolate_actions(clean_actions: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    """Return x_t=(1-t)x_clean+t*x_noise (LeRobot/openpi convention)."""
    t = _time_like(timestep, clean_actions)
    return (1 - t) * clean_actions + t * noise


def compute_target_velocity(clean_actions: Tensor, noise: Tensor) -> Tensor:
    return noise - clean_actions


def velocity_to_action_estimate(noisy_actions: Tensor, velocity: Tensor, timestep: Tensor) -> Tensor:
    """Recover x_clean=x_t-t*u_t for x_t=(1-t)x_clean+t*x_noise and u_t=x_noise-x_clean.

    Sampling starts at x_1=noise and integrates this velocity with negative dt toward t=0.
    """
    return noisy_actions - _time_like(timestep, noisy_actions) * velocity


def make_flow_training_sample(
    clean_actions: Tensor,
    action_is_pad: Tensor | None = None,
    noise: Tensor | None = None,
    timestep: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowTrainingSample:
    if noise is None:
        noise = torch.randn(
            clean_actions.shape, device=clean_actions.device, dtype=clean_actions.dtype, generator=generator
        )
    if timestep is None:
        timestep = torch.rand(clean_actions.shape[0], device=clean_actions.device, generator=generator)
    return FlowTrainingSample(
        clean_actions,
        noise,
        timestep,
        interpolate_actions(clean_actions, noise, timestep),
        compute_target_velocity(clean_actions, noise),
        action_is_pad,
    )


def compute_flow_loss(predicted: Tensor, target: Tensor, action_is_pad: Tensor | None = None) -> Tensor:
    loss = (predicted - target).square()
    if action_is_pad is None:
        return loss.mean()
    valid = ~action_is_pad.bool()
    while valid.ndim < loss.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(loss)
    return loss[valid].mean() if valid.any() else loss.sum() * 0
