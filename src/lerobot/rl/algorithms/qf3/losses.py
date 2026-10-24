# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QF3 actor objective from arXiv:2610.08789v1.

Labels in the paper source: eq:actor-cfm-loss, eq:clipped-reconstruction,
eq:actor-loss, eq:q-gradient-mask. The direct-Q ablation is the same objective
with the velocity clip removed. CFM uses the unclipped residual.

The mask is strict, ``|delta_i| < alpha``. ``torch.clamp`` passes a gradient at
exactly ``+/- alpha``, so the clipped residual used by Q is detached and the
gradient is multiplied by that indicator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lerobot.policies.flow_actor.modeling_flow_actor import euler_integrate

from .networks import TwinQ


@dataclass
class ActorTerms:
    """Tensors from one evaluation of the actor objective."""

    loss: Tensor
    cfm: Tensor
    q_mean: Tensor
    mask_fraction: Tensor
    identity_error: Tensor
    endpoint_gap: Tensor
    bound_violation: Tensor


def flow_path(noise: Tensor, replay_action: Tensor, flow_time: Tensor) -> tuple[Tensor, Tensor]:
    """Linear path. ``x_tau = (1-tau) eps + tau a_buf``, ``u_buf = a_buf - eps``."""
    x_tau = (1.0 - flow_time) * noise + flow_time * replay_action
    u_buf = replay_action - noise
    return x_tau, u_buf


def qf3_actor_terms(
    actor: nn.Module,
    q1: nn.Module,
    state: Tensor,
    replay_action: Tensor,
    noise: Tensor,
    flow_time: Tensor,
    *,
    alpha: float,
    lambda_cfm: float,
    use_q: bool,
    clip: bool,
) -> ActorTerms:
    """QF3 actor loss, the direct-Q ablation, or the CFM term alone.

    ``clip=True`` is eq:actor-loss. ``clip=False`` is the paper's direct-Q ablation.
    ``use_q=False`` keeps only the CFM term. The CFM residual is never clipped.
    ``L_cfm`` is the batch mean of the squared L2 from the main text, not the
    supplement's ``1/d_a`` form. ``Q`` is ``q1`` only, matching CleanRL TD3.
    """
    x_tau, u_buf = flow_path(noise, replay_action, flow_time)
    velocity = actor(state, x_tau, flow_time)
    delta = velocity - u_buf
    if clip:
        delta_clip = delta.clamp(-alpha, alpha)
        mask = (delta.abs() < alpha).to(dtype=delta.dtype)
        delta_for_q = delta_clip.detach() + (delta - delta.detach()) * mask
    else:
        delta_for_q = delta
    endpoint = replay_action + (1.0 - flow_time) * delta_for_q
    reconstruction = x_tau + (1.0 - flow_time) * u_buf
    cfm = delta.square().sum(dim=-1).mean()
    if use_q:
        q_values = q1(state, endpoint)
        loss = -q_values.mean() + lambda_cfm * cfm
        q_mean = q_values.mean().detach()
    else:
        with torch.no_grad():
            q_mean = q1(state, endpoint).mean()
        loss = lambda_cfm * cfm
    inside = delta.abs() < alpha
    allowed = (1.0 - flow_time) * alpha
    excess = (endpoint - replay_action).abs() - allowed
    return ActorTerms(
        loss=loss,
        cfm=cfm.detach(),
        q_mean=q_mean,
        mask_fraction=inside.float().mean().detach(),
        identity_error=(reconstruction - replay_action).abs().max().detach(),
        endpoint_gap=(endpoint - replay_action).abs().max().detach(),
        bound_violation=excess.max().clamp(min=0).detach(),
    )


def critic_td_loss(
    critic: TwinQ,
    target_actor: nn.Module,
    target_critic: TwinQ,
    state: Tensor,
    action: Tensor,
    reward: Tensor,
    next_state: Tensor,
    done: Tensor,
    euler_noise: Tensor,
    smooth_noise: Tensor,
    *,
    euler_steps: int,
    policy_noise: float,
    noise_clip: float,
    gamma: float,
) -> Tensor:
    """Twin-Q TD3 target. The next action is an Euler sample plus clipped noise.

    The action is not squashed with tanh and is not clamped to [-1, 1]. The paper
    rejects endpoint tanh and does not state an action bound. Noise scale, noise
    clip, and the min of the two target critics are the CleanRL TD3 target. The
    paper says the MuJoCo critic matches that baseline and does not restate the
    noise numbers.
    """
    with torch.no_grad():
        next_action = euler_integrate(target_actor, next_state, euler_noise, euler_steps)
        clipped_noise = (smooth_noise * policy_noise).clamp(-noise_clip, noise_clip)
        next_action = next_action + clipped_noise
        next_q1, next_q2 = target_critic(next_state, next_action)
        target = reward + gamma * (1.0 - done) * torch.minimum(next_q1, next_q2)
    q1, q2 = critic(state, action)
    return (q1 - target).square().mean() + (q2 - target).square().mean()
