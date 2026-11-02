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

"""QF3 algorithm: TD3 critic plus the clipped one-step flow actor loss.

One ``update`` consumes one replay batch. It always steps the critic. It steps
the actor when ``optimization_step % policy_frequency == 0``, which includes the
first call. The actor loss is ``qf3_actor_terms``. ``clip_velocity`` defaults
to True. Setting it False is the paper's direct-Q ablation.

``RLTrainer.training_step`` calls this update. ``FlowActorConfig`` carries the
SAC actor-learner fields, so ``python -m lerobot.rl.learner`` and
``python -m lerobot.rl.actor`` can train this algorithm. The actor still builds
its environment only through ``make_robot_env``.

The scalar twin Q and ``qf3_actor_terms`` live in this module.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from lerobot.lerobot_types import BatchType
from lerobot.policies.flow_actor.modeling_flow_actor import (
    FlowActorPolicy,
    FlowVelocityMLP,
    euler_integrate,
)
from lerobot.utils.constants import OBS_STATE

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_qf3 import QF3AlgorithmConfig


class QNetwork(nn.Module):
    """MLP Q(s, a) with SiLU hidden layers.

    The MuJoCo supplement says the critic is identical to the CleanRL TD3 baseline.
    CleanRL's Q network uses ReLU. SiLU is a departure from that stated identity.
    The from-scratch appendix specifies SiLU for the actor.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int | list[int]) -> None:
        super().__init__()
        hidden_dims = [hidden, hidden] if isinstance(hidden, int) else list(hidden)
        if not hidden_dims:
            raise ValueError("hidden width list is empty.")
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((state, action), dim=-1))


class TwinQ(nn.Module):
    """Two scalar critics. The QF3 actor uses q1 only."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int | list[int]) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden)
        self.q2 = QNetwork(state_dim, action_dim, hidden)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state, action), self.q2(state, action)


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """``target <- (1 - tau) target + tau source``."""
    for src, dst in zip(source.parameters(), target.parameters(), strict=True):
        dst.mul_(1.0 - tau).add_(src, alpha=tau)


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


class QF3Algorithm(RLAlgorithm):
    """Off-policy flow actor-critic. The policy must be a ``FlowActorPolicy``."""

    config_class = QF3AlgorithmConfig
    name = "qf3"

    def __init__(self, policy: nn.Module, config: QF3AlgorithmConfig) -> None:
        if not isinstance(policy, FlowActorPolicy):
            raise TypeError(f"QF3 requires FlowActorPolicy, got {type(policy).__name__}.")
        if not isinstance(config, QF3AlgorithmConfig):
            raise TypeError(f"QF3 requires QF3AlgorithmConfig, got {type(config).__name__}.")
        self.config = config
        self.policy = policy
        self._optimization_step = 0
        self.optimizers: dict[str, Optimizer] = {}

        state_dim, action_dim = _feature_dims(policy)
        device = torch.device(policy.config.device or "cpu")
        self.critic = TwinQ(state_dim, action_dim, config.hidden_dims)
        self.target_velocity: FlowVelocityMLP = copy.deepcopy(policy.velocity)
        self.target_critic = copy.deepcopy(self.critic)
        self._freeze(self.target_velocity)
        self._freeze(self.target_critic)
        self.critic.to(device)
        self.target_velocity.to(device)
        self.target_critic.to(device)

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        if "critic" not in self.optimizers or "actor" not in self.optimizers:
            raise RuntimeError("Call make_optimizers_and_scheduler before update.")
        batch = next(batch_iterator)
        device = next(self.critic.parameters()).device
        state = _as_tensor(batch["state"]).to(device)
        action = _as_tensor(batch["action"]).to(device)
        reward = _column(_as_tensor(batch["reward"])).to(device)
        next_state = _as_tensor(batch["next_state"]).to(device)
        done = _column(_as_tensor(batch["done"])).to(device)
        batch_size, action_dim = action.shape

        euler_noise = torch.randn(batch_size, action_dim, device=device)
        smooth_noise = torch.randn(batch_size, action_dim, device=device)
        td = critic_td_loss(
            self.critic,
            self.target_velocity,
            self.target_critic,
            state,
            action,
            reward,
            next_state,
            done,
            euler_noise,
            smooth_noise,
            euler_steps=self.config.euler_steps,
            policy_noise=self.config.policy_noise,
            noise_clip=self.config.noise_clip,
            gamma=self.config.gamma,
        )
        self.optimizers["critic"].zero_grad(set_to_none=True)
        td.backward()
        self.optimizers["critic"].step()

        stats = TrainingStats(losses={"loss_critic": float(td.detach())})
        if self.optimization_step % self.config.policy_frequency == 0:
            flow_noise = torch.randn(batch_size, action_dim, device=device)
            flow_time = torch.rand(batch_size, 1, device=device)
            terms = qf3_actor_terms(
                self.policy.velocity,
                self.critic.q1,
                state,
                action,
                flow_noise,
                flow_time,
                alpha=self.config.alpha,
                lambda_cfm=self.config.lambda_cfm,
                use_q=True,
                clip=self.config.clip_velocity,
            )
            self.optimizers["actor"].zero_grad(set_to_none=True)
            terms.loss.backward()
            self.optimizers["actor"].step()
            polyak_update(self.policy.velocity, self.target_velocity, self.config.polyak_tau)
            polyak_update(self.critic, self.target_critic, self.config.polyak_tau)
            stats.losses["loss_actor"] = float(terms.loss.detach())
            stats.losses["loss_cfm"] = float(terms.cfm)
            stats.extra["q_mean"] = float(terms.q_mean)
            stats.extra["mask_fraction"] = float(terms.mask_fraction)
            stats.extra["identity_error"] = float(terms.identity_error)
            stats.extra["bound_violation"] = float(terms.bound_violation)

        self.optimization_step += 1
        return stats

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        self.optimizers = {
            "actor": torch.optim.Adam(self.policy.velocity.parameters(), lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr),
        }
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        return {
            "policy": {key: value.detach().cpu() for key, value in self.policy.velocity.state_dict().items()}
        }

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        state = {key: value.to(device) for key, value in weights["policy"].items()}
        self.policy.velocity.load_state_dict(state)

    def state_dict(self) -> dict[str, Tensor]:
        bundle: dict[str, Tensor] = {}
        for key, value in self.critic.state_dict().items():
            bundle[f"critic.{key}"] = value.detach()
        for key, value in self.target_critic.state_dict().items():
            bundle[f"target_critic.{key}"] = value.detach()
        for key, value in self.target_velocity.state_dict().items():
            bundle[f"target_velocity.{key}"] = value.detach()
        return bundle

    def load_state_dict(self, state_dict: dict[str, Tensor], device: str | torch.device = "cpu") -> None:
        self.critic.load_state_dict(_strip(state_dict, "critic."), strict=True)
        self.target_critic.load_state_dict(_strip(state_dict, "target_critic."), strict=True)
        self.target_velocity.load_state_dict(_strip(state_dict, "target_velocity."), strict=True)
        self.critic.to(device)
        self.target_critic.to(device)
        self.target_velocity.to(device)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)


def _feature_dims(policy: FlowActorPolicy) -> tuple[int, int]:
    from lerobot.utils.constants import ACTION, OBS_STATE

    features_in = policy.config.input_features
    features_out = policy.config.output_features
    if features_in is None or features_out is None:
        raise ValueError("flow_actor features are unset.")
    return features_in[OBS_STATE].shape[0], features_out[ACTION].shape[0]


def _as_tensor(value: Tensor | dict[str, Tensor]) -> Tensor:
    if isinstance(value, dict):
        if OBS_STATE not in value:
            raise KeyError("observation dict is missing observation.state.")
        tensor = value[OBS_STATE]
    else:
        tensor = value
    if not isinstance(tensor, Tensor):
        raise TypeError(f"expected a tensor, got {type(tensor).__name__}.")
    return tensor


def _column(tensor: Tensor) -> Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor


def _strip(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
