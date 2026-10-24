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

``RLTrainer.training_step`` can call this update in-process. This class does not
start ``lerobot.rl.learner`` or ``lerobot.rl.actor``. ``FlowActorConfig`` does
not carry the SAC service fields those processes read.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from lerobot.lerobot_types import BatchType
from lerobot.policies.flow_actor.modeling_flow_actor import FlowActorPolicy, FlowVelocityMLP
from lerobot.utils.constants import OBS_STATE

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_qf3 import QF3AlgorithmConfig
from .losses import critic_td_loss, qf3_actor_terms
from .networks import TwinQ, polyak_update


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
