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
"""SAC (Soft Actor-Critic) algorithm.

This module encapsulates all SAC-specific training logic (critic, actor,
temperature, and discrete-critic updates) behind the ``RLAlgorithm`` interface.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.algorithms.base import (
    BatchType,
    RLAlgorithm,
    RLAlgorithmConfig,
    TrainingStats,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device


@RLAlgorithmConfig.register_subclass("sac")
@dataclass
class SACAlgorithmConfig(RLAlgorithmConfig):
    """SAC-specific hyper-parameters that control the update loop."""

    utd_ratio: int = 1
    policy_update_freq: int = 1
    clip_grad_norm: float = 40.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    @classmethod
    def from_policy_config(cls, policy_cfg) -> SACAlgorithmConfig:
        """Build from an existing ``SACConfig`` (cfg.policy) for backwards compat."""
        return cls(
            utd_ratio=policy_cfg.utd_ratio,
            policy_update_freq=policy_cfg.policy_update_freq,
            clip_grad_norm=policy_cfg.grad_clip_norm,
            actor_lr=policy_cfg.actor_lr,
            critic_lr=policy_cfg.critic_lr,
        )

    def build_algorithm(self, policy: torch.nn.Module) -> SACAlgorithm:
        return SACAlgorithm(policy=policy, config=self)


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic with optional discrete-critic head.

    Owns the ``SACPolicy`` and its optimizers.
    """

    def __init__(
        self,
        policy: SACPolicy,
        config: SACAlgorithmConfig,
    ):
        self.policy = policy
        self.config = config
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        batch = {k: v.unsqueeze(0) if isinstance(v, Tensor) else v for k, v in observation.items()}
        return self.policy.select_action(batch).squeeze(0)

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Run one full SAC update with UTD critic warm-up.

        Pulls ``utd_ratio`` batches from ``batch_iterator``.  The first
        ``utd_ratio - 1`` batches are used for critic-only warm-up steps;
        the last batch drives the full update (critic + actor + temperature).
        """
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            forward_batch = self._prepare_forward_batch(batch)

            critic_output = self.policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.critic_ensemble.parameters(),
                max_norm=self.config.clip_grad_norm,
            ).item()
            self.optimizers["critic"].step()

            if self.policy.config.num_discrete_actions is not None:
                discrete_critic_output = self.policy.forward(forward_batch, model="discrete_critic")
                loss_discrete = discrete_critic_output["loss_discrete_critic"]
                self.optimizers["discrete_critic"].zero_grad()
                loss_discrete.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.discrete_critic.parameters(),
                    max_norm=self.config.clip_grad_norm,
                ).item()
                self.optimizers["discrete_critic"].step()
            self.policy.update_target_networks()

        batch = next(batch_iterator)
        forward_batch = self._prepare_forward_batch(batch)

        critic_output = self.policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.critic_ensemble.parameters(),
            max_norm=self.config.clip_grad_norm,
        ).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            loss_critic=loss_critic.item(),
            grad_norms={"critic": critic_grad_norm},
        )

        if self.policy.config.num_discrete_actions is not None:
            discrete_critic_output = self.policy.forward(forward_batch, model="discrete_critic")
            loss_discrete = discrete_critic_output["loss_discrete_critic"]
            self.optimizers["discrete_critic"].zero_grad()
            loss_discrete.backward()
            dc_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.discrete_critic.parameters(),
                max_norm=self.config.clip_grad_norm,
            ).item()
            self.optimizers["discrete_critic"].step()
            stats.loss_discrete_critic = loss_discrete.item()
            stats.grad_norms["discrete_critic"] = dc_grad

        if self._optimization_step % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_freq):
                actor_output = self.policy.forward(forward_batch, model="actor")
                actor_loss = actor_output["loss_actor"]
                self.optimizers["actor"].zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(),
                    max_norm=self.config.clip_grad_norm,
                ).item()
                self.optimizers["actor"].step()

                temperature_output = self.policy.forward(forward_batch, model="temperature")
                temp_loss = temperature_output["loss_temperature"]
                self.optimizers["temperature"].zero_grad()
                temp_loss.backward()
                temp_grad = torch.nn.utils.clip_grad_norm_(
                    [self.policy.log_alpha],
                    max_norm=self.config.clip_grad_norm,
                ).item()
                self.optimizers["temperature"].step()

            stats.loss_actor = actor_loss.item()
            stats.loss_temperature = temp_loss.item()
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.policy.temperature

        self.policy.update_target_networks()

        self._optimization_step += 1
        return stats

    def _prepare_forward_batch(self, batch: BatchType) -> dict[str, Any]:
        """Build the dict expected by ``SACPolicy.forward()`` from a raw batch."""
        observations = batch["state"]
        next_observations = batch["next_state"]
        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": batch["state"],
            "next_state": batch["next_state"],
            "done": batch["done"],
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }
        if "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    def make_optimizers(self) -> dict[str, Optimizer]:
        """Create Adam optimizers for the SAC components and store them."""
        actor_params = [
            p
            for n, p in self.policy.actor.named_parameters()
            if not self.policy.config.shared_encoder or not n.startswith("encoder")
        ]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.policy.critic_ensemble.parameters(), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.policy.log_alpha], lr=self.config.critic_lr),
        }
        if self.policy.config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                self.policy.discrete_critic.parameters(), lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        """State-dicts to push to the actor process."""
        out: dict[str, Any] = {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }
        if hasattr(self.policy, "discrete_critic") and self.policy.discrete_critic is not None:
            out["discrete_critic"] = move_state_dict_to_device(
                self.policy.discrete_critic.state_dict(), device="cpu"
            )
        return out

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load state-dict(s) received from the learner (inverse of ``get_weights``)."""
        if "policy" in weights:
            actor_state = move_state_dict_to_device(weights["policy"], device=device)
            self.policy.actor.load_state_dict(actor_state)
        if (
            "discrete_critic" in weights
            and hasattr(self.policy, "discrete_critic")
            and self.policy.discrete_critic is not None
        ):
            dc_state = move_state_dict_to_device(weights["discrete_critic"], device=device)
            self.policy.discrete_critic.load_state_dict(dc_state)

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None
        observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)
        return observation_features, next_observation_features

    @property
    def optimization_step(self) -> int:
        return self._optimization_step

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = value
