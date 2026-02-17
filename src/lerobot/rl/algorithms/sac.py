# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""SAC (Soft Actor-Critic) algorithm extracted from the learner loop.

This module encapsulates all SAC-specific training logic (critic, actor,
temperature, and discrete-critic updates) behind the ``RLAlgorithm`` interface
so that the learner orchestrator is algorithm-agnostic.
"""

from __future__ import annotations

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
    SampleFn,
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


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic with optional discrete-critic head.

    Owns the ``SACPolicy`` and its optimizers.  The ``update`` method
    replicates the exact same gradient-step sequence as the original
    ``add_actor_information_and_train`` inner loop in ``learner.py``.
    """

    def __init__(
        self,
        policy: SACPolicy,
        config: SACAlgorithmConfig,
        optimizers: dict[str, Optimizer],
    ):
        self.policy = policy
        self.config = config
        self.optimizers = optimizers
        self._optimization_step: int = 0

    # -- RLAlgorithm interface ------------------------------------------------

    @torch.no_grad()
    def select_action(
        self, observation: dict[str, Tensor], deterministic: bool = False
    ) -> Tensor:
        batch = {
            k: v.unsqueeze(0) if isinstance(v, Tensor) else v
            for k, v in observation.items()
        }
        return self.policy.select_action(batch).squeeze(0)

    def update(self, sample_fn: SampleFn) -> TrainingStats:
        """Run one full SAC update with UTD critic warm-up.

        Calls ``sample_fn()`` exactly ``utd_ratio`` times to obtain fresh
        batches.  The first ``utd_ratio - 1`` batches are used for
        critic-only warm-up steps; the last batch drives the full update
        (critic + actor + temperature).

        Args:
            sample_fn: Zero-argument callable that returns a new training batch
                each time it is invoked.
        """
        # --- UTD warm-up: critic-only steps ---------------------------------
        for _ in range(self.config.utd_ratio - 1):
            batch = sample_fn()
            self._critic_warmup_step(batch)

        # --- Final step: full update ----------------------------------------
        batch = sample_fn()
        return self._full_step(batch)

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

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None
        observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)
        return observation_features, next_observation_features

    # -- Internal helpers (exact parity with old learner loop) ----------------

    def _build_forward_batch(
        self,
        batch: BatchType,
        observation_features: Tensor | None,
        next_observation_features: Tensor | None,
        include_complementary: bool = True,
    ) -> dict[str, Any]:
        forward_batch: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": batch["state"],
            "next_state": batch["next_state"],
            "done": batch["done"],
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }
        if include_complementary and "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    def _critic_step(self, forward_batch: dict[str, Any]) -> tuple[float, float]:
        """One critic gradient step; returns (loss, grad_norm)."""
        critic_output = self.policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.critic_ensemble.parameters(),
            max_norm=self.config.clip_grad_norm,
        ).item()
        self.optimizers["critic"].step()
        return loss_critic.item(), critic_grad_norm

    def _discrete_critic_step(self, forward_batch: dict[str, Any]) -> tuple[float, float]:
        """One discrete-critic gradient step; returns (loss, grad_norm)."""
        discrete_critic_output = self.policy.forward(forward_batch, model="discrete_critic")
        loss = discrete_critic_output["loss_discrete_critic"]
        self.optimizers["discrete_critic"].zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.discrete_critic.parameters(),
            max_norm=self.config.clip_grad_norm,
        ).item()
        self.optimizers["discrete_critic"].step()
        return loss.item(), grad_norm

    def _actor_step(self, forward_batch: dict[str, Any]) -> tuple[float, float]:
        actor_output = self.policy.forward(forward_batch, model="actor")
        loss = actor_output["loss_actor"]
        self.optimizers["actor"].zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(),
            max_norm=self.config.clip_grad_norm,
        ).item()
        self.optimizers["actor"].step()
        return loss.item(), grad_norm

    def _temperature_step(self, forward_batch: dict[str, Any]) -> tuple[float, float]:
        temperature_output = self.policy.forward(forward_batch, model="temperature")
        loss = temperature_output["loss_temperature"]
        self.optimizers["temperature"].zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [self.policy.log_alpha],
            max_norm=self.config.clip_grad_norm,
        ).item()
        self.optimizers["temperature"].step()
        return loss.item(), grad_norm

    def _critic_warmup_step(self, batch: BatchType) -> None:
        """Critic-only step used during the UTD warm-up phase.

        Replicates the first ``utd_ratio - 1`` iterations of the original
        learner loop: critic step, optional discrete critic step, target
        network update.
        """
        observations = batch["state"]
        next_observations = batch["next_state"]
        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch = self._build_forward_batch(
            batch, observation_features, next_observation_features, include_complementary=True,
        )

        self._critic_step(forward_batch)

        if self.policy.config.num_discrete_actions is not None:
            self._discrete_critic_step(forward_batch)

        self.policy.update_target_networks()

    def _full_step(self, batch: BatchType) -> TrainingStats:
        """Final UTD step: critic + optional actor/temperature + target update."""
        observations = batch["state"]
        next_observations = batch["next_state"]
        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch = self._build_forward_batch(
            batch, observation_features, next_observation_features, include_complementary=True,
        )

        # -- Critic ----------------------------------------------------------
        loss_critic, critic_grad_norm = self._critic_step(forward_batch)
        stats = TrainingStats(
            loss_critic=loss_critic,
            grad_norms={"critic": critic_grad_norm},
        )

        # -- Discrete critic (optional) --------------------------------------
        if self.policy.config.num_discrete_actions is not None:
            dc_loss, dc_grad = self._discrete_critic_step(forward_batch)
            stats.loss_discrete_critic = dc_loss
            stats.grad_norms["discrete_critic"] = dc_grad

        # -- Actor + temperature (at policy_update_freq) ---------------------
        if self._optimization_step % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_freq):
                actor_loss, actor_grad = self._actor_step(forward_batch)
                temp_loss, temp_grad = self._temperature_step(forward_batch)

            stats.loss_actor = actor_loss
            stats.loss_temperature = temp_loss
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.policy.temperature

        # -- Target network update -------------------------------------------
        self.policy.update_target_networks()

        self._optimization_step += 1
        return stats

    @property
    def optimization_step(self) -> int:
        return self._optimization_step

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = value
