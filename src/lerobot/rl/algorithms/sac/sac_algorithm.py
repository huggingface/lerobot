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

Thin orchestration layer that delegates all module creation and loss
computation to :class:`SACPolicy`.  This keeps the ``RLAlgorithm``
interface while ensuring the policy module tree is identical to the
known-good pre-refactor structure.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.algorithms.base import (
    BatchType,
    RLAlgorithm,
    TrainingStats,
)
from lerobot.rl.algorithms.sac.configuration_sac import SACAlgorithmConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic with optional discrete-critic head.

    All modules (critics, targets, temperature, loss functions) live on
    :class:`SACPolicy`.  This class only owns the update loop, optimizers,
    and weight-transfer helpers.
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
        self._device = get_device_from_parameters(self.policy)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Run one full SAC update with UTD critic warm-up.

        Mirrors the known-good learner loop from ``fix/re-enable-sac-normalization``:
        warm-up batches include ``complementary_info``; the final batch omits it.
        """
        clip = self.config.clip_grad_norm

        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            forward_batch = self._prepare_forward_batch(batch, include_complementary_info=True)

            critic_output = self.policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()

            if self.policy.config.num_discrete_actions is not None:
                dc_output = self.policy.forward(forward_batch, model="discrete_critic")
                loss_dc = dc_output["loss_discrete_critic"]
                self.optimizers["discrete_critic"].zero_grad()
                loss_dc.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.discrete_critic.parameters(), max_norm=clip)
                self.optimizers["discrete_critic"].step()

            self.policy.update_target_networks()

        # Final UTD step — omit complementary_info (matches working branch)
        batch = next(batch_iterator)
        forward_batch = self._prepare_forward_batch(batch, include_complementary_info=False)

        critic_output = self.policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.policy.critic_ensemble.parameters(), max_norm=clip
        ).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )

        if self.policy.config.num_discrete_actions is not None:
            dc_output = self.policy.forward(forward_batch, model="discrete_critic")
            loss_dc = dc_output["loss_discrete_critic"]
            self.optimizers["discrete_critic"].zero_grad()
            loss_dc.backward()
            dc_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.discrete_critic.parameters(), max_norm=clip
            ).item()
            self.optimizers["discrete_critic"].step()
            stats.losses["loss_discrete_critic"] = loss_dc.item()
            stats.grad_norms["discrete_critic"] = dc_grad

        if self._optimization_step % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_freq):
                actor_output = self.policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                self.optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), max_norm=clip
                ).item()
                self.optimizers["actor"].step()

                temp_output = self.policy.forward(forward_batch, model="temperature")
                loss_temp = temp_output["loss_temperature"]
                self.optimizers["temperature"].zero_grad()
                loss_temp.backward()
                temp_grad = torch.nn.utils.clip_grad_norm_([self.policy.log_alpha], max_norm=clip).item()
                self.optimizers["temperature"].step()

            stats.losses["loss_actor"] = loss_actor.item()
            stats.losses["loss_temperature"] = loss_temp.item()
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.policy.temperature

        self.policy.update_target_networks()

        self._optimization_step += 1
        return stats

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_forward_batch(
        self, batch: BatchType, *, include_complementary_info: bool = True
    ) -> dict[str, Any]:
        """Build the dict expected by ``policy.forward(batch, model=...)``."""
        observations = batch["state"]
        next_observations = batch["next_state"]

        observation_features, next_observation_features = self.get_observation_features(
            observations, next_observations
        )
        forward_batch: dict[str, Any] = {
            ACTION: batch[ACTION],
            "reward": batch["reward"],
            "state": observations,
            "next_state": next_observations,
            "done": batch["done"],
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }
        if include_complementary_info and "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def make_optimizers(self) -> dict[str, Optimizer]:
        optim_params = self.policy.get_optim_params()
        self.optimizers = {
            "actor": torch.optim.Adam(optim_params["actor"], lr=self.config.actor_lr),
            "critic": torch.optim.Adam(optim_params["critic"], lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([optim_params["temperature"]], lr=self.config.temperature_lr),
        }
        if self.policy.config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                optim_params["discrete_critic"], lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    # ------------------------------------------------------------------
    # Weight transfer (actor <-> learner)
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, Any]:
        """Send actor + discrete-critic state dicts (avoids encoder duplication)."""
        state_dicts: dict[str, Any] = {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }
        if self.policy.config.num_discrete_actions is not None:
            state_dicts["discrete_critic"] = move_state_dict_to_device(
                self.policy.discrete_critic.state_dict(), device="cpu"
            )
        return state_dicts

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        actor_sd = move_state_dict_to_device(weights["policy"], device=device)
        self.policy.actor.load_state_dict(actor_sd)
        if "discrete_critic" in weights and self.policy.config.num_discrete_actions is not None:
            dc_sd = move_state_dict_to_device(weights["discrete_critic"], device=device)
            self.policy.discrete_critic.load_state_dict(dc_sd)

    # ------------------------------------------------------------------
    # Observation feature caching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        with torch.no_grad():
            observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features
