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

import math
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.sac.modeling_sac import (
    DISCRETE_DIMENSION_INDEX,
    CriticEnsemble,
    CriticHead,
    DiscreteCritic,
    SACObservationEncoder,
    SACPolicy,
)
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

    Owns the ``SACPolicy`` and its optimizers.  All loss methods call
    ``self.policy(batch_dict)`` rather than reaching into ``self.policy.actor``
    directly, so any policy that returns ``{"action", "log_prob"}`` from its
    ``forward()`` is compatible.
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
        self._init_critic_encoder()
        self._init_critics()
        self._init_temperature()
        self._move_to_device()

    def _init_critic_encoder(self) -> None:
        """Build or share the encoder used by critics."""
        if self.config.shared_encoder:
            self.critic_encoder = self.policy.encoder
            self.policy.actor.encoder_is_shared = True
        else:
            self.critic_encoder = SACObservationEncoder(self.policy.config)

    def _init_critics(self) -> None:
        """Build critic ensemble, targets, and optional discrete critic."""
        action_dim = self.policy.config.output_features[ACTION].shape[0]
        input_dim = self.critic_encoder.output_dim + action_dim

        heads = [
            CriticHead(input_dim=input_dim, **asdict(self.config.critic_network_kwargs))
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.critic_encoder, ensemble=heads)

        target_heads = [
            CriticHead(input_dim=input_dim, **asdict(self.config.critic_network_kwargs))
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.critic_encoder, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critic_target()

    def _init_discrete_critic_target(self) -> None:
        """Build only the target discrete critic."""
        input_dim = self.critic_encoder.output_dim
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.critic_encoder,
            input_dim=input_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        # TODO: (kmeftah) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.policy.discrete_critic.state_dict())

    def _init_temperature(self) -> None:
        """Set up temperature parameter (log_alpha) and default target entropy."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))

        action_dim = self.policy.config.output_features[ACTION].shape[0]
        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _move_to_device(self) -> None:
        """Move algorithm-owned modules to the policy device."""
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.log_alpha = nn.Parameter(self.log_alpha.data.to(self._device))
        if hasattr(self, "discrete_critic_target"):
            self.discrete_critic_target.to(self._device)

    @property
    def temperature(self) -> float:
        return self.log_alpha.exp().item()

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Run one full SAC update with UTD critic warm-up.

        Pulls ``utd_ratio`` batches from ``batch_iterator``.  The first
        ``utd_ratio - 1`` batches are used for critic-only warm-up steps;
        the last batch drives the full update (critic + actor + temperature).
        """
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            forward_batch = self._prepare_forward_batch(batch)

            loss_critic = self._compute_loss_critic(forward_batch)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic_ensemble.parameters(),
                max_norm=self.config.clip_grad_norm,
            ).item()
            self.optimizers["critic"].step()

            if self.config.num_discrete_actions is not None:
                loss_discrete = self._compute_loss_discrete_critic(forward_batch)
                self.optimizers["discrete_critic"].zero_grad()
                loss_discrete.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.discrete_critic.parameters(),
                    max_norm=self.config.clip_grad_norm,
                ).item()
                self.optimizers["discrete_critic"].step()
            self._update_target_networks()

        batch = next(batch_iterator)
        forward_batch = self._prepare_forward_batch(batch)

        loss_critic = self._compute_loss_critic(forward_batch)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_ensemble.parameters(),
            max_norm=self.config.clip_grad_norm,
        ).item()
        self.optimizers["critic"].step()

        critic_loss_val = loss_critic.item()
        stats = TrainingStats(
            losses={"critic": critic_loss_val},
            grad_norms={"critic": critic_grad_norm},
        )

        if self.config.num_discrete_actions is not None:
            loss_discrete = self._compute_loss_discrete_critic(forward_batch)
            self.optimizers["discrete_critic"].zero_grad()
            loss_discrete.backward()
            dc_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.discrete_critic.parameters(),
                max_norm=self.config.clip_grad_norm,
            ).item()
            self.optimizers["discrete_critic"].step()
            stats.losses["discrete_critic"] = loss_discrete.item()
            stats.grad_norms["discrete_critic"] = dc_grad

        if self._optimization_step % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_freq):
                actor_loss = self._compute_loss_actor(forward_batch)
                self.optimizers["actor"].zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(),
                    max_norm=self.config.clip_grad_norm,
                ).item()
                self.optimizers["actor"].step()

                temp_loss = self._compute_loss_temperature(forward_batch)
                self.optimizers["temperature"].zero_grad()
                temp_loss.backward()
                temp_grad = torch.nn.utils.clip_grad_norm_(
                    [self.log_alpha],
                    max_norm=self.config.clip_grad_norm,
                ).item()
                self.optimizers["temperature"].step()

            stats.losses["actor"] = actor_loss.item()
            stats.losses["temperature"] = temp_loss.item()
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.temperature

        self._update_target_networks()

        self._optimization_step += 1
        return stats

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        obs_features = batch.get("observation_feature")
        next_obs_features = batch.get("next_observation_feature")

        with torch.no_grad():
            next_output = self.policy({"state": next_observations, "observation_feature": next_obs_features})
            next_actions = next_output["action"]
            next_log_probs = next_output["log_prob"]

            q_targets = self.critic_target(next_observations, next_actions, next_obs_features)

            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            min_q, _ = q_targets.min(dim=0)
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        if self.config.num_discrete_actions is not None:
            actions = actions[:, :DISCRETE_DIMENSION_INDEX]

        q_preds = self.critic_ensemble(observations, actions, obs_features)

        td_target_dup = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        critics_loss = (F.mse_loss(input=q_preds, target=td_target_dup, reduction="none").mean(dim=1)).sum()
        return critics_loss

    def _compute_loss_discrete_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        obs_features = batch.get("observation_feature")
        next_obs_features = batch.get("next_observation_feature")
        complementary_info = batch.get("complementary_info")

        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete).long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            next_discrete_qs = self.policy.discrete_critic(next_observations, next_obs_features)
            best_next_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            target_next_qs = self.discrete_critic_target(next_observations, next_obs_features)
            target_next_q = torch.gather(target_next_qs, dim=1, index=best_next_action).squeeze(-1)

            rewards_disc = rewards
            if discrete_penalties is not None:
                rewards_disc = rewards + discrete_penalties
            target_q = rewards_disc + (1 - done) * self.config.discount * target_next_q

        predicted_qs = self.policy.discrete_critic(observations, obs_features)
        predicted_q = torch.gather(predicted_qs, dim=1, index=actions_discrete).squeeze(-1)

        return F.mse_loss(input=predicted_q, target=target_q)

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        obs_features = batch.get("observation_feature")

        output = self.policy({"state": observations, "observation_feature": obs_features})
        actions_pi = output["action"]
        log_probs = output["log_prob"]

        q_preds = self.critic_ensemble(observations, actions_pi, obs_features)
        min_q = q_preds.min(dim=0)[0]

        return ((self.temperature * log_probs) - min_q).mean()

    def _compute_loss_temperature(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        obs_features = batch.get("observation_feature")

        with torch.no_grad():
            output = self.policy({"state": observations, "observation_feature": obs_features})
            log_probs = output["log_prob"]

        return (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()

    def _update_target_networks(self) -> None:
        tau = self.config.critic_target_update_weight
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))
        if self.config.num_discrete_actions is not None:
            for target_p, p in zip(
                self.discrete_critic_target.parameters(),
                self.policy.discrete_critic.parameters(),
                strict=True,
            ):
                target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))

    def _prepare_forward_batch(self, batch: BatchType) -> dict[str, Any]:
        """Build the dict expected by loss computation from a sampled batch."""
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
        if "complementary_info" in batch:
            forward_batch["complementary_info"] = batch["complementary_info"]
        return forward_batch

    def make_optimizers(self) -> dict[str, Optimizer]:
        """Create Adam optimizers for the SAC components and store them."""
        actor_params = [
            p
            for n, p in self.policy.actor.named_parameters()
            if not self.config.shared_encoder or not n.startswith("encoder")
        ]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.log_alpha], lr=self.config.temperature_lr),
        }
        if self.config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                self.policy.discrete_critic.parameters(), lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        """Policy state-dict to push to actors (includes actor + discrete critic)."""
        return move_state_dict_to_device(self.policy.state_dict(), device="cpu")

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load policy state-dict received from the learner."""
        state = move_state_dict_to_device(weights, device=device)
        self.policy.load_state_dict(state)

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.shared_encoder:
            return None, None
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None
        if not self.policy.encoder.has_images:
            return None, None
        observation_features = self.policy.encoder.get_cached_image_features(observations)
        next_observation_features = self.policy.encoder.get_cached_image_features(next_observations)
        return observation_features, next_observation_features
