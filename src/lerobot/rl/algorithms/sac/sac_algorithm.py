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

Owns critics, targets, temperature, and all loss computation.
Calls ``policy.actor(obs, features)`` **directly** (never ``policy(batch)``)
to keep the computation graph identical to the known-good monolithic policy.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import einops
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

    All critic / target / temperature modules and loss methods live here.
    The policy only provides the encoder and actor.  Every loss method calls
    ``self.policy.actor(obs, features)`` directly — never ``self.policy(batch)``
    — so the computation graph is identical to the known-good monolithic version.
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

        self._init_critics()
        self._init_temperature()
        self._move_to_device()

    # ------------------------------------------------------------------
    # Module initialisation — uses policy.encoder_critic directly
    # ------------------------------------------------------------------

    def _init_critics(self) -> None:
        """Build targets on top of the policy's base critics.

        The policy already created ``critic_ensemble`` (possibly compiled)
        and ``discrete_critic`` during ``__init__`` to preserve RNG order.
        We reference them here and create the corresponding target networks.
        """
        encoder = self.policy.encoder_critic

        self.critic_ensemble = self.policy.critic_ensemble

        action_dim = self.policy.config.output_features[ACTION].shape[0]
        input_dim = encoder.output_dim + action_dim

        target_heads = [
            CriticHead(input_dim=input_dim, **asdict(self.config.critic_network_kwargs))
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=encoder, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self) -> None:
        encoder = self.policy.encoder_critic
        self.discrete_critic = self.policy.discrete_critic

        self.discrete_critic_target = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_temperature(self) -> None:
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))

    def _move_to_device(self) -> None:
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.log_alpha = nn.Parameter(self.log_alpha.data.to(self._device))
        if hasattr(self, "discrete_critic"):
            self.discrete_critic.to(self._device)
            self.discrete_critic_target.to(self._device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return self.log_alpha.exp().item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """One full SAC update with UTD critic warm-up.

        Warm-up batches include complementary_info; the final batch omits it
        (matches the known-good learner loop).
        """
        clip = self.config.clip_grad_norm

        # --- UTD warm-up steps ---
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch, include_complementary_info=True)

            loss_critic = self._compute_loss_critic(fb)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()

            if self.config.num_discrete_actions is not None:
                loss_dc = self._compute_loss_discrete_critic(fb)
                self.optimizers["discrete_critic"].zero_grad()
                loss_dc.backward()
                torch.nn.utils.clip_grad_norm_(self.discrete_critic.parameters(), max_norm=clip)
                self.optimizers["discrete_critic"].step()

            self._update_target_networks()

        # --- Final UTD step (omit complementary_info) ---
        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch, include_complementary_info=False)

        loss_critic = self._compute_loss_critic(fb)
        self.optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip).item()
        self.optimizers["critic"].step()

        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic": critic_grad},
        )

        if self.config.num_discrete_actions is not None:
            loss_dc = self._compute_loss_discrete_critic(fb)
            self.optimizers["discrete_critic"].zero_grad()
            loss_dc.backward()
            dc_grad = torch.nn.utils.clip_grad_norm_(self.discrete_critic.parameters(), max_norm=clip).item()
            self.optimizers["discrete_critic"].step()
            stats.losses["loss_discrete_critic"] = loss_dc.item()
            stats.grad_norms["discrete_critic"] = dc_grad

        # --- Actor + temperature (at policy_update_freq) ---
        if self._optimization_step % self.config.policy_update_freq == 0:
            for _ in range(self.config.policy_update_freq):
                loss_actor = self._compute_loss_actor(fb)
                self.optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), max_norm=clip
                ).item()
                self.optimizers["actor"].step()

                loss_temp = self._compute_loss_temperature(fb)
                self.optimizers["temperature"].zero_grad()
                loss_temp.backward()
                temp_grad = torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=clip).item()
                self.optimizers["temperature"].step()

            stats.losses["loss_actor"] = loss_actor.item()
            stats.losses["loss_temperature"] = loss_temp.item()
            stats.grad_norms["actor"] = actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.temperature

        self._update_target_networks()

        self._optimization_step += 1
        return stats

    # ------------------------------------------------------------------
    # Loss methods — call policy.actor() DIRECTLY (never policy(batch))
    # ------------------------------------------------------------------

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        obs_features = batch.get("observation_feature")
        next_obs_features = batch.get("next_observation_feature")

        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.policy.actor(next_observations, next_obs_features)

            q_targets = self.critic_target(next_observations, next_action_preds, next_obs_features)

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
            next_discrete_qs = self.discrete_critic(next_observations, next_obs_features)
            best_next_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            target_next_qs = self.discrete_critic_target(next_observations, next_obs_features)
            target_next_q = torch.gather(target_next_qs, dim=1, index=best_next_action).squeeze(-1)

            rewards_disc = rewards
            if discrete_penalties is not None:
                rewards_disc = rewards + discrete_penalties
            target_q = rewards_disc + (1 - done) * self.config.discount * target_next_q

        predicted_qs = self.discrete_critic(observations, obs_features)
        predicted_q = torch.gather(predicted_qs, dim=1, index=actions_discrete).squeeze(-1)

        return F.mse_loss(input=predicted_q, target=target_q)

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        obs_features = batch.get("observation_feature")

        actions_pi, log_probs, _ = self.policy.actor(observations, obs_features)

        q_preds = self.critic_ensemble(observations, actions_pi, obs_features)
        min_q = q_preds.min(dim=0)[0]

        return ((self.temperature * log_probs) - min_q).mean()

    def _compute_loss_temperature(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        obs_features = batch.get("observation_feature")

        with torch.no_grad():
            _, log_probs, _ = self.policy.actor(observations, obs_features)

        return (-self.log_alpha.exp() * (log_probs + self.policy.target_entropy)).mean()

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _update_target_networks(self) -> None:
        tau = self.config.critic_target_update_weight
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))
        if self.config.num_discrete_actions is not None:
            for target_p, p in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_forward_batch(
        self, batch: BatchType, *, include_complementary_info: bool = True
    ) -> dict[str, Any]:
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
        actor_params = self.policy.get_optim_params()["actor"]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.log_alpha], lr=self.config.temperature_lr),
        }
        if self.config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                self.discrete_critic.parameters(), lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, Any]:
        """Send actor + discrete-critic state dicts (avoids encoder duplication)."""
        state_dicts: dict[str, Any] = {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }
        if self.config.num_discrete_actions is not None:
            state_dicts["discrete_critic"] = move_state_dict_to_device(
                self.discrete_critic.state_dict(), device="cpu"
            )
        return state_dicts

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        actor_sd = move_state_dict_to_device(weights["policy"], device=device)
        self.policy.actor.load_state_dict(actor_sd)
        if "discrete_critic" in weights and self.config.num_discrete_actions is not None:
            dc_sd = move_state_dict_to_device(weights["discrete_critic"], device=device)
            self.discrete_critic.load_state_dict(dc_sd)

    # ------------------------------------------------------------------
    # Observation feature caching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features
