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

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
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
    MLP,
    DiscreteCritic,
    SACObservationEncoder,
    SACPolicy,
    orthogonal_init,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.algorithms.base import BatchType, RLAlgorithm
from lerobot.rl.algorithms.configs import TrainingStats
from lerobot.rl.algorithms.sac.configuration_sac import SACAlgorithmConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """CriticEnsemble wraps multiple CriticHead modules into an ensemble."""

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions], dim=-1)

        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic. Owns critics, targets, temperature, and loss computation."""

    config_class = SACAlgorithmConfig
    name = "sac"

    def __init__(
        self,
        policy: SACPolicy,
        config: SACAlgorithmConfig,
    ):
        self.policy = policy
        self.config = config
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

        self._init_critics()
        self._init_temperature()

        self._device = torch.device(self.policy.config.device)
        self._move_to_device()

    # --- Init ---

    def _init_critics(self) -> None:
        encoder = self.policy.encoder_critic
        action_dim = self.policy.config.output_features[ACTION].shape[0]
        input_dim = encoder.output_dim + action_dim

        heads = [
            CriticHead(input_dim=input_dim, **asdict(self.config.critic_network_kwargs))
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=encoder, ensemble=heads)

        target_heads = [
            CriticHead(input_dim=input_dim, **asdict(self.config.critic_network_kwargs))
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=encoder, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        self.discrete_critic = None
        self.discrete_critic_target = None
        if self.config.num_discrete_actions is not None:
            self.discrete_critic, self.discrete_critic_target = self._init_discrete_critics(encoder)
            self.policy.discrete_critic = self.discrete_critic

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

    def _init_discrete_critics(self, encoder: SACObservationEncoder) -> tuple[DiscreteCritic, DiscreteCritic]:
        kw = asdict(self.config.discrete_critic_network_kwargs)
        dc = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.config.num_discrete_actions,
            **kw,
        )
        dc_target = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.config.num_discrete_actions,
            **kw,
        )
        dc_target.load_state_dict(dc.state_dict())
        return dc, dc_target

    def _init_temperature(self) -> None:
        self.log_alpha = nn.Parameter(torch.tensor([math.log(self.config.temperature_init)]))

    def _move_to_device(self) -> None:
        self.policy.to(self._device)
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.log_alpha = nn.Parameter(self.log_alpha.data.to(self._device))
        if self.discrete_critic is not None:
            self.discrete_critic.to(self._device)
            self.discrete_critic_target.to(self._device)

    @property
    def temperature(self) -> float:
        return self.log_alpha.exp().item()

    # --- Update ---

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        clip = self.config.clip_grad_norm

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

    # --- Losses ---

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

    # --- Target networks ---

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

    # --- Optimizers ---

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

    # --- Weight transfer ---

    def get_weights(self) -> dict[str, Any]:
        """Send actor + discrete-critic state dicts."""
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

    # --- Observation features ---

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features
