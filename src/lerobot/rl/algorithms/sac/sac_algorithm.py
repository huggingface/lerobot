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

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import (
    DISCRETE_DIMENSION_INDEX,
    MLP,
    DiscreteCritic,
    GaussianActorObservationEncoder,
    GaussianActorPolicy,
    orthogonal_init,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.types import BatchType
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_state_dict_to_device

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_sac import SACAlgorithmConfig


class SACAlgorithm(RLAlgorithm):
    """Soft Actor-Critic. Owns critics, targets, temperature, and loss computation."""

    config_class = SACAlgorithmConfig
    name = "sac"

    def __init__(
        self,
        policy: GaussianActorPolicy,
        config: SACAlgorithmConfig,
    ):
        self.config = config
        self.policy_config = config.policy_config
        self.policy = policy
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0

        action_dim = self.policy.config.output_features[ACTION].shape[0]
        self._init_critics(action_dim)
        self._init_temperature(action_dim)

        self._device = torch.device(self.policy.config.device)
        self._move_to_device()

    def _init_critics(self, action_dim) -> None:
        """Build critic ensemble, targets."""
        encoder = self.policy.encoder_critic

        heads = [
            CriticHead(
                input_dim=encoder.output_dim + action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=encoder, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=encoder.output_dim + action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=encoder, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # TODO(Khalil): Investigate and fix torch.compile
        # NOTE: torch.compile is disabled, policy does not converge when enabled.
        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        self.discrete_critic_target = None
        if self.policy_config.num_discrete_actions is not None:
            self.discrete_critic_target = self._init_discrete_critic_target(encoder)

    def _init_discrete_critic_target(self, encoder: GaussianActorObservationEncoder) -> DiscreteCritic:
        """Build target discrete critic (main network is owned by the policy)."""
        discrete_critic_target = DiscreteCritic(
            encoder=encoder,
            input_dim=encoder.output_dim,
            output_dim=self.policy_config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        # TODO(Khalil): Compile the discrete critic
        discrete_critic_target.load_state_dict(self.policy.discrete_critic.state_dict())
        return discrete_critic_target

    def _init_temperature(self, continuous_action_dim: int) -> None:
        """Set up temperature parameter (log_alpha) and target entropy."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            total_action_dim = continuous_action_dim + (
                1 if self.policy_config.num_discrete_actions is not None else 0
            )
            self.target_entropy = -total_action_dim / 2

    def _move_to_device(self) -> None:
        self.policy.to(self._device)
        self.critic_ensemble.to(self._device)
        self.critic_target.to(self._device)
        self.log_alpha = nn.Parameter(self.log_alpha.data.to(self._device))
        if self.discrete_critic_target is not None:
            self.discrete_critic_target.to(self._device)

    @property
    def temperature(self) -> float:
        """Return the current temperature value, always in sync with log_alpha."""
        return self.log_alpha.exp().item()

    def _critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def _discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.policy.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Run one SAC training step (critic / discrete-critic / actor / temperature).

        Pulls ``utd_ratio`` batches from ``batch_iterator``, computes the relevant
        losses, backpropagates each, and updates target networks.

        Args:
            batch_iterator: yields batches each containing
                - ``action``: Action tensor
                - ``reward``: Reward tensor
                - ``state``: Observations tensor dict
                - ``next_state``: Next observations tensor dict
                - ``done``: Done mask tensor
                - ``observation_feature``: Optional pre-computed observation features
                - ``next_observation_feature``: Optional pre-computed next observation features
                - ``complementary_info`` (optional): per-step extras like discrete penalties

        Returns:
            TrainingStats with per-component losses and grad norms.
        """
        clip = self.config.grad_clip_norm

        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch, include_complementary_info=True)

            loss_critic = self._compute_loss_critic(fb)
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_ensemble.parameters(), max_norm=clip)
            self.optimizers["critic"].step()

            if self.policy_config.num_discrete_actions is not None:
                loss_dc = self._compute_loss_discrete_critic(fb)
                self.optimizers["discrete_critic"].zero_grad()
                loss_dc.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.discrete_critic.parameters(), max_norm=clip)
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

        if self.policy_config.num_discrete_actions is not None:
            loss_dc = self._compute_loss_discrete_critic(fb)
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

    def _compute_loss_critic(self, batch: dict[str, Any]) -> Tensor:
        # Extract common components from batch
        observations = batch["state"]
        actions = batch[ACTION]
        observation_features = batch.get("observation_feature")
        # Extract critic-specific components
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        next_observation_features = batch.get("next_observation_feature")

        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.policy.actor(
                next_observations, next_observation_features
            )

            # 2- compute q targets
            q_targets = self._critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        if self.policy_config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self._critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def _compute_loss_discrete_critic(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        actions = batch[ACTION]
        rewards = batch["reward"]
        next_observations = batch["next_state"]
        done = batch["done"]
        observation_features = batch.get("observation_feature")
        next_observation_features = batch.get("next_observation_feature")
        complementary_info = batch.get("complementary_info")

        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self._discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self._discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self._discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def _compute_loss_actor(self, batch: dict[str, Any]) -> Tensor:
        observations = batch["state"]
        observation_features = batch.get("observation_feature")

        actions_pi, log_probs, _ = self.policy.actor(observations, observation_features)

        q_preds = self._critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss

    def _compute_loss_temperature(self, batch: dict[str, Any]) -> Tensor:
        """Compute the temperature loss"""
        observations = batch["state"]
        observation_features = batch.get("observation_feature")

        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.policy.actor(observations, observation_features)

        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def _update_target_networks(self) -> None:
        """Update target networks with exponential moving average"""
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(
                p.data * self.config.critic_target_update_weight
                + target_p.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.policy_config.num_discrete_actions is not None:
            for target_p, p in zip(
                self.discrete_critic_target.parameters(),
                self.policy.discrete_critic.parameters(),
                strict=True,
            ):
                target_p.data.copy_(
                    p.data * self.config.critic_target_update_weight
                    + target_p.data * (1.0 - self.config.critic_target_update_weight)
                )

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

    def make_optimizers_and_scheduler(self) -> dict[str, Optimizer]:
        """
        Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

        This function sets up Adam optimizers for:
        - The **actor network**, ensuring that only relevant parameters are optimized.
        - The **critic ensemble**, which evaluates the value function.
        - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

        It also initializes a learning rate scheduler, though currently, it is set to `None`.

        NOTE:
        - If the encoder is shared, its parameters are excluded from the actor's optimization process.
        - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

        Args:
            cfg: Configuration object containing hyperparameters.
            policy (nn.Module): The policy model containing the actor, critic, and temperature components.

        Returns:
            A dictionary mapping component names ("actor", "critic", "temperature")
            to their respective Adam optimizers.
        """
        actor_params = self.policy.get_optim_params()["actor"]
        self.optimizers = {
            "actor": torch.optim.Adam(actor_params, lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critic_ensemble.parameters(), lr=self.config.critic_lr),
            "temperature": torch.optim.Adam([self.log_alpha], lr=self.config.temperature_lr),
        }
        if self.policy_config.num_discrete_actions is not None:
            self.optimizers["discrete_critic"] = torch.optim.Adam(
                self.policy.discrete_critic.parameters(), lr=self.config.critic_lr
            )
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    def get_weights(self) -> dict[str, Any]:
        """Send actor + discrete-critic state dicts."""
        state_dicts: dict[str, Any] = {
            "policy": move_state_dict_to_device(self.policy.actor.state_dict(), device="cpu"),
        }
        if self.policy_config.num_discrete_actions is not None:
            state_dicts["discrete_critic"] = move_state_dict_to_device(
                self.policy.discrete_critic.state_dict(), device="cpu"
            )
        return state_dicts

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load actor + discrete-critic weights into the policy."""
        actor_sd = move_state_dict_to_device(weights["policy"], device=device)
        self.policy.actor.load_state_dict(actor_sd)
        if "discrete_critic" in weights and self.policy.discrete_critic is not None:
            discrete_sd = move_state_dict_to_device(weights["discrete_critic"], device=device)
            self.policy.discrete_critic.load_state_dict(discrete_sd)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Algorithm-owned trainable tensors.

        Encoder weights are stripped because they are owned by the policy
        (``policy.encoder_critic``) and already saved via ``policy.save_pretrained``.
        """
        bundle: dict[str, torch.Tensor] = {}
        for k, v in _strip_encoder_keys(self.critic_ensemble.state_dict()).items():
            bundle[f"critic_ensemble.{k}"] = v
        for k, v in _strip_encoder_keys(self.critic_target.state_dict()).items():
            bundle[f"critic_target.{k}"] = v
        if self.discrete_critic_target is not None:
            for k, v in _strip_encoder_keys(self.discrete_critic_target.state_dict()).items():
                bundle[f"discrete_critic_target.{k}"] = v
        bundle["log_alpha"] = self.log_alpha.detach()
        return bundle

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device = "cpu",
    ) -> None:
        """In-place load of algorithm-owned tensors.

        ``log_alpha`` is restored via ``Parameter.data.copy_`` so the
        ``temperature`` optimizer's reference to the parameter object stays
        valid after resume.
        """
        critic_ensemble_state = _split_prefix(state_dict, "critic_ensemble.")
        critic_target_state = _split_prefix(state_dict, "critic_target.")
        self.critic_ensemble.load_state_dict(critic_ensemble_state, strict=False)
        self.critic_target.load_state_dict(critic_target_state, strict=False)

        if self.discrete_critic_target is not None:
            discrete_target_state = _split_prefix(state_dict, "discrete_critic_target.")
            self.discrete_critic_target.load_state_dict(discrete_target_state, strict=False)

        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.log_alpha.device))

    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """
        Get observation features from the policy encoder. It act as cache for the observation features.
        when the encoder is frozen, the observation features are not updated.
        We can save compute by caching the observation features.

        Args:
            policy: The policy model
            observations: The current observations
            next_observations: The next observations

        Returns:
            tuple: observation_features, next_observation_features
        """

        if self.policy.config.vision_encoder_name is None or not self.policy.config.freeze_vision_encoder:
            return None, None

        with torch.no_grad():
            observation_features = self.policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = self.policy.actor.encoder.get_cached_image_features(next_observations)

        return observation_features, next_observation_features


def _strip_encoder_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop ``encoder.*`` keys from a critic-module state dict."""
    return {k: v for k, v in state.items() if not k.startswith("encoder.")}


def _split_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return the subset of ``state`` whose keys start with ``prefix``, prefix-stripped."""
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


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
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (GaussianActorObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
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
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values
