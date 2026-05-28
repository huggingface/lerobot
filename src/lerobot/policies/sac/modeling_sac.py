#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig, is_image_feature
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class SACPolicy(
    PreTrainedPolicy,
):
    config_class = SACConfig
    name = "sac"

    def __init__(
        self,
        config: SACConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

        # Residual mode: load frozen base policy. Bypass nn.Module __setattr__
        # so the base policy is NOT registered as a child module — keeps its
        # params out of state_dict / optim groups / gRPC parameter sync.
        object.__setattr__(self, "_base_policy", None)
        if getattr(self.config, "residual_mode", False):
            object.__setattr__(self, "_base_policy", self._load_base_policy())

        # Pre-build a (1, A) tensor of residual scales for fast broadcast.
        scale = self.config.residual_action_scale
        if isinstance(scale, (list, tuple)):
            assert len(scale) == continuous_action_dim, (
                f"residual_action_scale list length {len(scale)} != action_dim {continuous_action_dim}"
            )
            scale_tensor = torch.tensor(list(scale), dtype=torch.float32)
        else:
            scale_tensor = torch.full((continuous_action_dim,), float(scale), dtype=torch.float32)
        # store as buffer so it moves with .to(device) and survives ckpt load.
        self.register_buffer("residual_scale_buf", scale_tensor.view(1, -1))

    def _load_base_policy(self):
        """Load + freeze base policy for residual mode (called only when residual_mode=True)."""
        cfg = self.config
        if cfg.base_policy_path is None:
            raise ValueError("residual_mode=True requires base_policy_path")
        from lerobot.configs.policies import PreTrainedConfig
        base_cfg = PreTrainedConfig.from_pretrained(cfg.base_policy_path)
        if cfg.base_policy_type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as _Cls
        elif cfg.base_policy_type == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as _Cls
        else:
            raise ValueError(f"Unknown base_policy_type {cfg.base_policy_type}")
        base = _Cls.from_pretrained(cfg.base_policy_path, config=base_cfg)
        base.eval()
        if cfg.freeze_base_policy:
            for p in base.parameters():
                p.requires_grad = False
        base = base.to(cfg.device)
        # CRITICAL: load preprocessor so raw uint8 images + raw state get
        # normalized exactly like during BC training. Without this, ACT
        # outputs garbage (incl. wrong gripper). Postprocessor is intentionally
        # skipped: residual mode expects base_action in normalized [-1,1].
        from lerobot.policies.factory import make_pre_post_processors
        _pre, _ = make_pre_post_processors(
            policy_cfg=base_cfg, pretrained_path=cfg.base_policy_path,
        )
        object.__setattr__(base, "_pre_processor", _pre)
        return base

    @property
    def base_policy(self):
        return self._base_policy

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        # Residual mode: base_action is expected to already be in `batch`
        # (attached by the actor-process env wrapper before each obs flows in).
        # Here we just read it, sample the residual, and clip the combined action.
        if getattr(self.config, "residual_mode", False):
            base_action = batch["observation.base_action"]
            observations_features = None
            if self.shared_encoder and self.actor.encoder.has_images:
                observations_features = self.actor.encoder.get_cached_image_features(batch)
            residual, _, _ = self.actor(batch, observations_features)
            residual = residual * self.residual_scale_buf
            actions = torch.clamp(base_action + residual, -1.0, 1.0)
            if self.config.num_discrete_actions is not None:
                discrete_action_value = self.discrete_critic(batch, observations_features)
                discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
                actions = torch.cat([actions, discrete_action], dim=-1)
            return actions

        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)

        actions, _, _ = self.actor(batch, observations_features)

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)

        return actions

    def critic_forward(
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

    def discrete_critic_forward(
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
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch[ACTION]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                critic_warmup=bool(batch.get("critic_warmup", False)),
            )

            return {"loss_critic": loss_critic}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}
        if model == "actor":
            bc_observations = batch.get("bc_state")
            bc_actions = batch.get("bc_action")
            bc_observation_features = batch.get("bc_observation_feature")
            bc_weights = batch.get("bc_weights")
            bc_loss_weight = float(batch.get("bc_loss_weight", 0.0))

            actor_loss, info = self.compute_loss_actor(
                observations=observations,
                observation_features=observation_features,
                bc_observations=bc_observations,
                bc_actions=bc_actions,
                bc_observation_features=bc_observation_features,
                bc_weights=bc_weights,
                bc_loss_weight=bc_loss_weight,
            )
            out = {"loss_actor": actor_loss}
            out.update(info)
            return out

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    @property
    def temperature(self) -> float:
        """Return the current temperature value, always in sync with log_alpha."""
        return self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
        critic_warmup: bool = False,
    ) -> Tensor:
        with torch.no_grad():
            residual_mode = getattr(self.config, "residual_mode", False)
            if critic_warmup and residual_mode:
                # Bootstrap: TD target uses frozen base policy action (δ=0)
                # → critic learns Q^{π_base}. Entropy term inert (log_prob=0).
                next_action_preds = next_observations["observation.base_action"]
                next_log_probs = torch.zeros(
                    next_action_preds.shape[0],
                    device=next_action_preds.device,
                    dtype=next_action_preds.dtype,
                )
            else:
                next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

                # Residual mode: actor predicts residual; combine w/ next-step base_action and clip.
                if residual_mode:
                    next_base = next_observations["observation.base_action"]
                    next_action_preds = torch.clamp(
                        next_base + next_action_preds * self.residual_scale_buf,
                        -1.0,
                        1.0,
                    )

            # 2- compute q targets
            q_targets = self.critic_forward(
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

            if getattr(self.config, "clip_q_target_to_reward_range", False):
                td_target = torch.clamp(
                    td_target, self.config.q_target_clip_min, self.config.q_target_clip_max
                )

        # 3- compute predicted qs
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
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

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
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
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        bc_observations: dict[str, Tensor] | None = None,
        bc_actions: Tensor | None = None,
        bc_observation_features: Tensor | None = None,
        bc_weights: Tensor | None = None,
        bc_loss_weight: float = 0.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        if getattr(self.config, "residual_mode", False):
            base = observations["observation.base_action"]
            actions_pi = torch.clamp(
                base + actions_pi * self.residual_scale_buf,
                -1.0,
                1.0,
            )

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        sac_loss = ((self.temperature * log_probs) - min_q_preds).mean()

        info: dict[str, Tensor] = {"loss_actor_sac": sac_loss.detach()}

        if bc_loss_weight > 0.0 and bc_actions is not None and bc_observations is not None:
            # Reparameterize: take demo (obs, action) and score under current policy.
            # Use mean (no tanh) → invert tanh on demo action to get pre-squash target.
            obs_enc = self.actor.encoder(
                bc_observations,
                cache=bc_observation_features,
                detach=self.actor.encoder_is_shared,
            )
            outputs = self.actor.network(obs_enc)
            means = self.actor.mean_layer(outputs)
            if self.actor.fixed_std is None:
                log_std = self.actor.std_layer(outputs)
                std = torch.exp(log_std)
                std = torch.maximum(std, self.actor.std_min_buf)
                std = torch.minimum(std, self.actor.std_max_buf)
            else:
                std = self.actor.fixed_std.expand_as(means)

            dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)
            # Buffer may store [continuous..., discrete]; actor only models continuous dims.
            target = bc_actions[..., : self.actor.action_dim]
            if self.actor.use_tanh_squash:
                target = target.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            log_prob_demo = dist.log_prob(target)  # (B,)

            if bc_weights is None:
                bc_loss = -log_prob_demo.mean()
            else:
                w = bc_weights.to(log_prob_demo.device, log_prob_demo.dtype)
                # Weighted mean (weights pre-normalized to sum=B by RABCWeights).
                bc_loss = -(w * log_prob_demo).sum() / (w.sum() + 1e-6)

            info["loss_actor_bc"] = bc_loss.detach()
            actor_loss = sac_loss + bc_loss_weight * bc_loss
        else:
            actor_loss = sac_loss

        return actor_loss, info

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self) -> None:
        """Set up temperature parameter (log_alpha)."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.config = config
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        if self.config.vision_encoder_name is not None:
            self.image_encoder = PretrainedImageEncoder(self.config)
        else:
            self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = torch.zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = nn.ModuleDict()
        self.post_encoders = nn.ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=self.config.latent_dim),
                nn.Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = OBS_ENV_STATE in self.config.input_features
        self.has_state = OBS_STATE in self.config.input_features
        if self.has_env:
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        # Residual mode: concat raw base action vector at the encoder tail so
        # both actor (residual) and critic see the base action.
        if getattr(self.config, "residual_mode", False):
            out += int(self.config.output_features[ACTION].shape[0])
        self._out_dim = out

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            parts.append(self.state_encoder(obs[OBS_STATE]))
        if getattr(self.config, "residual_mode", False):
            parts.append(obs["observation.base_action"])
        if parts:
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Usage patterns:
        - Called in select_action()
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward()

        Args:
            obs: Dictionary of observation tensors containing image keys

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

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


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float | list[float] = -5,
        std_max: float | list[float] = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        # Per-dim std bounds: pass a scalar (broadcast) or a list of length
        # action_dim. Stored as buffers so they move with .to(device).
        def _to_bound(v, n):
            if isinstance(v, (list, tuple)):
                assert len(v) == n, f"std bound list len {len(v)} != action_dim {n}"
                return torch.tensor(list(v), dtype=torch.float32)
            return torch.full((n,), float(v), dtype=torch.float32)
        self.register_buffer("std_min_buf", _to_bound(std_min, action_dim))
        self.register_buffer("std_max_buf", _to_bound(std_max, action_dim))
        # Keep scalar attrs for code that may still read them (e.g. logging).
        self.std_min = std_min if isinstance(std_min, (int, float)) else float(min(std_min))
        self.std_max = std_max if isinstance(std_max, (int, float)) else float(max(std_max))
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            # Per-dim clamp via broadcasting buffers (shape [action_dim]).
            std = torch.maximum(std, self.std_min_buf)
            std = torch.minimum(std, self.std_max_buf)
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.image_enc_layers(x)
        return x


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: SACConfig):
        """Set up CNN encoder"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        enc_feat = self.image_enc_layers(x).last_hidden_state
        return enc_feat


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x
