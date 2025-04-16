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

# TODO: (1) better device management

import math
from dataclasses import asdict
from typing import Callable, List, Literal, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.utils import get_device_from_parameters

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class SACPolicy(
    PreTrainedPolicy,
):
    config_class = SACConfig
    name = "sac"

    def __init__(
        self,
        config: SACConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        continuous_action_dim = config.output_features["action"].shape[0]

        if config.dataset_stats is not None:
            input_normalization_params = _convert_normalization_params_to_tensor(config.dataset_stats)
            self.normalize_inputs = Normalize(
                config.input_features,
                config.normalization_mapping,
                input_normalization_params,
            )
        else:
            self.normalize_inputs = nn.Identity()

        if config.dataset_stats is not None:
            output_normalization_params = _convert_normalization_params_to_tensor(config.dataset_stats)

            # HACK: This is hacky and should be removed
            dataset_stats = dataset_stats or output_normalization_params
            self.normalize_targets = Normalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = Unnormalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
        else:
            self.normalize_targets = nn.Identity()
            self.unnormalize_outputs = nn.Identity()

        # NOTE: For images the encoder should be shared between the actor and critic
        if config.shared_encoder:
            encoder_critic = SACObservationEncoder(config, self.normalize_inputs)
            encoder_actor: SACObservationEncoder = encoder_critic
        else:
            encoder_critic = SACObservationEncoder(config, self.normalize_inputs)
            encoder_actor = SACObservationEncoder(config, self.normalize_inputs)
        self.shared_encoder = config.shared_encoder

        # Create a list of critic heads
        critic_heads = [
            CriticHead(
                input_dim=encoder_critic.output_dim + continuous_action_dim,
                **asdict(config.critic_network_kwargs),
            )
            for _ in range(config.num_critics)
        ]

        self.critic_ensemble = CriticEnsemble(
            encoder=encoder_critic,
            ensemble=critic_heads,
            output_normalization=self.normalize_targets,
        )

        # Create target critic heads as deepcopies of the original critic heads
        target_critic_heads = [
            CriticHead(
                input_dim=encoder_critic.output_dim + continuous_action_dim,
                **asdict(config.critic_network_kwargs),
            )
            for _ in range(config.num_critics)
        ]

        self.critic_target = CriticEnsemble(
            encoder=encoder_critic,
            ensemble=target_critic_heads,
            output_normalization=self.normalize_targets,
        )

        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        self.critic_ensemble = torch.compile(self.critic_ensemble)
        self.critic_target = torch.compile(self.critic_target)

        self.grasp_critic = None
        self.grasp_critic_target = None

        if config.num_discrete_actions is not None:
            # Create grasp critic
            self.grasp_critic = GraspCritic(
                encoder=encoder_critic,
                input_dim=encoder_critic.output_dim,
                output_dim=config.num_discrete_actions,
                **asdict(config.grasp_critic_network_kwargs),
            )

            # Create target grasp critic
            self.grasp_critic_target = GraspCritic(
                encoder=encoder_critic,
                input_dim=encoder_critic.output_dim,
                output_dim=config.num_discrete_actions,
                **asdict(config.grasp_critic_network_kwargs),
            )

            self.grasp_critic_target.load_state_dict(self.grasp_critic.state_dict())

            self.grasp_critic = torch.compile(self.grasp_critic)
            self.grasp_critic_target = torch.compile(self.grasp_critic_target)

        self.actor = Policy(
            encoder=encoder_actor,
            network=MLP(input_dim=encoder_actor.output_dim, **asdict(config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=config.shared_encoder,
            **asdict(config.policy_kwargs),
        )
        if config.target_entropy is None:
            discrete_actions_dim: Literal[1] | Literal[0] = (
                1 if config.num_discrete_actions is not None else 0
            )
            config.target_entropy = -np.prod(continuous_action_dim + discrete_actions_dim) / 2  # (-dim(A)/2)

        # TODO (azouitine): Handle the case where the temparameter is a fixed
        # TODO (michel-aractingi): Put the log_alpha in cuda by default because otherwise
        # it triggers "can't optimize a non-leaf Tensor"

        temperature_init = config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temperature_init)]))
        self.temperature = self.log_alpha.exp().item()

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
            optim_params["grasp_critic"] = self.grasp_critic.parameters()
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    def to(self, *args, **kwargs):
        """Override .to(device) method to involve moving the log_alpha fixed_std"""
        if self.actor.fixed_std is not None:
            self.actor.fixed_std = self.actor.fixed_std.to(*args, **kwargs)
        # self.log_alpha = self.log_alpha.to(*args, **kwargs)
        super().to(*args, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        # We cached the encoder output to avoid recomputing it
        observations_features = None
        if self.shared_encoder:
            observations_features = self.actor.encoder.get_cached_image_features(batch=batch, normalize=True)

        actions, _, _ = self.actor(batch, observations_features)
        actions = self.unnormalize_outputs({"action": actions})["action"]

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.grasp_critic(batch, observations_features)
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

    def grasp_critic_forward(self, observations, use_target=False, observation_features=None) -> torch.Tensor:
        """Forward pass through a grasp critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the grasp critic network
        """
        grasp_critic = self.grasp_critic_target if use_target else self.grasp_critic
        q_values = grasp_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "grasp_critic"] = "critic",
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
            model: Which model to compute the loss for ("actor", "critic", "grasp_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch["action"]
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
            )

            return {"loss_critic": loss_critic}

        if model == "grasp_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_grasp_critic = self.compute_loss_grasp_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_grasp_critic": loss_grasp_critic}
        if model == "actor":
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

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
            strict=False,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.grasp_critic_target.parameters(),
                self.grasp_critic.parameters(),
                strict=False,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

            # TODO: (maractingi, azouitine) This is to slow, we should find a way to do this in a more efficient way
            next_action_preds = self.unnormalize_outputs({"action": next_action_preds})["action"]

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
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

    def compute_loss_grasp_critic(
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

        gripper_penalties: Tensor | None = None
        if complementary_info is not None:
            gripper_penalties: Tensor | None = complementary_info.get("gripper_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_grasp_qs = self.grasp_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_grasp_action = torch.argmax(next_grasp_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_grasp_qs = self.grasp_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_grasp_q = torch.gather(
                target_next_grasp_qs, dim=1, index=best_next_grasp_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_gripper = rewards
            if gripper_penalties is not None:
                rewards_gripper = rewards + gripper_penalties
            target_grasp_q = rewards_gripper + (1 - done) * self.config.discount * target_next_grasp_q

        # Get predicted Q-values for current observations
        predicted_grasp_qs = self.grasp_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_grasp_q = torch.gather(predicted_grasp_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        grasp_critic_loss = F.mse_loss(input=predicted_grasp_q, target=target_grasp_q)
        return grasp_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.config.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        # TODO: (maractingi, azouitine) This is to slow, we should find a way to do this in a more efficient way
        actions_pi: Tensor = self.unnormalize_outputs({"action": actions_pi})["action"]

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig, input_normalizer: nn.Module):
        """
        Creates encoders for pixel and/or state modalities.
        """
        super().__init__()
        self.config = config

        self.freeze_image_encoder = config.freeze_vision_encoder

        self.input_normalization = input_normalizer
        self._out_dim = 0

        if any("observation.image" in key for key in config.input_features):
            self.camera_number = config.camera_number
            self.all_image_keys = sorted(
                [k for k in config.input_features if k.startswith("observation.image")]
            )

            self._out_dim += len(self.all_image_keys) * config.latent_dim

            if self.config.vision_encoder_name is not None:
                self.image_enc_layers = PretrainedImageEncoder(config)
                self.has_pretrained_vision_encoder = True
            else:
                self.image_enc_layers = DefaultImageEncoder(config)

            if self.freeze_image_encoder:
                freeze_image_encoder(self.image_enc_layers)

            # Separate components for each image stream
            self.spatial_embeddings = nn.ModuleDict()
            self.post_encoders = nn.ModuleDict()

            # determine the nb_channels, height and width of the image

            # Get first image key from input features
            image_key = next(key for key in config.input_features if key.startswith("observation.image"))  # noqa: SIM118
            dummy_batch = torch.zeros(1, *config.input_features[image_key].shape)
            with torch.inference_mode():
                dummy_output = self.image_enc_layers(dummy_batch)
                _, channels, height, width = dummy_output.shape

            for key in self.all_image_keys:
                # HACK: This a hack because the state_dict use . to separate the keys
                safe_key = key.replace(".", "_")
                # Separate spatial embedding per image
                self.spatial_embeddings[safe_key] = SpatialLearnedEmbeddings(
                    height=height,
                    width=width,
                    channel=channels,
                    num_features=config.image_embedding_pooling_dim,
                )
                # Separate post-encoder per image
                self.post_encoders[safe_key] = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(
                        in_features=channels * config.image_embedding_pooling_dim,
                        out_features=config.latent_dim,
                    ),
                    nn.LayerNorm(normalized_shape=config.latent_dim),
                    nn.Tanh(),
                )

        if "observation.state" in config.input_features:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(
                    in_features=config.input_features["observation.state"].shape[0],
                    out_features=config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=config.latent_dim),
                nn.Tanh(),
            )
            self._out_dim += config.latent_dim
        if "observation.environment_state" in config.input_features:
            self.env_state_enc_layers = nn.Sequential(
                nn.Linear(
                    in_features=config.input_features["observation.environment_state"].shape[0],
                    out_features=config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=config.latent_dim),
                nn.Tanh(),
            )
            self._out_dim += config.latent_dim

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        vision_encoder_cache: torch.Tensor | None = None,
        detach: bool = False,
    ) -> torch.Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        obs_dict = self.input_normalization(obs_dict)
        if len(self.all_image_keys) > 0:
            if vision_encoder_cache is None:
                vision_encoder_cache = self.get_cached_image_features(obs_dict, normalize=False)

            vision_encoder_cache = self.get_full_image_representation_with_cached_features(
                batch_image_cached_features=vision_encoder_cache, detach=detach
            )
            feat.append(vision_encoder_cache)

        if "observation.environment_state" in self.config.input_features:
            feat.append(self.env_state_enc_layers(obs_dict["observation.environment_state"]))
        if "observation.state" in self.config.input_features:
            feat.append(self.state_enc_layers(obs_dict["observation.state"]))

        features = torch.cat(tensors=feat, dim=-1)

        return features

    def get_cached_image_features(
        self, batch: dict[str, torch.Tensor], normalize: bool = True
    ) -> dict[str, torch.Tensor]:
        """Get the cached image features for a batch of observations, when the image encoder is frozen"""
        if normalize:
            batch = self.input_normalization(batch)

        # Sort keys for consistent ordering
        sorted_keys = sorted(self.all_image_keys)

        # Stack all images into a single batch
        batched_input = torch.cat([batch[key] for key in sorted_keys], dim=0)

        # Process through the image encoder in one pass
        batched_output = self.image_enc_layers(batched_input)

        # Split the output back into individual tensors
        image_features = torch.chunk(batched_output, chunks=len(sorted_keys), dim=0)

        # Create a dictionary mapping the original keys to their features
        result = {}
        for key, features in zip(sorted_keys, image_features, strict=True):
            result[key] = features

        return result

    def get_full_image_representation_with_cached_features(
        self,
        batch_image_cached_features: dict[str, torch.Tensor],
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Get the full image representation with the cached features, applying the post-encoder and the spatial embedding"""

        image_features = []
        for key in batch_image_cached_features:
            safe_key = key.replace(".", "_")
            x = self.spatial_embeddings[safe_key](batch_image_cached_features[key])
            x = self.post_encoders[safe_key](x)

            # The gradient of the image encoder is not needed to update the policy
            if detach:
                x = x.detach()
            image_features.append(x)

        image_features = torch.cat(image_features, dim=-1)
        return image_features

    @property
    def output_dim(self) -> int:
        """Returns the dimension of the encoder output"""
        return self._out_dim


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: Optional[float] = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        layers = []

        # First layer uses input_dim
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Add activation after first layer
        if dropout_rate is not None and dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())

        # Rest of the layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(hidden_dims[i]))

                # If we're at the final layer and a final activation is specified, use it
                if i + 1 == len(hidden_dims) and activate_final and final_activation is not None:
                    layers.append(
                        final_activation
                        if isinstance(final_activation, nn.Module)
                        else getattr(nn, final_activation)()
                    )
                else:
                    layers.append(
                        activations if isinstance(activations, nn.Module) else getattr(nn, activations)()
                    )

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
        dropout_rate: Optional[float] = None,
        init_final: Optional[float] = None,
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
    ┌──────────────────┬─────────────────────────────────────────────────────────┐
    │ Critic Ensemble  │                                                         │
    ├──────────────────┘                                                         │
    │                                                                            │
    │        ┌────┐             ┌────┐                               ┌────┐      │
    │        │ Q1 │             │ Q2 │                               │ Qn │      │
    │        └────┘             └────┘                               └────┘      │
    │  ┌──────────────┐    ┌──────────────┐                     ┌──────────────┐ │
    │  │              │    │              │                     │              │ │
    │  │    MLP 1     │    │    MLP 2     │                     │     MLP      │ │
    │  │              │    │              │       ...           │ num_critics  │ │
    │  │              │    │              │                     │              │ │
    │  └──────────────┘    └──────────────┘                     └──────────────┘ │
    │          ▲                   ▲                                    ▲        │
    │          └───────────────────┴───────┬────────────────────────────┘        │
    │                                      │                                     │
    │                                      │                                     │
    │                            ┌───────────────────┐                           │
    │                            │     Embedding     │                           │
    │                            │                   │                           │
    │                            └───────────────────┘                           │
    │                                      ▲                                     │
    │                                      │                                     │
    │                        ┌─────────────┴────────────┐                        │
    │                        │                          │                        │
    │                        │  SACObservationEncoder   │                        │
    │                        │                          │                        │
    │                        └──────────────────────────┘                        │
    │                                      ▲                                     │
    │                                      │                                     │
    │                                      │                                     │
    │                                      │                                     │
    └───────────────────────────┬────────────────────┬───────────────────────────┘
                                │    Observation     │
                                └────────────────────┘
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: List[CriticHead],
        output_normalization: nn.Module,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
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
        # NOTE: We normalize actions it helps for sample efficiency
        actions: dict[str, torch.tensor] = {"action": actions}
        # NOTE: Normalization layer took dict in input and outputs a dict that why
        actions = self.output_normalization(actions)["action"]
        actions = actions.to(device)

        obs_enc = self.encoder(observations, observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class GraspCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: Optional[float] = None,
        init_final: Optional[float] = None,
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
        # Move each tensor in observations to device by cloning first to avoid inplace operations
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, vision_encoder_cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        log_std_min: float = -5,
        log_std_max: float = 2,
        fixed_std: Optional[torch.Tensor] = None,
        init_final: Optional[float] = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(
            observations, vision_encoder_cache=observation_features, detach=self.encoder_is_shared
        )

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.log_std_min, self.log_std_max)  # Match JAX default clip
        else:
            log_std = self.fixed_std.expand_as(means)

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
        image_key = next(key for key in config.input_features.keys() if key.startswith("observation.image"))  # noqa: SIM118
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
        # Get first image key from input features

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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params


if __name__ == "__main__":
    # # Benchmark the CriticEnsemble performance
    # import time

    # # Configuration
    # num_critics = 10
    # batch_size = 32
    # action_dim = 7
    # obs_dim = 64
    # hidden_dims = [256, 256]
    # num_iterations = 100

    # print("Creating test environment...")

    # # Create a simple dummy encoder
    # class DummyEncoder(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.output_dim = obs_dim
    #         self.parameters_to_optimize = []

    #     def forward(self, obs):
    #         # Just return a random tensor of the right shape
    #         # In practice, this would encode the observations
    #         return torch.randn(batch_size, obs_dim, device=device)

    # # Create critic heads
    # print(f"Creating {num_critics} critic heads...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # critic_heads = [
    #     CriticHead(
    #         input_dim=obs_dim + action_dim,
    #         hidden_dims=hidden_dims,
    #     ).to(device)
    #     for _ in range(num_critics)
    # ]

    # # Create the critic ensemble
    # print("Creating CriticEnsemble...")
    # critic_ensemble = CriticEnsemble(
    #     encoder=DummyEncoder().to(device),
    #     ensemble=critic_heads,
    #     output_normalization=nn.Identity(),
    # ).to(device)

    # # Create random input data
    # print("Creating input data...")
    # obs_dict = {
    #     "observation.state": torch.randn(batch_size, obs_dim, device=device),
    # }
    # actions = torch.randn(batch_size, action_dim, device=device)

    # # Warmup run
    # print("Warming up...")
    # _ = critic_ensemble(obs_dict, actions)

    # # Time the forward pass
    # print(f"Running benchmark with {num_iterations} iterations...")
    # start_time = time.perf_counter()
    # for _ in range(num_iterations):
    #     q_values = critic_ensemble(obs_dict, actions)
    # end_time = time.perf_counter()

    # # Print results
    # elapsed_time = end_time - start_time
    # print(f"Total time: {elapsed_time:.4f} seconds")
    # print(f"Average time per iteration: {elapsed_time / num_iterations * 1000:.4f} ms")
    # print(f"Output shape: {q_values.shape}")  # Should be [num_critics, batch_size]

    # Verify that all critic heads produce different outputs
    # This confirms each critic head is unique
    # print("\nVerifying critic outputs are different:")
    # for i in range(num_critics):
    #     for j in range(i + 1, num_critics):
    #         diff = torch.abs(q_values[i] - q_values[j]).mean().item()
    #         print(f"Mean difference between critic {i} and {j}: {diff:.6f}")

    from lerobot.configs import parser

    @parser.wrap()
    def main(config: SACConfig):
        policy = SACPolicy(config=config)
        print("yolo")

    main()
