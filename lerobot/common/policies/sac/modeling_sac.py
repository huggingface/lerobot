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

from typing import Callable, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.utils import get_device_from_parameters


class SACPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "RL", "SAC"],
):
    name = "sac"

    def __init__(
        self,
        config: SACConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()

        if config is None:
            config = SACConfig()
        self.config = config
        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = nn.Identity()

        output_normalization_params = {}
        for outer_key, inner_dict in config.output_normalization_params.items():
            output_normalization_params[outer_key] = {}
            for key, value in inner_dict.items():
                output_normalization_params[outer_key][key] = torch.tensor(value)

        # HACK: This is hacky and should be removed
        dataset_stats = dataset_stats or output_normalization_params
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        # NOTE: For images the encoder should be shared between the actor and critic
        if config.shared_encoder:
            encoder_critic = SACObservationEncoder(config)
            encoder_actor: SACObservationEncoder = encoder_critic
        else:
            encoder_critic = SACObservationEncoder(config)
            encoder_actor = SACObservationEncoder(config)

        self.critic_ensemble = CriticEnsemble(
            encoder=encoder_critic,
            network_list=nn.ModuleList(
                [
                    MLP(
                        input_dim=encoder_critic.output_dim + config.output_shapes["action"][0],
                        **config.critic_network_kwargs,
                    )
                    for _ in range(config.num_critics)
                ]
            ),
        )

        self.critic_target = CriticEnsemble(
            encoder=encoder_critic,
            network_list=nn.ModuleList(
                [
                    MLP(
                        input_dim=encoder_critic.output_dim + config.output_shapes["action"][0],
                        **config.critic_network_kwargs,
                    )
                    for _ in range(config.num_critics)
                ]
            ),
        )

        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        self.actor = Policy(
            encoder=encoder_actor,
            network=MLP(input_dim=encoder_actor.output_dim, **config.actor_network_kwargs),
            action_dim=config.output_shapes["action"][0],
            encoder_is_shared=config.shared_encoder,
            **config.policy_kwargs,
        )
        if config.target_entropy is None:
            config.target_entropy = -np.prod(config.output_shapes["action"][0]) / 2  # (-dim(A)/2)

        # TODO (azouitine): Handle the case where the temparameter is a fixed
        # TODO (michel-aractingi): Put the log_alpha in cuda by default because otherwise
        # it triggers "can't optimize a non-leaf Tensor"
        self.log_alpha = torch.zeros(1, requires_grad=True, device=torch.device("cuda:0"))
        self.temperature = self.log_alpha.exp().item()

    def reset(self):
        """Reset the policy"""
        pass

    def to(self, *args, **kwargs):
        """Override .to(device) method to involve moving the log_alpha fixed_std"""
        if self.actor.fixed_std is not None:
            self.actor.fixed_std = self.actor.fixed_std.to(*args, **kwargs)
        self.log_alpha = self.log_alpha.to(*args, **kwargs)
        super().to(*args, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        actions, _, _ = self.actor(batch)
        actions = self.unnormalize_outputs({"action": actions})["action"]
        return actions

    def critic_forward(
        self, observations: dict[str, Tensor], actions: Tensor, use_target: bool = False
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
        q_values = critics(observations, actions)
        return q_values

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]: ...
    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=False
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )

    def compute_loss_critic(self, observations, actions, rewards, next_observations, done) -> Tensor:
        temperature = self.log_alpha.exp().item()
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations)

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations, actions=next_action_preds, use_target=True
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        q_preds = self.critic_forward(observations, actions, use_target=False)

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(1)
        ).sum()
        return critics_loss

    def compute_loss_temperature(self, observations) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.config.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(self, observations) -> Tensor:
        temperature = self.log_alpha.exp().item()

        actions_pi, log_probs, _ = self.actor(observations)

        q_preds = self.critic_forward(observations, actions_pi, use_target=False)
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((temperature * log_probs) - min_q_preds).mean()
        return actor_loss


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.ReLU(),
        activate_final: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        layers = []

        # First layer uses input_dim
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Add activation after first layer
        if dropout_rate is not None and dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        # layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())

        # Rest of the layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(hidden_dims[i]))
                layers.append(
                    activations if isinstance(activations, nn.Module) else getattr(nn, activations)()
                )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        encoder: Optional[nn.Module],
        network_list: nn.ModuleList,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network_list = network_list
        self.init_final = init_final

        self.parameters_to_optimize = []
        # Handle the case where a part of the encoder if frozen
        if self.encoder is not None:
            self.parameters_to_optimize += list(self.encoder.parameters_to_optimize)

        self.parameters_to_optimize += list(self.network_list.parameters())
        # Find the last Linear layer's output dimension
        for layer in reversed(network_list[0].net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        # Output layer
        self.output_layers = []
        if init_final is not None:
            for _ in network_list:
                output_layer = nn.Linear(out_features, 1)
                nn.init.uniform_(output_layer.weight, -init_final, init_final)
                nn.init.uniform_(output_layer.bias, -init_final, init_final)
                self.output_layers.append(output_layer)
        else:
            self.output_layers = []
            for _ in network_list:
                output_layer = nn.Linear(out_features, 1)
                orthogonal_init()(output_layer.weight)
                self.output_layers.append(output_layer)

        self.output_layers = nn.ModuleList(self.output_layers)
        self.parameters_to_optimize += list(self.output_layers.parameters())

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        actions = actions.to(device)

        obs_enc = observations if self.encoder is None else self.encoder(observations)

        inputs = torch.cat([obs_enc, actions], dim=-1)
        list_q_values = []
        for network, output_layer in zip(self.network_list, self.output_layers, strict=False):
            x = network(inputs)
            value = output_layer(x)
            list_q_values.append(value.squeeze(-1))
        return torch.stack(list_q_values)


class Policy(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
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
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.parameters_to_optimize = []

        self.parameters_to_optimize += list(self.network.parameters())

        if self.encoder is not None and not encoder_is_shared:
            self.parameters_to_optimize += list(self.encoder.parameters())
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

        self.parameters_to_optimize += list(self.mean_layer.parameters())
        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)
            self.parameters_to_optimize += list(self.std_layer.parameters())

    def forward(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode observations if encoder exists
        if  isinstance(observations, dict):
            obs_enc = observations["observation.state"] if self.encoder is None else self.encoder(observations)
        else:
            obs_enc = observations if self.encoder is None else self.encoder(observations)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            assert not torch.isnan(log_std).any(), "[ERROR] log_std became NaN after std_layer!"

            if self.use_tanh_squash:
                log_std = torch.tanh(log_std)
                log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
            else:
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.fixed_std.expand_as(means)

        # uses tanh activation function to squash the action to be in the range of [-1, 1]
        normal = torch.distributions.Normal(means, torch.exp(log_std))
        x_t = normal.rsample()  # Reparameterization trick (mean + std * N(0,1))
        log_probs = normal.log_prob(x_t)  # Base log probability before Tanh

        if self.use_tanh_squash:
            actions = torch.tanh(x_t)
            log_probs -= torch.log((1 - actions.pow(2)) + 1e-6)  # Adjust log-probs for Tanh
        else:
            actions = x_t  # No Tanh; raw Gaussian sample

        log_probs = log_probs.sum(-1)  # Sum over action dimensions
        means = torch.tanh(means) if self.use_tanh_squash else means
        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig):
        """
        Creates encoders for pixel and/or state modalities.
        """
        super().__init__()
        self.config = config
        self.has_pretrained_vision_encoder = False
        self.parameters_to_optimize = []

        self.aggregation_size: int = 0
        if "observation.image" in config.input_shapes:
            self.camera_number = config.camera_number

            if self.config.vision_encoder_name is not None:
                self.image_enc_layers = PretrainedImageEncoder(config)
                self.has_pretrained_vision_encoder = True
            else:
                self.image_enc_layers = DefaultImageEncoder(config)

            self.aggregation_size += config.latent_dim * self.camera_number

            if config.freeze_vision_encoder:
                freeze_image_encoder(self.image_enc_layers)
            else:
                self.parameters_to_optimize += list(self.image_enc_layers.parameters())

        if "observation.state" in config.input_shapes:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(
                    in_features=config.input_shapes["observation.state"][0], out_features=config.latent_dim
                ),
                nn.LayerNorm(normalized_shape=config.latent_dim),
                nn.Tanh(),
            )
            self.aggregation_size += config.latent_dim

            self.parameters_to_optimize += list(self.state_enc_layers.parameters())

        if "observation.environment_state" in config.input_shapes:
            self.env_state_enc_layers = nn.Sequential(
                nn.Linear(
                    in_features=config.input_shapes["observation.environment_state"][0],
                    out_features=config.latent_dim,
                ),
                nn.LayerNorm(normalized_shape=config.latent_dim),
                nn.Tanh(),
            )
            self.aggregation_size += config.latent_dim
            self.parameters_to_optimize += list(self.env_state_enc_layers.parameters())

        self.aggregation_layer = nn.Linear(in_features=self.aggregation_size, out_features=config.latent_dim)
        self.parameters_to_optimize += list(self.aggregation_layer.parameters())

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        # Concatenate all images along the channel dimension.
        image_keys = [k for k in self.config.input_shapes if k.startswith("observation.image")]
        for image_key in image_keys:
            enc_feat = self.image_enc_layers(obs_dict[image_key])

            # if not self.has_pretrained_vision_encoder:
            #     enc_feat = flatten_forward_unflatten(self.image_enc_layers, obs_dict[image_key])
            feat.append(enc_feat)
        if "observation.environment_state" in self.config.input_shapes:
            feat.append(self.env_state_enc_layers(obs_dict["observation.environment_state"]))
        if "observation.state" in self.config.input_shapes:
            feat.append(self.state_enc_layers(obs_dict["observation.state"]))

        features = torch.cat(tensors=feat, dim=-1)
        features = self.aggregation_layer(features)

        return features

    @property
    def output_dim(self) -> int:
        """Returns the dimension of the encoder output"""
        return self.config.latent_dim


class DefaultImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_enc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_shapes["observation.image"][0],
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
        dummy_batch = torch.zeros(1, *config.input_shapes["observation.image"])
        with torch.inference_mode():
            self.image_enc_out_shape = self.image_enc_layers(dummy_batch).shape[1:]
        self.image_enc_layers.extend(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.image_enc_out_shape), config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        return self.image_enc_layers(x)


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)
        self.image_enc_proj = nn.Sequential(
            nn.Linear(np.prod(self.image_enc_out_shape), config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Tanh(),
        )

    def _load_pretrained_vision_encoder(self, config):
        """Set up CNN encoder"""
        from transformers import AutoModel

        self.image_enc_layers = AutoModel.from_pretrained(config.vision_encoder_name)
        # self.image_enc_layers.pooler = Identity()

        if hasattr(self.image_enc_layers.config, "hidden_sizes"):
            self.image_enc_out_shape = self.image_enc_layers.config.hidden_sizes[-1]  # Last channel dimension
        elif hasattr(self.image_enc_layers, "fc"):
            self.image_enc_out_shape = self.image_enc_layers.fc.in_features
        else:
            raise ValueError("Unsupported vision encoder architecture, make sure you are using a CNN")
        return self.image_enc_layers, self.image_enc_out_shape

    def forward(self, x):
        # TODO: (maractingi, azouitine) check the forward pass of the pretrained model
        # doesn't reach the classifier layer because we don't need it
        enc_feat = self.image_enc_layers(x).pooler_output
        enc_feat = self.image_enc_proj(enc_feat.view(enc_feat.shape[0], -1))
        return enc_feat


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# TODO (azouitine): I think in our case this function is not usefull we should remove it
# after some investigation
# borrowed from tdmpc
def flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
    """Helper to temporarily flatten extra dims at the start of the image tensor.

    Args:
        fn: Callable that the image tensor will be passed to. It should accept (B, C, H, W) and return
            (B, *), where * is any number of dimensions.
        image_tensor: An image tensor of shape (**, C, H, W), where ** is any number of dimensions and
        can be more than 1 dimensions, generally different from *.
    Returns:
        A return value from the callable reshaped to (**, *).
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = torch.flatten(image_tensor, end_dim=-4)
    flat_out = fn(inp)
    return torch.reshape(flat_out, (*start_dims, *flat_out.shape[1:]))


if __name__ == "__main__":
    # Test the SACObservationEncoder
    import time

    config = SACConfig()
    config.num_critics = 10
    encoder = SACObservationEncoder(config)
    actor_encoder = SACObservationEncoder(config)
    encoder = torch.compile(encoder)
    critic_ensemble = CriticEnsemble(
        encoder=encoder,
        network_list=nn.ModuleList(
            [
                MLP(
                    input_dim=encoder.output_dim + config.output_shapes["action"][0],
                    **config.critic_network_kwargs,
                )
                for _ in range(config.num_critics)
            ]
        ),
    )
    actor = Policy(
        encoder=actor_encoder,
        network=MLP(input_dim=actor_encoder.output_dim, **config.actor_network_kwargs),
        action_dim=config.output_shapes["action"][0],
        encoder_is_shared=config.shared_encoder,
        **config.policy_kwargs,
    )
    encoder = encoder.to("cuda:0")
    critic_ensemble = torch.compile(critic_ensemble)
    critic_ensemble = critic_ensemble.to("cuda:0")
    actor = torch.compile(actor)
    actor = actor.to("cuda:0")
    obs_dict = {
        "observation.image": torch.randn(1, 3, 84, 84),
        "observation.state": torch.randn(1, 4),
    }
    actions = torch.randn(1, 2).to("cuda:0")
    obs_dict = {k: v.to("cuda:0") for k, v in obs_dict.items()}
    print("compiling...")
    # q_value = critic_ensemble(obs_dict, actions)
    action = actor(obs_dict)
    print("compiled")
    start = time.perf_counter()
    for _ in range(1000):
        # features = encoder(obs_dict)
        action = actor(obs_dict)
        # q_value = critic_ensemble(obs_dict, actions)
    print("Time taken:", time.perf_counter() - start)
