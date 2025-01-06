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

from collections import deque
from copy import deepcopy
from typing import Callable, Optional, Sequence, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.sac.configuration_sac import SACConfig


class SACPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "RL", "SAC"],
):
    name = "sac"

    def __init__(
        self, config: SACConfig | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
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
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        encoder_critic = SACObservationEncoder(config)
        encoder_actor = SACObservationEncoder(config)
        # Define networks
        critic_nets = []
        for _ in range(config.num_critics):
            critic_net = Critic(
                encoder=encoder_critic,
                network=MLP(
                    input_dim=encoder_critic.output_dim + config.output_shapes["action"][0],
                    **config.critic_network_kwargs
                )
            )
            critic_nets.append(critic_net)

        self.critic_ensemble = create_critic_ensemble(critic_nets, config.num_critics)
        self.critic_target = deepcopy(self.critic_ensemble)

        self.actor = Policy(
            encoder=encoder_actor,
            network=MLP(
                input_dim=encoder_actor.output_dim,
                **config.actor_network_kwargs
            ),
            action_dim=config.output_shapes["action"][0],
            **config.policy_kwargs
        )
        if config.target_entropy is None:
            config.target_entropy = -np.prod(config.output_shapes["action"][0]) #  (-dim(A))
        self.temperature = LagrangeMultiplier(init_value=config.temperature_init)    

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        queues are populated during rollout of the policy, they contain the n latest observations and actions
        """

        self._queues = {
            "observation.state": deque(maxlen=1),
            "action": deque(maxlen=1),
        }
        if "observation.image" in self.config.input_shapes:
            self._queues["observation.image"] = deque(maxlen=1)
        if "observation.environment_state" in self.config.input_shapes:
            self._queues["observation.environment_state"] = deque(maxlen=1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        actions, _ = self.actor(batch)
        actions = self.unnormalize_outputs({"action": actions})["action"]
        return actions
    
    def critic_forward(self, observations: dict[str, Tensor], actions: Tensor, use_target: bool = False) -> Tensor:
        """Forward pass through a critic network ensemble
        
        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics
        
        Returns:
            Tensor of Q-values from all critics
        """
        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = torch.stack([critic(observations, actions) for critic in critics])
        return q_values


    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Run the batch through the model and compute the loss.
        
        Returns a dictionary with loss as a tensor, and other information as native floats.
        """
        batch = self.normalize_inputs(batch)
        # batch shape is (b, 2, ...) where index 1 returns the current observation and 
        # the next observation for calculating the right td index. 
        actions = batch["action"][:, 0]
        rewards = batch["next.reward"][:, 0]
        observations = {}
        next_observations = {}
        for k in batch:
            if k.startswith("observation."):
                observations[k] = batch[k][:, 0]
                next_observations[k] = batch[k][:, 1]
       
        # perform image augmentation

        # reward bias from HIL-SERL code base 
        # add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]} in reward_batch
        
        # calculate critics loss
        # 1- compute actions from policy
        action_preds, log_probs = self.actor(next_observations)

        # 2- compute q targets
        q_targets = self.critic_forward(next_observations, action_preds, use_target=True)

        # subsample critics to prevent overfitting if use high UTD (update to date)
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(self.config.num_critics)
            indices = indices[:self.config.num_subsample_critics]
            q_targets = q_targets[indices]

        # critics subsample size
        min_q, _ = q_targets.min(dim=0)  # Get values from min operation

        # compute td target
        td_target = rewards + self.config.discount * min_q #+ self.config.discount * self.temperature() * log_probs # add entropy term

        # 3- compute predicted qs
        q_preds = self.critic_forward(observations, actions, use_target=False)

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        critics_loss = F.mse_loss(
            q_preds,  # shape: [num_critics, batch_size]
            einops.repeat(td_target, "b -> e b", e=q_preds.shape[0]), # expand td_target to match q_preds shape
            reduction="none"
        ).sum(0).mean()

        # critics_loss = (   
        #     F.mse_loss(
        #             q_preds,
        #             einops.repeat(td_target, "b -> e b", e=q_preds.shape[0]),
        #             reduction="none",
        #         ).sum(0)  # sum over ensemble
        #         # `q_preds_ensemble` depends on the first observation and the actions.
        #         * ~batch["observation.state_is_pad"][0]
        #         * ~batch["action_is_pad"]
        #         # q_targets depends on the reward and the next observations.
        #         * ~batch["next.reward_is_pad"]
        #         * ~batch["observation.state_is_pad"][1:]
        #     ).sum(0).mean()
        
        # calculate actors loss
        # 1- temperature
        temperature = self.temperature()
        # 2- get actions (batch_size, action_dim) and log probs (batch_size,)
        actions, log_probs = self.actor(observations)
        # 3- get q-value predictions
        with torch.inference_mode():
            q_preds = self.critic_forward(observations, actions, use_target=False)
        actor_loss = (
            -(q_preds - temperature * log_probs).mean()
            # * ~batch["observation.state_is_pad"][0]
            # * ~batch["action_is_pad"]
        ).mean()


        # calculate temperature loss
        # 1- calculate entropy
        entropy = -log_probs.mean()
        temperature_loss = self.temperature(
            lhs=entropy,
            rhs=self.config.target_entropy
        )

        loss = critics_loss + actor_loss + temperature_loss

        return {
                "critics_loss": critics_loss.item(),
                "actor_loss": actor_loss.item(),
                "temperature_loss": temperature_loss.item(),
                "temperature": temperature.item(),
                "entropy": entropy.item(),
                "loss": loss,
            }
 
    def update(self):
        # TODO: implement UTD update
        # First update only critics for utd_ratio-1 times
        #for critic_step in range(self.config.utd_ratio - 1):
            # only update critic and critic target
        # Then update critic, critic target, actor and temperature
        """Update target networks with exponential moving average"""
        with torch.no_grad():
            for target_critic, critic in zip(self.critic_target, self.critic_ensemble, strict=False):
                for target_param, param in zip(target_critic.parameters(), critic.parameters(), strict=False):
                    target_param.data.copy_(
                        target_param.data * self.config.critic_target_update_weight + 
                        param.data * (1.0 - self.config.critic_target_update_weight)
                    )
 
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
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
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())
        
        # Rest of the layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(hidden_dims[i]))
                layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())
                
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    
class Critic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        
        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        
        # Output layer
        if init_final is not None:
            self.output_layer = nn.Linear(out_features, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(out_features, 1)
            orthogonal_init()(self.output_layer.weight)
        
        self.to(self.device)

    def forward(
        self, 
        observations: dict[str, torch.Tensor], 
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Move each tensor in observations to device
        observations = {
            k: v.to(self.device) for k, v in observations.items()
        }
        actions = actions.to(self.device)
        
        obs_enc = observations if self.encoder is None else self.encoder(observations)
            
        inputs = torch.cat([obs_enc, actions], dim=-1)
        x = self.network(inputs)
        value = self.output_layer(x)
        return value.squeeze(-1)

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
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fixed_std = fixed_std.to(self.device) if fixed_std is not None else None
        self.use_tanh_squash = use_tanh_squash
        
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
        
        self.to(self.device)

    def forward(
        self, 
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
                
        # Encode observations if encoder exists
        obs_enc = observations if self.encoder is None else self.encoder(observations)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)
        
        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            if self.use_tanh_squash:
                log_std = torch.tanh(log_std)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.fixed_std.expand_as(means)
    
        # uses tahn activation function to squash the action to be in the range of [-1, 1]
        normal = torch.distributions.Normal(means, torch.exp(log_std))
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1)) 
        log_probs = normal.log_prob(x_t)
        if self.use_tanh_squash:
            actions = torch.tanh(x_t)
            log_probs -= torch.log((1 - actions.pow(2)) + 1e-6)
        log_probs = log_probs.sum(-1) # sum over action dim

        return actions, log_probs
    
    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        observations = observations.to(self.device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations.
    TODO(ke-wang): The original work allows for (1) stacking multiple history frames and (2) using pretrained resnet encoders.
    """

    def __init__(self, config: SACConfig):
        """
        Creates encoders for pixel and/or state modalities.
        """
        super().__init__()
        self.config = config

        if "observation.image" in config.input_shapes:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(
                    config.input_shapes["observation.image"][0], config.image_encoder_hidden_dim, 7, stride=2
                ),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
            )
            dummy_batch = torch.zeros(1, *config.input_shapes["observation.image"])
            with torch.inference_mode():
                out_shape = self.image_enc_layers(dummy_batch).shape[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), config.latent_dim),
                    nn.LayerNorm(config.latent_dim),
                    nn.Tanh(),
                )
            )
        if "observation.state" in config.input_shapes:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(config.input_shapes["observation.state"][0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Tanh(),
            )
        if "observation.environment_state" in config.input_shapes:
            self.env_state_enc_layers = nn.Sequential(
                nn.Linear(
                    config.input_shapes["observation.environment_state"][0], config.state_encoder_hidden_dim
                ),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Tanh(),
            )

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        # Concatenate all images along the channel dimension.
        image_keys = [k for k in self.config.input_shapes if k.startswith("observation.image")]
        for image_key in image_keys:
            feat.append(flatten_forward_unflatten(self.image_enc_layers, obs_dict[image_key]))
        if "observation.environment_state" in self.config.input_shapes:
            feat.append(self.env_state_enc_layers(obs_dict["observation.environment_state"]))
        if "observation.state" in self.config.input_shapes:
            feat.append(self.state_enc_layers(obs_dict["observation.state"]))
        # TODO(ke-wang): currently average over all features, concatenate all features maybe a better way
        return torch.stack(feat, dim=0).mean(0)
    
    @property
    def output_dim(self) -> int:
        """Returns the dimension of the encoder output"""
        return self.config.latent_dim


class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Sequence[int] = (),
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        init_value = torch.log(torch.exp(torch.tensor(init_value, device=self.device)) - 1)
            
        # Initialize the Lagrange multiplier as a parameter
        self.lagrange = nn.Parameter(
            torch.full(constraint_shape, init_value, dtype=torch.float32, device=self.device)
        )
        
        self.to(self.device)

    def forward(
        self, 
        lhs: Optional[torch.Tensor | float | int] = None, 
        rhs: Optional[torch.Tensor | float | int] = None
    ) -> torch.Tensor:
        # Get the multiplier value based on parameterization        
        multiplier = torch.nn.functional.softplus(self.lagrange)
                
        # Return the raw multiplier if no constraint values provided
        if lhs is None:
            return multiplier
            
        # Convert inputs to tensors and move to device
        lhs = torch.tensor(lhs, device=self.device) if not isinstance(lhs, torch.Tensor) else lhs.to(self.device)
        if rhs is not None:
            rhs = torch.tensor(rhs, device=self.device) if not isinstance(rhs, torch.Tensor) else rhs.to(self.device)
        else:
            rhs = torch.zeros_like(lhs, device=self.device)
            
        diff = lhs - rhs
        
        assert diff.shape == multiplier.shape, f"Shape mismatch: {diff.shape} vs {multiplier.shape}"
        
        return multiplier * diff


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


def create_critic_ensemble(critics: list[nn.Module], num_critics: int, device: str = "cuda") -> nn.ModuleList:
    """Creates an ensemble of critic networks"""
    assert len(critics) == num_critics, f"Expected {num_critics} critics, got {len(critics)}"
    return nn.ModuleList(critics).to(device)

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