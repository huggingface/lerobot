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

from lerobot.policies.dsrl.configuration_dsrl import DSRLConfig, is_image_feature
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

VALID_ACTION_POLICIES = ["diffusion", "smolvla", "pi0", "pi05"]


class DSRLPolicy(PreTrainedPolicy):
    """Diffusion Steering via Reinforcement Learning - Noise-Aliased (DSRL-NA) Policy.

    DSRL-NA is a reinforcement learning algorithm designed to fine-tune pretrained diffusion policies
    by operating in a latent-noise space rather than the action space directly.

    The policy consists of three main components:

    1. **Action Critic (QA: S × A → R)**
       - Ensemble of critics that estimate Q-values for (state, action) pairs
       - Trained via standard TD-learning on the action space
       - Can incorporate offline data with actions in the original action space

    2. **Noise Critic (QW: S × W → R)**
       - Single critic that estimates Q-values for (state, noise) pairs
       - Trained via distillation from the action critic
       - Samples noise w ~ N(0, I), generates actions via diffusion policy, then matches action critic
       - Exploits noise aliasing: multiple noise vectors can map to the same action

    3. **Noise Actor (πW: S → △W)**
       - Policy that outputs distributions over noise vectors
       - Trained to maximize Q-values from the noise critic
       - During inference, samples noise and feeds it to the diffusion policy

    Training order:
    1. Action critic: TD-learning on (s, a, r, s', done) transitions
    2. Noise critic: Distillation from action critic via diffusion policy
    3. Noise actor: Policy gradient using noise critic Q-values

    Reference: Wagenmaker et al., 2025, https://arxiv.org/abs/2506.15799
    """

    config_class = DSRLConfig
    name = "dsrl"

    def __init__(
        self,
        config: DSRLConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the action policy
        self._init_action_policy()

        # Initialize DSRL components
        self._init_encoders()
        self._init_action_critics()
        self._init_noise_critic()
        self._init_noise_actor()
        self._init_temperature()

    def get_optim_params(self) -> dict:
        optim_params = {
            "noise_actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic_action": self.action_critic_ensemble.parameters(),
            "critic_noise": self.noise_critic.parameters(),
            "temperature": self.log_alpha,
        }
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("DSRLPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select noise vector for inference/evaluation,
        pass it through the action policy to get the action.

        Args:
            batch: Dictionary of observations
        """
        observations_features = None
        if self.shared_encoder and self.noise_actor.encoder.has_images:
            observations_features = self.noise_actor.encoder.get_cached_image_features(batch)

        noise, _, _ = self.noise_actor(batch, observations_features)

        return self.action_policy(batch, noise)

    def action_critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through action critic ensemble: QA(s, a)

        Args:
            observations: Dictionary of observations (state s)
            actions: Action tensor in action space (a)
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features

        Returns:
            Tensor of Q-values from all critics, shape (num_critics, batch_size)
        """
        critics = self.action_critic_target if use_target else self.action_critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def noise_critic_forward(
        self,
        observations: dict[str, Tensor],
        noise: Tensor,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through noise critic: QW(s, w)

        Args:
            observations: Dictionary of observations (state s)
            noise: Noise tensor in latent-noise space (w)
            observation_features: Optional pre-computed observation features

        Returns:
            Tensor of Q-value, shape (batch_size, 1)
        """
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}

        # Encode observations
        obs_enc = self.encoder_critic(observations, cache=observation_features)

        # Concatenate with noise and pass through critic
        inputs = torch.cat([obs_enc, noise], dim=-1)
        q_value = self.noise_critic(inputs)

        return q_value

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["noise_actor", "critic_action", "critic_noise", "temperature"] = "critic_action",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model component.

        DSRL-NA has three main components:
        1. Action critic (QA): trained via TD-learning on action space
        2. Noise critic (QW): trained via distillation from action critic
        3. Noise actor (πW): trained to maximize Q-values in noise space

        Args:
            batch: Dictionary containing training data
            model: Which component to compute loss for

        Returns:
            Dictionary with the computed loss
        """
        # Extract common components from batch
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor | None = batch.get("observation_feature")

        if model == "critic_action":
            # 1. Action Critic: TD-learning on action space
            # Extract critic-specific components
            actions: Tensor = batch[ACTION]
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor | None = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic_action(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )
            return {"loss_critic": loss_critic}

        if model == "critic_noise":
            # 2. Noise Critic: Distillation from action critic
            loss_critic_noise = self.compute_loss_critic_noise(
                observations=observations,
                observation_features=observation_features,
            )
            return {"loss_critic_noise": loss_critic_noise}

        if model == "noise_actor":
            # 3. Noise Actor: Maximize Q-values in noise space
            loss_noise_actor = self.compute_loss_noise_actor(
                observations=observations,
                observation_features=observation_features,
            )
            return {"loss_noise_actor": loss_noise_actor}

        if model == "temperature":
            # Temperature: Entropy coefficient auto-tuning
            loss_temperature = self.compute_loss_temperature(
                observations=observations,
                observation_features=observation_features,
            )
            return {"loss_temperature": loss_temperature}

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.action_critic_target.parameters(),
            self.action_critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic_action(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_observations: dict[str, Tensor],
        done: Tensor,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute action critic TD-learning loss: QA(s, a) → r + γ Q̄A(s', a')

        Algorithm from paper:
        1. Current Q-value: QA(s, a) from current state-action pair
        2. Next action: a' = πW_dp(s', w') where w' ~ πW(s') from noise actor
        3. Target Q-value: r + γ min_i Q̄A_i(s', a') from target critic
        4. Loss: MSE between current and target Q-values

        """
        with torch.no_grad():
            # 1. Sample noise from noise actor: w' ~ πW(s')
            next_noise, next_log_probs, _ = self.noise_actor(next_observations, next_observation_features)

            # 2. Generate next actions
            # a' = πW_dp(s', w')
            next_action_preds = self.action_policy(next_observations, next_noise)

            # 3. Compute target Q-values: Q̄A(s', a')
            q_targets = self.action_critic_forward(
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
        q_preds = self.action_critic_forward(
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

    def compute_loss_critic_noise(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute the noise critic distillation loss: QW(s, w) → QA(s, πW_dp(s, w))

        Algorithm from paper:
        1. Sample noise w ~ N(0, I)
        2. Generate action a = πW_dp(s, w) using the pretrained base policy
        3. Get target Q-value from action critic: QA(s, a)
        4. Get predicted Q-value from noise critic: QW(s, w)
        5. Loss = MSE(QA(s, a), QW(s, w))

        Args:
            observations: Dictionary of observations (state s)
            observation_features: Optional pre-computed observation features

        Returns:
            MSE loss between noise critic and action critic predictions
        """

        batch_size = next(iter(observations.values())).shape[0]
        action_dim = self.config.output_features[ACTION].shape[0]

        # 1. Sample noise w ~ N(0, I)
        noise = torch.randn(batch_size, action_dim, device=get_device_from_parameters(self))

        with torch.no_grad():
            # 2. Generate action using base policy: a = πW_dp(s, w)
            actions = self.action_policy(observations, noise)

            # 3. Get target Q-values from action critic: QA(s, a)
            q_targets = self.action_critic_forward(
                observations=observations,
                actions=actions,
                use_target=False,  # Use current action critic, not target
                observation_features=observation_features,
            )
            # Average over ensemble critics
            q_targets = q_targets.mean(dim=0, keepdim=True)  # (1, batch_size)

        # 4. Get predicted Q-values from noise critic: QW(s, w)
        q_preds = self.noise_critic_forward(
            observations=observations,
            noise=noise,
            observation_features=observation_features,
        )  # (batch_size, 1)

        # 5. Compute MSE loss
        loss = F.mse_loss(q_preds.squeeze(-1), q_targets.squeeze(0))

        return loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.noise_actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_noise_actor(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute the noise actor loss: max_{πW} E_s QW(s, πW(s))

        Algorithm from paper:
        1. Sample noise w ~ πW(s) from the noise actor
        2. Evaluate Q-value: QW(s, w) using noise critic
        3. Maximize Q-value with entropy regularization

        The loss is: (temperature * log_prob(w) - QW(s, w))
        which when minimized maximizes the Q-value and adds entropy regularization

        Args:
            observations: Dictionary of observations (state s)
            observation_features: Optional pre-computed observation features

        Returns:
            Noise actor loss with entropy regularization
        """
        # 1. Sample noise w ~ πW(s) from noise actor
        noise, log_probs, _ = self.noise_actor(observations, observation_features)

        # 2. Evaluate QW(s, w) using noise critic
        q_values = self.noise_critic_forward(
            observations=observations,
            noise=noise,
            observation_features=observation_features,
        )  # (batch_size, 1)

        # 3. Compute loss: minimize (temperature * log_prob - Q_value)
        # This is equivalent to maximizing (Q_value - temperature * log_prob)
        noise_actor_loss = (self.temperature * log_probs - q_values.squeeze(-1)).mean()

        return noise_actor_loss

    def _init_action_policy(self):
        """Initialize the action policy."""

        action_policy = get_policy_class(self.config.action_policy_name)
        self.action_policy = action_policy.from_pretrained(self.config.action_policy_weights)
        self.action_policy.to(self.config.device)
        self.action_policy.eval()

    def _init_encoders(self):
        """Initialize shared or separate encoders for noise actor and critics."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = DSRLObservationEncoder(self.config)
        self.encoder_noise_actor = (
            self.encoder_critic if self.shared_encoder else DSRLObservationEncoder(self.config)
        )

    def _init_action_critics(self):
        """Build critic ensemble, targets."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.action_critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.action_critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        self.action_critic_target.load_state_dict(self.action_critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.action_critic_ensemble = torch.compile(self.action_critic_ensemble)
            self.action_critic_target = torch.compile(self.action_critic_target)

    def _init_noise_critic(self):
        """Build noise critic: QW(s, w)

        The noise critic is a single critic head (not an ensemble) that maps
        (state, noise) pairs to Q-values. It's trained via distillation from
        the action critic ensemble.

        """
        self.noise_critic = CriticHead(
            input_dim=self.encoder_critic.output_dim,
            **asdict(self.config.noise_critic_network_kwargs),
        )

        if self.config.use_torch_compile:
            self.noise_critic = torch.compile(self.noise_critic)

    def _init_noise_actor(self):
        """Initialize noise actor network and default target entropy."""
        self.noise_actor = Policy(
            encoder=self.encoder_noise_actor,
            network=MLP(
                input_dim=self.encoder_noise_actor.output_dim,
                **asdict(self.config.noise_actor_network_kwargs),
            ),
            action_dim=self.config.output_features[ACTION].shape[0],
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.noise_actor_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = self.config.output_features[ACTION].shape[0]
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()


class DSRLObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: DSRLConfig) -> None:
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
        reused across policy components (noise actor, critics), avoiding redundant forward passes.

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
    CriticEnsemble wraps multiple CriticHead modules into an ensemble for action-value estimation.

    This is the Action Critic (QA) that maps (state, action) pairs to Q-values.

    Args:
        encoder (DSRLObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: DSRLObservationEncoder,
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


class Policy(nn.Module):
    """Noise Actor (πW) that maps states to noise distributions.

    This is the noise actor πW: S → △W that outputs noise vectors in the latent-noise space.
    The noise is then fed to a pretrained diffusion policy to generate actions.

    The policy outputs a Gaussian distribution over noise vectors, and is trained to
    maximize Q-values from the noise critic.
    """

    def __init__(
        self,
        encoder: DSRLObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: DSRLObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
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
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
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
    def __init__(self, config: DSRLConfig):
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
    def __init__(self, config: DSRLConfig):
        super().__init__()

        self.image_enc_layers, self.image_enc_out_shape = self._load_pretrained_vision_encoder(config)

    def _load_pretrained_vision_encoder(self, config: DSRLConfig):
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
