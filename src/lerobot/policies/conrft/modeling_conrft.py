#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
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

"""
ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy

This module implements the ConRFT (Consistency-based Reinforced Fine-Tuning) approach
for fine-tuning Vision-Language-Action (VLA) models in robotic manipulation tasks.

ConRFT consists of two stages:
1. Cal-ConRFT (offline): Combines behavior cloning with Cal-QL for stable initialization
2. HIL-ConRFT (online): Human-in-the-loop reinforcement learning with consistency policy

Paper: https://arxiv.org/abs/2502.05450
"""

from dataclasses import asdict
from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.constants import ACTION
from lerobot.policies.conrft.configuration_conrft import ConRFTConfig
from lerobot.policies.normalize import NormalizeBuffer, UnnormalizeBuffer
from lerobot.policies.octo.modeling_octo import OctoPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.modeling_sac import (
    MLP,
    CriticEnsemble,
    CriticHead,
    SACObservationEncoder,
    _convert_normalization_params_to_tensor,
)
from lerobot.policies.utils import get_device_from_parameters


class ConRFTPolicy(PreTrainedPolicy):
    """ConRFT Policy for VLA model fine-tuning

    This policy implements the two-stage ConRFT approach:
    1. Offline Cal-ConRFT: Calibrated Q-learning with consistency BC loss
    2. Online HIL-ConRFT: Online RL with human interventions

    It integrates with LeRobot's infrastructure by inheriting from PreTrainedPolicy
    and following LeRobot's patterns for RL policies.
    """

    config_class = ConRFTConfig
    name = "conrft"

    def __init__(
        self,
        config: ConRFTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        self._init_normalization(dataset_stats)
        self._init_octo_policy()
        self._init_encoders()
        self._init_consistency_policy(continuous_action_dim)
        self._init_critics(continuous_action_dim)

        # Training stage and loss weights
        self.training_stage = "offline"
        self.bc_weight = config.bc_weight_offline
        self.q_weight = config.q_weight_offline

    def _init_normalization(self, dataset_stats):
        """Initialize input/output normalization modules."""
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        self.unnormalize_outputs = nn.Identity()
        if self.config.dataset_stats is not None:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = NormalizeBuffer(
                self.config.input_features, self.config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = NormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )
            self.unnormalize_outputs = UnnormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def _init_octo_policy(self):
        """Initialize Octo VLA policy."""
        # ConRFT always requires an Octo model
        self.octo_policy = OctoPolicy.from_pretrained(self.config.base_vla_model_path)
        if self.config.freeze_base_vla:
            for param in self.octo_policy.parameters():
                param.requires_grad = False

    def _init_encoders(self):
        """Initialize shared or separate encoders for consistency policy and critic."""
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        self.encoder_actor = OctoEncodingWrapper(
            self.octo_policy,
            use_proprio=self.config.use_proprio,
            state_dim=self.config.state_dim,
            proprio_latent_dim=self.config.proprio_latent_dim,
        )

    def _init_consistency_policy(self, continuous_action_dim):
        # Calculate input dimension: Octo embeddings + proprioception (if enabled) + time + action
        self.actor_input_dim = self.octo_policy.config.token_embedding_size  # Octo embedding size
        if self.config.use_proprio:
            self.actor_input_dim += self.config.proprio_latent_dim  # Add proprioception dimension

        self.consistency_policy = ConsistencyPolicy(
            encoder=self.encoder_actor,
            network=MLP(
                input_dim=self.actor_input_dim + self.config.time_dim + continuous_action_dim,
                hidden_dims=[self.config.consistency_hidden_dim, self.config.consistency_hidden_dim],
            ),
            t_network=TimeMLP(t_dim=self.config.time_dim),
            action_dim=continuous_action_dim,
            sigma_data=self.config.sigma_data,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            rho=self.config.rho,
            num_scales=self.config.num_scales,
            clip_denoised=self.config.clip_denoised,
            consistency_hidden_dim=self.config.consistency_hidden_dim,
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
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )

        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_ensemble_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_ensemble_target = torch.compile(self.critic_ensemble_target)
            self.consistency_policy = torch.compile(self.consistency_policy)

    def _encode_state(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode observations into state representation using VLA"""
        return self.encoder_actor(batch)

    def get_optim_params(self) -> dict:
        """Return parameters for optimization"""
        params = {
            "consistency_policy": self.consistency_policy.parameters(),
            "critic": self.critic_ensemble.parameters(),
        }

        if self.encoder_actor is not None and not self.config.freeze_base_vla:
            params["encoder_actor"] = self.encoder_actor.parameters()

        return params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("ConRFTPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        self.eval()

        # Normalize inputs
        st = batch["state"] if "state" in batch else batch
        st = self.normalize_inputs(st)

        # Generate action using consistency policy
        action, _ = self.consistency_policy(st)

        # Unnormalize action
        action = self.unnormalize_outputs({ACTION: action})[ACTION]

        return action

    def forward(
        self,
        batch: dict[str, Tensor],
        model: Literal["actor", "critic", "cal_ql", "bc"] = "critic",
    ) -> dict[str, Tensor]:
        """Forward pass for training

        Args:
            batch: Training batch containing states, actions, rewards, etc.
            model: Which component to compute loss for

        Returns:
            Dictionary containing the computed loss and metrics
        """
        if model == "critic":
            return self.compute_critic_loss(batch)
        elif model == "actor":
            return self.compute_actor_loss(batch)
        elif model == "cal_ql":
            return self.compute_cal_ql_loss(batch)
        elif model == "bc":
            return self.compute_bc_loss(batch)
        else:
            raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with soft updates"""
        tau = self.config.soft_target_update_rate
        for target_param, param in zip(
            self.critic_ensemble_target.parameters(), self.critic_ensemble.parameters(), strict=False
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def set_training_stage(self, stage: str):
        """Switch between offline and online training stages"""
        assert stage in ["offline", "online"], f"Invalid stage: {stage}"
        self.training_stage = stage

        if stage == "offline":
            self.bc_weight = self.config.bc_weight_offline
            self.q_weight = self.config.q_weight_offline
        else:
            self.bc_weight = self.config.bc_weight_online
            self.q_weight = self.config.q_weight_online

    def compute_critic_loss(self, batch):
        """Compute standard TD loss for critic (online stage)"""
        observations = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]
        next_observations = batch["next_state"]
        observation_features = batch.get("observation_feature")
        next_observation_features = batch.get("next_observation_feature")
        # action_embeddings = batch.get("action_embeddings")
        next_action_embeddings = batch.get("next_action_embeddings")

        # Compute target Q-values
        with torch.no_grad():
            next_action, _ = self.consistency_policy(
                next_observations, action_embeddings=next_action_embeddings
            )
            target_q_values = self.critic_ensemble_target(
                next_observations, next_action, next_observation_features
            )
            target_q = torch.min(target_q_values, dim=0)[0]
            target_value = rewards + self.config.discount * (1 - dones) * target_q

        # Current Q-values
        current_q_values = self.critic_ensemble(observations, actions, observation_features)
        # current_q1, current_q2 = current_q_values[0], current_q_values[1]

        # TD loss
        # loss_critic = (F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)) / 2
        # TODO(lilkm): check this loss
        loss_critic = F.mse_loss(current_q_values, target_value.unsqueeze(0).expand_as(current_q_values))

        return {
            "loss_critic": loss_critic,
            # "q1_mean": current_q1.mean(),
            # "q2_mean": current_q2.mean(),
            "target_q_mean": target_value.mean(),
        }

    def compute_cal_ql_loss(self, batch):
        """Compute Calibrated Q-Learning loss (offline stage)"""
        observations = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]
        mc_returns = batch.get("mc_returns", rewards)  # Use MC returns if available
        next_observations = batch["next_state"]
        observation_features = batch.get("observation_feature")
        next_observation_features = batch.get("next_observation_feature")
        action_embeddings = batch.get("action_embeddings")
        next_action_embeddings = batch.get("next_action_embeddings")

        batch_size = actions.shape[0]

        # Get action dimension after potential stripping
        action_dim = actions.shape[1]

        # Standard TD loss
        with torch.no_grad():
            next_action, _ = self.consistency_policy(
                observations=next_observations, action_embeddings=action_embeddings
            )
            # next_action from consistency_policy is already the correct dimension (continuous only)

            target_q_values = self.critic_ensemble_target(
                next_observations, next_action, next_observation_features
            )

            # TODO(lilkm): Get indices before forward pass to avoid unnecessary computation
            # Subsample critics
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)[: self.config.num_subsample_critics]
                target_q_values = target_q_values[indices]

            target_q = torch.min(target_q_values, dim=0)[0]
            target_value = rewards + self.config.discount * (1 - dones) * target_q

        # Current Q-values
        current_q_values = self.critic_ensemble(observations, actions, observation_features)

        # TODO(lilkm): Get indices before forward pass to avoid unnecessary computation
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(self.config.num_critics)[: self.config.num_subsample_critics]
            current_q_values = current_q_values[indices]
            critic_size = self.config.num_subsample_critics
        else:
            critic_size = self.config.num_critics

        # Expand target values to match critic ensemble size
        target_values_expanded = target_value.unsqueeze(0).expand(critic_size, -1)

        # TD loss
        td_loss = F.mse_loss(current_q_values, target_values_expanded)

        # CQL penalty for OOD actions
        num_random = self.config.cql_n_actions

        # Sample random actions
        if self.config.cql_action_sample_method == "uniform":
            random_actions = torch.rand(batch_size, num_random, action_dim, device=actions.device) * 2 - 1
        elif self.config.cql_action_sample_method == "normal":
            random_actions = torch.randn(batch_size, num_random, action_dim, device=actions.device)
        else:
            raise NotImplementedError

        # Vectorized: sample num_random actions in one call using the repeat axis
        policy_actions, _ = self.consistency_policy(observations, repeat=num_random)
        # policy_actions: [B, num_random, action_dim]

        next_actions_cql, _ = self.consistency_policy(next_observations, repeat=num_random)
        # next_actions_cql: [B, num_random, action_dim]

        # Combine all sampled actions [random, current, next]
        all_sampled_actions = torch.cat([random_actions, policy_actions, next_actions_cql], dim=1)

        # Compute Q-values for all sampled actions
        cql_q_samples = []
        for i in range(all_sampled_actions.shape[1]):
            q_vals = self.critic_ensemble(observations, all_sampled_actions[:, i], observation_features)
            # TODO(lilkm): Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                q_vals = q_vals[indices]
            cql_q_samples.append(q_vals.unsqueeze(-1))
        cql_q_samples = torch.cat(cql_q_samples, dim=-1)  # Shape: [critic_size, batch_size, n_actions*3]

        # Cal-QL: Apply lower bound using MC returns
        n_actions_for_calql = num_random * 3
        mc_lower_bound = mc_returns.unsqueeze(0).unsqueeze(-1).expand(critic_size, -1, n_actions_for_calql)

        # Count bound violations for logging
        num_vals = cql_q_samples.numel()
        calql_bound_rate = (cql_q_samples < mc_lower_bound).float().sum() / num_vals

        # Apply the bound
        cql_q_samples = torch.maximum(cql_q_samples, mc_lower_bound)

        # Add current Q-values and apply temperature scaling
        current_q_expanded = current_q_values.unsqueeze(-1)  # [critic_size, batch_size, 1]
        cql_q_samples = torch.cat([cql_q_samples, current_q_expanded], dim=-1)

        # Subtract log(num_samples) * temperature
        cql_q_samples = (
            cql_q_samples - torch.log(torch.tensor(cql_q_samples.shape[-1])) * self.config.cql_temp
        )

        # Compute logsumexp of OOD actions
        cql_ood_values = torch.logsumexp(cql_q_samples / self.config.cql_temp, dim=-1) * self.config.cql_temp

        # CQL difference
        cql_q_diff = cql_ood_values - current_q_values
        cql_q_diff = torch.clamp(cql_q_diff, self.config.cql_clip_diff_min, self.config.cql_clip_diff_max)

        # CQL loss
        cql_loss = cql_q_diff.mean()

        # Total loss
        loss_critic = td_loss + self.config.cql_alpha * cql_loss

        return {
            "loss_critic": loss_critic,
            "td_loss": td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": self.config.cql_alpha,
            "cql_diff": cql_q_diff.mean(),
            "calql_bound_rate": calql_bound_rate,
            "cql_ood_values": cql_ood_values.mean(),
            "predicted_qs": current_q_values.mean(),
            "target_qs": target_values_expanded.mean(),
        }

    def compute_bc_loss(self, batch):
        """Compute consistency-based BC loss"""
        observations = batch["state"]
        actions = batch["action"]
        batch_size = actions.shape[0]
        device = actions.device
        action_embeddings = batch.get("action_embeddings")

        # Sample random diffusion step (matching JAX: indices from 0 to num_scales-1)
        indices = torch.randint(0, self.config.num_scales - 1, (batch_size,), device=device)

        # Compute sigma values using the same formula as JAX
        t = self.config.sigma_max ** (1 / self.config.rho) + indices / (self.config.num_scales - 1) * (
            self.config.sigma_min ** (1 / self.config.rho) - self.config.sigma_max ** (1 / self.config.rho)
        )
        t = t**self.config.rho

        # Add noise to actions
        noise = torch.randn_like(actions)
        dims = actions.ndim
        x_t = actions + noise * append_dims(t, dims)

        # Forward pass through consistency policy
        denoised_action, _ = self.consistency_policy(
            observations, x_t=x_t, sigmas=t, action_embeddings=action_embeddings
        )

        # Compute SNR and weightings
        snrs = get_snr(t)
        weights = get_weightings("karras", snrs, self.config.sigma_data)

        # Compute weighted reconstruction loss
        recon_diffs = (denoised_action - actions) ** 2
        recon_loss = (mean_flat(recon_diffs) * weights).mean()

        return {
            "loss_bc": recon_loss,
            "action_mse": F.mse_loss(denoised_action, actions),
            "recon_loss": recon_loss,
        }

    def compute_actor_loss(self, batch):
        """Compute actor loss combining BC and Q losses"""
        # Get BC loss
        bc_output = self.compute_bc_loss(batch)
        bc_loss = bc_output["loss_bc"]

        # Get Q loss
        observations = batch["state"]
        observation_features = batch.get("observation_feature")
        # next_observation_features = batch.get("next_observation_feature")
        action_embeddings = batch.get("action_embeddings")
        # next_action_embeddings = batch.get("next_action_embeddings")

        policy_action, _ = self.consistency_policy(observations, action_embeddings=action_embeddings)

        q_values = self.critic_ensemble(observations, policy_action, observation_features)
        # TODO(lilkm): Get indices before forward pass to avoid unnecessary computation
        if self.config.num_subsample_critics is not None:
            indices = torch.randperm(self.config.num_critics)[: self.config.num_subsample_critics]
            q_values = q_values[indices]
        q_value = q_values.mean(dim=0)
        q_loss = -q_value.mean()  # Negative for gradient ascent

        # Combined loss
        loss_actor = self.bc_weight * bc_loss + self.q_weight * q_loss

        return {
            "loss_actor": loss_actor,
            "bc_loss": bc_loss,
            "q_loss": q_loss,
            "bc_weight": self.bc_weight,
            "q_weight": self.q_weight,
            "q_mean": q_value.mean(),
        }


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=time.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings
        return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, t_dim: int):
        super().__init__()
        self.t_dim = t_dim
        self.sin_pos_emb = SinusoidalPosEmb(t_dim)
        self.net = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2),
            nn.Tanh(),  # Changed from nn.Tanh() to match JAX implementation (swish)
            nn.Linear(t_dim * 2, t_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self.sin_pos_emb(t)
        return self.net(t)


class OctoEncodingWrapper(nn.Module):
    """Wrapper around Octo transformer to extract action embeddings for ConRFT."""

    def __init__(
        self,
        octo_policy: OctoPolicy,
        use_proprio: bool = True,
        state_dim: int = 18,
        proprio_latent_dim: int = 64,
    ):
        super().__init__()
        self.octo_policy = octo_policy
        self.octo_transformer = octo_policy.model.octo_transformer
        self.text_processor = octo_policy.text_processor
        self.use_proprio = use_proprio
        self.state_dim = state_dim
        self.proprio_latent_dim = proprio_latent_dim

        # Create proprioception encoder if needed
        self.proprio_encoder = None
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(state_dim, self.proprio_latent_dim),
                nn.LayerNorm(self.proprio_latent_dim),
                nn.Tanh(),
            )

    def get_cached_action_embeddings(
        self, observations: dict[str, Tensor], normalize: bool = False
    ) -> dict[str, Tensor]:
        """Extract and cache action embeddings from Octo transformer.

        This function processes observations through the Octo transformer once and returns
        the resulting action embeddings. When the Octo model is frozen, these embeddings can be safely cached and
        reused across policy components, avoiding redundant forward passes.

        Args:
            observations: Dictionary of observation tensors
            normalize: Whether to normalize observations before encoding (currently unused for Octo)

        Returns:
            Tensor containing the cached action embeddings
        """
        # Get batch size from observations
        batch_size = next(iter(observations.values())).shape[0]

        # Create empty tasks for the entire batch
        raw_tasks = [""] * batch_size

        # Prepare batch in Octo format with proper batch size
        prepared_batch = self.octo_policy._prepare_batch(observations, raw_tasks=raw_tasks)
        obs, task_dict, _, _, timestep_pad_mask = prepared_batch

        # Get transformer outputs
        transformer_outputs = self.octo_transformer(obs, task_dict, timestep_pad_mask)

        # Extract action embeddings (readout_action tokens)
        action_embeddings = transformer_outputs["readout_action"]  # TimestepGroup object

        # Extract the actual tensor from TimestepGroup
        # TimestepGroup has .tokens attribute containing the tensor
        if hasattr(action_embeddings, "tokens"):
            action_embeddings = action_embeddings.tokens

        # Mean over tokens and take last timestep
        action_embeddings = action_embeddings.mean(dim=-2)  # Mean over tokens
        action_embeddings = action_embeddings[:, -1, :]  # Take last timestep

        return action_embeddings

    def forward(
        self,
        observations: dict[str, Tensor],
        tasks: dict[str, Tensor] | None = None,
        action_embeddings: Tensor | None = None,
        stop_gradient: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Extract action embeddings from Octo transformer and concatenate with proprioception"""
        if action_embeddings is None:
            # Get batch size from observations
            batch_size = next(iter(observations.values())).shape[0]

            # Prepare batch in Octo format with proper batch size
            if tasks and "language_instruction" in tasks:
                raw_tasks = tasks["language_instruction"]
            else:
                # Create empty tasks for the entire batch
                raw_tasks = [""] * batch_size

            prepared_batch = self.octo_policy._prepare_batch(observations, raw_tasks=raw_tasks)
            obs, task_dict, _, _, timestep_pad_mask = prepared_batch

            # TODO(lilkm): add masking when training
            # # Apply masking to wrist image when not stopping gradient (like JAX)
            # if not stop_gradient:
            #     # Create mask with 20% probability of masking wrist image
            #     mask_prob = 0.2
            #     mask = torch.rand(batch_size, device=device) < mask_prob
            #     # Expand mask to match image_wrist dimensions
            #     mask_expanded = mask.view(batch_size, 1, 1, 1, 1)
            #     image_wrist = torch.where(mask_expanded, torch.zeros_like(image_wrist), image_wrist)

            # Get transformer outputs
            transformer_outputs = self.octo_transformer(obs, task_dict, timestep_pad_mask)

            # Extract action embeddings (readout_action tokens)
            action_embeddings = transformer_outputs["readout_action"]  # TimestepGroup object

            # Extract the actual tensor from TimestepGroup
            # TimestepGroup has .tokens attribute containing the tensor
            if hasattr(action_embeddings, "tokens"):
                action_embeddings = action_embeddings.tokens

            # TODO(lilkm): check this
            # Mean over tokens and take last timestep like JAX
            action_embeddings = action_embeddings.mean(dim=-2)  # Mean over tokens
            action_embeddings = action_embeddings[:, -1, :]  # Take last timestep

            # # Flatten to [batch_size, embedding_dim] for consistency policy
            # # action_embeddings shape: [batch_size, horizon, n_tokens, embedding_dim]
            # # We want [batch_size, embedding_dim], so take first timestep and first token
            # if action_embeddings.dim() == 4:
            #     action_embeddings = action_embeddings[:, 0, 0, :]  # Take first timestep, first token
            # elif action_embeddings.dim() == 3:
            #     action_embeddings = action_embeddings.squeeze(1)  # Remove window dimension

        encoded = action_embeddings

        if stop_gradient:
            action_embeddings = action_embeddings.detach()

        # Add proprioception
        if self.use_proprio and "observation.state" in observations:
            state = observations["observation.state"]

            # TODO(lilkm): implement state stacking
            # # Handle state stacking like JAX
            # if self.enable_stacking:
            #     import einops
            #     # Combine stacking and channels into a single dimension
            #     if len(state.shape) == 2:
            #         state = einops.rearrange(state, "T C -> (T C)")
            #         # If encoded is 1D, we need to handle it
            #         if len(encoded.shape) == 1:
            #             encoded = encoded.unsqueeze(0)
            #     elif len(state.shape) == 3:
            #         state = einops.rearrange(state, "B T C -> B (T C)")

            state_encoded = self.proprio_encoder(state)
            encoded = torch.cat([encoded, state_encoded], dim=-1)

        return encoded, action_embeddings


class ConsistencyPolicy(nn.Module):
    """Consistency Policy network for action generation

    This implements the consistency distillation approach for diffusion models,
    allowing single-step generation while maintaining the quality of multi-step diffusion.
    """

    def __init__(
        self,
        encoder: OctoEncodingWrapper,
        network: nn.Module,
        t_network: nn.Module,
        action_dim: int,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_scales: int = 40,
        clip_denoised: bool = True,
        consistency_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.t_network = t_network
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.clip_denoised = clip_denoised

        # Generate Karras noise schedule
        self.register_buffer("sigmas", get_sigmas_karras(num_scales, sigma_min, sigma_max, rho))

        # Action head for final output
        self.action_head = nn.Linear(consistency_hidden_dim, action_dim)

    def forward(
        self,
        observations: dict[str, Tensor],
        tasks: dict[str, Tensor] | None = None,
        action_embeddings: Tensor | None = None,
        x_t: Tensor | None = None,
        sigmas: Tensor | None = None,
        repeat: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of consistency policy"""
        obs_enc, action_embeddings = self.encoder(
            observations, tasks=tasks, action_embeddings=action_embeddings
        )

        device = get_device_from_parameters(self)
        batch_size = obs_enc.shape[0]

        # Ensure proper dimensions
        if obs_enc.ndim == 1:
            obs_enc = obs_enc.unsqueeze(0)

        # Handle repeat for multiple samples
        if repeat > 1:
            obs_enc = einops.repeat(obs_enc, "b f -> b r f", r=repeat)

        if x_t is None and sigmas is None:
            x_shape = (batch_size, repeat, self.action_dim) if repeat > 1 else (batch_size, self.action_dim)
            x_t = torch.randn(x_shape, device=device) * self.sigma_max
            sigmas = self.sigmas[0] * torch.ones(batch_size, device=device)
            x_0 = self.base_network(x_t, sigmas, obs_enc, repeat)
        else:
            x_0 = self.base_network(x_t, sigmas, obs_enc, repeat)

        if self.clip_denoised:
            x_0 = torch.clamp(x_0, -1, 1)

        return x_0, action_embeddings

    def base_network(
        self,
        x_t: Tensor,
        sigmas: Tensor,
        obs_enc: Tensor,
        repeat: int = 1,
    ) -> Tensor:
        # Get scaling factors and ensure proper dimensions
        c_skip, c_out, c_in = (
            append_dims(x, x_t.ndim)
            for x in get_scalings_for_boundary_condition(sigmas, self.sigma_data, self.sigma_min)
        )

        # Time embedding
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        t_embed = self.t_network(rescaled_t)

        # Handle repeat for time embedding
        cont_axis = 1
        if repeat > 1:
            t_embed = einops.repeat(t_embed, "b t -> b r t", r=repeat)
            cont_axis = 2

        # Forward pass
        outputs = self.network(torch.cat([c_in * x_t, t_embed, obs_enc], dim=cont_axis))

        # Final action output
        denoised = self.action_head(outputs)
        denoised = c_out * denoised + c_skip * x_t

        return denoised

    def get_features(self, observations):
        """Extract features from observations"""
        return self.encoder(observations)


def get_sigmas_karras(num_scales, sigma_min, sigma_max, rho, device="cpu"):
    """Generate Karras noise schedule"""
    ramp = torch.linspace(0, 1, num_scales, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # Append zero for the final step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    return sigmas


def get_scalings_for_boundary_condition(sigma, sigma_data, sigma_min):
    """Get c_skip, c_out, c_in scalings for boundary condition"""
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = (sigma - sigma_min) * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    return c_skip, c_out, c_in


def get_snr(sigmas):
    """Compute signal-to-noise ratio"""
    return sigmas**-2


def get_weightings(weighting, snrs, sigma_data):
    """Get loss weightings based on SNR"""
    if weighting == "karras":
        return snrs + 1.0 / sigma_data**2
    else:
        raise NotImplementedError(f"Weighting {weighting} not implemented")


def append_dims(x, target_dims):
    """Append dimensions to tensor"""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Cannot append {dims_to_append} dimensions")
    return x.view(*x.shape, *((1,) * dims_to_append))


def mean_flat(tensor):
    """Take mean over all dimensions except batch"""
    return tensor.mean(dim=list(range(1, tensor.ndim)))
