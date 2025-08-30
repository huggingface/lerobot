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

from typing import Literal, Optional

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
    CriticEnsemble,
    CriticHead,
    DiscreteCritic as GraspCritic,
    MLP,
    SACObservationEncoder,
    _convert_normalization_params_to_tensor,
)
from lerobot.policies.utils import get_device_from_parameters

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


def get_sigmas_karras(num_scales, sigma_min, sigma_max, rho):
    """Generate Karras noise schedule"""
    ramp = torch.linspace(0, 1, num_scales)
    min_inv_rho = sigma_min**(1 / rho)
    max_inv_rho = sigma_max**(1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    # Append zero for the final step
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas


def get_scalings_for_boundary_condition(sigma, sigma_data, sigma_min):
    """Get c_skip, c_out, c_in scalings for boundary condition"""
    c_skip = sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    c_out = (sigma - sigma_min) * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
    c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
    return c_skip, c_out, c_in


def compute_critic_loss(policy, batch):
    """Compute standard TD loss for critic (online stage)"""
    observations = batch["state"]
    actions = batch["action"]
    rewards = batch["reward"]
    dones = batch["done"]
    next_observations = batch["next_state"]
    obs_feat = batch.get("observation_feature")
    nxt_feat = batch.get("next_observation_feature")

    # Handle discrete actions (gripper) like SAC does
    if policy.config.num_discrete_actions is not None:
        # Strip discrete action dimension for critic (same as SAC)
        actions = actions[:, :DISCRETE_DIMENSION_INDEX]

    # Compute target Q-values
    with torch.no_grad():
        next_action, _ = policy.consistency_policy(next_observations, training=False)
        target_q_values = policy.critic_target(next_observations, next_action, nxt_feat)
        target_q = torch.min(target_q_values, dim=0)[0]
        target_value = rewards + policy.config.discount * (1 - dones) * target_q

    # Current Q-values
    current_q_values = policy.critic_ensemble(observations, actions, obs_feat)
    current_q1, current_q2 = current_q_values[0], current_q_values[1]

    # TD loss
    loss_critic = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

    return {
        "loss_critic": loss_critic,
        "q1_mean": current_q1.mean(),
        "q2_mean": current_q2.mean(),
        "target_q_mean": target_value.mean(),
    }


def compute_cal_ql_loss(policy, batch):
    """Compute Calibrated Q-Learning loss (offline stage)"""
    observations = batch["state"]
    actions = batch["action"]
    rewards = batch["reward"]
    dones = batch["done"]
    mc_returns = batch.get("mc_returns", rewards)  # Use MC returns if available
    next_observations = batch["next_state"]
    obs_feat = batch.get("observation_feature")
    nxt_feat = batch.get("next_observation_feature")

    batch_size = actions.shape[0]
    
    # Handle discrete actions (gripper) like SAC does
    if policy.config.num_discrete_actions is not None:
        # Strip discrete action dimension for critic (same as SAC)
        actions = actions[:, :DISCRETE_DIMENSION_INDEX]

    # Get action dimension after potential stripping
    action_dim = actions.shape[1]

    # Standard TD loss
    with torch.no_grad():
        next_action, _ = policy.consistency_policy(next_observations, training=False)
        # next_action from consistency_policy is already the correct dimension (continuous only)

        target_q_values = policy.critic_target(next_observations, next_action, nxt_feat)

        # Subsample critics
        if policy.config.critic_subsample_size is not None:
            indices = torch.randperm(policy.config.critic_ensemble_size)[:policy.config.critic_subsample_size]
            target_q_values = target_q_values[indices]

        target_q = torch.min(target_q_values, dim=0)[0]
        target_value = rewards + policy.config.discount * (1 - dones) * target_q

    # Current Q-values
    current_q_values = policy.critic_ensemble(observations, actions, obs_feat)

    if policy.config.critic_subsample_size is not None:
        indices = torch.randperm(policy.config.critic_ensemble_size)[:policy.config.critic_subsample_size]
        current_q_values = current_q_values[indices]
        critic_size = policy.config.critic_subsample_size
    else:
        critic_size = policy.config.critic_ensemble_size

    # Expand target values to match critic ensemble size
    target_values_expanded = target_value.unsqueeze(0).expand(critic_size, -1)

    # TD loss
    td_loss = F.mse_loss(current_q_values, target_values_expanded)

    # CQL penalty for OOD actions
    num_random = policy.config.cql_n_actions

    # Sample random actions
    if policy.config.cql_action_sample_method == "uniform":
        random_actions = torch.rand(batch_size, num_random, action_dim, device=actions.device) * 2 - 1
    elif policy.config.cql_action_sample_method == "normal":
        random_actions = torch.randn(batch_size, num_random, action_dim, device=actions.device)
    else:
        raise NotImplementedError

    # Sample actions from current policy
    policy_actions = []
    for _ in range(num_random):
        policy_action, _ = policy.consistency_policy(observations, training=False)
        # policy_action is already the correct dimension (continuous only)
        policy_actions.append(policy_action.unsqueeze(1))
    policy_actions = torch.cat(policy_actions, dim=1)

    # Sample next actions
    next_actions_cql = []
    for _ in range(num_random):
        next_action_cql, _ = policy.consistency_policy(next_observations, training=False)
        # next_action_cql is already the correct dimension (continuous only)
        next_actions_cql.append(next_action_cql.unsqueeze(1))
    next_actions_cql = torch.cat(next_actions_cql, dim=1)

    # Combine all sampled actions [random, current, next]
    all_sampled_actions = torch.cat([random_actions, policy_actions, next_actions_cql], dim=1)

    # Compute Q-values for all sampled actions
    cql_q_samples = []
    for i in range(all_sampled_actions.shape[1]):
        q_vals = policy.critic_ensemble(observations, all_sampled_actions[:, i], obs_feat)
        if policy.config.critic_subsample_size is not None:
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
    cql_q_samples = cql_q_samples - torch.log(torch.tensor(cql_q_samples.shape[-1])) * policy.config.cql_temp

    # Compute logsumexp of OOD actions
    cql_ood_values = torch.logsumexp(cql_q_samples / policy.config.cql_temp, dim=-1) * policy.config.cql_temp

    # CQL difference
    cql_q_diff = cql_ood_values - current_q_values
    cql_q_diff = torch.clamp(cql_q_diff, policy.config.cql_clip_diff_min, policy.config.cql_clip_diff_max)

    # CQL loss
    cql_loss = cql_q_diff.mean()
    
    # Total loss
    loss_critic = td_loss + policy.config.cql_alpha * cql_loss

    return {
        "loss_critic": loss_critic,
        "td_loss": td_loss,
        "cql_loss": cql_loss,
        "cql_alpha": policy.config.cql_alpha,
        "cql_diff": cql_q_diff.mean(),
        "calql_bound_rate": calql_bound_rate,
        "cql_ood_values": cql_ood_values.mean(),
        "predicted_qs": current_q_values.mean(),
        "target_qs": target_values_expanded.mean(),
    }


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


def compute_bc_loss(policy, batch):
    """Compute consistency-based BC loss"""
    observations = batch["state"]
    actions = batch["action"]
    batch_size = actions.shape[0]
    device = actions.device

    # Sample random diffusion step (matching JAX: indices from 0 to num_scales-1)
    indices = torch.randint(0, policy.config.num_scales - 1, (batch_size,), device=device)

    # Compute sigma values using the same formula as JAX
    t = (policy.config.sigma_max**(1 / policy.config.rho) + indices / (policy.config.num_scales - 1) * (policy.config.sigma_min**(1 / policy.config.rho) - policy.config.sigma_max**(1 / policy.config.rho)))
    t = t**policy.config.rho

    # Add noise to actions
    noise = torch.randn_like(actions)
    dims = actions.ndim
    x_t = actions + noise * append_dims(t, dims)

    # Forward pass through consistency policy
    denoised_action, _ = policy.consistency_policy(
        observations, noisy_action=x_t, sigma=t, training=True
    )

    # Compute SNR and weightings
    snrs = get_snr(t)
    weights = get_weightings("karras", snrs, policy.config.sigma_data)

    # Compute weighted reconstruction loss
    recon_diffs = (denoised_action - actions) ** 2
    recon_loss = (mean_flat(recon_diffs) * weights).mean()

    return {
        "loss_bc": recon_loss,
        "action_mse": F.mse_loss(denoised_action, actions),
        "recon_loss": recon_loss,
    }


def compute_actor_loss(policy, batch):
    """Compute actor loss combining BC and Q losses"""
    # Get BC loss
    bc_output = compute_bc_loss(policy, batch)
    bc_loss = bc_output["loss_bc"]

    # Get Q loss
    observations = batch["state"]
    obs_feat = batch.get("observation_feature")
    policy_action, _ = policy.consistency_policy(observations, training=False)

    q_values = policy.critic_ensemble(observations, policy_action, obs_feat)
    if policy.config.critic_subsample_size is not None:
        indices = torch.randperm(policy.config.critic_ensemble_size)[
            : policy.config.critic_subsample_size
        ]
        q_values = q_values[indices]
    q_value = torch.min(q_values, dim=0)[0]
    q_loss = -q_value.mean()  # Negative for gradient ascent

    # Combined loss
    loss_actor = policy.bc_weight * bc_loss + policy.q_weight * q_loss

    return {
        "loss_actor": loss_actor,
        "bc_loss": bc_loss,
        "q_loss": q_loss,
        "bc_weight": policy.bc_weight,
        "q_weight": policy.q_weight,
        "q_mean": q_value.mean(),
    }


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
        self.config = config

        # Initialize normalization - follow SAC's pattern exactly
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        self.unnormalize_outputs = nn.Identity()
        if config.dataset_stats is not None:
            params = _convert_normalization_params_to_tensor(config.dataset_stats)
            self.normalize_inputs = NormalizeBuffer(
                config.input_features, config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = NormalizeBuffer(
                config.output_features, config.normalization_mapping, stats
            )
            self.unnormalize_outputs = UnnormalizeBuffer(
                config.output_features, config.normalization_mapping, stats
            )

        # Initialize encoders
        self.encoder_critic = SACObservationEncoder(config, self.normalize_inputs)

        # ConRFT always requires an Octo model
        octo_policy = OctoPolicy.from_pretrained(config.base_vla_model_path)
        if config.freeze_base_vla:
            for param in octo_policy.parameters():
                param.requires_grad = False
        self.encoder_actor = OctoEncodingWrapper(
            octo_policy,
            use_proprio=config.use_proprio,
            proprio_latent_dim=config.proprio_latent_dim
        )

        # Get dimensions
        action_dim = config.output_features["action"].shape[0]

        # Initialize consistency policy
        # Calculate input dimension: Octo embeddings + proprioception (if enabled) + time + action
        actor_input_dim = octo_policy.config.token_embedding_size  # Octo embedding size
        if config.use_proprio:
            actor_input_dim += config.proprio_latent_dim  # Add proprioception dimension

        self.consistency_policy = ConsistencyPolicy(
            encoder=self.encoder_actor,
            network=MLP(
                input_dim=actor_input_dim + config.time_dim + action_dim,
                hidden_dims=[config.consistency_hidden_dim, config.consistency_hidden_dim],
            ),
            t_network=TimeMLP(t_dim=config.time_dim),
            action_dim=action_dim,
            sigma_data=config.sigma_data,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            num_scales=config.num_scales,
            clip_denoised=config.clip_denoised,
        )

        # Initialize critic networks (twin Q-functions)
        critic_input_dim = self.encoder_critic.output_dim + action_dim
        critic_hidden_dims = [config.critic_hidden_dim, config.critic_hidden_dim] # [256, 256]
        critics = [CriticHead(input_dim=critic_input_dim, hidden_dims=critic_hidden_dims) for _ in range(2)]
        self.critic_ensemble = CriticEnsemble(self.encoder_critic, critics, self.normalize_targets)

        target_encoder = SACObservationEncoder(config, self.normalize_inputs)
        target_critics = [
            CriticHead(input_dim=critic_input_dim, hidden_dims=critic_hidden_dims) for _ in range(2)
        ]
        # Create a separate NormalizeBuffer for the target critic to avoid sharing state
        if config.dataset_stats is not None:
            target_normalize_buffer = NormalizeBuffer(
                config.output_features, config.normalization_mapping, stats
            )
        else:
            target_normalize_buffer = nn.Identity()
        self.critic_target = CriticEnsemble(target_encoder, target_critics, target_normalize_buffer)

        # Initialize discrete critic (grasp critic) only if needed
        if config.num_discrete_actions is not None:
            self.grasp_critic = GraspCritic(
                encoder=self.encoder_critic,
                input_dim=self.encoder_critic.output_dim,
                hidden_dims=critic_hidden_dims, # [256, 256]
                output_dim=config.num_discrete_actions,
            )
            self.grasp_critic_target = GraspCritic(
                encoder=target_encoder,
                input_dim=target_encoder.output_dim,
                hidden_dims=critic_hidden_dims,
                output_dim=config.num_discrete_actions,
            )
            # Copy weights to target
            self.grasp_critic_target.load_state_dict(self.grasp_critic.state_dict())

        # Copy weights to target
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # Training stage and loss weights
        self.training_stage = "offline"
        self.bc_weight = config.bc_weight_offline
        self.q_weight = config.q_weight_offline

    def _encode_state(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode observations into state representation using VLA"""
        if self.encoder_actor is not None:
            return self.encoder_actor(batch)
        else:
            # When no VLA is used, the actor shares the critic's encoder.
            return self.encoder_critic(batch), None

    def get_optim_params(self) -> dict:
        """Return parameters for optimization"""
        params = {
            "consistency_policy": self.consistency_policy.parameters(),
            "critic": self.critic_ensemble.parameters(),
        }

        if self.config.num_discrete_actions is not None:
            params["grasp_critic"] = self.grasp_critic.parameters()

        if self.encoder_actor is not None and not self.config.freeze_base_vla:
            params["encoder_actor"] = self.encoder_actor.parameters()

        return params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions - not used in ConRFT"""
        raise NotImplementedError("ConRFT does not support action chunking")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        self.eval()

        # Normalize inputs
        st = batch["state"] if "state" in batch else batch
        st = self.normalize_inputs(st)

        # Generate action using consistency policy
        action, _ = self.consistency_policy(st, training=False)

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
            return compute_critic_loss(self, batch)
        elif model == "actor":
            return compute_actor_loss(self, batch)
        elif model == "cal_ql":
            return compute_cal_ql_loss(self, batch)
        elif model == "bc":
            return compute_bc_loss(self, batch)
        else:
            raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with soft updates"""
        tau = self.config.soft_target_update_rate
        for target_param, param in zip(self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.grasp_critic_target.parameters(), self.grasp_critic.parameters(), strict=False
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
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self.sin_pos_emb(t)
        return self.net(t)


class OctoEncodingWrapper(nn.Module):
    """Wrapper around Octo transformer to extract action embeddings for ConRFT."""

    def __init__(self, octo_policy: OctoPolicy, use_proprio: bool = True, proprio_latent_dim: int = 64):
        super().__init__()
        self.octo_policy = octo_policy
        self.octo_transformer = octo_policy.model.octo_transformer
        self.text_processor = octo_policy.text_processor
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim

        # Proprioception encoder will be created dynamically
        self.proprio_encoder = None

    def forward(
        self,
        observations: dict[str, Tensor],
        tasks: Optional[dict[str, Tensor]] = None,
        action_embeddings: Optional[Tensor] = None,
        stop_gradient: bool = True,
    ) -> tuple[Tensor, Optional[Tensor]]:
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

            # Get transformer outputs
            if stop_gradient:
                with torch.no_grad():
                    transformer_outputs = self.octo_transformer(obs, task_dict, timestep_pad_mask)
            else:
                transformer_outputs = self.octo_transformer(obs, task_dict, timestep_pad_mask)

            # Extract action embeddings (readout_action tokens)
            action_embeddings = transformer_outputs["readout_action"]  # TimestepGroup object

            # Extract the actual tensor from TimestepGroup
            # TimestepGroup has .tokens attribute containing the tensor
            if hasattr(action_embeddings, 'tokens'):
                action_embeddings = action_embeddings.tokens

            # Flatten to [batch_size, embedding_dim] for consistency policy
            # action_embeddings shape: [batch_size, horizon, n_tokens, embedding_dim]
            # We want [batch_size, embedding_dim], so take first timestep and first token
            if action_embeddings.dim() == 4:
                action_embeddings = action_embeddings[:, 0, 0, :]  # Take first timestep, first token
            elif action_embeddings.dim() == 3:
                action_embeddings = action_embeddings.squeeze(1)  # Remove window dimension

        encoded = action_embeddings

        # Add proprioception
        if self.use_proprio and "observation.state" in observations:
            state = observations["observation.state"]

            # Create proprioception encoder dynamically if not exists
            if self.proprio_encoder is None:
                state_dim = state.shape[-1]
                self.proprio_encoder = nn.Sequential(
                    nn.Linear(state_dim, self.proprio_latent_dim),
                    nn.LayerNorm(self.proprio_latent_dim),
                    nn.Tanh()
                ).to(state.device)

            if stop_gradient:
                with torch.no_grad():
                    state_encoded = self.proprio_encoder(state)
            else:
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
        # Use the last hidden dimension from the network configuration
        self.action_head = nn.Linear(256, action_dim)  # consistency_hidden_dim from config

    def forward(
        self,
        observations: dict[str, Tensor],
        tasks: Optional[dict[str, Tensor]] = None,
        action_embeddings: Optional[Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of consistency policy"""
        if self.encoder is not None:
            state_encoding, action_embeddings = self.encoder(observations, tasks, action_embeddings)
        else:
            state_encoding = observations
            action_embeddings = None

        batch_size = state_encoding.shape[0]
        device = state_encoding.device

        if training:
            assert noisy_action is not None and sigma is not None
            x_t = noisy_action
        else:
            # Sample from noise for inference
            x_t = torch.randn(batch_size, self.action_dim, device=device) * self.sigma_max
            sigma = self.sigmas[0].expand(batch_size).to(device)

        # Get scaling factors
        c_skip, c_out, c_in = get_scalings_for_boundary_condition(sigma, self.sigma_data, self.sigma_min)

        # Reshape for broadcasting
        c_skip = c_skip.view(-1, 1)
        c_out = c_out.view(-1, 1)
        c_in = c_in.view(-1, 1)

        # Time embedding
        rescaled_t = 1000 * 0.25 * torch.log(sigma + 1e-44)
        t_embed = self.t_network(rescaled_t.view(-1, 1))

        # Ensure all tensors are 2D for concatenation
        c_in_x_t = c_in * x_t  # [batch_size, action_dim]

        # Flatten state_encoding if it has extra dimensions
        if state_encoding.dim() > 2:
            state_encoding = state_encoding.view(batch_size, -1)

        # Ensure t_embed is also 2D
        if t_embed.dim() > 2:
            t_embed = t_embed.view(batch_size, -1)

        # Network forward pass
        network_input = torch.cat([c_in_x_t, t_embed, state_encoding], dim=1)
        network_output = self.network(network_input)

        # Final dense layer for action output
        action_output = self.action_head(network_output)

        # Apply boundary condition
        denoised = c_out * action_output + c_skip * x_t

        if self.clip_denoised and not training:
            denoised = torch.clamp(denoised, -1, 1)

        return denoised, action_embeddings
