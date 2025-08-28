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

from collections import deque
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.constants import ACTION
from lerobot.policies.conrft.configuration_conrft import ConRFTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.octo.modeling_octo import OctoPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.modeling_sac import (
    CriticEnsemble,
    CriticHead,
    DiscreteCritic as GraspCritic,
    MLP,
    SACObservationEncoder,
)


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
    observations = {k: v for k, v in batch.items() if "observation" in k}
    actions = batch["action"]
    rewards = batch["reward"]
    dones = batch["done"]
    next_observations = {k.replace("next_", ""): v for k, v in batch.items() if "next_observation" in k}

    # Compute target Q-values
    with torch.no_grad():
        next_action, _ = policy.consistency_policy(next_observations, training=False)
        target_q_values = policy.critic_target(next_observations, next_action)
        target_q = torch.min(target_q_values, dim=0)[0]
        target_value = rewards + policy.config.discount * (1 - dones) * target_q

    # Current Q-values
    current_q_values = policy.critic(observations, actions)
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
    observations = {k: v for k, v in batch.items() if "observation" in k}
    actions = batch["action"]
    rewards = batch["reward"]
    dones = batch["done"]
    mc_returns = batch.get("mc_returns", rewards)  # Use MC returns if available
    next_observations = {k.replace("next_", ""): v for k, v in batch.items() if "next_observation" in k}

    batch_size = actions.shape[0]

    # Standard TD loss
    with torch.no_grad():
        next_action, _ = policy.consistency_policy(next_observations, training=False)
        target_q_values = policy.critic_target(next_observations, next_action)
        target_q = torch.min(target_q_values, dim=0)[0]
        target_value = rewards + policy.config.discount * (1 - dones) * target_q

    current_q_values = policy.critic(observations, actions)
    current_q1, current_q2 = current_q_values[0], current_q_values[1]

    td_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

    # CQL penalty for OOD actions
    num_random = policy.config.cql_n_actions

    # Sample random actions
    random_actions = torch.rand(batch_size, num_random, actions.shape[1], device=actions.device) * 2 - 1

    # Sample actions from current policy
    policy_actions = []
    for _ in range(num_random):
        policy_action, _ = policy.consistency_policy(observations, training=False)
        policy_actions.append(policy_action.unsqueeze(1))
    policy_actions = torch.cat(policy_actions, dim=1)

    # Sample next actions
    next_actions_cql = []
    for _ in range(num_random):
        next_action_cql, _ = policy.consistency_policy(next_observations, training=False)
        next_actions_cql.append(next_action_cql.unsqueeze(1))
    next_actions_cql = torch.cat(next_actions_cql, dim=1)

    # Combine all sampled actions
    all_actions = torch.cat([random_actions, policy_actions, next_actions_cql], dim=1)

    # Compute Q-values for all sampled actions
    cql_q_values = []
    for i in range(all_actions.shape[1]):
        q_values = policy.critic(observations, all_actions[:, i])
        cql_q_values.append(torch.min(q_values, dim=0)[0].unsqueeze(-1))
    cql_q_values = torch.cat(cql_q_values, dim=1)

    # Apply Cal-QL bound using MC returns
    mc_lower_bound = mc_returns.unsqueeze(1).expand(-1, cql_q_values.shape[1])
    cql_q_values = torch.maximum(cql_q_values, mc_lower_bound)

    # Add current Q-values
    current_q = torch.min(torch.stack([current_q1, current_q2]), dim=0)[0].unsqueeze(1)
    all_q_values = torch.cat([cql_q_values, current_q], dim=1)

    # CQL loss (logsumexp)
    cql_loss = (
        torch.logsumexp(all_q_values / policy.config.cql_temp, dim=1).mean() * policy.config.cql_temp -
        current_q.squeeze(1).mean()
    )

    # Total loss
    loss_critic = td_loss + policy.config.cql_alpha * cql_loss

    return {
        "loss_critic": loss_critic,
        "td_loss": td_loss,
        "cql_loss": cql_loss,
        "q1_mean": current_q1.mean(),
        "q2_mean": current_q2.mean(),
    }


def compute_bc_loss(policy, batch):
    """Compute consistency-based BC loss"""
    observations = {k: v for k, v in batch.items() if "observation" in k}
    actions = batch["action"]
    batch_size = actions.shape[0]

    # Sample random diffusion step
    m = torch.randint(1, policy.config.num_scales, (batch_size, ), device=actions.device)

    # Get noise levels for sampled steps
    sigma_m = policy.consistency_policy.sigmas[m].to(actions.device)

    # Add noise to actions
    noise = torch.randn_like(actions)
    noisy_action = actions + sigma_m.view(-1, 1) * noise

    # Forward pass through consistency policy
    denoised_action, _ = policy.consistency_policy(
        observations, noisy_action=noisy_action, sigma=sigma_m, training=True
    )

    # Compute L2 loss
    bc_loss = F.mse_loss(denoised_action, actions)

    return {
        "loss_bc": bc_loss,
        "action_mse": F.mse_loss(denoised_action, actions),
    }


def compute_actor_loss(policy, batch):
    """Compute actor loss combining BC and Q losses"""
    # Get BC loss
    bc_output = compute_bc_loss(policy, batch)
    bc_loss = bc_output["loss_bc"]

    # Get Q loss
    observations = {k: v for k, v in batch.items() if "observation" in k}
    policy_action, _ = policy.consistency_policy(observations, training=False)

    q_values = policy.critic(observations, policy_action)
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

        # Initialize normalization
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Initialize encoders
        self.encoder_critic = SACObservationEncoder(config, self.normalize_inputs)
        self.encoder_actor = None
        if config.base_vla_model_path:
            octo_policy = OctoPolicy.from_pretrained(config.base_vla_model_path)
            if config.freeze_base_vla:
                for param in octo_policy.parameters():
                    param.requires_grad = False
            self.encoder_actor = OctoEncodingWrapper(octo_policy)

        # Get dimensions
        action_dim = config.output_features["action"].shape[0]

        # Initialize consistency policy
        actor_input_dim = (
            self.encoder_actor.encoder.config.token_embedding_size
            if self.encoder_actor
            else self.encoder_critic.output_dim
        )
        self.consistency_policy = ConsistencyPolicy(
            encoder=self.encoder_actor,
            network=MLP(
                input_dim=actor_input_dim + config.consistency_hidden_dim + action_dim,
                hidden_dims=[config.consistency_hidden_dim, config.consistency_hidden_dim],
            ),
            t_network=TimeMLP(t_dim=config.consistency_hidden_dim),
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
        hidden_dims = [config.critic_hidden_dim, config.critic_hidden_dim]
        critics = [CriticHead(input_dim=critic_input_dim, hidden_dims=hidden_dims) for _ in range(2)]
        self.critic = CriticEnsemble(self.encoder_critic, critics, self.normalize_targets)

        target_encoder = SACObservationEncoder(config, self.normalize_inputs)
        target_critics = [
            CriticHead(input_dim=critic_input_dim, hidden_dims=hidden_dims) for _ in range(2)
        ]
        self.critic_target = CriticEnsemble(target_encoder, target_critics, self.normalize_targets)

        # Initialize grasp critic
        self.grasp_critic = GraspCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            hidden_dims=[config.critic_hidden_dim, config.critic_hidden_dim],
            output_dim=3,
        )
        self.grasp_critic_target = GraspCritic(
            encoder=target_encoder,
            input_dim=target_encoder.output_dim,
            hidden_dims=[config.critic_hidden_dim, config.critic_hidden_dim],
            output_dim=3,
        )

        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.grasp_critic_target.load_state_dict(self.grasp_critic.state_dict())

        # Training stage and loss weights
        self.training_stage = "offline"
        self.bc_weight = config.bc_weight_offline
        self.q_weight = config.q_weight_offline

        self.reset()

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
            "critic": self.critic.parameters(),
            "grasp_critic": self.grasp_critic.parameters(),
        }

        if self.encoder_actor is not None and not self.config.freeze_base_vla:
            params["encoder_actor"] = self.encoder_actor.parameters()

        return params

    def reset(self):
        """Reset the policy state"""
        self._queues = {
            ACTION: deque(maxlen=1),  # ConRFT doesn't use action chunking
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions - not used in ConRFT"""
        raise NotImplementedError("ConRFT does not support action chunking")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        self.eval()

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Generate action using consistency policy
        action, _ = self.consistency_policy(batch, training=False)

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
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

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
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters(), strict=False):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
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

    def update_offline(self, batch, optimizers):
        """Perform one offline update step."""
        # Update critic with Cal-QL loss
        critic_info = self.forward(batch, model="cal_ql")
        critic_loss = critic_info["loss_critic"]

        optimizers["critic"].zero_grad()
        critic_loss.backward()
        optimizers["critic"].step()

        # Update policy with combined BC and Q loss
        actor_info = self.forward(batch, model="actor")
        actor_loss = actor_info["loss_actor"]

        optimizers["consistency_policy"].zero_grad()
        actor_loss.backward()
        optimizers["consistency_policy"].step()

        self.update_target_networks()

        return {**critic_info, **actor_info}

    def update_online(self, batch, optimizers):
        """Perform one online update step."""
        # Update critic with standard TD loss
        critic_info = self.forward(batch, model="critic")
        critic_loss = critic_info["loss_critic"]

        optimizers["critic"].zero_grad()
        critic_loss.backward()
        optimizers["critic"].step()

        # Update policy with combined BC and Q loss
        actor_info = self.forward(batch, model="actor")
        actor_loss = actor_info["loss_actor"]

        optimizers["consistency_policy"].zero_grad()
        actor_loss.backward()
        optimizers["consistency_policy"].step()

        self.update_target_networks()

        return {**critic_info, **actor_info}


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
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

    def __init__(self, encoder: OctoPolicy):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        observations: dict[str, Tensor],
        tasks: Optional[dict[str, Tensor]] = None,
        action_embeddings: Optional[Tensor] = None,
    ):
        return self.encoder.get_action_embeddings(observations, tasks, action_embeddings)


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

        # Network forward pass
        network_input = torch.cat([c_in * x_t, t_embed, state_encoding], dim=1)
        network_output = self.network(network_input)

        # Apply boundary condition
        denoised = c_out * network_output + c_skip * x_t

        if self.clip_denoised and not training:
            denoised = torch.clamp(denoised, -1, 1)

        return denoised, action_embeddings
