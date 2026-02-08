#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
QC-FQL (Q-Chunking with Fitted Q-Learning) Policy Implementation.

This implements the QC-FQL algorithm from "Reinforcement Learning with Action Chunking"
(Li et al., 2025). QC-FQL combines action chunking with flow-based policies and
optimal transport for effective offline-to-online RL.

Key components:
1. FlowMatchingBehaviorPolicy: Captures behavior distribution using flow matching
2. ChunkedCritic: Q-function operating on action chunks
3. NoiseConditionedPolicy: Policy network with noise conditioning
4. DistillationLoss: Wasserstein distance upper bound between policy and behavior

Algorithm Overview:
------------------
1. Train behavior policy π_b using flow matching on offline data
2. Train critic Q(s, a_chunk) using TD learning with n-step backups (n = chunk_size)
3. Train policy π_θ using:
   - Q-maximization: max E[Q(s, π_θ(s, noise))]
   - Distillation constraint: min W_2(π_θ, π_b) via upper bound
"""

import math
from typing import Callable

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.qc_fql.configuration_qc_fql import QCFQLConfig


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None = None,
        activation: Callable = nn.SiLU(),
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Don't add activation after last hidden layer if we have output layer
            if i < len(hidden_dims) - 1 or output_dim is not None:
                layers.append(activation)
            
            in_dim = hidden_dim
        
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FlowMatchingBehaviorPolicy(nn.Module):
    """
    Behavior policy using flow matching to capture complex behavior distributions.
    
    Flow matching learns a velocity field v_t(a) such that following the ODE:
        da/dt = v_t(a)
    transforms noise a_0 ~ N(0, I) to data samples a_1 ~ π_b(·|s).
    
    The velocity field is parameterized by a neural network that takes:
    - Current action a_t
    - Time t ∈ [0, 1]
    - State s
    
    and predicts the velocity v_t(a_t; s).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dims: list[int],
        sigma: float = 0.001,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.sigma = sigma
        
        # Total action dimension for chunk
        total_action_dim = action_dim * action_chunk_size
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        
        # Velocity field network
        # Input: [state, action_chunk, time_embedding]
        input_dim = state_dim + total_action_dim + 128
        self.velocity_net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=total_action_dim,
        )
    
    def forward(self, state: Tensor, action_chunk: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field v_t(a_t; s).
        
        Args:
            state: (batch, state_dim)
            action_chunk: (batch, action_chunk_size, action_dim) or (batch, total_action_dim)
            t: (batch,) time in [0, 1]
            
        Returns:
            velocity: (batch, total_action_dim)
        """
        batch_size = state.shape[0]
        
        # Flatten action chunk if needed
        if action_chunk.dim() == 3:
            action_chunk = action_chunk.reshape(batch_size, -1)
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # Concatenate inputs
        x = torch.cat([state, action_chunk, t_emb], dim=-1)
        
        # Predict velocity
        velocity = self.velocity_net(x)
        return velocity
    
    def compute_loss(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        """
        Compute flow matching loss (conditional flow matching).
        
        The loss is: E_t ||v_t(a_t) - (a_1 - a_0)||^2
        where a_t = (1-t) * a_0 + t * a_1
        
        Args:
            state: (batch, state_dim)
            action_chunk: (batch, action_chunk_size, action_dim) target actions
            
        Returns:
            loss: scalar
        """
        batch_size = state.shape[0]
        
        # Flatten action chunk
        if action_chunk.dim() == 3:
            action_chunk = action_chunk.reshape(batch_size, -1)
        
        # Sample noise
        a_0 = torch.randn_like(action_chunk) * self.sigma
        a_1 = action_chunk
        
        # Sample time
        t = torch.rand(batch_size, device=state.device)
        
        # Interpolate
        t_expanded = t.unsqueeze(-1)
        a_t = (1 - t_expanded) * a_0 + t_expanded * a_1
        
        # Target velocity
        target_velocity = a_1 - a_0
        
        # Predict velocity
        pred_velocity = self.forward(state, a_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        state: Tensor,
        num_steps: int = 10,
        num_samples: int = 1,
    ) -> Tensor:
        """
        Sample action chunks using Euler integration.
        
        Args:
            state: (batch, state_dim) or (batch,)
            num_steps: Number of integration steps
            num_samples: Number of action samples per state
            
        Returns:
            actions: (batch, num_samples, action_chunk_size, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        total_action_dim = self.action_dim * self.action_chunk_size
        
        # Expand state for multiple samples
        if num_samples > 1:
            state = einops.repeat(state, 'b d -> (b n) d', n=num_samples)
        
        # Initialize from noise
        a_t = torch.randn(batch_size * num_samples, total_action_dim, device=state.device)
        
        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(batch_size * num_samples, device=state.device) * (i * dt)
            velocity = self.forward(state, a_t, t)
            a_t = a_t + dt * velocity
        
        # Reshape to (batch, num_samples, chunk_size, action_dim)
        actions = a_t.reshape(batch_size, num_samples, self.action_chunk_size, self.action_dim)
        return actions


class ChunkedCritic(nn.Module):
    """
    Critic operating on action chunks.
    
    Q(s, a_chunk) estimates the value of executing the entire action chunk
    starting from state s. This enables unbiased n-step backups where n equals
    the chunk size.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        total_action_dim = action_dim * action_chunk_size
        input_dim = state_dim + total_action_dim
        
        self.q_net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
        )
    
    def forward(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        """
        Compute Q-value for state-action chunk pair.
        
        Args:
            state: (batch, state_dim)
            action_chunk: (batch, action_chunk_size, action_dim) or (batch, total_action_dim)
            
        Returns:
            q_value: (batch, 1)
        """
        if action_chunk.dim() == 3:
            action_chunk = action_chunk.reshape(action_chunk.shape[0], -1)
        
        x = torch.cat([state, action_chunk], dim=-1)
        return self.q_net(x)


class CriticEnsemble(nn.Module):
    """Ensemble of ChunkedCritics for conservative estimation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dims: list[int],
        num_critics: int,
    ):
        super().__init__()
        self.critics = nn.ModuleList([
            ChunkedCritic(state_dim, action_dim, action_chunk_size, hidden_dims)
            for _ in range(num_critics)
        ])
        self.num_critics = num_critics
    
    def forward(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        """
        Compute Q-values from all critics.
        
        Returns:
            q_values: (num_critics, batch, 1)
        """
        q_values = torch.stack([critic(state, action_chunk) for critic in self.critics], dim=0)
        return q_values
    
    def get_min_q(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        """Get minimum Q-value across ensemble (for target computation)."""
        q_values = self.forward(state, action_chunk)
        return q_values.min(dim=0)[0]


class NoiseConditionedPolicy(nn.Module):
 """
    Noise-conditioned action prediction model.
    
    This is the policy network that predicts action chunks given state and noise.
    It uses the same architecture as the behavior policy but is trained to maximize
    Q-values while staying close to the behavior policy (via distillation).
    
    The policy takes noise ε ~ N(0, I) and state s, and outputs an action chunk:
        a = π_θ(s, ε)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        
        total_action_dim = action_dim * action_chunk_size
        
        # Input: [state, noise]
        input_dim = state_dim + total_action_dim
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=total_action_dim,
        )
    
    def forward(self, state: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        Predict action chunk from state and noise.
        
        Args:
            state: (batch, state_dim)
            noise: (batch, total_action_dim) or None (will be sampled)
            
        Returns:
            action_chunk: (batch, action_chunk_size, action_dim)
        """
        batch_size = state.shape[0]
        total_action_dim = self.action_dim * self.action_chunk_size
        
        if noise is None:
            noise = torch.randn(batch_size, total_action_dim, device=state.device)
        
        x = torch.cat([state, noise], dim=-1)
        action_chunk = self.net(x)
        
        # Reshape to (batch, chunk_size, action_dim)
        action_chunk = action_chunk.reshape(batch_size, self.action_chunk_size, self.action_dim)
        return action_chunk


class QCFQLPolicy(PreTrainedPolicy):
    """
    QC-FQL Policy: Q-Chunking with Fitted Q-Learning.
    
    This policy implements the QC-FQL algorithm which combines:
    1. Action chunking for temporally coherent exploration
    2. Flow matching for behavior policy
    3. Noise-conditioned policy for Q-maximization
    4. Distillation loss for Wasserstein constraint
    
    The training involves three loss terms:
    - L_behavior: Flow matching loss for behavior policy
    - L_critic: TD loss on action chunks (unbiased n-step backup)
    - L_actor: Q-maximization + distillation to behavior policy
    
    Reference: Li et al., "Reinforcement Learning with Action Chunking", 2025
    """
    
    config_class = QCFQLConfig
    name = "qc_fql"
    
    def __init__(self, config: QCFQLConfig):
        super().__init__(config)
        self.config = config
        
        # Infer dimensions from features if not provided
        if config.state_dim is None or config.action_dim is None:
            config.state_dim, config.action_dim = self._infer_dimensions()
        
        # Initialize networks
        self._init_behavior_policy()
        self._init_critics()
        self._init_policy()
        
        # Training state
        self.total_steps = 0
        self.pretrain_phase = True
    
    def _infer_dimensions(self) -> tuple[int, int]:
        """Infer state and action dimensions from config features."""
        # This would be implemented based on how features are specified
        # For now, return dummy values
        return 128, 16  # (state_dim, action_dim)
    
    def _init_behavior_policy(self):
        """Initialize flow-matching behavior policy."""
        self.behavior_policy = FlowMatchingBehaviorPolicy(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            action_chunk_size=self.config.action_chunk_size,
            hidden_dims=self.config.flow_matching_hidden_dims,
            sigma=self.config.flow_matching_sigma,
        )
    
    def _init_critics(self):
        """Initialize critic ensemble and targets."""
        self.critic_ensemble = CriticEnsemble(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            action_chunk_size=self.config.action_chunk_size,
            hidden_dims=self.config.critic_hidden_dims,
            num_critics=self.config.num_critics,
        )
        
        # Target network
        self.critic_target = CriticEnsemble(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            action_chunk_size=self.config.action_chunk_size,
            hidden_dims=self.config.critic_hidden_dims,
            num_critics=self.config.num_critics,
        )
        
        # Copy weights
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())
        
        # Freeze target
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def _init_policy(self):
        """Initialize noise-conditioned policy."""
        self.policy = NoiseConditionedPolicy(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            action_chunk_size=self.config.action_chunk_size,
            hidden_dims=self.config.actor_hidden_dims,
        )
    
    def get_optim_params(self) -> dict:
        """Return parameters for optimization."""
        return {
            "behavior_policy": self.behavior_policy.parameters(),
            "critic": self.critic_ensemble.parameters(),
            "policy": self.policy.parameters(),
        }
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select action for inference.
        
        Returns the first action from the predicted action chunk.
        """
        state = batch["observation.state"]
        
        # Predict action chunk
        action_chunk = self.policy(state)
        
        # Return first action
        return action_chunk[:, 0, :]
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Predict full action chunk.
        
        Returns:
            action_chunk: (batch, action_chunk_size, action_dim)
        """
        state = batch["observation.state"]
        return self.policy(state)
    
    def forward(self, batch: dict[str, Tensor], model: str = "all") -> dict[str, Tensor]:
        """
        Compute losses for training.
        
        Args:
            batch: Dictionary containing:
                - observation.state: (batch, state_dim)
                - action: (batch, action_chunk_size, action_dim)
                - reward: (batch,)
                - next_observation.state: (batch, state_dim)
                - done: (batch,)
            model: Which model to compute loss for ('behavior', 'critic', 'policy', 'all')
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        if model in ["behavior", "all"]:
            losses["loss_behavior"] = self.compute_loss_behavior(batch)
        
        if model in ["critic", "all"]:
            losses["loss_critic"] = self.compute_loss_critic(batch)
        
        if model in ["policy", "all"]:
            losses["loss_policy"] = self.compute_loss_policy(batch)
        
        return losses
    
    def compute_loss_behavior(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Compute flow matching loss for behavior policy.
        
        L_behavior = E[||v_t(a_t) - (a_1 - a_0)||^2]
        """
        state = batch["observation.state"]
        action_chunk = batch["action"]
        return self.behavior_policy.compute_loss(state, action_chunk)
    
    def compute_loss_critic(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Compute TD loss for critic ensemble.
        
        Uses n-step backup where n = action_chunk_size.
        L_critic = E[(Q(s, a) - (r + γ * min Q'(s', π(s'))))^2]
        """
        state = batch["observation.state"]
        action_chunk = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_observation.state"]
        done = batch["done"]
        
        # Current Q-values
        q_values = self.critic_ensemble(state, action_chunk)  # (num_critics, batch, 1)
        
        # Target Q-value
        with torch.no_grad():
            # Sample action from policy for next state
            next_action_chunk = self.policy(next_state)
            
            # Min Q-value from target ensemble
            next_q = self.critic_target.get_min_q(next_state, next_action_chunk)
            
            # TD target (n-step backup where n = chunk_size)
            # In practice, reward should be the sum of n-step rewards
            target_q = reward + (1 - done) * (self.config.discount ** self.config.action_chunk_size) * next_q
        
        # TD loss for each critic
        target_q = einops.repeat(target_q.squeeze(-1), 'b -> c b', c=q_values.shape[0])
        loss = F.mse_loss(q_values.squeeze(-1), target_q)
        
        return loss
    
    def compute_loss_policy(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Compute policy loss: Q-maximization + distillation.
        
        L_policy = -E[Q(s, π(s, ε))] + λ * distillation(π, π_b)
        
        The distillation loss is an upper bound on the 2-Wasserstein distance.
        """
        state = batch["observation.state"]
        batch_size = state.shape[0]
        
        # Sample noise
        total_action_dim = self.config.action_dim * self.config.action_chunk_size
        noise = torch.randn(batch_size, total_action_dim, device=state.device)
        
        # Predict action chunk
        action_chunk = self.policy(state, noise)
        
        # Q-maximization (use mean of ensemble for policy gradient)
        q_values = self.critic_ensemble(state, action_chunk)
        q_mean = q_values.mean(dim=0).squeeze(-1)
        loss_q = -q_mean.mean()
        
        # Distillation loss (Wasserstein upper bound)
        # Sample from behavior policy using flow matching
        with torch.no_grad():
            behavior_action = self.behavior_policy.sample(
                state,
                num_steps=self.config.flow_matching_num_inference_steps,
                num_samples=1,
            ).squeeze(1)  # (batch, chunk_size, action_dim)
        
        # Distillation: match policy output to behavior samples
        loss_distill = F.mse_loss(action_chunk, behavior_action)
        
        # Combined loss
        loss = loss_q + self.config.distillation_weight * loss_distill
        
        return loss
    
    def update_target_networks(self):
        """Update target networks using exponential moving average."""
        tau = self.config.critic_target_update_tau
        for target_param, param in zip(self.critic_target.parameters(), self.critic_ensemble.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def reset(self):
        """Reset policy state."""
        self.total_steps = 0
        self.pretrain_phase = True
