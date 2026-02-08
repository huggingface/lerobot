#!/usr/bin/env python3
"""
π*₀.₆ RECAP Policy for LeRobot

RECAP: RL with Experience and Corrections via Advantage-conditioned Policies

This integrates π*₀.₆ with LeRobot's:
- Dataset format (LeRobotDataset)
- Training loop
- Evaluation framework
- Visualization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device,
    infer_dtype,
    _parse_bridge_dataset,
    get_prev_and_future_frames,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_pi_star_recap import PiStarRECAPConfig


class PiStarRECAPPolicy(PreTrainedPolicy):
    """
    π*₀.₆ RECAP Policy
    
    Combines:
    - VLA architecture (VLM + Action Expert)
    - IQL for offline RL
    - Advantage-conditioned policy
    - RECAP data mixing
    """
    
    name = "pi_star_recap"
    config_class = PiStarRECAPConfig
    
    def __init__(
        self,
        config: PiStarRECAPConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.config = config
        
        # Build model components
        self._build_model()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Training step counter
        self.global_step = 0
        
    def _build_model(self):
        """Build π*₀.₆ model components"""
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        
        # VLM Backbone (Gemma or Qwen)
        self.vlm = AutoModel.from_pretrained(
            self.config.vlm_model_name,
            torch_dtype=getattr(torch, self.config.dtype),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.vlm_model_name)
        
        # Vision processor
        self.vision_processor = AutoProcessor.from_pretrained(
            self.config.vlm_model_name
        )
        
        # Get dimensions
        vlm_hidden_size = self.vlm.config.hidden_size
        
        # Q-Networks (IQL)
        self.q_networks = nn.ModuleList([
            QNetwork(vlm_hidden_size, self.config)
            for _ in range(self.config.num_q_networks)
        ])
        
        # Target Q-networks
        self.q_target_networks = nn.ModuleList([
            QNetwork(vlm_hidden_size, self.config)
            for _ in range(self.config.num_q_networks)
        ])
        self._update_target_networks(tau=1.0)  # Hard initialize
        
        # V-Network (IQL)
        self.v_network = VNetwork(vlm_hidden_size, self.config)
        
        # Policy (Advantage-conditioned Flow Matching)
        self.policy = AdvantageConditionedPolicy(vlm_hidden_size, self.config)
        
        # Freeze VLM if specified
        if self.config.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif self.config.freeze_vision_encoder:
            # Only freeze vision components
            if hasattr(self.vlm, 'vision_model'):
                for param in self.vlm.vision_model.parameters():
                    param.requires_grad = False
        
        if self.config.train_expert_only and hasattr(self.vlm, 'language_model'):
            # Freeze language model, only train action expert
            for param in self.vlm.language_model.parameters():
                param.requires_grad = False
    
    def _init_optimizers(self):
        """Initialize separate optimizers for IQL components"""
        # Q-network optimizer
        self.q_optimizer = torch.optim.AdamW(
            self.q_networks.parameters(),
            lr=self.config.q_lr,
            betas=self.config.q_betas,
            weight_decay=self.config.q_weight_decay,
        )
        
        # V-network optimizer
        self.v_optimizer = torch.optim.AdamW(
            self.v_network.parameters(),
            lr=self.config.v_lr,
            betas=self.config.v_betas,
            weight_decay=self.config.v_weight_decay,
        )
        
        # Policy optimizer
        policy_params = self.policy.parameters()
        if not self.config.freeze_vlm and not self.config.train_expert_only:
            # Also train VLM projections if not frozen
            policy_params = list(policy_params) + [
                p for p in self.vlm.parameters() if p.requires_grad
            ]
        
        self.policy_optimizer = torch.optim.AdamW(
            policy_params,
            lr=self.config.policy_lr,
            betas=self.config.policy_betas,
            weight_decay=self.config.policy_weight_decay,
        )
        
        # Schedulers
        self.q_scheduler = self._create_scheduler(self.q_optimizer)
        self.v_scheduler = self._create_scheduler(self.v_optimizer)
        self.policy_scheduler = self._create_scheduler(self.policy_optimizer)
    
    def _create_scheduler(self, optimizer):
        """Create cosine decay scheduler with warmup"""
        from lerobot.optim.schedulers import CosineDecayWithWarmupScheduler
        
        return CosineDecayWithWarmupScheduler(
            optimizer,
            warmup_steps=self.config.scheduler_warmup_steps,
            decay_steps=self.config.scheduler_decay_steps,
            decay_lr=self.config.scheduler_decay_lr,
        )
    
    def _update_target_networks(self, tau: float = None):
        """Soft update target networks"""
        if tau is None:
            tau = self.config.target_update_tau
        
        for q, q_target in zip(self.q_networks, self.q_target_networks):
            for param, target_param in zip(q.parameters(), q_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
    
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            batch: Dictionary containing:
                - OBS_IMAGES: [B, n_obs_steps, C, H, W]
                - OBS_STATE: [B, n_obs_steps, state_dim]
                - ACTION: [B, chunk_size, action_dim]
                - Other metadata including data_type
        Returns:
            dict with losses
        """
        # Encode observations through VLM
        context = self._encode_observations(batch)
        
        # Get actions and rewards
        actions = batch[ACTION]
        rewards = batch.get("rewards", torch.zeros(actions.shape[0], 1, device=actions.device))
        dones = batch.get("dones", torch.zeros(actions.shape[0], 1, device=actions.device))
        
        # Get data type weights for RECAP
        type_weights = self._get_data_type_weights(batch)
        
        # === IQL Losses ===
        
        # 1. Value loss (expectile regression)
        v_loss = self._compute_v_loss(context, actions, type_weights)
        
        # 2. Q-loss (Bellman)
        q_loss = self._compute_q_loss(context, actions, rewards, dones, type_weights)
        
        # 3. Policy loss (advantage-weighted flow matching)
        policy_loss = self._compute_policy_loss(context, actions, type_weights)
        
        return {
            "loss": (
                self.config.v_loss_weight * v_loss +
                self.config.q_loss_weight * q_loss +
                self.config.policy_loss_weight * policy_loss
            ),
            "v_loss": v_loss.detach(),
            "q_loss": q_loss.detach(),
            "policy_loss": policy_loss.detach(),
        }
    
    def _encode_observations(self, batch: dict) -> torch.Tensor:
        """Encode observations through VLM"""
        images = batch[OBS_IMAGES][:, 0]  # Take first obs step
        
        # Process images through vision encoder
        # This is simplified - actual implementation depends on VLM architecture
        pixel_values = self.vision_processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.vlm.device)
        
        # Get VLM embeddings
        with torch.cuda.amp.autocast(enabled=self.config.dtype == "bfloat16"):
            vision_outputs = self.vlm.vision_model(pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            
            # Pool or project to get context
            context = image_embeds.mean(dim=1)  # Simplified pooling
        
        return context
    
    def _get_data_type_weights(self, batch: dict) -> torch.Tensor:
        """Get RECAP data type weights"""
        data_types = batch.get("data_types", ["demo"] * len(batch[ACTION]))
        
        weights = []
        for dtype in data_types:
            if dtype == "demo":
                weights.append(self.config.demo_weight)
            elif dtype == "auto":
                weights.append(self.config.auto_weight)
            elif dtype == "intervention":
                weights.append(self.config.intervention_weight)
            else:
                weights.append(1.0)
        
        return torch.tensor(weights, device=batch[ACTION].device).unsqueeze(1)
    
    def _compute_v_loss(self, context, actions, type_weights):
        """Compute expectile regression loss for V-network"""
        with torch.no_grad():
            # Target Q-value
            q_values = torch.stack([q(context, actions) for q in self.q_target_networks], dim=0)
            q_target = q_values.min(dim=0)[0]  # Conservative estimate
        
        v_value = self.v_network(context)
        
        # Asymmetric squared loss (expectile)
        diff = q_target - v_value
        expectile = self.config.iql_expectile
        v_loss = torch.where(
            diff > 0,
            expectile * diff ** 2,
            (1 - expectile) * diff ** 2,
        )
        v_loss = (v_loss * type_weights).mean()
        
        return v_loss
    
    def _compute_q_loss(self, context, actions, rewards, dones, type_weights):
        """Compute Bellman loss for Q-networks"""
        # Current Q
        q_pred = self.q_networks[0](context, actions)
        
        # Target: r + γ * V(s')
        with torch.no_grad():
            # For simplicity, use same context as next context
            # In practice, should encode next_obs
            next_v = self.v_network(context)
            q_target = rewards + self.config.iql_discount * next_v * (1 - dones)
        
        q_loss = F.mse_loss(q_pred, q_target, reduction="none")
        q_loss = (q_loss * type_weights).mean()
        
        return q_loss
    
    def _compute_policy_loss(self, context, actions, type_weights):
        """Compute advantage-weighted flow matching loss"""
        batch_size = actions.shape[0]
        device = actions.device
        
        # Compute advantage
        with torch.no_grad():
            q_values = torch.stack([q(context, actions) for q in self.q_networks], dim=0)
            q_value = q_values.min(dim=0)[0]
            v_value = self.v_network(context)
            advantage = q_value - v_value
        
        # Advantage weighting (AWR style)
        weights = torch.exp(advantage / self.config.iql_temperature)
        weights = torch.clamp(weights, max=20.0)  # Prevent extreme weights
        
        # Flow matching loss
        timesteps = torch.rand(batch_size, 1, device=device)
        noise = torch.randn_like(actions)
        noisy_actions = timesteps.unsqueeze(-1) * actions + (1 - timesteps.unsqueeze(-1)) * noise
        
        # True velocity
        true_velocity = (actions - noisy_actions) / (1 - timesteps.unsqueeze(-1) + 1e-8)
        
        # Predict velocity with advantage conditioning
        if self.config.use_advantage_conditioning:
            pred_velocity = self.policy(noisy_actions, timesteps, context, advantage)
        else:
            pred_velocity = self.policy(noisy_actions, timesteps, context, None)
        
        # Weighted MSE
        flow_loss = F.mse_loss(pred_velocity, true_velocity, reduction="none")
        flow_loss = (flow_loss * weights.unsqueeze(-1).unsqueeze(-1) * type_weights.unsqueeze(-1)).mean()
        
        return flow_loss
    
    def update(self):
        """Update optimizers and schedulers"""
        # Gradient clipping for policy
        if self.config.policy_grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.policy_grad_clip_norm,
            )
        
        # Optimizer steps
        self.q_optimizer.step()
        self.v_optimizer.step()
        self.policy_optimizer.step()
        
        # Scheduler steps
        self.q_scheduler.step()
        self.v_scheduler.step()
        self.policy_scheduler.step()
        
        # Zero gradients
        self.q_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        
        # Update target networks
        self.global_step += 1
        if self.global_step % self.config.target_update_period == 0:
            self._update_target_networks()
    
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action for inference
        
        Args:
            batch: Observation batch
        Returns:
            action: [B, action_dim]
        """
        self.eval()
        
        with torch.no_grad():
            # Encode observations
            context = self._encode_observations(batch)
            
            # Sample actions using flow matching
            batch_size = context.shape[0]
            device = context.device
            
            # Start from noise
            actions = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
            )
            
            # ODE solver (Euler)
            num_steps = self.config.num_inference_steps
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t = torch.ones(batch_size, 1, device=device) * i * dt
                
                # Compute advantage for conditioning
                if self.config.use_advantage_conditioning:
                    with torch.no_grad():
                        q_values = torch.stack([q(context, actions) for q in self.q_networks], dim=0)
                        q_value = q_values.min(dim=0)[0]
                        v_value = self.v_network(context)
                        advantage = (q_value - v_value) * self.config.eval_advantage_scale
                else:
                    advantage = None
                
                velocity = self.policy(actions, t, context, advantage)
                actions = actions + dt * velocity
        
        return actions[:, 0]  # Return first action
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        return {
            "q_lr": self.q_optimizer.param_groups[0]["lr"],
            "v_lr": self.v_optimizer.param_groups[0]["lr"],
            "policy_lr": self.policy_optimizer.param_groups[0]["lr"],
            "global_step": self.global_step,
        }


class QNetwork(nn.Module):
    """Q-network for IQL"""
    
    def __init__(self, input_dim: int, config: PiStarRECAPConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + config.max_action_dim * config.chunk_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Flatten actions
        actions_flat = actions.view(actions.shape[0], -1)
        x = torch.cat([context, actions_flat], dim=-1)
        return self.net(x)


class VNetwork(nn.Module):
    """V-network for IQL"""
    
    def __init__(self, input_dim: int, config: PiStarRECAPConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class AdvantageConditionedPolicy(nn.Module):
    """Advantage-conditioned flow matching policy"""
    
    def __init__(self, input_dim: int, config: PiStarRECAPConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.action_embed = nn.Linear(config.max_action_dim, 256)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.context_embed = nn.Linear(input_dim, 256)
        
        if config.use_advantage_conditioning:
            self.advantage_embed = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
            )
        
        # Transformer
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True,
            ),
            num_layers=4,
        )
        
        # Output head
        self.velocity_head = nn.Linear(256, config.max_action_dim)
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        advantage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Embed
        action_emb = self.action_embed(noisy_actions)
        time_emb = self.time_embed(timestep).unsqueeze(1).expand(-1, noisy_actions.shape[1], -1)
        context_emb = self.context_embed(context).unsqueeze(1)
        
        features = action_emb + time_emb
        
        if advantage is not None and self.config.use_advantage_conditioning:
            adv_emb = self.advantage_embed(advantage).unsqueeze(1).expand(-1, noisy_actions.shape[1], -1)
            features = features + adv_emb
        
        # Transform
        features = self.transformer(features, context_emb.expand(-1, features.shape[1], -1))
        
        # Predict velocity
        velocity = self.velocity_head(features)
        return velocity
