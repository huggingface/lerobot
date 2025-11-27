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
Reward-Aligned Behavior Cloning (RA-BC) utilities.

RA-BC uses a pre-trained reward model (e.g., SARM) to compute progress-based weights
for training samples, emphasizing high-quality demonstrations and down-weighting
suboptimal ones.
"""

import logging
import torch
import torch.nn as nn


class RABCWeightComputer:
    """
    Computes RA-BC weights for training batches using a pre-trained reward model.
    
    Uses Welford's online algorithm for numerically stable running statistics
    and applies soft weighting based on progress deltas.
    
    Args:
        reward_model: Pre-trained reward model (e.g., SARM)
        kappa: Hard threshold for high-quality samples (default: 0.01)
        epsilon: Small constant for numerical stability (default: 1e-6)
        device: Device to run reward model on
    """
    
    def __init__(
        self,
        reward_model: nn.Module,
        kappa: float = 0.01,
        epsilon: float = 1e-6,
        device: torch.device = None,
    ):
        self.reward_model = reward_model
        self.reward_model.eval()  # Always in eval mode
        self.kappa = kappa
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Running statistics (Welford's algorithm)
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0
        
        logging.info(f"RA-BC WeightComputer initialized with kappa={kappa}, epsilon={epsilon}")
    
    def _update_stats(self, deltas: torch.Tensor):
        """Update running statistics using Welford's online algorithm."""
        for delta in deltas:
            self.count += 1
            delta_val = delta.item()
            delta_mean = delta_val - self.mean
            self.mean += delta_mean / self.count
            delta_m2 = delta_val - self.mean
            self.m2 += delta_mean * delta_m2
    
    def _compute_weights(self, deltas: torch.Tensor) -> torch.Tensor:
        """Compute RA-BC weights from progress deltas."""
        if self.count < 2:
            # Not enough data, use uniform weights
            return torch.ones_like(deltas)
        
        # Get running statistics
        mean = max(self.mean, 0.0)  # Clamp mean to non-negative
        variance = self.m2 / (self.count - 1)
        std = torch.tensor(variance).sqrt().item()
        
        # Compute soft weights
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        weights = (deltas - lower_bound) / (4 * std + self.epsilon)
        weights = torch.clamp(weights, 0.0, 1.0)
        
        # Apply hard threshold
        high_quality_mask = deltas > self.kappa
        weights = torch.where(high_quality_mask, torch.ones_like(weights), weights)
        
        return weights
    
    @torch.no_grad()
    def compute_batch_weights(self, batch: dict, chunk_size: int = 1) -> torch.Tensor:
        """
        Compute RA-BC weights for a training batch.
        
        This function:
        1. Extracts current and next observations from the batch
        2. Computes rewards using the reward model
        3. Calculates progress deltas
        4. Updates running statistics
        5. Returns normalized weights
        
        Args:
            batch: Training batch containing observations and next_observation
            chunk_size: Size of action chunks for computing deltas (default: 1)
            
        Returns:
            Weights tensor (batch_size,) normalized to sum to batch_size
        """
        observation = batch.get('observation', batch)
        batch_size = next(iter(observation.values())).shape[0]
        
        # Extract features needed for reward computation
        # These should already be encoded by the preprocessor
        if 'video_features' not in observation or 'text_features' not in observation:
            logging.warning("RA-BC: Missing video/text features, using uniform weights")
            return torch.ones(batch_size, device=self.device)
        
        video_cur = observation['video_features'].to(self.device)
        text_cur = observation['text_features'].to(self.device)
        state_cur = observation.get('state_features', None)
        if state_cur is not None:
            state_cur = state_cur.to(self.device)
        
        # Get next observation
        next_obs = batch.get('next_observation') or batch.get('next_state')
        
        video_next = next_obs['video_features'].to(self.device)
        text_next = next_obs.get('text_features', text_cur).to(self.device)
        state_next = next_obs.get('state_features')
        if state_next is not None:
            state_next = state_next.to(self.device)
        
        # Compute rewards for current and next temporal windows
        rewards_cur = self._compute_rewards(text_cur, video_cur, state_cur)
        rewards_next = self._compute_rewards(text_next, video_next, state_next)
        
        # Calculate progress deltas
        progress_deltas = rewards_next - rewards_cur
        
        self._update_stats(progress_deltas)
        
        # Compute weights
        weights = self._compute_weights(progress_deltas)
        
        # Normalize weights to sum to batch_size (maintains effective batch size)
        weight_sum = weights.sum() + self.epsilon
        weights = weights * batch_size / weight_sum
        
        return weights
    
    def _compute_rewards(
        self,
        text_features: torch.Tensor,
        video_features: torch.Tensor,
        state_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute rewards for video features.
        
        Handles both single-frame and multi-frame video features:
        - Single-frame: (B, D) - adds temporal dimension
        - Multi-frame: (B, T, D) - uses as-is
        
        Args:
            text_features: Text embeddings (B, D_text)
            video_features: Video embeddings, either (B, D) or (B, T, D)
            state_features: Optional state embeddings (B, T, D_state) or None
            
        Returns:
            Rewards tensor (B,)
        """
        batch_size = video_features.shape[0]
        
        # Handle both single-frame and multi-frame features
        if video_features.dim() == 3:  # (B, T, D)
            # Multi-frame: use all frames
            if hasattr(self.reward_model, 'calculate_rewards'):
                rewards = self.reward_model.calculate_rewards(
                    text_features, video_features, state_features,
                    return_all_frames=False
                )
            else:
                # Fallback for models without calculate_rewards
                rewards = torch.zeros(batch_size, device=self.device)
        else:  # (B, D)
            # Single frame: add temporal dimension
            if hasattr(self.reward_model, 'calculate_rewards'):
                rewards = self.reward_model.calculate_rewards(
                    text_features, video_features.unsqueeze(1), state_features,
                    return_all_frames=False
                )
            else:
                rewards = torch.zeros(batch_size, device=self.device)
        

        if isinstance(rewards, tuple):
            rewards = rewards[0]
        
        # Ensure tensor format
        rewards = torch.tensor(rewards, device=self.device) if isinstance(rewards, (list, tuple)) else rewards
        
        return rewards
    
    def get_stats(self) -> dict:
        """Get current running statistics."""
        std = torch.tensor(self.m2 / (self.count - 1)).sqrt().item() if self.count > 1 else 0.0
        return {
            'mean': self.mean,
            'std': std,
            'count': self.count,
        }
