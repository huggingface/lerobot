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

The reward model must be preprocessed before it can compute rewards. Each reward model
type (SARM, etc.) may have different preprocessing requirements:
- SARM: Requires CLIP-encoded image/text features (512-dim vectors)
"""

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class RABCWeightComputer:
    """
    Computes RA-BC weights for training batches using a pre-trained reward model.

    Uses Welford's online algorithm for numerically stable running statistics
    and applies soft weighting based on progress values.

    Args:
        reward_model: Pre-trained reward model (e.g., SARM)
        preprocessor: Callable that preprocesses raw batch into encoded features.
                      For SARM, use make_sarm_pre_post_processors() to create this.
        image_key: Key for image data in batch (e.g., "observation.images.top")
        state_key: Key for state data in batch (e.g., "observation.state")
        kappa: Hard threshold for high-quality samples (default: 0.01)
        epsilon: Small constant for numerical stability (default: 1e-6)
        device: Device to run reward model on
        head_mode: For dual-head models (SARM), which head to use for rewards.
                   Options: "sparse", "dense", or None (uses model's default).
    """

    def __init__(
        self,
        reward_model: nn.Module,
        preprocessor: Callable | None = None,
        image_key: str = "observation.images.top",
        state_key: str = "observation.state",
        kappa: float = 0.01,
        epsilon: float = 1e-6,
        device: torch.device = None,
        head_mode: str | None = None,
    ):
        self.reward_model = reward_model
        self.reward_model.eval()  # Always in eval mode
        self.preprocessor = preprocessor
        self.image_key = image_key
        self.state_key = state_key
        self.kappa = kappa
        self.epsilon = epsilon
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_mode = head_mode  # "sparse", "dense", or None (use model default)

        # Running statistics 
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0

        logging.info(
            f"RA-BC WeightComputer initialized {"with preprocessor" if preprocessor else "without preprocessor"}, "
            f"kappa={kappa}, epsilon={epsilon}, head_mode={"{head_mode}" if head_mode else ""}"
        )

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
        """Compute RA-BC weights from progress deltas.
        
        Following paper Eq. 8-9:
        - Soft weight: ˜wi = clip((ri − (µ − 2σ)) / (4σ + ε), 0, 1)
        - Final weight: wi = 1{ri > κ} + 1{0 ≤ ri ≤ κ}˜wi
        
        This means:
        - ri > κ: weight = 1 (clearly good progress)
        - 0 ≤ ri ≤ κ: weight = soft_weight (moderate progress)
        - ri < 0: weight = 0 (negative progress = bad)
        """
        if self.count < 2:
            # Not enough data, use uniform weights
            return torch.ones_like(deltas)

        # Get running statistics
        mean = max(self.mean, 0.0)  # Clamp mean to non-negative
        variance = self.m2 / (self.count - 1)
        std = max(np.sqrt(variance), self.epsilon)

        # Compute soft weights: ˜wi = clip((ri − (µ − 2σ)) / (4σ + ε), 0, 1)
        lower_bound = mean - 2 * std
        soft_weights = (deltas - lower_bound) / (4 * std + self.epsilon)
        soft_weights = torch.clamp(soft_weights, 0.0, 1.0)

        # Apply paper's Eq. 9: wi = 1{ri > κ} + 1{0 ≤ ri ≤ κ}˜wi
        # This means:
        # - If ri > kappa: weight = 1
        # - If 0 <= ri <= kappa: weight = soft_weight
        # - If ri < 0: weight = 0
        weights = torch.zeros_like(deltas)
        
        # High quality: ri > kappa → weight = 1
        high_quality_mask = deltas > self.kappa
        weights = torch.where(high_quality_mask, torch.ones_like(weights), weights)
        
        # Moderate quality: 0 <= ri <= kappa → weight = soft_weight
        moderate_mask = (deltas >= 0) & (deltas <= self.kappa)
        weights = torch.where(moderate_mask, soft_weights, weights)
        
        # Negative progress: ri < 0 → weight = 0 (already initialized to 0)

        return weights

    @torch.no_grad()
    def compute_batch_weights(self, batch: dict) -> torch.Tensor:
        """
        Compute RA-BC weights for a training batch.

        This function:
        1. Preprocesses raw observations using the provided preprocessor
        2. Computes progress for current and next frames using the reward model
        3. Calculates progress deltas (ri = progress_next - progress_current)
        4. Updates running statistics
        5. Returns normalized weights

        For SARM, the preprocessor handles CLIP encoding of images and text.
        The reward model predicts cumulative progress [0, 1], and we compute
        deltas to measure progress made at each timestep.

        Args:
            batch: Training batch containing raw observations (images, state, task, etc.)

        Returns:
            Weights tensor (batch_size,) normalized to sum to batch_size
        """
        # Determine batch size from available keys
        batch_size = self._get_batch_size(batch)

        if self.preprocessor is None:
            logging.warning("RA-BC: No preprocessor provided, using uniform weights")
            return torch.ones(batch_size, device=self.device)

        # Compute progress deltas for each sample in the batch
        # The preprocessor expects single samples, so we loop over the batch
        all_deltas = []
        for i in range(batch_size):
            try:
                processed = self._preprocess_single_sample(batch, i)

                video_features = processed.get("video_features")
                text_features = processed.get("text_features")
                state_features = processed.get("state_features")

                if video_features is None or text_features is None:
                    all_deltas.append(0.0)
                    continue

                # Move to device
                video_features = video_features.to(self.device)
                text_features = text_features.to(self.device)
                if state_features is not None:
                    state_features = state_features.to(self.device)

                # Compute progress delta for this sample
                # If video_features has temporal dimension (T > 1), compute delta
                # between last and first frame. Otherwise, use progress directly.
                delta = self._compute_progress_delta(text_features, video_features, state_features)
                all_deltas.append(delta)

            except Exception as e:
                logging.debug(f"RA-BC: Preprocessing sample {i} failed ({e}), using zero delta")
                all_deltas.append(0.0)

        deltas = torch.tensor(all_deltas, device=self.device)

        # Update running statistics with deltas
        self._update_stats(deltas)

        # Compute weights from progress deltas
        weights = self._compute_weights(deltas)

        # Normalize weights to sum to batch_size (maintains effective batch size)
        weight_sum = weights.sum() + self.epsilon
        weights = weights * batch_size / weight_sum

        return weights

    def _get_batch_size(self, batch: dict) -> int:
        """Determine batch size from batch dictionary."""
        # Try common keys
        for key in [self.image_key, self.state_key, "action"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, (torch.Tensor, np.ndarray)):
                    return val.shape[0]

        # Try nested observation dict
        obs = batch.get("observation", {})
        for val in obs.values():
            if isinstance(val, (torch.Tensor, np.ndarray)):
                return val.shape[0]

        # Fallback
        return 1

    def _preprocess_single_sample(self, batch: dict, sample_idx: int) -> dict:
        """Preprocess a single sample from the batch using the provided preprocessor.

        Builds the input dict expected by the preprocessor and calls it.
        Following the pattern from sarm_inference_visualization.py.

        Args:
            batch: Full training batch
            sample_idx: Index of the sample to preprocess

        Returns:
            Preprocessed dict with encoded features
        """
        # Build single-sample dict for preprocessor
        # The preprocessor expects keys like image_key, state_key, task, index, episode_index
        preprocess_input = {}

        # Add image data for this sample
        if self.image_key in batch:
            img_data = batch[self.image_key]
            if isinstance(img_data, torch.Tensor):
                # Extract single sample from batch
                if img_data.dim() == 5:  # (B, T, C, H, W)
                    img_data = img_data[sample_idx]  # (T, C, H, W)
                elif img_data.dim() == 4:  # (B, C, H, W)
                    img_data = img_data[sample_idx]  # (C, H, W)
                elif img_data.dim() == 3:  # (C, H, W) - single sample already
                    pass
            preprocess_input[self.image_key] = img_data

        # Add state data for this sample
        if self.state_key in batch:
            state_data = batch[self.state_key]
            if isinstance(state_data, torch.Tensor):
                if state_data.dim() == 3:  # (B, T, D)
                    state_data = state_data[sample_idx]  # (T, D)
                elif state_data.dim() == 2:  # (B, D)
                    state_data = state_data[sample_idx]  # (D,)
                elif state_data.dim() == 1:  # (D,) - single sample already
                    pass
            preprocess_input[self.state_key] = state_data

        # Add task description for this sample
        task = batch.get("task", "")
        if isinstance(task, list):
            task = task[sample_idx] if sample_idx < len(task) else (task[0] if task else "")
        preprocess_input["task"] = task

        # Add frame and episode indices for this sample
        index = batch.get("index", 0)
        if isinstance(index, (torch.Tensor, np.ndarray)):
            index = int(index[sample_idx]) if index.ndim > 0 else int(index)
        elif isinstance(index, list):
            index = index[sample_idx] if sample_idx < len(index) else index[0]
        preprocess_input["index"] = index

        episode_index = batch.get("episode_index", 0)
        if isinstance(episode_index, (torch.Tensor, np.ndarray)):
            episode_index = int(episode_index[sample_idx]) if episode_index.ndim > 0 else int(episode_index)
        elif isinstance(episode_index, list):
            episode_index = episode_index[sample_idx] if sample_idx < len(episode_index) else episode_index[0]
        preprocess_input["episode_index"] = episode_index

        # Call preprocessor
        processed = self.preprocessor(preprocess_input)

        return processed

    def _compute_progress_delta(
        self,
        text_features: torch.Tensor,
        video_features: torch.Tensor,
        state_features: torch.Tensor | None,
    ) -> float:
        """
        Compute progress delta using the reward model.

        Following paper Eq. 6: ri = ϕ(o^{t+Δ}) − ϕ(o^t)
        
        If video_features has temporal dimension (T > 1), we compute:
        - progress_start = reward model prediction for first frame
        - progress_end = reward model prediction for last frame
        - delta = progress_end - progress_start

        If only single frame, we use the progress value directly as a proxy
        (assumes higher progress = better demonstration quality).

        Args:
            text_features: Encoded text features (1, D) or (D,)
            video_features: Encoded video features (T, D) or (1, T, D)
            state_features: Encoded state features or None

        Returns:
            Progress delta (float)
        """
        # Ensure proper dimensions for video features
        if video_features.dim() == 1:  # (D,) -> (1, 1, D)
            video_features = video_features.unsqueeze(0).unsqueeze(0)
        elif video_features.dim() == 2:  # (T, D) -> (1, T, D)
            video_features = video_features.unsqueeze(0)
        # Now video_features is (1, T, D)
        
        T = video_features.shape[1]
        
        if not hasattr(self.reward_model, "calculate_rewards"):
            logging.warning("RA-BC: Reward model doesn't have calculate_rewards, using zero delta")
            return 0.0

        if T > 1:
            # Compute delta: progress(last_frame) - progress(first_frame)
            # Get progress for first frame
            first_frame = video_features[:, :1, :]  # (1, 1, D)
            progress_start = self.reward_model.calculate_rewards(
                text_features.unsqueeze(0) if text_features.dim() == 1 else text_features,
                first_frame,
                state_features[:, :1, :] if state_features is not None and state_features.dim() == 3 else state_features,
                return_all_frames=False,
                head_mode=self.head_mode,
            )
            
            # Get progress for last frame
            last_frame = video_features[:, -1:, :]  # (1, 1, D)
            progress_end = self.reward_model.calculate_rewards(
                text_features.unsqueeze(0) if text_features.dim() == 1 else text_features,
                last_frame,
                state_features[:, -1:, :] if state_features is not None and state_features.dim() == 3 else state_features,
                return_all_frames=False,
                head_mode=self.head_mode,
            )
            
            # Extract scalar values
            progress_start = self._extract_scalar(progress_start)
            progress_end = self._extract_scalar(progress_end)
            
            return progress_end - progress_start
        else:
            # Single frame: use progress directly as proxy for quality
            # Higher progress generally indicates better demonstration
            progress = self.reward_model.calculate_rewards(
                text_features.unsqueeze(0) if text_features.dim() == 1 else text_features,
                video_features,
                state_features,
                return_all_frames=False,
                head_mode=self.head_mode,
            )
            return self._extract_scalar(progress)
    
    def _extract_scalar(self, value) -> float:
        """Extract scalar float from various return types."""
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, np.ndarray):
            value = value.flatten()[0]
        elif isinstance(value, torch.Tensor):
            value = value.flatten()[0].item()
        return float(value)

    def get_stats(self) -> dict:
        """Get current running statistics."""
        std = np.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0
        return {
            "mean": self.mean,
            "std": std,
            "count": self.count,
        }
