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
Real-Time Chunking (RTC) implementation for LeRobot.

Based on Physical Intelligence's Kinetix implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214
"""

from typing import Any

import torch
from torch import Tensor


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(
        self,
        chunk_size: int,
        soft_mask_length: int = 5,
        beta: float = 1.0,
        prefix_attention_schedule: str = "linear",
        device: torch.device | None = None,
    ):
        """Initialize RTC processor.

        Args:
            chunk_size: Size of action chunks
            soft_mask_length: Number of actions to soft mask in overlap regions
            beta: Maximum guidance weight for prefix attention
            prefix_attention_schedule: Schedule for prefix attention weights ("linear", "exp", "constant")
            device: PyTorch device for computations
        """
        self.chunk_size = chunk_size
        self.soft_mask_length = soft_mask_length
        self.beta = beta
        self.prefix_attention_schedule = prefix_attention_schedule
        self.device = device or torch.device("cpu")

        # Cache for previous chunk
        self.previous_chunk: Tensor | None = None
        self.chunk_step = 0

    def reset(self):
        """Reset the RTC processor state."""
        self.previous_chunk = None
        self.chunk_step = 0

    def calculate_velocity(self, actions: Tensor, dt: float = 0.1, method: str = "finite_diff") -> Tensor:
        """Calculate velocity from action sequences.

        Args:
            actions: Action tensor of shape (batch_size, sequence_length, action_dim)
            dt: Time step between actions in seconds
            method: Velocity calculation method ("finite_diff", "central_diff")

        Returns:
            Velocity tensor of shape (batch_size, sequence_length-1, action_dim)
        """
        if actions.dim() != 3:
            raise ValueError(f"Expected 3D actions tensor, got {actions.dim()}D")

        if actions.size(1) < 2:
            raise ValueError(f"Need at least 2 timesteps, got {actions.size(1)}")

        actions = actions.to(self.device)

        if method == "finite_diff":
            velocities = (actions[:, 1:, :] - actions[:, :-1, :]) / dt
        elif method == "central_diff":
            if actions.size(1) < 3:
                raise ValueError("Central difference requires at least 3 timesteps")
            velocities = (actions[:, 2:, :] - actions[:, :-2, :]) / (2 * dt)
        else:
            raise ValueError(f"Unknown velocity calculation method: {method}")

        return velocities

    def get_prefix_attention_weights(self, length: int) -> Tensor:
        """Generate prefix attention weights based on schedule.

        Args:
            length: Number of timesteps to generate weights for

        Returns:
            Tensor of attention weights
        """
        if self.prefix_attention_schedule == "linear":
            # Linearly decreasing weights
            weights = torch.linspace(self.beta, 0, length, device=self.device)
        elif self.prefix_attention_schedule == "exp":
            # Exponentially decreasing weights
            x = torch.linspace(0, 1, length, device=self.device)
            weights = self.beta * torch.exp(-5 * x)  # -5 controls decay rate
        elif self.prefix_attention_schedule == "constant":
            # Uniform weights
            weights = torch.full((length,), self.beta / length, device=self.device)
        else:
            raise ValueError(f"Unknown prefix attention schedule: {self.prefix_attention_schedule}")

        return weights

    def apply_rtc_masking(self, current_chunk: Tensor, overlap_length: int | None = None) -> Tensor:
        """Apply RTC soft masking to action chunk with previous chunk overlap.

        Args:
            current_chunk: Current action chunk (batch_size, chunk_size, action_dim)
            overlap_length: Number of overlapping timesteps (defaults to soft_mask_length)

        Returns:
            Masked action chunk with prefix attention applied
        """
        if self.previous_chunk is None:
            # No previous chunk, return current as-is
            self.previous_chunk = current_chunk.clone()
            return current_chunk

        overlap_length = overlap_length or self.soft_mask_length
        overlap_length = min(overlap_length, current_chunk.size(1), self.previous_chunk.size(1))

        if overlap_length <= 0:
            self.previous_chunk = current_chunk.clone()
            return current_chunk

        # Get prefix attention weights
        weights = self.get_prefix_attention_weights(overlap_length)
        weights = weights.view(1, -1, 1)  # Shape for broadcasting

        # Apply weighted combination in overlap region
        masked_chunk = current_chunk.clone()
        masked_chunk[:, :overlap_length] = (
            weights * self.previous_chunk[:, -overlap_length:]
            + (1 - weights) * current_chunk[:, :overlap_length]
        )

        # Update previous chunk cache
        self.previous_chunk = masked_chunk.clone()

        return masked_chunk

    def compute_velocity_penalty(
        self, actions: Tensor, dt: float = 0.1, target_smoothness: float = 1.0
    ) -> Tensor:
        """Compute velocity-based penalty for action smoothness.

        Args:
            actions: Action sequence tensor
            dt: Time step between actions
            target_smoothness: Target smoothness level (lower is smoother)

        Returns:
            Scalar penalty value
        """
        velocities = self.calculate_velocity(actions, dt)

        # Calculate velocity magnitude variation
        velocity_magnitudes = torch.norm(velocities, dim=-1)

        # Compute smoothness metric (coefficient of variation)
        mean_vel = velocity_magnitudes.mean(dim=1)
        std_vel = velocity_magnitudes.std(dim=1)

        smoothness = torch.where(mean_vel > 1e-8, std_vel / mean_vel, torch.zeros_like(mean_vel))

        # Compute penalty
        penalty = torch.mean(torch.relu(smoothness - target_smoothness))

        return penalty

    def process_action_chunk(
        self,
        action_chunk: Tensor,
        apply_masking: bool = True,
        compute_velocity: bool = False,
        dt: float = 0.1,
    ) -> dict[str, Any]:
        """Process an action chunk with RTC techniques.

        Args:
            action_chunk: Raw action chunk from policy
            apply_masking: Whether to apply RTC masking
            compute_velocity: Whether to compute velocity information
            dt: Time step for velocity calculation

        Returns:
            Dictionary containing:
            - "actions": Processed action chunk
            - "velocities": (optional) Computed velocities
            - "smoothness": (optional) Smoothness metric
        """
        result = {}

        # Apply RTC masking if enabled
        if apply_masking:
            action_chunk = self.apply_rtc_masking(action_chunk)

        result["actions"] = action_chunk

        # Compute velocity information if requested
        if compute_velocity and action_chunk.size(1) > 1:
            velocities = self.calculate_velocity(action_chunk, dt)
            result["velocities"] = velocities

            # Compute smoothness metric
            velocity_magnitudes = torch.norm(velocities, dim=-1)
            mean_vel = velocity_magnitudes.mean(dim=1)
            std_vel = velocity_magnitudes.std(dim=1)
            smoothness = torch.where(mean_vel > 1e-8, std_vel / mean_vel, torch.zeros_like(mean_vel))
            result["smoothness"] = smoothness

        self.chunk_step += 1

        return result

    def apply_velocity_smoothing(
        self, velocities: Tensor, window_size: int = 5, method: str = "moving_average"
    ) -> Tensor:
        """Apply smoothing to velocity profiles.

        Args:
            velocities: Velocity tensor to smooth
            window_size: Size of smoothing window (must be odd)
            method: Smoothing method ("moving_average", "gaussian")

        Returns:
            Smoothed velocity tensor of same shape
        """
        if window_size % 2 == 0:
            raise ValueError(f"Window size must be odd, got {window_size}")

        if window_size >= velocities.size(1):
            raise ValueError(f"Window size {window_size} too large for sequence length {velocities.size(1)}")

        velocities = velocities.to(self.device)

        if method == "moving_average":
            padding = window_size // 2
            kernel = torch.ones(1, 1, window_size, device=self.device) / window_size

            batch_size, seq_len, action_dim = velocities.shape
            velocities_reshaped = velocities.transpose(1, 2).reshape(-1, 1, seq_len)

            smoothed = torch.nn.functional.conv1d(velocities_reshaped, kernel, padding=padding)

            smoothed = smoothed.reshape(batch_size, action_dim, seq_len).transpose(1, 2)

        elif method == "gaussian":
            padding = window_size // 2
            sigma = window_size / 6.0

            x = torch.arange(window_size, device=self.device, dtype=torch.float32)
            x = x - window_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, window_size)

            batch_size, seq_len, action_dim = velocities.shape
            velocities_reshaped = velocities.transpose(1, 2).reshape(-1, 1, seq_len)

            smoothed = torch.nn.functional.conv1d(velocities_reshaped, kernel, padding=padding)

            smoothed = smoothed.reshape(batch_size, action_dim, seq_len).transpose(1, 2)

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return smoothed
