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

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field


# Pydantic Models for SARM subtask Annotation
class Timestamp(BaseModel):
    """Timestamp in MM:SS or SS format"""

    start: str = Field(description="Start timestamp (MM:SS or just seconds)")
    end: str = Field(description="End timestamp (MM:SS or just seconds)")


class Subtask(BaseModel):
    """Individual subtask/stage - must use EXACT names from provided list"""

    name: str = Field(description="Subtask name - MUST match one from the predefined list exactly")
    timestamps: Timestamp


class SubtaskAnnotation(BaseModel):
    """Complete annotation for a robot manipulation episode"""

    subtasks: list[Subtask] = Field(description="List of all subtasks in temporal order")


def compute_temporal_proportions(annotations: dict[int, Any], fps: int = 30) -> dict[str, float]:
    """
    Compute dataset-level temporal proportions (priors) for each subtask.

    Implements SARM Paper Formula (1):
        ᾱ_k = (1/M) × Σ_i (L_{i,k} / T_i)

    where:
        - M is the number of trajectories (episodes)
        - L_{i,k} is the duration of subtask k in trajectory i
        - T_i is the total duration of trajectory i

    This averages the proportions of each subtask within each trajectory,
    giving equal weight to all trajectories regardless of their absolute length.

    Args:
        annotations: Dict mapping episode index to SubtaskAnnotation object.
            Each annotation has a .subtasks list where each subtask has:
            - .name: subtask name
            - .timestamps.start: start time as "MM:SS" string
            - .timestamps.end: end time as "MM:SS" string
        fps: Frames per second (unused, kept for API compatibility)

    Returns:
        Dict mapping subtask name to its temporal proportion (ᾱ_k).
        Proportions are normalized to sum to 1.0.
    """
    subtask_proportions: dict[str, list[float]] = {}

    for annotation in annotations.values():
        total_duration = 0
        durations: dict[str, int] = {}

        for subtask in annotation.subtasks:
            start_parts = subtask.timestamps.start.split(":")
            end_parts = subtask.timestamps.end.split(":")

            start_seconds = (
                int(start_parts[0]) * 60 + int(start_parts[1])
                if len(start_parts) == 2
                else int(start_parts[0])
            )
            end_seconds = (
                int(end_parts[0]) * 60 + int(end_parts[1]) if len(end_parts) == 2 else int(end_parts[0])
            )

            duration = end_seconds - start_seconds
            durations[subtask.name] = duration
            total_duration += duration

        # Calculate L_{i,k} / T_i for each subtask in this trajectory
        if total_duration > 0:
            for name, duration in durations.items():
                if name not in subtask_proportions:
                    subtask_proportions[name] = []
                subtask_proportions[name].append(duration / total_duration)

    if not subtask_proportions:
        return {}

    # Average across trajectories: (1/M) × Σ_i (L_{i,k} / T_i)
    avg_proportions = {name: sum(props) / len(props) for name, props in subtask_proportions.items()}

    # Normalize to ensure sum = 1
    total = sum(avg_proportions.values())
    if total > 0:
        avg_proportions = {name: prop / total for name, prop in avg_proportions.items()}

    return avg_proportions


def compute_tau(
    current_frame: int | float,
    subtask_start: int | float,
    subtask_end: int | float,
) -> float:
    """
    Compute within-subtask normalized time τ_t.

    Implements part of SARM Paper Formula (2):
        τ_t = (t - s_k) / (e_k - s_k) ∈ [0, 1]

    where:
        - t is the current frame
        - s_k is the start frame of subtask k
        - e_k is the end frame of subtask k

    Args:
        current_frame: Current frame index (t)
        subtask_start: Start frame of the subtask (s_k)
        subtask_end: End frame of the subtask (e_k)

    Returns:
        Within-subtask progress τ_t ∈ [0, 1]
    """
    subtask_duration = subtask_end - subtask_start

    if subtask_duration <= 0:
        return 1.0

    tau = (current_frame - subtask_start) / subtask_duration

    return float(np.clip(tau, 0.0, 1.0))


def compute_cumulative_progress_batch(
    tau: torch.Tensor | float,
    stage_indices: torch.Tensor | int,
    alpha: torch.Tensor | Sequence[float],
    cumulative_prior: torch.Tensor | None = None,
) -> torch.Tensor | float:
    """
    Compute cumulative progress: y_t = P_{k-1} + ᾱ_k × τ_t ∈ [0, 1] (SARM Formula 2/4).

    Where P_{k-1} = sum of previous proportions, ᾱ_k = proportion for subtask k.
    Used for both training labels (τ from annotations) and inference (τ̂ from model).

    Supports scalar (tau: float, stage_indices: int, alpha: list) or
    batched tensor inputs (tau: Tensor, stage_indices: Tensor, alpha: Tensor).

    Args:
        tau: Within-subtask progress ∈ [0,1]. Scalar or Tensor (..., 1).
        stage_indices: Current subtask index (0-indexed). Scalar or Tensor (...).
        alpha: Temporal proportions (num_stages,) or Sequence[float].
        cumulative_prior: Optional precomputed cumulative priors (num_stages + 1,).

    Returns:
        Cumulative progress ∈ [0,1]. Scalar float or Tensor (..., 1).
    """
    if not isinstance(tau, torch.Tensor):
        if not alpha:
            raise ValueError("alpha (temporal_proportions) cannot be empty")

        if isinstance(alpha, torch.Tensor):
            alpha_list = alpha.tolist()
        else:
            alpha_list = list(alpha)

        if stage_indices < 0 or stage_indices >= len(alpha_list):
            raise ValueError(f"stage_indices {stage_indices} out of range for {len(alpha_list)} subtasks")

        # P_{k-1} = sum of proportions for subtasks 0 to k-1
        P_k_minus_1 = sum(alpha_list[:stage_indices])

        # ᾱ_k = proportion for current subtask
        alpha_k = alpha_list[stage_indices]

        # y_t = P_{k-1} + ᾱ_k × τ_t
        y_t = P_k_minus_1 + alpha_k * tau

        return float(np.clip(y_t, 0.0, 1.0))

    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32)

    # Compute cumulative_prior if not provided
    if cumulative_prior is None:
        cumulative_prior = torch.zeros(len(alpha) + 1, dtype=alpha.dtype, device=alpha.device)
        cumulative_prior[1:] = torch.cumsum(alpha, dim=0)

    # P_{k-1} for each predicted stage
    P_k_minus_1 = cumulative_prior[stage_indices]

    # ᾱ_k for each predicted stage
    alpha_k = alpha[stage_indices]

    # ŷ = P_{k-1} + ᾱ_k × τ̂
    progress = P_k_minus_1.unsqueeze(-1) + alpha_k.unsqueeze(-1) * tau

    return progress


def pad_state_to_max_dim(state: torch.Tensor, max_state_dim: int) -> torch.Tensor:
    """Pad the state tensor's last dimension to max_state_dim with zeros."""
    current_dim = state.shape[-1]
    if current_dim >= max_state_dim:
        return state[..., :max_state_dim]  # Truncate if larger

    # Pad with zeros on the right
    padding = (0, max_state_dim - current_dim)  # (left, right) for last dim
    return F.pad(state, padding, mode="constant", value=0)


def temporal_proportions_to_breakpoints(
    temporal_proportions: dict[str, float] | list[float] | None,
    subtask_names: list[str] | None = None,
) -> list[float] | None:
    """
    Convert temporal proportions to cumulative breakpoints for normalization.
    
    Reference: SARM paper uses temporal proportions (α̅_k) to weight each stage.
    The breakpoints are cumulative sums: [0, α̅_1, α̅_1+α̅_2, ..., 1.0]
    
    Example:
        proportions = {"task1": 0.1, "task2": 0.2, "task3": 0.7}
        breakpoints = [0.0, 0.1, 0.3, 1.0]
    
    Args:
        temporal_proportions: Dict mapping subtask names to proportions, or list of proportions
        subtask_names: Optional ordered list of subtask names (required if dict provided)
        
    Returns:
        List of cumulative breakpoints [0.0, ..., 1.0] with length num_stages + 1
    """
    if temporal_proportions is None:
        return None
    
    # Convert dict to ordered list if needed
    if isinstance(temporal_proportions, dict):
        if subtask_names is not None:
            # Use provided order
            proportions = [temporal_proportions.get(name, 0.0) for name in subtask_names]
        else:
            # Use dict order (Python 3.7+ preserves insertion order)
            proportions = list(temporal_proportions.values())
    else:
        proportions = list(temporal_proportions)
    
    # Normalize to ensure sum = 1.0
    total = sum(proportions)
    if total > 0 and abs(total - 1.0) > 1e-6:
        proportions = [p / total for p in proportions]
    
    # Compute cumulative breakpoints
    breakpoints = [0.0]
    cumsum = 0.0
    for prop in proportions:
        cumsum += prop
        breakpoints.append(cumsum)
    
    # Ensure last breakpoint is exactly 1.0
    breakpoints[-1] = 1.0
    
    return breakpoints


def normalize_sparse(
    x: float | torch.Tensor,
    num_stages: int | None = None,
    breakpoints: list[float] | None = None,
    temporal_proportions: dict[str, float] | list[float] | None = None,
    subtask_names: list[str] | None = None,
) -> float | torch.Tensor:
    """
    Normalize sparse stage+tau reward to [0, 1] with custom breakpoints.
    
    Reference: raw_data_utils.py normalize_sparse()
    
    Maps stage index + within-stage tau to normalized progress [0, 1].
    The breakpoints are designed to give appropriate weight to each stage
    based on their importance in the task (using temporal proportions).
    
    Priority: breakpoints > temporal_proportions > linear fallback
    
    Args:
        x: Raw reward value (stage index + tau) where stage ∈ [0, num_stages-1] and tau ∈ [0, 1)
        num_stages: Number of stages (required if breakpoints/proportions not provided)
        breakpoints: Optional custom breakpoints list of length num_stages + 1.
            Example for 6 stages: [0.0, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0]
        temporal_proportions: Optional temporal proportions dict/list to compute breakpoints.
            Example: {"task1": 0.1, "task2": 0.3, "task3": 0.6} -> breakpoints [0, 0.1, 0.4, 1.0]
        subtask_names: Optional ordered list of subtask names (for dict proportions)
        
    Returns:
        Normalized progress value ∈ [0, 1]
    """
    # Priority: explicit breakpoints > temporal_proportions > linear fallback
    if breakpoints is not None:
        num_stages = len(breakpoints) - 1
    elif temporal_proportions is not None:
        breakpoints = temporal_proportions_to_breakpoints(temporal_proportions, subtask_names)
        num_stages = len(breakpoints) - 1
    elif num_stages is not None:
        # Linear fallback: evenly space breakpoints
        breakpoints = [i / num_stages for i in range(num_stages + 1)]
    else:
        raise ValueError("Either num_stages, breakpoints, or temporal_proportions must be provided")
    
    if isinstance(x, torch.Tensor):
        result = torch.zeros_like(x)
        for i in range(num_stages):
            mask = (x >= i) & (x < i + 1)
            tau_in_stage = x - i  # tau ∈ [0, 1) within stage
            result[mask] = breakpoints[i] + tau_in_stage[mask] * (breakpoints[i + 1] - breakpoints[i])
        # Handle exactly at num_stages (complete)
        result[x >= num_stages] = 1.0
        return result.clamp(0.0, 1.0)
    else:
        # Scalar version
        if x < 0:
            return 0.0
        if x >= num_stages:
            return 1.0
        stage = int(x)
        tau = x - stage
        return breakpoints[stage] + tau * (breakpoints[stage + 1] - breakpoints[stage])


def normalize_dense(
    x: float | torch.Tensor,
    num_stages: int | None = None,
    breakpoints: list[float] | None = None,
    temporal_proportions: dict[str, float] | list[float] | None = None,
    subtask_names: list[str] | None = None,
) -> float | torch.Tensor:
    """
    Normalize dense stage+tau reward to [0, 1] with custom breakpoints.
    
    Reference: raw_data_utils.py normalize_dense()
    
    Maps stage index + within-stage tau to normalized progress [0, 1].
    Different breakpoints than sparse to reflect finer-grained stages.
    
    Priority: breakpoints > temporal_proportions > linear fallback
    
    Args:
        x: Raw reward value (stage index + tau)
        num_stages: Number of dense stages (required if breakpoints/proportions not provided)
        breakpoints: Optional custom breakpoints list of length num_stages + 1.
            Example for 8 stages: [0.0, 0.08, 0.37, 0.53, 0.67, 0.72, 0.81, 0.9, 1.0]
        temporal_proportions: Optional temporal proportions dict/list to compute breakpoints.
            Example: {"task1": 0.1, "task2": 0.3, "task3": 0.6} -> breakpoints [0, 0.1, 0.4, 1.0]
        subtask_names: Optional ordered list of subtask names (for dict proportions)
        
    Returns:
        Normalized progress value ∈ [0, 1]
    """
    # Priority: explicit breakpoints > temporal_proportions > linear fallback
    if breakpoints is not None:
        num_stages = len(breakpoints) - 1
    elif temporal_proportions is not None:
        breakpoints = temporal_proportions_to_breakpoints(temporal_proportions, subtask_names)
        num_stages = len(breakpoints) - 1
    elif num_stages is not None:
        # Linear fallback: evenly space breakpoints
        breakpoints = [i / num_stages for i in range(num_stages + 1)]
    else:
        raise ValueError("Either num_stages, breakpoints, or temporal_proportions must be provided")
    
    if isinstance(x, torch.Tensor):
        result = torch.zeros_like(x)
        for i in range(num_stages):
            mask = (x >= i) & (x < i + 1)
            tau_in_stage = x - i
            result[mask] = breakpoints[i] + tau_in_stage[mask] * (breakpoints[i + 1] - breakpoints[i])
        result[x >= num_stages] = 1.0
        return result.clamp(0.0, 1.0)
    else:
        if x < 0:
            return 0.0
        if x >= num_stages:
            return 1.0
        stage = int(x)
        tau = x - stage
        return breakpoints[stage] + tau * (breakpoints[stage + 1] - breakpoints[stage])


class RegressionConfidenceSmoother:
    """
    Confidence-weighted smoothing for SARM inference predictions.
    
    Reference: pred_smoother.py RegressionConfidenceSmoother
    
    Uses a sliding window of past predictions weighted by their confidence
    to produce smoother output during inference. Low-confidence predictions
    are rejected and the previous smoothed value is returned.
    
    This helps reduce jitter in real-time reward estimation.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        beta: float = 3.0,
        eps: float = 1e-6,
        low_conf_th: float = 0.9,
        value_range: tuple[float, float] | None = None,
    ):
        """
        Initialize the smoother.
        
        Reference: pred_smoother.py lines 8-25
        
        Args:
            window_size: Number of past predictions to keep
            beta: Exponent for confidence weighting (higher = more weight on high-conf)
            eps: Small constant for numerical stability
            low_conf_th: Confidence threshold below which predictions are rejected
            value_range: Optional (min, max) for value normalization
        """
        from collections import deque
        
        self.window_size = window_size
        self.beta = beta
        self.eps = eps
        self.low_conf_th = low_conf_th
        self.value_range = value_range
        
        self.hist_vals: deque = deque(maxlen=window_size)
        self.hist_confs: deque = deque(maxlen=window_size)
        self.last_smoothed: float | None = None

    def reset(self):
        """Clear history and reset state."""
        self.hist_vals.clear()
        self.hist_confs.clear()
        self.last_smoothed = None

    def _normalize_val(self, v: float) -> float:
        """Normalize value to [0, 1] if value_range is set."""
        if self.value_range is None:
            return float(v)
        vmin, vmax = self.value_range
        if vmax <= vmin:
            return float(v)
        return (float(v) - vmin) / (vmax - vmin)

    def update(self, value: float, confidence: float) -> float:
        """
        Update smoother with new prediction and return smoothed value.
        
        Reference: pred_smoother.py lines 37-68
        
        Args:
            value: Raw predicted value (e.g., normalized reward)
            confidence: Prediction confidence (e.g., max stage probability)
            
        Returns:
            Smoothed value based on confidence-weighted history
        """
        # Sanitize inputs
        val_t = self._normalize_val(value)
        conf_t = max(0.0, min(1.0, float(confidence)))

        # Compute baseline from history only
        if self.hist_vals:
            weights = [max(self.eps, c) ** self.beta for c in self.hist_confs]
            wsum = sum(weights) + self.eps
            baseline = sum(w * v for w, v in zip(weights, self.hist_vals)) / wsum
        else:
            baseline = val_t

        if self.last_smoothed is None:
            self.last_smoothed = baseline

        # Low confidence: skip update, return last smoothed
        if conf_t < self.low_conf_th:
            return self.last_smoothed

        # Accept point, update history
        self.hist_vals.append(val_t)
        self.hist_confs.append(conf_t)

        # Recompute smoothed with current point included
        weights = [max(self.eps, c) ** self.beta for c in self.hist_confs]
        wsum = sum(weights) + self.eps
        smoothed_item = sum(w * v for w, v in zip(weights, self.hist_vals)) / wsum

        self.last_smoothed = smoothed_item
        return smoothed_item
