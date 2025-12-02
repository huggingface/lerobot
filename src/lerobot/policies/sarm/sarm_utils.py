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

import numpy as np
import torch
import torch.nn.functional as F
from typing import Sequence, Any
from pydantic import BaseModel, Field

# Pydantic Models for SARM-style Annotation
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
            
            start_seconds = int(start_parts[0]) * 60 + int(start_parts[1]) if len(start_parts) == 2 else int(start_parts[0])
            end_seconds = int(end_parts[0]) * 60 + int(end_parts[1]) if len(end_parts) == 2 else int(end_parts[0])
            
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
    avg_proportions = {
        name: sum(props) / len(props)
        for name, props in subtask_proportions.items()
    }
    
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
    Compute cumulative normalized progress from within-subtask progress.
    
    This function implements the core formula used in SARM for both:
    
    **Formula 2 (Training labels):**
        y_t = P_{k-1} + ᾱ_k × τ_t ∈ [0, 1]
        
        Used to compute ground-truth progress labels from subtask annotations.
        - τ_t comes from annotated frame position: τ_t = (t - s_k) / (e_k - s_k)
        - k is the known subtask from annotations
        
    **Formula 4 (Inference predictions):**
        ŷ_{1:N} = P̂_{k-1, 1:N} + ᾱ_{k, 1:N} × τ̂_{1:N} ∈ [0, 1]
        
        Used to convert model outputs to cumulative progress during inference.
        - τ̂ comes from the subtask MLP head (conditioned on predicted stage)
        - k = Ŝ is the predicted stage from Formula 3: Ŝ = argmax(softmax(Ψ))
    
    The formulas are mathematically identical; only the source of inputs differs:
    - Training: τ and k from annotations → ground-truth labels
    - Inference: τ̂ and Ŝ from model → predicted progress
    
    where:
        - P_{k-1} = Σ_{j=1}^{k-1} ᾱ_j is the cumulative prior (sum of previous proportions)
        - ᾱ_k is the temporal proportion for subtask k (from Formula 1)
        - τ is within-subtask progress ∈ [0, 1]
    
    This ensures:
        - y at start of subtask k = P_{k-1}
        - y at end of subtask k = P_k
    
    Supports both scalar and batched tensor inputs:
        - Scalar: tau (float), stage_indices (int), alpha (list/sequence)
        - Batch: tau (Tensor), stage_indices (Tensor), alpha (Tensor), cumulative_prior (Tensor)
    
    Args:
        tau: Within-subtask progress τ ∈ [0, 1]. 
             For training: computed from frame position in annotated subtask.
             For inference: predicted by subtask MLP head.
             Scalar float or Tensor with shape (..., 1)
        stage_indices: Index of current subtask k (0-indexed).
             For training: known from annotations.
             For inference: predicted via argmax(stage_probs) (Formula 3).
             Scalar int or Tensor with shape (...)
        alpha: Temporal proportions ᾱ with shape (num_stages,) or Sequence[float].
             Computed from dataset annotations using Formula 1.
        cumulative_prior: Optional. Cumulative priors P with shape (num_stages + 1,)
             where cumulative_prior[k] = P_k = Σ_{j=1}^{k} ᾱ_j.
             If None, will be computed from alpha.
        
    Returns:
        Cumulative progress y ∈ [0, 1]. 
        Scalar float if inputs are scalar, otherwise Tensor with shape (..., 1)
    """    
    if not isinstance(tau, torch.Tensor):
        if not alpha:
            raise ValueError("alpha (temporal_proportions) cannot be empty")
        
        if isinstance(alpha, torch.Tensor):
            alpha_list = alpha.tolist()
        else:
            alpha_list = list(alpha)
        
        if stage_indices < 0 or stage_indices >= len(alpha_list):
            raise ValueError(
                f"stage_indices {stage_indices} out of range "
                f"for {len(alpha_list)} subtasks"
            )
        
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
    return F.pad(state, padding, mode='constant', value=0)
