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

import random

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def find_stage_and_tau(
    current_frame: int,
    episode_length: int,
    subtask_names: list | None,
    subtask_start_frames: list | None,
    subtask_end_frames: list | None,
    global_subtask_names: list,
    temporal_proportions: dict,
    return_combined: bool = False,
) -> tuple[int, float] | float:
    """Find stage and within-stage progress (tau) for a frame.

    Args:
        current_frame: Frame index relative to episode start
        episode_length: Total frames in episode
        subtask_names: Subtask names for this episode (None for single_stage)
        subtask_start_frames: Subtask start frames
        subtask_end_frames: Subtask end frames
        global_subtask_names: Global list of all subtask names
        temporal_proportions: Dict of temporal proportions
        return_combined: If True, return stage+tau as float; else (stage_idx, tau) tuple

    Returns:
        Float (stage.tau) if return_combined, else (stage_idx, tau) tuple
    """
    stage_idx, tau = 0, 0.0
    num_stages = len(global_subtask_names)

    # Single-stage mode: linear progress from 0 to 1
    if num_stages == 1:
        tau = min(1.0, max(0.0, current_frame / max(episode_length - 1, 1)))
    elif subtask_names is None:
        pass  # stage_idx=0, tau=0.0
    elif current_frame < subtask_start_frames[0]:
        pass  # Before first subtask: stage_idx=0, tau=0.0
    elif current_frame > subtask_end_frames[-1]:
        stage_idx, tau = num_stages - 1, 0.999  # After last subtask
    else:
        # Find which subtask this frame belongs to
        found = False
        for name, start, end in zip(subtask_names, subtask_start_frames, subtask_end_frames, strict=True):
            if start <= current_frame <= end:
                stage_idx = global_subtask_names.index(name) if name in global_subtask_names else 0
                tau = compute_tau(current_frame, start, end)
                found = True
                break
        # Frame between subtasks - use previous subtask's end state
        if not found:
            for j in range(len(subtask_names) - 1):
                if subtask_end_frames[j] < current_frame < subtask_start_frames[j + 1]:
                    name = subtask_names[j]
                    stage_idx = global_subtask_names.index(name) if name in global_subtask_names else j
                    tau = 1.0
                    break

    if return_combined:
        # Clamp to avoid overflow at end
        if stage_idx >= num_stages - 1 and tau >= 1.0:
            return num_stages - 1 + 0.999
        return stage_idx + tau
    return stage_idx, tau


def compute_absolute_indices(
    frame_idx: int,
    ep_start: int,
    ep_end: int,
    n_obs_steps: int,
    frame_gap: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute absolute frame indices with clamping for bidirectional observation sequence.

    Bidirectional sampling centered on target frame:
    - Before: [-frame_gap * half_steps, ..., -frame_gap] (half_steps frames)
    - Current: [0] (1 frame)
    - After: [frame_gap, ..., frame_gap * half_steps] (half_steps frames)
    - Total: n_obs_steps + 1 frames

    Out-of-bounds frames are clamped (duplicated from boundary).

    Args:
        frame_idx: Target frame index (center frame of sequence)
        ep_start: Episode start index
        ep_end: Episode end index (exclusive)
        n_obs_steps: Number of observation steps (must be even for symmetric sampling)
        frame_gap: Gap between observation frames

    Returns:
        Tuple of (indices, out_of_bounds_flags)
    """
    half_steps = n_obs_steps // 2

    # Bidirectional deltas: past + current + future
    past_deltas = [-frame_gap * i for i in range(half_steps, 0, -1)]
    future_deltas = [frame_gap * i for i in range(1, half_steps + 1)]
    delta_indices = past_deltas + [0] + future_deltas

    frames = []
    out_of_bounds = []

    for delta in delta_indices:
        target_idx = frame_idx + delta
        # Clamp to episode bounds (duplicate boundary frames for out-of-bounds)
        clamped_idx = max(ep_start, min(ep_end - 1, target_idx))
        frames.append(clamped_idx)
        # Flag as out-of-bounds if clamping occurred
        out_of_bounds.append(1 if target_idx != clamped_idx else 0)

    return torch.tensor(frames), torch.tensor(out_of_bounds)


def apply_rewind_augmentation(
    frame_idx: int,
    ep_start: int,
    n_obs_steps: int,
    max_rewind_steps: int,
    frame_gap: int = 30,
    rewind_step: int | None = None,
) -> tuple[int, list[int]]:
    """
    Generate rewind frame indices for temporal augmentation.

    Rewind simulates going backwards through previously seen frames,
    starting from before the earliest observation frame (for bidirectional sampling).
    Appends reversed frames after the observation sequence.

    Args:
        frame_idx: Target frame index (center of bidirectional observation window)
        ep_start: Episode start index
        n_obs_steps: Number of observation steps
        max_rewind_steps: Maximum rewind steps
        frame_gap: Gap between frames
        rewind_step: If provided, use this exact rewind step (for deterministic behavior).
                     If None, sample randomly.

    Returns:
        Tuple of (rewind_step, rewind_indices)
    """
    # For bidirectional sampling, earliest obs frame is at frame_idx - half_steps * frame_gap
    half_steps = n_obs_steps // 2
    earliest_obs_frame = frame_idx - half_steps * frame_gap

    # Required history: frames before earliest observation frame
    if earliest_obs_frame <= ep_start:
        return 0, []  # No history before observation window

    # Max valid rewind steps based on available history before earliest obs frame
    available_history = earliest_obs_frame - ep_start
    max_valid_step = available_history // frame_gap
    max_rewind = min(max_rewind_steps, max(0, max_valid_step))

    if max_rewind <= 0:
        return 0, []

    # Sample rewind steps if not provided
    rewind_step = random.randint(1, max_rewind) if rewind_step is None else min(rewind_step, max_rewind)

    if rewind_step == 0:
        return 0, []

    # Generate rewind indices going backwards from earliest obs frame
    # rewind_indices[0] is closest to obs window, rewind_indices[-1] is furthest back
    rewind_indices = []
    for i in range(1, rewind_step + 1):
        idx = earliest_obs_frame - i * frame_gap
        idx = max(ep_start, idx)  # Clamp to episode start
        rewind_indices.append(idx)

    return rewind_step, rewind_indices


def compute_tau(current_frame: int | float, subtask_start: int | float, subtask_end: int | float) -> float:
    """Compute τ_t = (t - s_k) / (e_k - s_k) ∈ [0, 1]. Returns 1.0 for zero-duration subtasks."""
    duration = subtask_end - subtask_start
    if duration <= 0:
        return 1.0
    return float(np.clip((current_frame - subtask_start) / duration, 0.0, 1.0))


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
    """Convert temporal proportions to cumulative breakpoints for normalization."""
    if temporal_proportions is None:
        return None

    if isinstance(temporal_proportions, dict):
        if subtask_names is not None:
            proportions = [temporal_proportions.get(name, 0.0) for name in subtask_names]
        else:
            proportions = list(temporal_proportions.values())
    else:
        proportions = list(temporal_proportions)

    total = sum(proportions)
    if total > 0 and abs(total - 1.0) > 1e-6:
        proportions = [p / total for p in proportions]

    breakpoints = [0.0]
    cumsum = 0.0
    for prop in proportions:
        cumsum += prop
        breakpoints.append(cumsum)
    breakpoints[-1] = 1.0

    return breakpoints


def normalize_stage_tau(
    x: float | torch.Tensor,
    num_stages: int | None = None,
    breakpoints: list[float] | None = None,
    temporal_proportions: dict[str, float] | list[float] | None = None,
    subtask_names: list[str] | None = None,
) -> float | torch.Tensor:
    """
    Normalize stage+tau reward to [0, 1] with custom breakpoints.

    Maps stage index + within-stage tau to normalized progress [0, 1].
    The breakpoints are designed to give appropriate weight to each stage
    based on their importance in the task (using temporal proportions).

    Priority: breakpoints > temporal_proportions > linear fallback

    Args:
        x: Raw reward value (stage index + tau) where stage ∈ [0, num_stages-1] and tau ∈ [0, 1)
        num_stages: Number of stages (required if breakpoints/proportions not provided)
        breakpoints: Optional custom breakpoints list of length num_stages + 1.
        temporal_proportions: Optional temporal proportions dict/list to compute breakpoints.
        subtask_names: Optional ordered list of subtask names (for dict proportions)

    Returns:
        Normalized progress value ∈ [0, 1]
    """
    if breakpoints is not None:
        num_stages = len(breakpoints) - 1
    elif temporal_proportions is not None:
        breakpoints = temporal_proportions_to_breakpoints(temporal_proportions, subtask_names)
        num_stages = len(breakpoints) - 1
    elif num_stages is not None:
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
