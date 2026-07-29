# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Statistics for processor-derived relative end-effector features."""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.processor.relative_ee_processor import absolute_ee_to_relative
from lerobot.utils.constants import ACTION, OBS_STATE

from .compute_stats import RunningQuantileStats

logger = logging.getLogger(__name__)


def compute_relative_ee_stats(hf_dataset, chunk_size: int) -> dict[str, dict[str, np.ndarray]]:
    """Compute statistics for the model-facing relative EE action and state."""
    actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            "Relative EE requires action shape [frames, 7] with [xyz, axis-angle, gripper], "
            f"got {actions.shape}"
        )
    if chunk_size < 1:
        raise ValueError(f"Relative EE requires chunk_size >= 1, got {chunk_size}")

    episode_indices = np.asarray(hf_dataset["episode_index"])
    action_stats = RunningQuantileStats()
    state_stats = RunningQuantileStats()
    num_action_targets = 0
    num_states = 0

    for episode_index in np.unique(episode_indices):
        frame_indices = np.flatnonzero(episode_indices == episode_index)
        if len(frame_indices) == 0:
            continue
        episode_actions = torch.from_numpy(actions[frame_indices])

        previous_indices = torch.arange(len(frame_indices)).sub(1).clamp_min(0)
        state_pair = torch.stack([episode_actions[previous_indices], episode_actions], dim=1)
        current = episode_actions[:, None, :].expand_as(state_pair)
        relative_state = absolute_ee_to_relative(current, state_pair).flatten(start_dim=1)
        state_stats.update(relative_state.numpy())
        num_states += len(relative_state)

        for batch_start in range(0, len(frame_indices), 20_000):
            base_indices = torch.arange(batch_start, min(batch_start + 20_000, len(frame_indices)))
            offsets = torch.arange(chunk_size)
            target_indices = base_indices[:, None] + offsets[None, :]
            valid = target_indices < len(frame_indices)
            targets = episode_actions[target_indices[valid]]
            references = episode_actions[base_indices[:, None].expand_as(target_indices)[valid]]
            relative_actions = absolute_ee_to_relative(references, targets)
            action_stats.update(relative_actions.numpy())
            num_action_targets += len(relative_actions)

    if num_states < 2 or num_action_targets < 2:
        raise ValueError("Relative EE statistics require at least two selected dataset frames")

    logger.info(
        "Computed relative EE statistics from %d states and %d unpadded action targets",
        num_states,
        num_action_targets,
    )
    return {
        ACTION: action_stats.get_statistics(),
        OBS_STATE: state_stats.get_statistics(),
    }
