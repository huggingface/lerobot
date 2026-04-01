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
Add ``observation.state`` to an existing UMI LeRobot dataset and recompute
stats for pi0 training with relative EE actions.

UMI datasets already contain ``action`` (absolute EE pose from SLAM) and
images. This script derives ``observation.state`` from the action column
and recomputes statistics with relative action stats.

State-Action Offset:
UMI SLAM produces a single trajectory of EE poses stored as ``action``.
We derive ``observation.state`` from the same trajectory with a
configurable offset:

  state[t] = action[t - STATE_ACTION_OFFSET]

With offset=0, state equals action at the same timestep. With offset=1,
state is the previous timestep's action — representing where the gripper
*arrived* (the result of the previous command), which is what the robot
knows at decision time. Offset=1 is the typical UMI convention.

For the first frame(s) of each episode where t < offset, we use the
earliest available action (action[0]) as state.

After adding state, train with standard lerobot-train:
    lerobot-train \\
        --dataset.repo_id=<your_dataset> \\
        --policy.type=pi0 \\
        --policy.use_relative_actions=true \\
        --policy.relative_exclude_joints='["gripper"]' \\
        --policy.pretrained_path=lerobot/pi0_base

Usage:
    python convert_umi_dataset.py
"""

from __future__ import annotations

import logging

import numpy as np

from lerobot.datasets.dataset_tools import add_features, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────

HF_DATASET_ID = "<hf_username>/<dataset_repo_id>"

# Offset between state and action indices within each episode.
#   0 → state[t] = action[t]       (same instant)
#   1 → state[t] = action[t-1]     (state lags by 1 step — typical for UMI)
STATE_ACTION_OFFSET = 1

# Joint names to keep absolute (not converted to relative).
RELATIVE_EXCLUDE_JOINTS: list[str] = ["gripper"]

# pi0 chunk size (for relative stats computation).
CHUNK_SIZE = 50


# ── Build state from action with offset ──────────────────────────────────


def build_state_array(dataset: LeRobotDataset, offset: int) -> np.ndarray:
    """Derive observation.state from the action column with a per-episode offset.

    For each frame t in an episode:
        state[t] = action[max(0, t - offset)]  (clamped to episode start)
    """
    hf = dataset.hf_dataset
    actions = np.array(hf["action"], dtype=np.float32)
    episode_indices = np.array(hf["episode_index"])
    frame_indices = np.array(hf["frame_index"])

    states = np.empty_like(actions)

    for ep_idx in np.unique(episode_indices):
        ep_mask = episode_indices == ep_idx
        ep_global_indices = np.where(ep_mask)[0]
        ep_actions = actions[ep_global_indices]
        ep_frames = frame_indices[ep_global_indices]

        sort_order = np.argsort(ep_frames)
        ep_global_indices = ep_global_indices[sort_order]
        ep_actions = ep_actions[sort_order]

        n = len(ep_actions)
        for local_t in range(n):
            source_t = max(0, local_t - offset)
            states[ep_global_indices[local_t]] = ep_actions[source_t]

    return states


def main():
    logger.info(f"Loading dataset {HF_DATASET_ID}")
    dataset = LeRobotDataset(HF_DATASET_ID)

    if "observation.state" in dataset.features:
        logger.warning("observation.state already exists — skipping add_features")
        augmented = dataset
    else:
        logger.info(f"Building observation.state from action with offset={STATE_ACTION_OFFSET}")
        state_array = build_state_array(dataset, offset=STATE_ACTION_OFFSET)

        action_meta = dataset.features["action"]
        state_feature_info = {
            "dtype": "float32",
            "shape": list(action_meta["shape"]),
            "names": action_meta.get("names"),
        }

        augmented = add_features(
            dataset,
            features={
                "observation.state": (state_array, state_feature_info),
            },
        )
        logger.info("observation.state added")

    logger.info("Recomputing stats with relative action statistics...")
    recompute_stats(
        augmented,
        relative_action=True,
        relative_exclude_joints=RELATIVE_EXCLUDE_JOINTS,
        chunk_size=CHUNK_SIZE,
    )

    logger.info(f"Dataset ready at {augmented.root}")
    logger.info(
        "Train with:\n"
        "  lerobot-train \\\n"
        f"    --dataset.repo_id={augmented.repo_id} \\\n"
        "    --policy.type=pi0 \\\n"
        "    --policy.use_relative_actions=true \\\n"
        f"    --policy.relative_exclude_joints='{RELATIVE_EXCLUDE_JOINTS}' \\\n"
        "    --policy.pretrained_path=lerobot/pi0_base"
    )


if __name__ == "__main__":
    main()
