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
Add ``observation.state`` to an existing LeRobot dataset.

pi0 with ``use_relative_actions=True`` requires ``observation.state`` to
compute relative actions (action − state) on the fly. This script adds
that feature when it doesn't already exist.

Two modes for deriving ``observation.state``:

  1. **From an existing feature** (``STATE_SOURCE_FEATURE``):
     Copies an existing column (e.g. ``observation.joints`` or
     ``observation.pose``) to ``observation.state``.

  2. **From action with offset** (``STATE_SOURCE_FEATURE = None``):
     Derives state from the action column with a per-episode offset:
       state[t] = action[t - STATE_ACTION_OFFSET]

After running this script, recompute relative action stats via the CLI:

    lerobot-edit-dataset \\
        --repo_id <your_dataset> \\
        --operation.type recompute_stats \\
        --operation.relative_action true \\
        --operation.chunk_size 50 \\
        --operation.relative_exclude_joints "['gripper']" \\
        --push_to_hub true

Usage:
    python convert_umi_dataset.py
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HF_DATASET_ID = ""

# Source for observation.state. Options:
#   - A feature name (e.g. "observation.joints", "observation.pose") to copy
#     an existing column. Must have the same shape as "action".
#   - None to derive state from action with STATE_ACTION_OFFSET.
STATE_SOURCE_FEATURE: str | None = "observation.joints"

# Only used when STATE_SOURCE_FEATURE is None.
#   0 → state[t] = action[t]       (same instant)
#   1 → state[t] = action[t-1]     (state lags by 1 step)
STATE_ACTION_OFFSET = 1

# Push the augmented dataset to the Hugging Face Hub.
PUSH_TO_HUB = True


def _build_state_from_feature(dataset: LeRobotDataset, source_feature: str) -> Callable:
    """Return a callable that copies values from an existing feature."""
    hf = dataset.hf_dataset
    source_values = hf[source_feature]

    episode_indices = np.array(hf["episode_index"])
    frame_indices = np.array(hf["frame_index"])
    key_to_global = {(int(episode_indices[i]), int(frame_indices[i])): i for i in range(len(episode_indices))}

    def _get_state(row_dict: dict, ep_idx: int, frame_idx: int):
        return source_values[key_to_global[(ep_idx, frame_idx)]]

    return _get_state


def _build_state_from_action_offset(dataset: LeRobotDataset, offset: int) -> Callable:
    """Return a callable that derives state from action with a per-episode offset.

    state[t] = action[max(0, t - offset)]  (clamped to episode start)
    """
    hf = dataset.hf_dataset
    episode_indices = np.array(hf["episode_index"])
    frame_indices = np.array(hf["frame_index"])

    ep_sorted: dict[int, list[tuple[int, int]]] = {}
    for ep_idx in np.unique(episode_indices):
        ep_mask = episode_indices == ep_idx
        ep_globals = np.where(ep_mask)[0]
        ep_frames = frame_indices[ep_globals]
        order = np.argsort(ep_frames)
        ep_sorted[int(ep_idx)] = [(int(ep_frames[o]), int(ep_globals[o])) for o in order]

    ep_frame_to_local: dict[int, dict[int, int]] = {}
    for ep_idx, sorted_list in ep_sorted.items():
        ep_frame_to_local[ep_idx] = {frame: local for local, (frame, _) in enumerate(sorted_list)}

    actions = hf["action"]

    def _get_state(row_dict: dict, ep_idx: int, frame_idx: int):
        local_t = ep_frame_to_local[ep_idx][frame_idx]
        source_local = max(0, local_t - offset)
        _, source_global = ep_sorted[ep_idx][source_local]
        return actions[source_global]

    return _get_state


def main():
    logger.info(f"Loading dataset {HF_DATASET_ID}")
    dataset = LeRobotDataset(HF_DATASET_ID)

    if "observation.state" in dataset.features:
        logger.info("observation.state already exists — nothing to do")
        return

    action_meta = dataset.features["action"]
    logger.info(f"Action shape: {action_meta['shape']}, names: {action_meta.get('names')}")

    if STATE_SOURCE_FEATURE is not None:
        if STATE_SOURCE_FEATURE not in dataset.features:
            raise ValueError(
                f"Source feature '{STATE_SOURCE_FEATURE}' not found. "
                f"Available: {list(dataset.features.keys())}"
            )
        source_meta = dataset.features[STATE_SOURCE_FEATURE]
        logger.info(f"Copying {STATE_SOURCE_FEATURE} → observation.state")
        state_fn = _build_state_from_feature(dataset, STATE_SOURCE_FEATURE)
        state_feature_info = {
            "dtype": "float32",
            "shape": list(source_meta["shape"]),
            "names": source_meta.get("names"),
        }
    else:
        logger.info(f"Deriving observation.state from action with offset={STATE_ACTION_OFFSET}")
        state_fn = _build_state_from_action_offset(dataset, offset=STATE_ACTION_OFFSET)
        state_feature_info = {
            "dtype": "float32",
            "shape": list(action_meta["shape"]),
            "names": action_meta.get("names"),
        }

    augmented = add_features(
        dataset,
        features={"observation.state": (state_fn, state_feature_info)},
    )
    logger.info("observation.state added")

    if PUSH_TO_HUB:
        logger.info(f"Pushing to Hub: {augmented.repo_id}")
        augmented.push_to_hub()

    logger.info(
        f"Done. Now recompute relative action stats:\n"
        "  lerobot-edit-dataset \\\n"
        f"    --repo_id {augmented.repo_id} \\\n"
        "    --operation.type recompute_stats \\\n"
        "    --operation.relative_action true \\\n"
        "    --operation.chunk_size 50 \\\n"
        "    --operation.relative_exclude_joints \"['gripper']\" \\\n"
        "    --push_to_hub true"
    )


if __name__ == "__main__":
    main()
