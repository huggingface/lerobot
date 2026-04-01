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

pi0 uses ``observation.state`` as its proprioceptive input AND for
relative action conversion (action − state). This script creates
``observation.state`` by concatenating one or more existing features.

Ordering matters: the features whose dimensions correspond to ``action``
must come FIRST, because ``RelativeActionsProcessorStep`` subtracts
``state[:action_dim]`` from the action. Extra state dimensions (e.g. EE
pose) are appended after and are seen by the model but not used for
relative conversion.

Example: action = [proximal, distal], and we want the model to also see
the EE pose:

    STATE_SOURCE_FEATURES = ["observation.joints", "observation.pose"]
    → observation.state = [proximal, distal, x, y, z, ax, ay, az]

The relative conversion uses state[:2] = [proximal, distal] to subtract
from action[:2], and the model sees all 8 dimensions.

After running this script, recompute relative action stats:

    lerobot-edit-dataset \\
        --repo_id <your_dataset> \\
        --operation.type recompute_stats \\
        --operation.relative_action true \\
        --operation.chunk_size 50 \\
        --operation.relative_exclude_joints "[]" \\
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

# Output repo ID. Set to None for default "<input>_modified".
OUTPUT_REPO_ID: str | None = None

# Features to concatenate into observation.state. Order matters:
# action-matching features FIRST, then extra proprioception.
# Set to a single string to copy one feature directly.
STATE_SOURCE_FEATURES: list[str] | str = ["observation.joints", "observation.pose"]

# Only used when STATE_SOURCE_FEATURES is None:
# derive state from action with a per-episode offset.
STATE_ACTION_OFFSET = 1

# Push the augmented dataset to the Hugging Face Hub.
PUSH_TO_HUB = True


def _build_global_index(dataset: LeRobotDataset) -> dict[tuple[int, int], int]:
    hf = dataset.hf_dataset
    ep = np.array(hf["episode_index"])
    fr = np.array(hf["frame_index"])
    return {(int(ep[i]), int(fr[i])): i for i in range(len(ep))}


def _build_state_from_features(dataset: LeRobotDataset, source_features: list[str]) -> Callable:
    """Concatenate multiple features into observation.state."""
    hf = dataset.hf_dataset
    key_to_global = _build_global_index(dataset)

    columns = [hf[feat] for feat in source_features]

    def _get_state(row_dict: dict, ep_idx: int, frame_idx: int):
        g = key_to_global[(ep_idx, frame_idx)]
        parts = []
        for col in columns:
            val = col[g]
            if hasattr(val, "tolist"):
                flat = val.tolist()
                if isinstance(flat, list):
                    parts.extend(flat)
                else:
                    parts.append(flat)
            elif isinstance(val, list):
                parts.extend(val)
            else:
                parts.append(float(val))
        return parts

    return _get_state


def _build_state_from_action_offset(dataset: LeRobotDataset, offset: int) -> Callable:
    """Derive state from action with a per-episode offset."""
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

    if STATE_SOURCE_FEATURES is not None:
        source_list = (
            [STATE_SOURCE_FEATURES] if isinstance(STATE_SOURCE_FEATURES, str) else list(STATE_SOURCE_FEATURES)
        )
        for feat in source_list:
            if feat not in dataset.features:
                raise ValueError(f"Feature '{feat}' not found. Available: {list(dataset.features.keys())}")

        # Compute combined shape and names
        total_dim = 0
        all_names = []
        for feat in source_list:
            meta = dataset.features[feat]
            total_dim += meta["shape"][0]
            names = meta.get("names")
            if names:
                all_names.extend(names)

        logger.info(
            f"Concatenating {source_list} → observation.state (shape=[{total_dim}], names={all_names})"
        )
        state_fn = _build_state_from_features(dataset, source_list)
        state_feature_info = {
            "dtype": "float32",
            "shape": [total_dim],
            "names": all_names or None,
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
        repo_id=OUTPUT_REPO_ID,
    )
    logger.info("observation.state added")

    if PUSH_TO_HUB:
        logger.info(f"Pushing to Hub: {augmented.repo_id}")
        augmented.push_to_hub()

    logger.info(
        f"Done. Dataset at: {augmented.root}\n"
        "Now recompute relative action stats:\n"
        "  lerobot-edit-dataset \\\n"
        f"    --repo_id {augmented.repo_id} \\\n"
        "    --operation.type recompute_stats \\\n"
        "    --operation.relative_action true \\\n"
        "    --operation.chunk_size 50 \\\n"
        '    --operation.relative_exclude_joints "[]" \\\n'
        "    --push_to_hub true"
    )


if __name__ == "__main__":
    main()
