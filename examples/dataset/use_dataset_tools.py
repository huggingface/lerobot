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
Example script demonstrating dataset tools utilities.

This script shows how to:
1. Delete episodes from a dataset
2. Split a dataset into train/val sets
3. Add/remove features
4. Merge datasets

Usage:
    python examples/use_dataset_tools.py
"""

import numpy as np

from lerobot.datasets.dataset_tools import (
    add_feature,
    delete_episodes,
    merge_datasets,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # Load an existing dataset (replace with your dataset)
    dataset = LeRobotDataset("lerobot/pusht")

    print(f"Original dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"Features: {list(dataset.meta.features.keys())}")

    # Example 1: Delete episodes
    print("\n1. Deleting episodes 0 and 2...")
    filtered_dataset = delete_episodes(dataset, episode_indices=[0, 2], repo_id="pusht_filtered")
    print(f"Filtered dataset: {filtered_dataset.meta.total_episodes} episodes")

    # Example 2: Split dataset
    print("\n2. Splitting dataset into train/val...")
    splits = split_dataset(
        dataset,
        splits={"train": 0.8, "val": 0.2},
    )
    print(f"Train split: {splits['train'].meta.total_episodes} episodes")
    print(f"Val split: {splits['val'].meta.total_episodes} episodes")

    # Example 3: Add a feature
    print("\n3. Adding a reward feature...")

    # Method 1: Pre-computed values
    reward_values = np.random.randn(dataset.meta.total_frames).astype(np.float32)
    dataset_with_reward = add_feature(
        dataset,
        feature_name="reward",
        feature_values=reward_values,
        feature_info={
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        repo_id="pusht_with_reward",
    )

    # Method 2: Using a callable
    def compute_success(frame_dict, episode_idx, frame_idx):
        # Example: mark last 10 frames of each episode as successful
        episode_length = 10  # You'd get this from episode metadata
        return float(frame_idx >= episode_length - 10)

    dataset_with_success = add_feature(
        dataset_with_reward,
        feature_name="success",
        feature_values=compute_success,
        feature_info={
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        repo_id="pusht_with_reward_and_success",
    )

    print(f"New features: {list(dataset_with_success.meta.features.keys())}")

    # Example 4: Remove features
    print("\n4. Removing the success feature...")
    dataset_cleaned = remove_feature(dataset_with_success, feature_names="success", repo_id="pusht_cleaned")
    print(f"Features after removal: {list(dataset_cleaned.meta.features.keys())}")

    # Example 5: Merge datasets
    print("\n5. Merging train and val splits back together...")
    merged = merge_datasets([splits["train"], splits["val"]], output_repo_id="pusht_merged")
    print(f"Merged dataset: {merged.meta.total_episodes} episodes")

    # Example 6: Complex workflow
    print("\n6. Complex workflow example...")

    # Remove a camera if dataset has multiple
    if len(dataset.meta.camera_keys) > 1:
        camera_to_remove = dataset.meta.camera_keys[0]
        print(f"Removing camera: {camera_to_remove}")
        dataset_no_cam = remove_feature(
            dataset, feature_names=camera_to_remove, repo_id="pusht_no_first_camera"
        )
        print(f"Remaining cameras: {dataset_no_cam.meta.camera_keys}")

    print("\nDone! Check ~/.cache/huggingface/lerobot/ for the created datasets.")


if __name__ == "__main__":
    main()
