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
    python examples/dataset/use_dataset_tools.py
"""

import numpy as np

from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    dataset = LeRobotDataset("lerobot/pusht")

    print(f"Original dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"Features: {list(dataset.meta.features.keys())}")

    print("\n1. Deleting episodes 0 and 2...")
    filtered_dataset = delete_episodes(dataset, episode_indices=[0, 2], repo_id="lerobot/pusht_filtered")
    print(f"Filtered dataset: {filtered_dataset.meta.total_episodes} episodes")

    print("\n2. Splitting dataset into train/val...")
    splits = split_dataset(
        dataset,
        splits={"train": 0.8, "val": 0.2},
    )
    print(f"Train split: {splits['train'].meta.total_episodes} episodes")
    print(f"Val split: {splits['val'].meta.total_episodes} episodes")

    print("\n3. Adding features...")

    reward_values = np.random.randn(dataset.meta.total_frames).astype(np.float32)

    def compute_success(row_dict, episode_index, frame_index):
        episode_length = 10
        return float(frame_index >= episode_length - 10)

    dataset_with_features = add_features(
        dataset,
        features={
            "reward": (
                reward_values,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
            "success": (
                compute_success,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        repo_id="lerobot/pusht_with_features",
    )

    print(f"New features: {list(dataset_with_features.meta.features.keys())}")

    print("\n4. Removing the success feature...")
    dataset_cleaned = remove_feature(
        dataset_with_features, feature_names="success", repo_id="lerobot/pusht_cleaned"
    )
    print(f"Features after removal: {list(dataset_cleaned.meta.features.keys())}")

    print("\n5. Using modify_features to add and remove features simultaneously...")
    dataset_modified = modify_features(
        dataset_with_features,
        add_features={
            "discount": (
                np.ones(dataset.meta.total_frames, dtype=np.float32) * 0.99,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        remove_features="reward",
        repo_id="lerobot/pusht_modified",
    )
    print(f"Modified features: {list(dataset_modified.meta.features.keys())}")

    print("\n6. Merging train and val splits back together...")
    merged = merge_datasets([splits["train"], splits["val"]], output_repo_id="lerobot/pusht_merged")
    print(f"Merged dataset: {merged.meta.total_episodes} episodes")

    print("\n7. Complex workflow example...")

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
