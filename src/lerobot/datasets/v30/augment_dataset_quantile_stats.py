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
This script augments existing LeRobot datasets with quantile statistics.

Most datasets created before the quantile feature was added do not contain
quantile statistics (q01, q10, q50, q90, q99) in their metadata. This script:

1. Loads an existing LeRobot dataset in v3.0 format
2. Checks if it already contains quantile statistics
3. If missing, computes quantile statistics for all features
4. Updates the dataset metadata with the new quantile statistics

Usage:

```bash
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
```
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.utils.utils import init_logging


def has_quantile_stats(stats: dict[str, dict] | None, quantile_list_keys: list[str] | None = None) -> bool:
    """Check if dataset statistics already contain quantile information.

    Args:
        stats: Dataset statistics dictionary

    Returns:
        True if quantile statistics are present, False otherwise
    """
    if quantile_list_keys is None:
        quantile_list_keys = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]

    if stats is None:
        return False

    for feature_stats in stats.values():
        if any(q_key in feature_stats for q_key in quantile_list_keys):
            return True

    return False


def load_episode_data(dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Load episode data by accessing the underlying HuggingFace dataset.

    Args:
        dataset: The LeRobot dataset
        episode_idx: Index of the episode to load

    Returns:
        Dictionary containing episode data for each feature
    """

    episode_info = dataset.meta.episodes[episode_idx]
    episode_length = episode_info["length"]

    start_idx = sum(dataset.meta.episodes[i]["length"] for i in range(episode_idx))
    end_idx = start_idx + episode_length

    episode_data = {}

    episode_slice = dataset.hf_dataset.select(range(start_idx, end_idx))

    for key, feature_info in dataset.features.items():
        if feature_info["dtype"] == "string":
            continue

        if feature_info["dtype"] in ["image", "video"]:
            image_paths = []
            for row in episode_slice:
                if key in row:
                    relative_path = row[key]
                    if isinstance(relative_path, str):
                        absolute_path = str(dataset.meta.root / relative_path)
                        image_paths.append(absolute_path)

            if image_paths:
                episode_data[key] = image_paths
        else:
            arrays = []
            for row in episode_slice:
                if key in row:
                    arrays.append(np.array(row[key]))

            if arrays:
                episode_data[key] = np.stack(arrays)

    return episode_data


def compute_quantile_stats_for_dataset(dataset: LeRobotDataset) -> dict[str, dict]:
    """Compute quantile statistics for all episodes in the dataset.

    Args:
        dataset: The LeRobot dataset to compute statistics for

    Returns:
        Dictionary containing aggregated statistics with quantiles
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    episode_stats_list = []

    for episode_idx in range(dataset.num_episodes):
        episode_data = load_episode_data(dataset, episode_idx)
        ep_stats = compute_episode_stats(episode_data, dataset.features)
        episode_stats_list.append(ep_stats)

    if not episode_stats_list:
        raise ValueError("No episode data found for computing statistics")

    logging.info(f"Aggregating statistics from {len(episode_stats_list)} episodes")
    return aggregate_stats(episode_stats_list)


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
    """
    logging.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )

    if has_quantile_stats(dataset.meta.stats):
        logging.info("Dataset already contains quantile statistics. No action needed.")
        return

    logging.info("Dataset does not contain quantile statistics. Computing them now...")

    new_stats = compute_quantile_stats_for_dataset(dataset)

    logging.info("Updating dataset metadata with new quantile statistics")
    dataset.meta.stats = new_stats

    write_stats(new_stats, dataset.meta.root)

    logging.info("Successfully updated dataset with quantile statistics")
    dataset.push_to_hub()


def main():
    """Main function to run the augmentation script."""
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with quantile statistics")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'lerobot/pusht')",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Local root directory for the dataset",
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
    )


if __name__ == "__main__":
    main()
