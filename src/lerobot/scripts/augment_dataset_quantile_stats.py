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
python src/lerobot/scripts/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
```
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from requests import HTTPError
from tqdm import tqdm

from lerobot.datasets import (
    CODEBASE_VERSION,
    DEFAULT_QUANTILES,
    LeRobotDataset,
    write_stats,
)
from lerobot.datasets.compute_stats import RunningQuantileStats
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


def compute_quantile_stats_for_dataset(dataset: LeRobotDataset, skip_images: bool = False) -> dict[str, dict]:
    """Compute quantile statistics for all episodes in the dataset.

    Uses a single RunningQuantileStats per feature across all episodes to estimate
    global quantiles from histograms, avoiding aggregation of per-episode quantile
    summaries. Estimates are subject to histogram discretization and rebinning error.

    Args:
        dataset: The LeRobot dataset to compute statistics for
        skip_images: If True, skip image/video features (useful when image stats
            are already correct and recomputation is expensive)

    Returns:
        Dictionary containing statistics with histogram-based global quantile estimates
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    # Maintain one RunningQuantileStats per feature across all episodes
    running_stats: dict[str, RunningQuantileStats] = {}
    feature_meta: dict[str, dict] = {}  # track dtype/axes info per feature

    for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
        start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
        end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

        collected_data: dict[str, list] = {}
        for idx in range(start_idx, end_idx):
            item = dataset[idx]
            for key, value in item.items():
                if key not in dataset.features:
                    continue
                if key not in collected_data:
                    collected_data[key] = []
                collected_data[key].append(value)

        for key, data_list in collected_data.items():
            if dataset.features[key]["dtype"] in {"string", "language"}:
                continue

            is_image_video = dataset.features[key]["dtype"] in ["image", "video"]
            if skip_images and is_image_video:
                continue

            data = torch.stack(data_list).cpu().numpy()

            if is_image_video:
                if data.dtype == np.uint8:
                    data = data.astype(np.float32) / 255.0
                # Reshape image (N, C, H, W) → (N*H*W, C) for per-channel stats
                _, c, _, _ = data.shape
                reshaped = data.transpose(0, 2, 3, 1).reshape(-1, c)
            else:
                reshaped = data.reshape(-1, data.shape[-1]) if data.ndim > 1 else data.reshape(-1, 1)

            if key not in running_stats:
                running_stats[key] = RunningQuantileStats()
                feature_meta[key] = {"is_image_video": is_image_video}

            running_stats[key].update(reshaped)

    if not running_stats:
        raise ValueError("No episode data found for computing statistics")

    # Finalize stats from the running accumulators
    aggregated_stats: dict[str, dict] = {}
    for key, rs in running_stats.items():
        stats = rs.get_statistics()

        if feature_meta[key]["is_image_video"]:
            # Expand dims to match (C, 1, 1) layout expected for image stats
            for stat_key in stats:
                if stat_key != "count":
                    stats[stat_key] = stats[stat_key][:, np.newaxis, np.newaxis]

        aggregated_stats[key] = stats

    logging.info(f"Computed global histogram statistics from {len(running_stats)} features")
    return aggregated_stats


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    skip_images: bool = False,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        skip_images: If True, skip image/video features and preserve their existing stats
    """
    logging.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )

    if not overwrite and has_quantile_stats(dataset.meta.stats):
        logging.info("Dataset already contains quantile statistics. No action needed.")
        return

    logging.info("Dataset does not contain quantile statistics. Computing them now...")

    new_stats = compute_quantile_stats_for_dataset(dataset, skip_images=skip_images)

    if skip_images and dataset.meta.stats:
        for key in dataset.meta.stats:
            if key not in new_stats:
                new_stats[key] = dataset.meta.stats[key]

    logging.info("Updating dataset metadata with new quantile statistics")
    dataset.meta.stats = new_stats

    write_stats(new_stats, dataset.meta.root)

    logging.info("Successfully updated dataset with quantile statistics")
    dataset.push_to_hub()

    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        logging.info(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
        pass
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=None, repo_type="dataset")


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quantile statistics if they already exist",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image/video features and preserve their existing stats",
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        skip_images=args.skip_images,
    )


if __name__ == "__main__":
    main()
