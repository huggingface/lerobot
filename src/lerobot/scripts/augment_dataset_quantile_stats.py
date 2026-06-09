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
import concurrent.futures
import logging
import os
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
    aggregate_stats,
    get_feature_stats,
    write_stats,
)
from lerobot.datasets.compute_stats import sample_indices
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


def process_single_episode(dataset: LeRobotDataset, episode_idx: int, use_sampling: bool = True) -> dict:
    """Process a single episode and return its statistics.

    Args:
        dataset: The LeRobot dataset
        episode_idx: Index of the episode to process
        use_sampling: If True, sub-sample image/video frames per episode to bound
            memory. If False, use every frame (exact, higher memory).

    Returns:
        Dictionary containing episode statistics
    """
    logging.info(f"Computing stats for episode {episode_idx}")

    start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    episode_len = end_idx - start_idx

    # Images/video are the memory hog, so sub-sample those frames per episode;
    # numeric columns are cheap, so read them in full (exact).
    image_keys = [k for k in dataset.features if dataset.features[k]["dtype"] in ("image", "video")]
    numeric_keys = [
        k for k in dataset.features if dataset.features[k]["dtype"] not in ("image", "video", "string")
    ]

    collected_data: dict[str, list] = {}

    # Numeric features: every frame, read directly from the underlying table.
    if numeric_keys:
        numeric_cols = dataset.hf_dataset.select_columns(numeric_keys)[start_idx:end_idx]
        for key in numeric_keys:
            collected_data[key] = [torch.as_tensor(v) for v in numeric_cols[key]]

    # Image/video features: decode only a sampled subset of frames.
    if image_keys:
        sampled_offsets = sample_indices(episode_len) if use_sampling else list(range(episode_len))
        for offset in sampled_offsets:
            item = dataset[start_idx + offset]
            for key in image_keys:
                if key in item:
                    collected_data.setdefault(key, []).append(item[key])

    ep_stats = {}
    for key, data_list in collected_data.items():
        if dataset.features[key]["dtype"] == "string":
            continue

        data = torch.stack(data_list).cpu().numpy()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0

            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            data, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
        )

        if dataset.features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def compute_quantile_stats_for_dataset(dataset: LeRobotDataset, use_sampling: bool = True) -> dict[str, dict]:
    """Compute quantile statistics for all episodes in the dataset.

    Args:
        dataset: The LeRobot dataset to compute statistics for
        use_sampling: If True, sub-sample image/video frames per episode to bound
            memory. If False, use every frame (exact, higher memory).

    Returns:
        Dictionary containing aggregated statistics with quantiles

    Note:
        Video decoding operations are not thread-safe, so we process episodes sequentially
        when video keys are present. For datasets without videos, we use parallel processing
        with ThreadPoolExecutor for better performance.
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    episode_stats_list = []
    has_videos = len(dataset.meta.video_keys) > 0

    if has_videos:
        logging.info("Dataset contains video keys - using sequential processing for thread safety")
        for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
            ep_stats = process_single_episode(dataset, episode_idx, use_sampling)
            episode_stats_list.append(ep_stats)
    else:
        logging.info("Dataset has no video keys - using parallel processing for better performance")
        max_workers = min(dataset.num_episodes, int(os.environ.get("LEROBOT_STATS_MAX_WORKERS", 16)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(process_single_episode, dataset, episode_idx, use_sampling): episode_idx
                for episode_idx in range(dataset.num_episodes)
            }

            episode_results = {}
            with tqdm(total=dataset.num_episodes, desc="Processing episodes") as pbar:
                for future in concurrent.futures.as_completed(future_to_episode):
                    episode_idx = future_to_episode[future]
                    ep_stats = future.result()
                    episode_results[episode_idx] = ep_stats
                    pbar.update(1)

        for episode_idx in range(dataset.num_episodes):
            if episode_idx in episode_results:
                episode_stats_list.append(episode_results[episode_idx])

    if not episode_stats_list:
        raise ValueError("No episode data found for computing statistics")

    logging.info(f"Aggregating statistics from {len(episode_stats_list)} episodes")
    return aggregate_stats(episode_stats_list)


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    use_sampling: bool = True,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        use_sampling: If True, sub-sample image/video frames per episode to bound
            memory. If False, use every frame (exact, higher memory).
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

    new_stats = compute_quantile_stats_for_dataset(dataset, use_sampling=use_sampling)

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
        "--no-sampling",
        action="store_true",
        help=(
            "Compute stats over every frame (exact, higher memory). By default, "
            "image/video frames are sub-sampled per episode to bound memory."
        ),
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        use_sampling=not args.no_sampling,
    )


if __name__ == "__main__":
    main()
