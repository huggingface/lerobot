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

Statistics are accumulated into a single running histogram per feature across
all episodes rather than aggregating per-episode quantile summaries. The
resulting quantiles are histogram approximations, subject to discretization and
range-rebinning error; image/video frames are sampled by default.

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
    get_feature_stats,
    write_stats,
)
from lerobot.datasets.compute_stats import RunningQuantileStats, sample_indices
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


def collect_episode_arrays(
    dataset: LeRobotDataset,
    episode_idx: int,
    use_sampling: bool = True,
    skip_images: bool = False,
) -> dict[str, tuple[np.ndarray, int]]:
    """Collect one episode's frames per feature, flattened to (num_samples, dim).

    Args:
        dataset: The LeRobot dataset
        episode_idx: Index of the episode to read
        use_sampling: If True, sub-sample image/video frames to bound memory.
            If False, use every frame (higher memory).
        skip_images: If True, skip image/video features entirely.

    Returns:
        Mapping of feature name to that episode's values and the number of frames
        they came from (which differs from the row count for image features).
    """
    start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    episode_len = end_idx - start_idx

    # Images/video are the memory hog, so sub-sample those frames per episode;
    # numeric columns are cheap, so read them in full (exact).
    image_keys = [k for k in dataset.features if dataset.features[k]["dtype"] in ("image", "video")]
    numeric_keys = [
        k
        for k in dataset.features
        if dataset.features[k]["dtype"] not in ("image", "video", "string", "language")
    ]

    collected_data: dict[str, list] = {}

    # Numeric features: every frame, read directly from the underlying table.
    if numeric_keys:
        numeric_cols = dataset.hf_dataset.select_columns(numeric_keys)[start_idx:end_idx]
        for key in numeric_keys:
            collected_data[key] = [torch.as_tensor(v) for v in numeric_cols[key]]

    # Image/video features: decode only a sampled subset of frames.
    if image_keys and not skip_images:
        sampled_offsets = sample_indices(episode_len) if use_sampling else list(range(episode_len))
        for offset in sampled_offsets:
            item = dataset[start_idx + offset]
            for key in image_keys:
                if key in item:
                    collected_data.setdefault(key, []).append(item[key])

    episode_arrays: dict[str, tuple[np.ndarray, int]] = {}
    for key, data_list in collected_data.items():
        data = torch.stack(data_list).cpu().numpy()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            # (N, C, H, W) -> (N * H * W, C) so quantiles are computed per channel.
            channels = data.shape[1]
            values = data.transpose(0, 2, 3, 1).reshape(-1, channels)
        else:
            values = data.reshape(-1, data.shape[-1]) if data.ndim > 1 else data.reshape(-1, 1)
        episode_arrays[key] = (values, len(data_list))

    return episode_arrays


def compute_quantile_stats_for_dataset(
    dataset: LeRobotDataset,
    use_sampling: bool = True,
    skip_images: bool = False,
) -> dict[str, dict]:
    """Compute whole-dataset statistics with one running histogram per feature.

    Args:
        dataset: The LeRobot dataset to compute statistics for
        use_sampling: If True, sub-sample image/video frames per episode to bound
            memory. If False, use every frame (higher memory).
        skip_images: If True, skip image/video features and leave their stats untouched.

    Returns:
        Dictionary containing statistics with histogram-based global quantile estimates

    Note:
        Episodes are accumulated sequentially because the running accumulators are
        shared across all of them.
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    running_stats: dict[str, RunningQuantileStats] = {}
    frame_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    # Kept only while a feature has a single row, so it can still be finalized.
    single_row_arrays: dict[str, np.ndarray] = {}

    for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
        episode_arrays = collect_episode_arrays(
            dataset, episode_idx, use_sampling=use_sampling, skip_images=skip_images
        )
        for key, (array, num_frames) in episode_arrays.items():
            running_stats.setdefault(key, RunningQuantileStats()).update(array)
            frame_counts[key] = frame_counts.get(key, 0) + num_frames
            row_counts[key] = row_counts.get(key, 0) + len(array)
            if row_counts[key] < 2:
                single_row_arrays[key] = array
            else:
                single_row_arrays.pop(key, None)

    if not running_stats:
        raise ValueError("No episode data found for computing statistics")

    aggregated_stats: dict[str, dict] = {}
    for key, accumulator in running_stats.items():
        if row_counts[key] < 2:
            # Histograms need at least two samples; mirror get_feature_stats' basic-stats path.
            stats = get_feature_stats(single_row_arrays[key], axis=0, keepdims=False)
        else:
            stats = accumulator.get_statistics()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            # Image stats are stored as (C, 1, 1) to broadcast over height and width.
            stats = {k: v if k == "count" else v[:, np.newaxis, np.newaxis] for k, v in stats.items()}
        # `get_feature_stats` counts frames, not the per-channel rows the accumulator sees.
        stats["count"] = np.array([frame_counts[key]])
        aggregated_stats[key] = stats

    logging.info(f"Computed global histogram statistics for {len(aggregated_stats)} features")
    return aggregated_stats


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    use_sampling: bool = True,
    skip_images: bool = False,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        use_sampling: If True, sub-sample image/video frames per episode to bound
            memory. If False, use every frame (higher memory).
        skip_images: If True, skip image/video features and keep their existing stats
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

    new_stats = compute_quantile_stats_for_dataset(
        dataset, use_sampling=use_sampling, skip_images=skip_images
    )

    if skip_images and dataset.meta.stats:
        for key, feature_stats in dataset.meta.stats.items():
            new_stats.setdefault(key, feature_stats)

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
            "Compute stats over every frame (higher memory). By default, "
            "image/video frames are sub-sampled per episode to bound memory."
        ),
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
        use_sampling=not args.no_sampling,
        skip_images=args.skip_images,
    )


if __name__ == "__main__":
    main()
