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

The quantiles computed here are TRUE GLOBAL quantiles over the whole dataset:
each episode's data is streamed into a per-feature running histogram and those
histograms are merged across episodes (``RunningQuantileStats.merge``). This is
the statistically correct way to combine quantiles and matches what
quantile-normalized policies (e.g. pi0 / pi0.5) expect. It deliberately avoids
``aggregate_stats``, whose count-weighted averaging of per-episode quantiles
biases the distribution tails inward.

Usage:

```bash
python src/lerobot/scripts/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
```
"""

import argparse
import concurrent.futures
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
    RunningQuantileStats,
    compute_feature_running_stats,
    feature_stats_from_running,
    write_stats,
)
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


def process_single_episode(dataset: LeRobotDataset, episode_idx: int) -> dict[str, dict]:
    """Build per-feature running histogram accumulators for a single episode.

    Returns a mapping ``feature_key -> info`` where ``info`` carries the
    episode's :class:`RunningQuantileStats` accumulator (``None`` if the episode
    has fewer than 2 usable samples) plus the metadata needed to finalize it
    after all episodes are merged. Returning accumulators rather than finished
    stats lets the caller merge histograms across episodes for *global*
    quantiles, instead of averaging per-episode quantiles.

    Args:
        dataset: The LeRobot dataset
        episode_idx: Index of the episode to process

    Returns:
        Dictionary mapping each numerical feature to its accumulator + metadata.
    """
    logging.info(f"Computing stats for episode {episode_idx}")

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

    ep_running: dict[str, dict] = {}
    for key, data_list in collected_data.items():
        if dataset.features[key]["dtype"] in {"string", "language"}:
            continue

        data = torch.stack(data_list).cpu().numpy()
        is_image = dataset.features[key]["dtype"] in ["image", "video"]
        if is_image:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0

            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        running_stats, sample_count = compute_feature_running_stats(
            data, axis=axes_to_reduce, quantile_list=DEFAULT_QUANTILES
        )

        ep_running[key] = {
            "running_stats": running_stats,
            "sample_count": sample_count,
            "axis": axes_to_reduce,
            "keepdims": keepdims,
            "original_shape": data.shape,
            "is_image": is_image,
        }

    return ep_running


def _merge_episode_running_stats(
    ep_running: dict[str, dict],
    merged_running: dict[str, RunningQuantileStats],
    feature_meta: dict[str, dict],
) -> None:
    """Merge one episode's accumulators into the dataset-level accumulators.

    Mutates ``merged_running`` and ``feature_meta`` in place. Must be called
    from a single thread (the merge itself is not thread-safe).
    """
    for key, info in ep_running.items():
        running_stats = info["running_stats"]
        if running_stats is None:
            # Episode too short for histogram quantiles; its handful of frames are
            # negligible for global quantiles, so skip rather than bias the result.
            continue

        if key not in merged_running:
            merged_running[key] = running_stats
            feature_meta[key] = {
                "axis": info["axis"],
                "keepdims": info["keepdims"],
                "original_shape": info["original_shape"],
                "is_image": info["is_image"],
                "total_count": int(info["sample_count"]),
            }
        else:
            merged_running[key].merge(running_stats)
            feature_meta[key]["total_count"] += int(info["sample_count"])


def compute_quantile_stats_for_dataset(dataset: LeRobotDataset) -> dict[str, dict]:
    """Compute TRUE global statistics (incl. quantiles) for the whole dataset.

    Each episode is streamed into per-feature running histograms which are then
    merged across episodes (:meth:`RunningQuantileStats.merge`). The merged
    histograms yield global quantiles accurate to histogram resolution -- unlike
    averaging per-episode quantiles. ``mean``/``std``/``min``/``max`` are exact.
    Only one merged accumulator per feature is retained at a time, so memory
    stays bounded regardless of dataset size.

    Args:
        dataset: The LeRobot dataset to compute statistics for

    Returns:
        Dictionary mapping feature keys to their global statistics dictionaries.

    Note:
        Video decoding operations are not thread-safe, so we process episodes
        sequentially when video keys are present. For datasets without videos,
        episodes are processed in parallel and merged as results arrive.
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    merged_running: dict[str, RunningQuantileStats] = {}
    feature_meta: dict[str, dict] = {}
    has_videos = len(dataset.meta.video_keys) > 0

    if has_videos:
        logging.info("Dataset contains video keys - using sequential processing for thread safety")
        for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
            ep_running = process_single_episode(dataset, episode_idx)
            _merge_episode_running_stats(ep_running, merged_running, feature_meta)
    else:
        logging.info("Dataset has no video keys - using parallel processing for better performance")
        max_workers = min(dataset.num_episodes, 16)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(process_single_episode, dataset, episode_idx): episode_idx
                for episode_idx in range(dataset.num_episodes)
            }

            with tqdm(total=dataset.num_episodes, desc="Processing episodes") as pbar:
                # Episodes are computed in worker threads; merging happens here in the
                # main thread as each result arrives, keeping memory bounded.
                for future in concurrent.futures.as_completed(future_to_episode):
                    _merge_episode_running_stats(future.result(), merged_running, feature_meta)
                    pbar.update(1)

    if not merged_running:
        raise ValueError("No episode data found for computing statistics")

    logging.info(f"Finalizing global statistics for {len(merged_running)} features")
    new_stats: dict[str, dict] = {}
    for key, running_stats in merged_running.items():
        meta = feature_meta[key]
        feature_stats = feature_stats_from_running(
            running_stats,
            sample_count=meta["total_count"],
            axis=meta["axis"],
            keepdims=meta["keepdims"],
            original_shape=meta["original_shape"],
        )
        if meta["is_image"]:
            feature_stats = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in feature_stats.items()
            }
        new_stats[key] = feature_stats

    return new_stats


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
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

    new_stats = compute_quantile_stats_for_dataset(dataset)

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

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
