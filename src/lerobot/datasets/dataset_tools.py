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

"""Dataset tools utilities for LeRobotDataset.

This module provides utilities for:
- Deleting episodes from datasets
- Splitting datasets into multiple smaller datasets
- Adding/removing features from datasets
- Merging datasets (wrapper around aggregate functionality)
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_parquet_file_size_in_mb,
    get_video_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


def delete_episodes(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Delete episodes from a LeRobotDataset and create a new dataset.

    Args:
        dataset: The source LeRobotDataset.
        episode_indices: List of episode indices to delete.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_filtered" to original.

    Returns:
        LeRobotDataset: New dataset with episodes removed.
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    # Validate episode indices
    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    logging.info(f"Deleting {len(episode_indices)} episodes from dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_filtered"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from dataset")

    # Create new dataset
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    # Process episodes
    episode_mapping = {}  # old_idx -> new_idx

    for new_idx, old_idx in tqdm(enumerate(episodes_to_keep), desc="Processing episodes"):
        episode_mapping[old_idx] = new_idx

    # Copy data files and update indices
    _copy_and_reindex_data(dataset, new_meta, episode_mapping)

    # Copy video files if present
    if dataset.meta.video_keys:
        _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

    # Create new dataset instance
    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    logging.info(f"Created new dataset with {len(episodes_to_keep)} episodes")
    return new_dataset


def split_dataset(
    dataset: LeRobotDataset,
    splits: dict[str, list[int]] | dict[str, float],
    output_dir: str | Path | None = None,
) -> dict[str, LeRobotDataset]:
    """Split a LeRobotDataset into multiple smaller datasets.

    Args:
        dataset: The source LeRobotDataset to split.
        splits: Either a dict mapping split names to episode indices, or a dict mapping
                split names to fractions (must sum to <= 1.0).
        output_dir: Base directory for output datasets. If None, uses default location.

    Returns:
        dict[str, LeRobotDataset]: Dictionary mapping split names to new datasets.

    Examples:
        # Split by specific episodes
        splits = {"train": [0, 1, 2], "val": [3, 4]}
        datasets = split_dataset(dataset, splits)

        # Split by fractions
        splits = {"train": 0.8, "val": 0.2}
        datasets = split_dataset(dataset, splits)
    """
    if not splits:
        raise ValueError("No splits provided")

    # Convert fractions to episode indices if needed
    if all(isinstance(v, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(dataset.meta.total_episodes, splits)

    # Validate episodes
    all_episodes = set()
    for split_name, episodes in splits.items():
        if not episodes:
            raise ValueError(f"Split '{split_name}' has no episodes")
        episode_set = set(episodes)
        if episode_set & all_episodes:
            raise ValueError("Episodes cannot appear in multiple splits")
        all_episodes.update(episode_set)

    # Validate all episodes are valid
    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = all_episodes - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / dataset.repo_id

    result_datasets = {}

    for split_name, episodes in splits.items():
        logging.info(f"Creating split '{split_name}' with {len(episodes)} episodes")

        # Create repo_id for split
        split_repo_id = f"{dataset.repo_id}_{split_name}"
        split_output_dir = output_dir / split_repo_id

        # Create episode mapping
        episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes))}

        # Create new dataset metadata
        new_meta = LeRobotDatasetMetadata.create(
            repo_id=split_repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
            root=split_output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
        )

        # Copy data and videos
        _copy_and_reindex_data(dataset, new_meta, episode_mapping)
        if dataset.meta.video_keys:
            _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

        # Create new dataset instance
        new_dataset = LeRobotDataset(
            repo_id=split_repo_id,
            root=split_output_dir,
            image_transforms=dataset.image_transforms,
            delta_timestamps=dataset.delta_timestamps,
            tolerance_s=dataset.tolerance_s,
        )

        result_datasets[split_name] = new_dataset

    return result_datasets


def merge_datasets(
    datasets: list[LeRobotDataset],
    output_repo_id: str,
    output_dir: str | Path | None = None,
) -> LeRobotDataset:
    """Merge multiple LeRobotDatasets into a single dataset.

    This is a wrapper around the aggregate_datasets functionality with a cleaner API.

    Args:
        datasets: List of LeRobotDatasets to merge.
        output_repo_id: Repository ID for the merged dataset.
        output_dir: Directory to save the merged dataset. If None, uses default location.

    Returns:
        LeRobotDataset: The merged dataset.
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / output_repo_id

    # Extract repo_ids and roots
    repo_ids = [ds.repo_id for ds in datasets]
    roots = [ds.root for ds in datasets]

    # Call aggregate_datasets
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_dir,
    )

    # Create and return the merged dataset
    merged_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=datasets[0].image_transforms,
        delta_timestamps=datasets[0].delta_timestamps,
        tolerance_s=datasets[0].tolerance_s,
    )

    return merged_dataset


def add_feature(
    dataset: LeRobotDataset,
    feature_name: str,
    feature_values: np.ndarray | torch.Tensor | Callable,
    feature_info: dict,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Add a new feature to a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        feature_name: Name of the new feature.
        feature_values: Either:
            - Array/tensor of shape (num_frames, ...) with values for each frame
            - Callable that takes (frame_dict, episode_index, frame_index) and returns feature value
        feature_info: Dictionary with feature metadata (dtype, shape, names).
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        LeRobotDataset: New dataset with the added feature.
    """
    if feature_name in dataset.meta.features:
        raise ValueError(f"Feature '{feature_name}' already exists in dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # Validate feature_info
    required_keys = {"dtype", "shape"}
    if not required_keys.issubset(feature_info.keys()):
        raise ValueError(f"feature_info must contain keys: {required_keys}")

    # Create new features dict
    new_features = dataset.meta.features.copy()
    new_features[feature_name] = feature_info

    # Create new dataset metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    # Process data with new feature
    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features={feature_name: (feature_values, feature_info)},
    )

    # Copy videos if present
    if dataset.meta.video_keys:
        _copy_videos(dataset, new_meta)

    # Create new dataset instance
    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


def remove_feature(
    dataset: LeRobotDataset,
    feature_names: str | list[str],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Remove features from a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        feature_names: Name(s) of features to remove. Can be a single string or list.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        LeRobotDataset: New dataset with features removed.
    """
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Validate features exist
    for name in feature_names:
        if name not in dataset.meta.features:
            raise ValueError(f"Feature '{name}' not found in dataset")

    # Check if trying to remove required features
    required_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    if any(name in required_features for name in feature_names):
        raise ValueError(f"Cannot remove required features: {required_features}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # Create new features dict
    new_features = {k: v for k, v in dataset.meta.features.items() if k not in feature_names}

    # Check if removing video features
    video_keys_to_remove = [name for name in feature_names if name in dataset.meta.video_keys]

    # Check if videos will remain after removal
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in video_keys_to_remove]

    # Create new dataset metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    # Process data with removed features
    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        remove_features=feature_names,
    )

    # Copy videos (excluding removed ones)
    if new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=video_keys_to_remove)

    # Create new dataset instance
    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


# Helper functions


def _fractions_to_episode_indices(
    total_episodes: int,
    splits: dict[str, float],
) -> dict[str, list[int]]:
    """Convert split fractions to episode indices."""
    if sum(splits.values()) > 1.0:
        raise ValueError("Split fractions must sum to <= 1.0")

    indices = list(range(total_episodes))
    result = {}
    start_idx = 0

    for split_name, fraction in splits.items():
        num_episodes = int(total_episodes * fraction)
        end_idx = start_idx + num_episodes
        if split_name == list(splits.keys())[-1]:  # Last split gets remaining episodes
            end_idx = total_episodes
        result[split_name] = indices[start_idx:end_idx]
        start_idx = end_idx

    return result


def _copy_and_reindex_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> None:
    """Copy data files and reindex episodes."""
    # Get unique data files from episodes to keep
    file_paths = set()
    for old_idx in episode_mapping:
        file_paths.add(src_dataset.meta.get_data_file_path(old_idx))

    # Track global index
    global_index = 0
    chunk_idx, file_idx = 0, 0

    # Process each data file
    for src_path in tqdm(sorted(file_paths), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)

        # Filter to keep only mapped episodes
        mask = df["episode_index"].isin(episode_mapping.keys())
        df = df[mask].copy()

        if len(df) == 0:
            continue

        # Update episode indices
        df["episode_index"] = df["episode_index"].map(episode_mapping)

        # Update global index to be continuous
        df["index"] = range(global_index, global_index + len(df))
        global_index += len(df)

        # Update task indices if needed
        if dst_meta.tasks is None:
            # Get unique tasks from filtered data
            task_indices = df["task_index"].unique()
            tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in task_indices]
            dst_meta.save_episode_tasks(list(set(tasks)))

        # Remap task indices
        task_mapping = {}
        for old_task_idx in df["task_index"].unique():
            task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
            new_task_idx = dst_meta.get_task_index(task_name)
            task_mapping[old_task_idx] = new_task_idx
        df["task_index"] = df["task_index"].map(task_mapping)

        # Save processed data
        chunk_idx, file_idx = _save_data_chunk(df, dst_meta, chunk_idx, file_idx)

    # Process episodes metadata
    _copy_and_reindex_episodes_metadata(src_dataset, dst_meta, episode_mapping)


def _copy_and_reindex_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> None:
    """Copy video files and update metadata."""
    for video_key in src_dataset.meta.video_keys:
        video_files = set()
        for old_idx in episode_mapping:
            video_files.add(src_dataset.meta.get_video_file_path(old_idx, video_key))

        chunk_idx, file_idx = 0, 0

        for src_path in tqdm(sorted(video_files), desc=f"Processing {video_key} videos"):
            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # For simplicity, copy entire video files
            # In production, you might want to extract only relevant segments
            shutil.copy(src_dataset.root / src_path, dst_path)

            # Update indices for next file
            file_size = get_video_size_in_mb(dst_path)
            if file_size >= DEFAULT_VIDEO_FILE_SIZE_IN_MB * 0.9:  # 90% threshold
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)


def _copy_and_reindex_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> None:
    """Copy and reindex episodes metadata."""
    all_stats = []
    frame_offset = 0

    for old_idx, new_idx in tqdm(
        sorted(episode_mapping.items(), key=lambda x: x[1]), desc="Processing episodes metadata"
    ):
        # Get episode from source
        src_episode = src_dataset.meta.episodes[old_idx]

        # Create episode dict
        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],  # Already a list of task names
            "length": src_episode["length"],
        }

        episode_metadata = {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": frame_offset,
            "dataset_to_index": frame_offset + src_episode["length"],
        }

        # Update frame offset for next episode
        frame_offset += src_episode["length"]

        # Copy stats metadata
        for key in src_episode:
            if key.startswith("stats/"):
                episode_dict[key] = src_episode[key]

        # Add episode metadata
        stats_dict = {
            key.replace("stats/", ""): value
            for key, value in episode_dict.items()
            if key.startswith("stats/")
        }
        all_stats.append(stats_dict)

        # Calculate stats from dict
        episode_stats = {}
        for key in dst_meta.features:
            if key in stats_dict:
                episode_stats[key] = stats_dict[key]

        dst_meta.save_episode(
            new_idx, episode_dict["length"], episode_dict["tasks"], episode_stats, episode_metadata
        )

    # Aggregate all stats
    if all_stats:
        aggregated_stats = aggregate_stats(all_stats)
        write_stats(aggregated_stats, dst_meta.root)


def _save_data_chunk(
    df: pd.DataFrame,
    meta: LeRobotDatasetMetadata,
    chunk_idx: int = 0,
    file_idx: int = 0,
) -> tuple[int, int]:
    """Save a data chunk and return updated indices."""
    path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(meta.image_keys) > 0:
        to_parquet_with_hf_images(df, path)
    else:
        df.to_parquet(path)

    # Check if we need to rotate files
    file_size = get_parquet_file_size_in_mb(path)
    if file_size >= DEFAULT_DATA_FILE_SIZE_IN_MB * 0.9:  # 90% threshold
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    return chunk_idx, file_idx


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
) -> None:
    """Copy data while adding or removing features."""
    # Get all unique data files
    file_paths = set()
    for ep_idx in range(dataset.meta.total_episodes):
        file_paths.add(dataset.meta.get_data_file_path(ep_idx))

    frame_idx = 0

    # Process each data file
    for src_path in tqdm(sorted(file_paths), desc="Processing data files"):
        df = pd.read_parquet(dataset.root / src_path)

        # Remove features
        if remove_features:
            df = df.drop(columns=remove_features, errors="ignore")

        # Add features
        if add_features:
            for feature_name, (values, _) in add_features.items():
                if callable(values):
                    # Compute values for each frame
                    feature_values = []
                    for _, row in df.iterrows():
                        ep_idx = row["episode_index"]
                        frame_in_ep = row["frame_index"]
                        value = values(row.to_dict(), ep_idx, frame_in_ep)
                        # Convert numpy arrays to scalars for single-element arrays
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        feature_values.append(value)
                    df[feature_name] = feature_values
                else:
                    # Use provided values
                    end_idx = frame_idx + len(df)
                    # Convert to list to ensure proper shape handling
                    feature_slice = values[frame_idx:end_idx]
                    if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                        # Flatten single-element arrays to scalars for pandas
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = feature_slice
                    frame_idx = end_idx

        # Save chunk
        _save_data_chunk(df, new_meta)

    # Copy episodes metadata and update stats
    _copy_episodes_metadata_and_stats(dataset, new_meta)


def _copy_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    exclude_keys: list[str] | None = None,
) -> None:
    """Copy video files, optionally excluding certain keys."""
    if exclude_keys is None:
        exclude_keys = []

    for video_key in src_dataset.meta.video_keys:
        if video_key in exclude_keys:
            continue

        # Get all video files for this key
        video_files = set()
        for ep_idx in range(src_dataset.meta.total_episodes):
            video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))

        # Copy video files
        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            # Maintain same structure
            rel_path = src_path.relative_to(src_dataset.root)
            dst_path = dst_meta.root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_episodes_metadata_and_stats(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
) -> None:
    """Copy episodes metadata and recalculate stats."""
    # Copy tasks
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    # Copy episodes metadata files
    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    # Update info
    dst_meta.info.update(
        {
            "total_episodes": src_dataset.meta.total_episodes,
            "total_frames": src_dataset.meta.total_frames,
            "total_tasks": src_dataset.meta.total_tasks,
            "splits": src_dataset.meta.info.get("splits", {"train": f"0:{src_dataset.meta.total_episodes}"}),
        }
    )

    # Update video info if needed
    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            if key in src_dataset.meta.features:
                dst_meta.info["features"][key]["info"] = src_dataset.meta.info["features"][key].get(
                    "info", {}
                )

    write_info(dst_meta.info, dst_meta.root)

    # Recalculate stats if features changed
    if set(dst_meta.features.keys()) != set(src_dataset.meta.features.keys()):
        # Need to recalculate stats
        logging.info("Recalculating dataset statistics...")
        # This is a simplified version - in production you'd want to properly recalculate
        if src_dataset.meta.stats:
            new_stats = {}
            for key in dst_meta.features:
                if key in src_dataset.meta.stats:
                    new_stats[key] = src_dataset.meta.stats[key]
            write_stats(new_stats, dst_meta.root)
    else:
        # Copy existing stats
        if src_dataset.meta.stats:
            write_stats(src_dataset.meta.stats, dst_meta.root)
