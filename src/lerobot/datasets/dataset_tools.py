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

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    get_parquet_file_size_in_mb,
    load_episodes,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


def _load_episode_with_stats(src_dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Load a single episode's metadata including stats from parquet file.

    Args:
        src_dataset: Source dataset
        episode_idx: Episode index to load

    Returns:
        dict containing episode metadata and stats
    """
    ep_meta = src_dataset.meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = src_dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]

    return episode_row.to_dict()


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
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    logging.info(f"Deleting {len(episode_indices)} episodes from dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from dataset")

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

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
    splits: dict[str, float | list[int]],
    output_dir: str | Path | None = None,
) -> dict[str, LeRobotDataset]:
    """Split a LeRobotDataset into multiple smaller datasets.

    Args:
        dataset: The source LeRobotDataset to split.
        splits: Either a dict mapping split names to episode indices, or a dict mapping
                split names to fractions (must sum to <= 1.0).
        output_dir: Base directory for output datasets. If None, uses default location.

    Examples:
      Split by specific episodes
        splits = {"train": [0, 1, 2], "val": [3, 4]}
        datasets = split_dataset(dataset, splits)

      Split by fractions
        splits = {"train": 0.8, "val": 0.2}
        datasets = split_dataset(dataset, splits)
    """
    if not splits:
        raise ValueError("No splits provided")

    if all(isinstance(v, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(dataset.meta.total_episodes, splits)

    all_episodes = set()
    for split_name, episodes in splits.items():
        if not episodes:
            raise ValueError(f"Split '{split_name}' has no episodes")
        episode_set = set(episodes)
        if episode_set & all_episodes:
            raise ValueError("Episodes cannot appear in multiple splits")
        all_episodes.update(episode_set)

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = all_episodes - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if output_dir is not None:
        output_dir = Path(output_dir)

    result_datasets = {}

    for split_name, episodes in splits.items():
        logging.info(f"Creating split '{split_name}' with {len(episodes)} episodes")

        split_repo_id = f"{dataset.repo_id}_{split_name}"

        split_output_dir = (
            output_dir / split_name if output_dir is not None else HF_LEROBOT_HOME / split_repo_id
        )

        episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes))}

        new_meta = LeRobotDatasetMetadata.create(
            repo_id=split_repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
            root=split_output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        video_metadata = None
        if dataset.meta.video_keys:
            video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

        data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

        _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

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
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / output_repo_id

    repo_ids = [ds.repo_id for ds in datasets]
    roots = [ds.root for ds in datasets]

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_dir,
    )

    merged_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=datasets[0].image_transforms,
        delta_timestamps=datasets[0].delta_timestamps,
        tolerance_s=datasets[0].tolerance_s,
    )

    return merged_dataset


def modify_features(
    dataset: LeRobotDataset,
    add_features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]] | None = None,
    remove_features: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Modify a LeRobotDataset by adding and/or removing features in a single pass.

    This is the most efficient way to modify features, as it only copies the dataset once
    regardless of how many features are being added or removed.

    Args:
        dataset: The source LeRobotDataset.
        add_features: Optional dict mapping feature names to (feature_values, feature_info) tuples.
        remove_features: Optional feature name(s) to remove. Can be a single string or list.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with features modified.

    Example:
        new_dataset = modify_features(
            dataset,
            add_features={
                "reward": (reward_array, {"dtype": "float32", "shape": [1], "names": None}),
            },
            remove_features=["old_feature"],
            output_dir="./output",
        )
    """
    if add_features is None and remove_features is None:
        raise ValueError("Must specify at least one of add_features or remove_features")

    remove_features_list: list[str] = []
    if remove_features is not None:
        remove_features_list = [remove_features] if isinstance(remove_features, str) else remove_features

    if add_features:
        required_keys = {"dtype", "shape"}
        for feature_name, (_, feature_info) in add_features.items():
            if feature_name in dataset.meta.features:
                raise ValueError(f"Feature '{feature_name}' already exists in dataset")

            if not required_keys.issubset(feature_info.keys()):
                raise ValueError(f"feature_info for '{feature_name}' must contain keys: {required_keys}")

    if remove_features_list:
        for name in remove_features_list:
            if name not in dataset.meta.features:
                raise ValueError(f"Feature '{name}' not found in dataset")

        required_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if any(name in required_features for name in remove_features_list):
            raise ValueError(f"Cannot remove required features: {required_features}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_features = dataset.meta.features.copy()

    if remove_features_list:
        for name in remove_features_list:
            new_features.pop(name, None)

    if add_features:
        for feature_name, (_, feature_info) in add_features.items():
            new_features[feature_name] = feature_info

    video_keys_to_remove = [name for name in remove_features_list if name in dataset.meta.video_keys]
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in video_keys_to_remove]

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features=add_features,
        remove_features=remove_features_list if remove_features_list else None,
    )

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=video_keys_to_remove if video_keys_to_remove else None)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


def add_features(
    dataset: LeRobotDataset,
    features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Add multiple features to a LeRobotDataset in a single pass.

    This is more efficient than calling add_feature() multiple times, as it only
    copies the dataset once regardless of how many features are being added.

    Args:
        dataset: The source LeRobotDataset.
        features: Dictionary mapping feature names to (feature_values, feature_info) tuples.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with all features added.

    Example:
        features = {
            "task_embedding": (task_emb_array, {"dtype": "float32", "shape": [384], "names": None}),
            "cam1_embedding": (cam1_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
            "cam2_embedding": (cam2_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
        }
        new_dataset = add_features(dataset, features, output_dir="./output", repo_id="my_dataset")
    """
    if not features:
        raise ValueError("No features provided")

    return modify_features(
        dataset=dataset,
        add_features=features,
        remove_features=None,
        output_dir=output_dir,
        repo_id=repo_id,
    )


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
        New dataset with features removed.
    """
    return modify_features(
        dataset=dataset,
        add_features=None,
        remove_features=feature_names,
        output_dir=output_dir,
        repo_id=repo_id,
    )


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
        if num_episodes == 0:
            logging.warning(f"Split '{split_name}' has no episodes, skipping...")
            continue
        end_idx = start_idx + num_episodes
        if split_name == list(splits.keys())[-1]:
            end_idx = total_episodes
        result[split_name] = indices[start_idx:end_idx]
        start_idx = end_idx

    return result


def _copy_and_reindex_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """Copy and filter data files, only modifying files with deleted episodes.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its data file metadata (chunk_index, file_index, etc.)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    file_to_episodes: dict[Path, set[int]] = {}
    for old_idx in episode_mapping:
        file_path = src_dataset.meta.get_data_file_path(old_idx)
        if file_path not in file_to_episodes:
            file_to_episodes[file_path] = set()
        file_to_episodes[file_path].add(old_idx)

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    if dst_meta.tasks is None:
        all_task_indices = set()
        for src_path in file_to_episodes:
            df = pd.read_parquet(src_dataset.root / src_path)
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            task_series: pd.Series = df[mask]["task_index"]
            all_task_indices.update(task_series.unique().tolist())
        tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in all_task_indices]
        dst_meta.save_episode_tasks(list(set(tasks)))

    task_mapping = {}
    for old_task_idx in range(len(src_dataset.meta.tasks)):
        task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)

        all_episodes_in_file = set(df["episode_index"].unique())
        episodes_to_keep = file_to_episodes[src_path]

        if all_episodes_in_file == episodes_to_keep:
            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]
        else:
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            df = df[mask].copy().reset_index(drop=True)

            if len(df) == 0:
                continue

            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, dst_meta)

        for ep_old_idx in episodes_to_keep:
            ep_new_idx = episode_mapping[ep_old_idx]
            ep_df = df[df["episode_index"] == ep_new_idx]
            episode_data_metadata[ep_new_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(df)

    return episode_data_metadata


def _keep_episodes_from_video_with_av(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[float, float]],
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Keep only specified episodes from a video file using PyAV.

    This function decodes frames from specified time ranges and re-encodes them with
    properly reset timestamps to ensure monotonic progression.

    Args:
        input_path: Source video file path.
        output_path: Destination video file path.
        episodes_to_keep: List of (start_time, end_time) tuples for episodes to keep.
        fps: Frame rate of the video.
        vcodec: Video codec to use for encoding.
        pix_fmt: Pixel format for output video.
    """
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))

    # Check if video stream exists.
    if not in_container.streams.video:
        raise ValueError(
            f"No video streams found in {input_path}. "
            "The video file may be corrupted or empty. "
            "Try re-downloading the dataset or checking the video file."
        )

    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")

    # Convert fps to Fraction for PyAV compatibility.
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream(vcodec, rate=fps_fraction)

    # PyAV type stubs don't distinguish video streams from audio/subtitle streams.
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt

    # Set time_base to match the frame rate for proper timestamp handling.
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    # Create set of (start, end) ranges for fast lookup.
    # Convert to a sorted list for efficient checking.
    time_ranges = sorted(episodes_to_keep)

    # Track frame index for setting PTS and current range being processed.
    frame_count = 0
    range_idx = 0

    # Read through entire video once and filter frames.
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # Get frame timestamp.
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

            # Check if frame is in any of our desired time ranges.
            # Skip ranges that have already passed.
            while range_idx < len(time_ranges) and frame_time >= time_ranges[range_idx][1]:
                range_idx += 1

            # If we've passed all ranges, stop processing.
            if range_idx >= len(time_ranges):
                break

            # Check if frame is in current range.
            start_ts, end_ts = time_ranges[range_idx]
            if frame_time < start_ts:
                continue

            # Frame is in range - create a new frame with reset timestamps.
            # We need to create a copy to avoid modifying the original.
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            # Encode and mux the frame.
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            frame_count += 1

    # Flush encoder.
    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def _copy_and_reindex_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> dict[int, dict]:
    """Copy and filter video files, only re-encoding files with deleted episodes.

    For video files that only contain kept episodes, we copy them directly.
    For files with mixed kept/deleted episodes, we use PyAV filters to efficiently
    re-encode only the desired segments.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its video metadata (chunk_index, file_index, timestamps)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}

    for video_key in src_dataset.meta.video_keys:
        logging.info(f"Processing videos for {video_key}")

        if dst_meta.video_path is None:
            raise ValueError("Destination metadata has no video_path defined")

        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for old_idx in episode_mapping:
            src_ep = src_dataset.meta.episodes[old_idx]
            chunk_idx = src_ep[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep[f"videos/{video_key}/file_index"]
            file_key = (chunk_idx, file_idx)
            if file_key not in file_to_episodes:
                file_to_episodes[file_key] = []
            file_to_episodes[file_key].append(old_idx)

        for (src_chunk_idx, src_file_idx), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()), desc=f"Processing {video_key} video files"
        ):
            all_episodes_in_file = [
                ep_idx
                for ep_idx in range(src_dataset.meta.total_episodes)
                if src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/chunk_index") == src_chunk_idx
                and src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/file_index") == src_file_idx
            ]

            episodes_to_keep_set = set(episodes_in_file)
            all_in_file_set = set(all_episodes_in_file)

            if all_in_file_set == episodes_to_keep_set:
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video_path, dst_video_path)

                for old_idx in episodes_in_file:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                        f"videos/{video_key}/from_timestamp"
                    ]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                        f"videos/{video_key}/to_timestamp"
                    ]
            else:
                # Build list of time ranges to keep, in sorted order.
                sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
                episodes_to_keep_ranges: list[tuple[float, float]] = []

                for old_idx in sorted_keep_episodes:
                    src_ep = src_dataset.meta.episodes[old_idx]
                    from_ts = src_ep[f"videos/{video_key}/from_timestamp"]
                    to_ts = src_ep[f"videos/{video_key}/to_timestamp"]
                    episodes_to_keep_ranges.append((from_ts, to_ts))

                # Use PyAV filters to efficiently re-encode only the desired segments.
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(
                    f"Re-encoding {video_key} (chunk {src_chunk_idx}, file {src_file_idx}) "
                    f"with {len(episodes_to_keep_ranges)} episodes"
                )
                _keep_episodes_from_video_with_av(
                    src_video_path,
                    dst_video_path,
                    episodes_to_keep_ranges,
                    src_dataset.meta.fps,
                    vcodec,
                    pix_fmt,
                )

                cumulative_ts = 0.0
                for old_idx in sorted_keep_episodes:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    ep_length = src_ep["length"]
                    ep_duration = ep_length / src_dataset.meta.fps

                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = cumulative_ts
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = (
                        cumulative_ts + ep_duration
                    )

                    cumulative_ts += ep_duration

    return episodes_video_metadata


def _copy_and_reindex_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    data_metadata: dict[int, dict],
    video_metadata: dict[int, dict] | None = None,
) -> None:
    """Copy and reindex episodes metadata using provided data and video metadata.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices
        data_metadata: Dict mapping new episode index to its data file metadata
        video_metadata: Optional dict mapping new episode index to its video metadata
    """
    from lerobot.datasets.utils import flatten_dict

    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    all_stats = []
    total_frames = 0

    for old_idx, new_idx in tqdm(
        sorted(episode_mapping.items(), key=lambda x: x[1]), desc="Processing episodes metadata"
    ):
        src_episode_full = _load_episode_with_stats(src_dataset, old_idx)

        src_episode = src_dataset.meta.episodes[old_idx]

        episode_meta = data_metadata[new_idx].copy()

        if video_metadata and new_idx in video_metadata:
            episode_meta.update(video_metadata[new_idx])

        # Extract episode statistics from parquet metadata.
        # Note (maractingi): When pandas/pyarrow serializes numpy arrays with shape (3, 1, 1) to parquet,
        # they are being deserialized as nested object arrays like:
        #   array([array([array([0.])]), array([array([0.])]), array([array([0.])])])
        # This happens particularly with image/video statistics. We need to detect and flatten
        # these nested structures back to proper (3, 1, 1) arrays so aggregate_stats can process them.
        episode_stats = {}
        for key in src_episode_full:
            if key.startswith("stats/"):
                stat_key = key.replace("stats/", "")
                parts = stat_key.split("/")
                if len(parts) == 2:
                    feature_name, stat_name = parts
                    if feature_name not in episode_stats:
                        episode_stats[feature_name] = {}

                    value = src_episode_full[key]

                    if feature_name in src_dataset.meta.features:
                        feature_dtype = src_dataset.meta.features[feature_name]["dtype"]
                        if feature_dtype in ["image", "video"] and stat_name != "count":
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                flat_values = []
                                for item in value:
                                    while isinstance(item, np.ndarray):
                                        item = item.flatten()[0]
                                    flat_values.append(item)
                                value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                            elif isinstance(value, np.ndarray) and value.shape == (3,):
                                value = value.reshape(3, 1, 1)

                    episode_stats[feature_name][stat_name] = value

        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],
            "length": src_episode["length"],
        }
        episode_dict.update(episode_meta)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        dst_meta._save_episode_metadata(episode_dict)

        total_frames += src_episode["length"]

    dst_meta._close_writer()

    dst_meta.info.update(
        {
            "total_episodes": len(episode_mapping),
            "total_frames": total_frames,
            "total_tasks": len(dst_meta.tasks) if dst_meta.tasks is not None else 0,
            "splits": {"train": f"0:{len(episode_mapping)}"},
        }
    )
    write_info(dst_meta.info, dst_meta.root)

    if not all_stats:
        logging.warning("No statistics found to aggregate")
        return

    logging.info(f"Aggregating statistics for {len(all_stats)} episodes")
    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in dst_meta.features}
    write_stats(filtered_stats, dst_meta.root)


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """Write DataFrame to parquet

    This ensures images are properly embedded and the file can be loaded correctly by HF datasets.
    """
    from lerobot.datasets.utils import embed_images, get_hf_features_from_features

    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _save_data_chunk(
    df: pd.DataFrame,
    meta: LeRobotDatasetMetadata,
    chunk_idx: int = 0,
    file_idx: int = 0,
) -> tuple[int, int, dict[int, dict]]:
    """Save a data chunk and return updated indices and episode metadata.

    Returns:
        tuple: (next_chunk_idx, next_file_idx, episode_metadata_dict)
            where episode_metadata_dict maps episode_index to its data file metadata
    """
    path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet(df, path, meta)

    episode_metadata = {}
    for ep_idx in df["episode_index"].unique():
        ep_df = df[df["episode_index"] == ep_idx]
        episode_metadata[ep_idx] = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": int(ep_df["index"].min()),
            "dataset_to_index": int(ep_df["index"].max() + 1),
        }

    file_size = get_parquet_file_size_in_mb(path)
    if file_size >= DEFAULT_DATA_FILE_SIZE_IN_MB * 0.9:
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    return chunk_idx, file_idx, episode_metadata


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
) -> None:
    """Copy data while adding or removing features."""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        if remove_features:
            df = df.drop(columns=remove_features, errors="ignore")

        if add_features:
            end_idx = frame_idx + len(df)
            for feature_name, (values, _) in add_features.items():
                if callable(values):
                    feature_values = []
                    for _, row in df.iterrows():
                        ep_idx = row["episode_index"]
                        frame_in_ep = row["frame_index"]
                        value = values(row.to_dict(), ep_idx, frame_in_ep)
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        feature_values.append(value)
                    df[feature_name] = feature_values
                else:
                    feature_slice = values[frame_idx:end_idx]
                    if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = feature_slice
            frame_idx = end_idx

        # Write using the same chunk/file structure as source
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

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

        video_files = set()
        for ep_idx in range(len(src_dataset.meta.episodes)):
            try:
                video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))
            except KeyError:
                continue

        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            dst_path = dst_meta.root / src_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_episodes_metadata_and_stats(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
) -> None:
    """Copy episodes metadata and recalculate stats."""
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    dst_meta.info.update(
        {
            "total_episodes": src_dataset.meta.total_episodes,
            "total_frames": src_dataset.meta.total_frames,
            "total_tasks": src_dataset.meta.total_tasks,
            "splits": src_dataset.meta.info.get("splits", {"train": f"0:{src_dataset.meta.total_episodes}"}),
        }
    )

    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            if key in src_dataset.meta.features:
                dst_meta.info["features"][key]["info"] = src_dataset.meta.info["features"][key].get(
                    "info", {}
                )

    write_info(dst_meta.info, dst_meta.root)

    if set(dst_meta.features.keys()) != set(src_dataset.meta.features.keys()):
        logging.info("Recalculating dataset statistics...")
        if src_dataset.meta.stats:
            new_stats = {}
            for key in dst_meta.features:
                if key in src_dataset.meta.stats:
                    new_stats[key] = src_dataset.meta.stats[key]
            write_stats(new_stats, dst_meta.root)
    else:
        if src_dataset.meta.stats:
            write_stats(src_dataset.meta.stats, dst_meta.root)
