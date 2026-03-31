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

import concurrent
import gc
import logging
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats, compute_mp4_video_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    get_parquet_file_size_in_mb,
    load_episodes,
    load_stats,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import (
    _get_codec_options,
    concatenate_video_files,
    encode_video_frames,
    get_video_duration_in_s,
    get_video_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_IMAGE


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
                "video": (video_path, {"dtype": "video", "shape": [96, 96, 3], "video_info": ...})
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

    video_keys_to_add = [k for k, v in (add_features or {}).items() if v[1]["dtype"] == "video"]
    remaining_video_keys += video_keys_to_add

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    _copy_episodes_metadata_and_stats(dataset, new_meta)

    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features=add_features,
        remove_features=remove_features_list if remove_features_list else None,
    )

    if new_meta.video_keys:
        new_meta.load_metadata()
        _copy_videos_with_feature_changes(
            dataset,
            new_meta,
            add_features=add_features,
            exclude_keys=video_keys_to_remove if video_keys_to_remove else None,
        )

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
    episodes_to_keep: list[tuple[int, int]],
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Keep only specified episodes from a video file using PyAV.

    This function decodes frames from specified frame ranges and re-encodes them with
    properly reset timestamps to ensure monotonic progression.

    Args:
        input_path: Source video file path.
        output_path: Destination video file path.
        episodes_to_keep: List of (start_frame, end_frame) tuples for episodes to keep.
            Ranges are half-open intervals: [start_frame, end_frame), where start_frame
            is inclusive and end_frame is exclusive.
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
    frame_ranges = sorted(episodes_to_keep)

    # Track frame index for setting PTS and current range being processed.
    src_frame_count = 0
    frame_count = 0
    range_idx = 0

    # Read through entire video once and filter frames.
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # Check if frame is in any of our desired frame ranges.
            # Skip ranges that have already passed.
            while range_idx < len(frame_ranges) and src_frame_count >= frame_ranges[range_idx][1]:
                range_idx += 1

            # If we've passed all ranges, stop processing.
            if range_idx >= len(frame_ranges):
                break

            # Check if frame is in current range.
            start_frame = frame_ranges[range_idx][0]

            if src_frame_count < start_frame:
                src_frame_count += 1
                continue

            # Frame is in range - create a new frame with reset timestamps.
            # We need to create a copy to avoid modifying the original.
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            # Encode and mux the frame.
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            src_frame_count += 1
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
                # Build list of frame ranges to keep, in sorted order.
                sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
                episodes_to_keep_ranges: list[tuple[int, int]] = []
                for old_idx in sorted_keep_episodes:
                    src_ep = src_dataset.meta.episodes[old_idx]
                    from_frame = round(src_ep[f"videos/{video_key}/from_timestamp"] * src_dataset.meta.fps)
                    to_frame = round(src_ep[f"videos/{video_key}/to_timestamp"] * src_dataset.meta.fps)
                    assert src_ep["length"] == to_frame - from_frame, (
                        f"Episode length mismatch: {src_ep['length']} vs {to_frame - from_frame}"
                    )
                    episodes_to_keep_ranges.append((from_frame, to_frame))

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
    update_file: bool = True,
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
    if file_size >= meta.data_files_size_in_mb * 0.9 and update_file:
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, meta.chunks_size)

    return chunk_idx, file_idx, episode_metadata


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
) -> None:
    """Copy data while adding or removing features, dynamically chunking based on size limits."""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    chunk_idx = 0
    file_idx = 0
    current_chunk_dfs = []

    current_frames_in_buffer = 0
    bytes_per_frame = None

    all_episodes_metadata = {}
    all_episodes_stats = []

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        for ep_idx in tqdm(
            df["episode_index"].unique(), desc=f"Processing episodes of {src_path.stem}", leave=False
        ):
            # slice first (minimal memory footprint)
            ep_df = df[df["episode_index"] == ep_idx].copy()

            # modify features on the slice
            if remove_features:
                ep_df = ep_df.drop(columns=remove_features, errors="ignore")

            if add_features:
                episode_data_for_stats = {}
                features_for_stats = {}
                end_idx = frame_idx + len(ep_df)
                for feature_name, (values, feature_info) in add_features.items():
                    if feature_info["dtype"] in ["video", "image"]:
                        continue
                    if callable(values):
                        feature_values = []
                        for _, row in ep_df.iterrows():
                            val = values(row.to_dict(), ep_idx, row["frame_index"])
                            if isinstance(val, np.ndarray) and val.size == 1:
                                val = val.item()
                            feature_values.append(val)
                        ep_df[feature_name] = feature_values
                    else:
                        feature_slice = values[frame_idx:end_idx]
                        if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                            # (100, 1)    -> (100,)
                            # (10, 1, 5)  -> (10, 5)
                            ep_df[feature_name] = list(feature_slice.squeeze(axis=1))
                        else:
                            ep_df[feature_name] = list(feature_slice)

                    episode_data_for_stats[feature_name] = np.stack(ep_df[feature_name].values)
                    features_for_stats[feature_name] = feature_info

                if episode_data_for_stats:
                    episode_feature_stats = compute_episode_stats(
                        episode_data=episode_data_for_stats, features=features_for_stats
                    )
                    all_episodes_stats.append(episode_feature_stats)

                frame_idx = end_idx

            if bytes_per_frame is None and len(ep_df) > 0:
                # get the one row dimension of the dataset to estimate the size
                bytes_per_frame = 0
                single_row_df = ep_df.iloc[0:1]

                for col in ep_df.columns:
                    val = single_row_df[col].iloc[0]

                    # if the object knows its exact C-buffer size, use nbytes
                    if hasattr(val, "nbytes"):
                        bytes_per_frame += val.nbytes
                    else:
                        # else, fallback on pandas deep-scan
                        bytes_per_frame += single_row_df[[col]].memory_usage(deep=True).sum()

            current_chunk_dfs.append(ep_df)
            current_frames_in_buffer += len(ep_df)

            total_mem_bytes = current_frames_in_buffer * (bytes_per_frame or 0)
            estimated_mb = total_mem_bytes / (1024 * 1024)

            # flush if too big
            if estimated_mb >= new_meta.data_files_size_in_mb * 0.9:
                combined_df = pd.concat(current_chunk_dfs, ignore_index=True)

                # _save_data_chunk natively writes the file and iterates the file_idx for the next loop
                chunk_idx, file_idx, ep_meta = _save_data_chunk(
                    combined_df, new_meta, chunk_idx, file_idx, update_file=False
                )
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, new_meta.chunks_size)

                all_episodes_metadata.update(ep_meta)

                # nuclear deallocation
                del combined_df
                current_chunk_dfs.clear()
                current_frames_in_buffer = 0
                gc.collect()

        # nuclear deallocation
        del df
        gc.collect()

    # flush any remaining episodes left in the buffer at the end
    if current_chunk_dfs:
        combined_df = pd.concat(current_chunk_dfs, ignore_index=True)
        chunk_idx, file_idx, ep_meta = _save_data_chunk(combined_df, new_meta, chunk_idx, file_idx)
        all_episodes_metadata.update(ep_meta)

        del combined_df
        gc.collect()

    logging.info("Updating episode metadata to reflect new data chunking")
    episodes_ds = load_episodes(new_meta.root)
    episodes_ds = episodes_ds.map(lambda row: {**row, **all_episodes_metadata.get(row["episode_index"], {})})
    new_meta.episodes = episodes_ds
    write_episodes(episodes_ds, new_meta.root)

    logging.info("Updating new features stats")
    all_episodes_stats = aggregate_stats(all_episodes_stats)
    stats = load_stats(new_meta.root)
    stats.update(all_episodes_stats)
    new_meta.stats = stats
    write_stats(stats, new_meta.root)


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

    logging.info(f"{', '.join([k for k in src_dataset.meta.video_keys if k not in exclude_keys])} copied")
    logging.info(f"{', '.join(exclude_keys)} not copied")


def _encode_episode_slice_worker(
    source_path: str,
    out_path: str,
    start_frame: int,
    end_frame: int,
    target_fps: int,
    target_vcodec: str,
    video_options: dict,
    pix_fmt: str,
) -> None:
    """Worker function to encode a single episode slice in a separate process."""
    # Silence PyAV within this specific process
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    frames_to_write = end_frame - start_frame
    start_time_s = start_frame / target_fps

    with av.open(out_path, "w") as output_container:
        output_stream = output_container.add_stream(target_vcodec, target_fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.time_base = Fraction(1, target_fps)

        with av.open(source_path, "r") as input_container:
            input_stream = input_container.streams.video[0]
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height

            seek_target = int(start_time_s * av.time_base)
            input_container.seek(seek_target, backward=True, any_frame=False)

            frames_written = 0

            for frame in input_container.decode(input_stream):
                frame_time_s = float(frame.time) if frame.time is not None else 0.0

                if frame_time_s < start_time_s - 1e-4:
                    continue

                frame.pts = frames_written
                frame.time_base = Fraction(1, target_fps)

                packets = output_stream.encode(frame)
                for packet in packets:
                    output_container.mux(packet)

                frames_written += 1
                if frames_written >= frames_to_write:
                    break

        packets = output_stream.encode()
        for packet in packets:
            output_container.mux(packet)


def _copy_videos_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    exclude_keys: list[str] | None = None,
    encoder_threads: int | None = None,
    max_workers: int | None = None,
    fast_decode: int = 0,
    vcodec: str | None = None,
    pix_fmt: str = "yuv420p",
) -> None:
    """Copy video files, optionally excluding certain keys."""
    logging.info("Copying existing video features in the dataset")
    _copy_videos(dataset, new_meta, exclude_keys)
    os.environ["SVT_LOG"] = "1"

    logging.info("Adding new video features in the dataset")
    if add_features is None:
        return
    add_video_features = {fn: (p, fi) for fn, (p, fi) in add_features.items() if fi["dtype"] == "video"}
    if len(add_video_features) == 0:
        logging.info("No video features to add")
        return

    tmp_videos_root = new_meta.root / "tmp_videos"

    target_fps = dataset.fps
    target_vcodec = dataset.vcodec if dataset.vcodec is not None else vcodec
    video_options = _get_codec_options(target_vcodec)

    if fast_decode:
        key = "svtav1-params" if target_vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if target_vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    logging.getLogger("libav").setLevel(av.logging.ERROR)

    max_threads_per_encoder = 2
    max_concurrent_workers = 12

    total_cores = os.cpu_count() or 4

    if max_workers is None and encoder_threads is None:
        # if not set, prioritize workers over encoder (throughput over single video parallelization)
        encoder_threads = max_threads_per_encoder
        max_workers = max(1, min(total_cores // encoder_threads, max_concurrent_workers))

    elif max_workers is None and encoder_threads is not None:
        max_workers = max(1, min(total_cores // encoder_threads, max_concurrent_workers))

    elif max_workers is not None and encoder_threads is None:
        encoder_threads = max(1, min(total_cores // max_workers, max_threads_per_encoder))

    if encoder_threads is not None:
        if target_vcodec == "libsvtav1":
            lp_param = f"lp={encoder_threads}"
            if "svtav1-params" in video_options:
                video_options["svtav1-params"] += f":{lp_param}"
            else:
                video_options["svtav1-params"] = lp_param
        else:
            video_options["threads"] = str(encoder_threads)

    if (target_vcodec == "libsvtav1" or target_vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {target_vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    max_size_mb = dataset.meta.video_files_size_in_mb
    chunks_size = dataset.meta.chunks_size

    num_episodes = len(dataset.meta.episodes)

    for feature_name, (path, _) in add_video_features.items():
        ep_paths = []

        # create tmp videos directory
        tmp_videos = tmp_videos_root / feature_name
        tmp_videos.mkdir(parents=True, exist_ok=True)

        # create per episode sliced videos using multiprocessing
        futures = []

        # create per episode sliced videos
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for episode in dataset.meta.episodes:
                out_path = tmp_videos / f"episode_{episode['episode_index']}.mp4"
                ep_paths.append(out_path)

                start_frame = episode["dataset_from_index"]
                end_frame = episode["dataset_to_index"]

                future = executor.submit(
                    _encode_episode_slice_worker,
                    str(path),
                    str(out_path),
                    start_frame,
                    end_frame,
                    target_fps,
                    target_vcodec,
                    video_options,
                    pix_fmt,
                )
                futures.append(future)

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Creating tmp per episode videos of {feature_name}",
            ):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error slicing an episode: {e}")
                    raise

        # used to create episodes metadata
        col_chunk = [0] * num_episodes
        col_file = [0] * num_episodes
        col_from = [0.0] * num_episodes
        col_to = [0.0] * num_episodes

        chunk_idx = 0
        file_idx = 0

        cum_size_mb = 0.0
        start_idx = 0
        from_timestep = 0

        all_episodes_stats = []

        for ep_idx in tqdm(range(num_episodes), desc=f"Organizing {feature_name} videos in chunk and files"):
            ep_path = tmp_videos / f"episode_{ep_idx}.mp4"
            ep_size_mb = os.path.getsize(ep_path) / (1024 * 1024)

            # if is the first iteration or the current ep_file fit in the file-xxx
            if cum_size_mb == 0 or cum_size_mb + ep_size_mb <= max_size_mb:  # and ep_idx != (num_episodes-1):
                # update cum_size in order to do the same check for the following episode
                cum_size_mb += ep_size_mb

                col_from[ep_idx] = from_timestep
                col_to[ep_idx] = from_timestep + get_video_duration_in_s(ep_path)
                from_timestep = col_to[ep_idx]
            # the current ep_file do not fit in the file-xxx
            else:
                # concatenate and save videos until ep_idx (excluded)
                out_path = new_meta.root / new_meta.video_path.format(
                    video_key=feature_name, chunk_index=chunk_idx, file_index=file_idx
                )
                concatenate_video_files(ep_paths[start_idx:ep_idx], out_path)
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunks_size)
                # set the cumulative size to include the current ep_file
                cum_size_mb = ep_size_mb
                # start the following concatenation from the current ep_file
                start_idx = ep_idx

                col_from[ep_idx] = 0.0
                col_to[ep_idx] = get_video_duration_in_s(ep_path)
                from_timestep = col_to[ep_idx]

            col_file[ep_idx] = file_idx
            col_chunk[ep_idx] = chunk_idx

            stats = compute_mp4_video_stats(ep_path)
            all_episodes_stats.append({feature_name: stats})

        # save the last videos
        out_path = new_meta.root / new_meta.video_path.format(
            video_key=feature_name, chunk_index=chunk_idx, file_index=file_idx
        )
        concatenate_video_files(ep_paths[start_idx:num_episodes], out_path)

        logging.info(f"Adding {feature_name} index information into new dataset metadata")
        # update episodes
        episodes_ds = new_meta.episodes
        episodes_ds = episodes_ds.add_column(f"videos/{feature_name}/chunk_index", col_chunk)
        episodes_ds = episodes_ds.add_column(f"videos/{feature_name}/file_index", col_file)
        episodes_ds = episodes_ds.add_column(f"videos/{feature_name}/from_timestamp", col_from)
        episodes_ds = episodes_ds.add_column(f"videos/{feature_name}/to_timestamp", col_to)
        new_meta.episodes = episodes_ds

        shutil.rmtree(tmp_videos)

    shutil.rmtree(tmp_videos_root)

    # restore logging
    av.logging.restore_default_callback()

    logging.info("Updating episode metadata to reflect new data chunking")
    write_episodes(new_meta.episodes, new_meta.root)

    logging.info("Updating new features stats")
    final_stats = aggregate_stats(all_episodes_stats)
    stats = load_stats(new_meta.root)
    stats.update(final_stats)
    new_meta.stats = stats
    write_stats(stats, new_meta.root)


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


def _save_episode_images_for_video(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_index: int,
    num_workers: int = 4,
) -> None:
    """Save images from a specific episode and camera to disk for video encoding.

    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        img_key: The image key (camera) to extract
        episode_index: Index of the episode to save
        num_workers: Number of threads for parallel image saving
    """
    # Create directory
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset without torch format for PIL image access
    hf_dataset = dataset.hf_dataset.with_format(None)

    # Select only this camera's images
    imgs_dataset = hf_dataset.select_columns(img_key)

    # Get episode start and end indices
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Get all items for this episode
    episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

    # Define function to save a single image
    def save_single_image(i_item_tuple):
        i, item = i_item_tuple
        img = item[img_key]
        # Use frame-XXXXXX.png format to match encode_video_frames expectations
        img.save(str(imgs_dir / f"frame-{i:06d}.png"), quality=100)
        return i

    # Save images with proper naming convention for encode_video_frames (frame-XXXXXX.png)
    items = list(enumerate(episode_dataset))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_image, item) for item in items]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred


def _save_batch_episodes_images(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_indices: list[int],
    num_workers: int = 4,
) -> list[float]:
    """Save images from multiple episodes to disk for batch video encoding.

    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        img_key: The image key (camera) to extract
        episode_indices: List of episode indices to save
        num_workers: Number of threads for parallel image saving

    Returns:
        List of episode durations in seconds
    """
    imgs_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset = dataset.hf_dataset.with_format(None)
    imgs_dataset = hf_dataset.select_columns(img_key)

    # Define function to save a single image with global frame index
    # Defined once outside the loop to avoid repeated closure creation
    def save_single_image(i_item_tuple, base_frame_idx, img_key_param):
        i, item = i_item_tuple
        img = item[img_key_param]
        # Use global frame index for naming
        img.save(str(imgs_dir / f"frame-{base_frame_idx + i:06d}.png"), quality=100)
        return i

    episode_durations = []
    frame_idx = 0

    for ep_idx in episode_indices:
        # Get episode range
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        episode_length = to_idx - from_idx
        episode_durations.append(episode_length / dataset.fps)

        # Get episode images
        episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

        # Save images
        items = list(enumerate(episode_dataset))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(save_single_image, item, frame_idx, img_key) for item in items]
            for future in as_completed(futures):
                future.result()

        frame_idx += episode_length

    return episode_durations


def _iter_episode_batches(
    episode_indices: list[int],
    episode_lengths: dict[int, int],
    size_per_frame_mb: float,
    video_file_size_limit: float,
    max_episodes: int | None,
    max_frames: int | None,
):
    """Generator that yields batches of episode indices for video encoding.

    Groups episodes into batches that respect size and memory constraints:
    - Stays under video file size limit
    - Respects maximum episodes per batch (if specified)
    - Respects maximum frames per batch (if specified)

    Args:
        episode_indices: List of episode indices to batch
        episode_lengths: Dictionary mapping episode index to episode length
        size_per_frame_mb: Estimated size per frame in MB
        video_file_size_limit: Maximum video file size in MB
        max_episodes: Maximum number of episodes per batch (None = no limit)
        max_frames: Maximum number of frames per batch (None = no limit)

    Yields:
        List of episode indices for each batch
    """
    batch_episodes = []
    estimated_size = 0.0
    total_frames = 0

    for ep_idx in episode_indices:
        ep_length = episode_lengths[ep_idx]
        ep_estimated_size = ep_length * size_per_frame_mb

        # we check if adding this episode would exceed any constraint
        would_exceed_size = estimated_size > 0 and estimated_size + ep_estimated_size >= video_file_size_limit
        would_exceed_episodes = max_episodes is not None and len(batch_episodes) >= max_episodes
        would_exceed_frames = max_frames is not None and total_frames + ep_length > max_frames

        if batch_episodes and (would_exceed_size or would_exceed_episodes or would_exceed_frames):
            # yield current batch before adding this episode
            yield batch_episodes
            # start a new batch with current episode
            batch_episodes = [ep_idx]
            estimated_size = ep_estimated_size
            total_frames = ep_length
        else:
            # add to current batch
            batch_episodes.append(ep_idx)
            estimated_size += ep_estimated_size
            total_frames += ep_length

    # yield final batch if not empty
    if batch_episodes:
        yield batch_episodes


def _estimate_frame_size_via_calibration(
    dataset: LeRobotDataset,
    img_key: str,
    episode_indices: list[int],
    temp_dir: Path,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    g: int,
    crf: int,
    fast_decode: int,
    num_calibration_frames: int = 30,
) -> float:
    """Estimate MB per frame by encoding a small calibration sample.

    Encodes a representative sample of frames using the exact codec parameters
    to measure actual compression ratio, which is more accurate than heuristics.

    Args:
        dataset: Source dataset with images.
        img_key: Image key to calibrate (e.g., "observation.images.top").
        episode_indices: List of episode indices being processed.
        temp_dir: Temporary directory for calibration files.
        fps: Frames per second for video encoding.
        vcodec: Video codec (libsvtav1, h264, hevc).
        pix_fmt: Pixel format (yuv420p, etc.).
        g: GOP size (group of pictures).
        crf: Constant Rate Factor (quality).
        fast_decode: Fast decode tuning parameter.
        num_calibration_frames: Number of frames to use for calibration (default: 30).

    Returns:
        Estimated size in MB per frame based on actual encoding.
    """
    calibration_dir = temp_dir / "calibration" / img_key
    calibration_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Select a representative episode (prefer middle episode if available)
        calibration_ep_idx = episode_indices[len(episode_indices) // 2]

        # Get episode range
        from_idx = dataset.meta.episodes["dataset_from_index"][calibration_ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][calibration_ep_idx]
        episode_length = to_idx - from_idx

        # Use up to num_calibration_frames from this episode
        num_frames = min(num_calibration_frames, episode_length)

        # Get frames from dataset
        hf_dataset = dataset.hf_dataset.with_format(None)
        sample_indices = range(from_idx, from_idx + num_frames)

        # Save calibration frames
        for i, idx in enumerate(sample_indices):
            img = hf_dataset[idx][img_key]
            img.save(str(calibration_dir / f"frame-{i:06d}.png"), quality=100)

        # Encode calibration video
        calibration_video_path = calibration_dir / "calibration.mp4"
        encode_video_frames(
            imgs_dir=calibration_dir,
            video_path=calibration_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            fast_decode=fast_decode,
            overwrite=True,
        )

        # Measure actual compressed size
        video_size_bytes = calibration_video_path.stat().st_size
        video_size_mb = video_size_bytes / BYTES_PER_MIB
        size_per_frame_mb = video_size_mb / num_frames

        logging.info(
            f"  Calibration: {num_frames} frames -> {video_size_mb:.2f} MB "
            f"= {size_per_frame_mb:.4f} MB/frame for {img_key}"
        )

        return size_per_frame_mb

    finally:
        # Clean up calibration files
        if calibration_dir.exists():
            shutil.rmtree(calibration_dir)


def _copy_data_without_images(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_indices: list[int],
    img_keys: list[str],
) -> None:
    """Copy data files without image columns.

    Args:
        src_dataset: Source dataset
        dst_meta: Destination metadata
        episode_indices: Episodes to include
        img_keys: Image keys to remove
    """
    from lerobot.datasets.utils import DATA_DIR

    data_dir = src_dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    episode_set = set(episode_indices)

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        # Filter to only include selected episodes
        df = df[df["episode_index"].isin(episode_set)].copy()

        if len(df) == 0:
            continue

        # Remove image columns
        columns_to_drop = [col for col in img_keys if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # Get chunk and file indices from path
        relative_path = src_path.relative_to(src_dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        # Write to destination without pandas index
        dst_path = dst_meta.root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)


# Video conversion constants
BYTES_PER_KIB = 1024
BYTES_PER_MIB = BYTES_PER_KIB * BYTES_PER_KIB


def modify_tasks(
    dataset: LeRobotDataset,
    new_task: str | None = None,
    episode_tasks: dict[int, str] | None = None,
) -> LeRobotDataset:
    """Modify tasks in a LeRobotDataset.

    This function allows you to either:
    1. Set a single task for the entire dataset (using `new_task`)
    2. Set specific tasks for specific episodes (using `episode_tasks`)

    You can combine both: `new_task` sets the default, and `episode_tasks` overrides
    specific episodes.

    The dataset is modified in-place, updating only the task-related files:
    - meta/tasks.parquet
    - data/**/*.parquet (task_index column)
    - meta/episodes/**/*.parquet (tasks column)
    - meta/info.json (total_tasks)

    Args:
        dataset: The source LeRobotDataset to modify.
        new_task: A single task string to apply to all episodes. If None and episode_tasks
            is also None, raises an error.
        episode_tasks: Optional dict mapping episode indices to their task strings.
            Overrides `new_task` for specific episodes.


    Examples:
        Set a single task for all episodes:
            dataset = modify_tasks(dataset, new_task="Pick up the cube")

        Set different tasks for specific episodes:
            dataset = modify_tasks(
                dataset,
                episode_tasks={0: "Task A", 1: "Task B", 2: "Task A"}
            )

        Set a default task with overrides:
            dataset = modify_tasks(
                dataset,
                new_task="Default task",
                episode_tasks={5: "Special task for episode 5"}
            )
    """
    if new_task is None and episode_tasks is None:
        raise ValueError("Must specify at least one of new_task or episode_tasks")

    if episode_tasks is not None:
        valid_indices = set(range(dataset.meta.total_episodes))
        invalid = set(episode_tasks.keys()) - valid_indices
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")

    # Ensure episodes metadata is loaded
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    # Build the mapping from episode index to task string
    episode_to_task: dict[int, str] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        if episode_tasks and ep_idx in episode_tasks:
            episode_to_task[ep_idx] = episode_tasks[ep_idx]
        elif new_task is not None:
            episode_to_task[ep_idx] = new_task
        else:
            # Keep original task if not overridden and no default provided
            original_tasks = dataset.meta.episodes[ep_idx]["tasks"]
            if not original_tasks:
                raise ValueError(f"Episode {ep_idx} has no tasks and no default task was provided")
            episode_to_task[ep_idx] = original_tasks[0]

    # Collect all unique tasks and create new task mapping
    unique_tasks = sorted(set(episode_to_task.values()))
    new_task_df = pd.DataFrame({"task_index": list(range(len(unique_tasks)))}, index=unique_tasks)
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}

    logging.info(f"Modifying tasks in {dataset.repo_id}")
    logging.info(f"New tasks: {unique_tasks}")

    root = dataset.root

    # Update data files - modify task_index column
    logging.info("Updating data files...")
    data_dir = root / DATA_DIR

    for parquet_path in tqdm(sorted(data_dir.rglob("*.parquet")), desc="Updating data"):
        df = pd.read_parquet(parquet_path)

        # Build a mapping from episode_index to new task_index for rows in this file
        episode_indices_in_file = df["episode_index"].unique()
        ep_to_new_task_idx = {
            ep_idx: task_to_index[episode_to_task[ep_idx]] for ep_idx in episode_indices_in_file
        }

        # Update task_index column
        df["task_index"] = df["episode_index"].map(ep_to_new_task_idx)
        df.to_parquet(parquet_path, index=False)

    # Update episodes metadata - modify tasks column
    logging.info("Updating episodes metadata...")
    episodes_dir = root / "meta" / "episodes"

    for parquet_path in tqdm(sorted(episodes_dir.rglob("*.parquet")), desc="Updating episodes"):
        df = pd.read_parquet(parquet_path)

        # Update tasks column
        df["tasks"] = df["episode_index"].apply(lambda ep_idx: [episode_to_task[ep_idx]])
        df.to_parquet(parquet_path, index=False)

    # Write new tasks.parquet
    write_tasks(new_task_df, root)

    # Update info.json
    dataset.meta.info["total_tasks"] = len(unique_tasks)
    write_info(dataset.meta.info, root)

    # Reload metadata to reflect changes
    dataset.meta.tasks = new_task_df
    dataset.meta.episodes = load_episodes(root)

    logging.info(f"Tasks: {unique_tasks}")

    return dataset


def convert_image_to_video_dataset(
    dataset: LeRobotDataset,
    output_dir: Path,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fast_decode: int = 0,
    episode_indices: list[int] | None = None,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
) -> LeRobotDataset:
    """Convert image-to-video dataset.

    Creates a new LeRobotDataset with images encoded as videos, following the proper
    LeRobot dataset structure with videos stored in chunked MP4 files.

    Args:
        dataset: The source LeRobot dataset with images
        output_dir: Directory to save the new video dataset
        repo_id: Repository ID for the new dataset (default: original_id + "_video")
        vcodec: Video codec (default: libsvtav1)
        pix_fmt: Pixel format (default: yuv420p)
        g: Group of pictures size (default: 2)
        crf: Constant rate factor (default: 30)
        fast_decode: Fast decode tuning (default: 0)
        episode_indices: List of episode indices to convert (None = all episodes)
        num_workers: Number of threads for parallel processing (default: 4)
        max_episodes_per_batch: Maximum episodes per video batch to avoid memory issues (None = no limit)
        max_frames_per_batch: Maximum frames per video batch to avoid memory issues (None = no limit)

    Returns:
        New LeRobotDataset with images encoded as videos
    """
    # Check that it's an image dataset
    if len(dataset.meta.video_keys) > 0:
        raise ValueError(
            f"This operation is for image datasets only. Video dataset provided: {dataset.repo_id}"
        )

    # Get all image keys
    hf_dataset = dataset.hf_dataset.with_format(None)
    img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]

    if len(img_keys) == 0:
        raise ValueError(f"No image keys found in dataset {dataset.repo_id}")

    # Determine which episodes to process
    if episode_indices is None:
        episode_indices = list(range(dataset.meta.total_episodes))

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_video"

    logging.info(
        f"Converting {len(episode_indices)} episodes with {len(img_keys)} cameras from {dataset.repo_id}"
    )
    logging.info(f"Video codec: {vcodec}, pixel format: {pix_fmt}, GOP: {g}, CRF: {crf}")

    # Create new features dict, converting image features to video features
    new_features = {}
    for key, value in dataset.meta.features.items():
        if key not in img_keys:
            new_features[key] = value
        else:
            # Convert image key to video format
            new_features[key] = value.copy()
            new_features[key]["dtype"] = "video"  # Change dtype from "image" to "video"
            # Video info will be updated after episodes are encoded

    # Create new metadata for video dataset
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Create temporary directory for image extraction
    temp_dir = output_dir / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process all episodes and batch encode videos
    # Use dictionary for O(1) episode metadata lookups instead of O(n) linear search
    all_episode_metadata = {}
    fps = int(dataset.fps)

    try:
        # Build episode metadata entries first
        logging.info("Building episode metadata...")
        cumulative_frame_idx = 0
        for ep_idx in episode_indices:
            src_episode = dataset.meta.episodes[ep_idx]
            ep_length = src_episode["length"]
            ep_meta = {
                "episode_index": ep_idx,
                "length": ep_length,
                "dataset_from_index": cumulative_frame_idx,
                "dataset_to_index": cumulative_frame_idx + ep_length,
            }
            if "data/chunk_index" in src_episode:
                ep_meta["data/chunk_index"] = src_episode["data/chunk_index"]
                ep_meta["data/file_index"] = src_episode["data/file_index"]
            all_episode_metadata[ep_idx] = ep_meta
            cumulative_frame_idx += ep_length

        # Process each camera and batch encode multiple episodes together
        video_file_size_limit = new_meta.video_files_size_in_mb

        # Pre-compute episode lengths for batching
        episode_lengths = {ep_idx: dataset.meta.episodes["length"][ep_idx] for ep_idx in episode_indices}

        for img_key in tqdm(img_keys, desc="Processing cameras"):
            # Estimate size per frame by encoding a small calibration sample
            # This provides accurate compression ratio for the specific codec parameters
            size_per_frame_mb = _estimate_frame_size_via_calibration(
                dataset=dataset,
                img_key=img_key,
                episode_indices=episode_indices,
                temp_dir=temp_dir,
                fps=fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
                g=g,
                crf=crf,
                fast_decode=fast_decode,
            )

            logging.info(f"Processing camera: {img_key}")
            chunk_idx, file_idx = 0, 0
            cumulative_timestamp = 0.0

            # Process episodes in batches to stay under size limit
            for batch_episodes in _iter_episode_batches(
                episode_indices=episode_indices,
                episode_lengths=episode_lengths,
                size_per_frame_mb=size_per_frame_mb,
                video_file_size_limit=video_file_size_limit,
                max_episodes=max_episodes_per_batch,
                max_frames=max_frames_per_batch,
            ):
                total_frames_in_batch = sum(episode_lengths[idx] for idx in batch_episodes)
                logging.info(
                    f"  Encoding batch of {len(batch_episodes)} episodes "
                    f"({batch_episodes[0]}-{batch_episodes[-1]}) = {total_frames_in_batch} frames"
                )

                # Save images for all episodes in this batch
                imgs_dir = temp_dir / f"batch_{chunk_idx}_{file_idx}" / img_key
                episode_durations = _save_batch_episodes_images(
                    dataset=dataset,
                    imgs_dir=imgs_dir,
                    img_key=img_key,
                    episode_indices=batch_episodes,
                    num_workers=num_workers,
                )

                # Encode all batched episodes into single video
                video_path = new_meta.root / new_meta.video_path.format(
                    video_key=img_key, chunk_index=chunk_idx, file_index=file_idx
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)

                encode_video_frames(
                    imgs_dir=imgs_dir,
                    video_path=video_path,
                    fps=fps,
                    vcodec=vcodec,
                    pix_fmt=pix_fmt,
                    g=g,
                    crf=crf,
                    fast_decode=fast_decode,
                    overwrite=True,
                )

                # Clean up temporary images
                shutil.rmtree(imgs_dir)

                # Update metadata for each episode in the batch
                for ep_idx, duration in zip(batch_episodes, episode_durations, strict=True):
                    from_timestamp = cumulative_timestamp
                    to_timestamp = cumulative_timestamp + duration
                    cumulative_timestamp = to_timestamp

                    # Find episode metadata entry and add video metadata (O(1) dictionary lookup)
                    ep_meta = all_episode_metadata[ep_idx]
                    ep_meta[f"videos/{img_key}/chunk_index"] = chunk_idx
                    ep_meta[f"videos/{img_key}/file_index"] = file_idx
                    ep_meta[f"videos/{img_key}/from_timestamp"] = from_timestamp
                    ep_meta[f"videos/{img_key}/to_timestamp"] = to_timestamp

                # Move to next video file for next batch
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, new_meta.chunks_size)
                cumulative_timestamp = 0.0

        # Copy and transform data files (removing image columns)
        _copy_data_without_images(dataset, new_meta, episode_indices, img_keys)

        # Save episode metadata
        episodes_df = pd.DataFrame(list(all_episode_metadata.values()))
        episodes_path = new_meta.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_path, index=False)

        # Update metadata info
        new_meta.info["total_episodes"] = len(episode_indices)
        new_meta.info["total_frames"] = sum(ep["length"] for ep in all_episode_metadata.values())
        new_meta.info["total_tasks"] = dataset.meta.total_tasks
        new_meta.info["splits"] = {"train": f"0:{len(episode_indices)}"}

        # Update video info for all image keys (now videos)
        # We need to manually set video info since update_video_info() checks video_keys first
        for img_key in img_keys:
            if not new_meta.features[img_key].get("info", None):
                video_path = new_meta.root / new_meta.video_path.format(
                    video_key=img_key, chunk_index=0, file_index=0
                )
                new_meta.info["features"][img_key]["info"] = get_video_info(video_path)

        write_info(new_meta.info, new_meta.root)

        # Copy stats and tasks
        if dataset.meta.stats is not None:
            # Remove image stats
            new_stats = {k: v for k, v in dataset.meta.stats.items() if k not in img_keys}
            write_stats(new_stats, new_meta.root)

        if dataset.meta.tasks is not None:
            write_tasks(dataset.meta.tasks, new_meta.root)

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    logging.info(f"Completed converting {dataset.repo_id} to video format")
    logging.info(f"New dataset saved to: {output_dir}")

    # Return new dataset
    return LeRobotDataset(repo_id=repo_id, root=output_dir)
