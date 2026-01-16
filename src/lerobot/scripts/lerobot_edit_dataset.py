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
Edit LeRobot datasets using various transformation tools.

This script allows you to delete episodes, split datasets, merge datasets,
remove features, and convert image datasets to video format.
When new_repo_id is specified, creates a new dataset.

Usage Examples:

Delete episodes 0, 2, and 5 from a dataset:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Delete episodes and save to a new dataset:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_filtered \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Split dataset by fractions:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.8, "val": 0.2}'

Split dataset by episode indices:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": [0, 1, 2, 3], "val": [4, 5]}'

Split into more than two splits:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.6, "val": 0.2, "test": 0.2}'

Merge multiple datasets:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

Remove camera feature:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht \
        --operation.type remove_feature \
        --operation.feature_names "['observation.images.top']"

Convert image dataset to video format (saves locally):
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht_image \
        --operation.type convert_to_video \
        --operation.output_dir /path/to/output/pusht_video

Convert image dataset and save with new repo_id:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht_image \
        --new_repo_id lerobot/pusht_video \
        --operation.type convert_to_video

Convert and push to hub:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht_image \
        --new_repo_id lerobot/pusht_video \
        --operation.type convert_to_video \
        --push_to_hub true

Using JSON config file:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --config_path path/to/edit_config.json
"""

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.datasets.dataset_tools import (
    delete_episodes,
    merge_datasets,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    get_file_size_in_mb,
    update_chunk_file_indices,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, encode_video_frames, get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_IMAGE
from lerobot.utils.utils import init_logging


@dataclass
class DeleteEpisodesConfig:
    type: str = "delete_episodes"
    episode_indices: list[int] | None = None


@dataclass
class SplitConfig:
    type: str = "split"
    splits: dict[str, float | list[int]] | None = None


@dataclass
class MergeConfig:
    type: str = "merge"
    repo_ids: list[str] | None = None


@dataclass
class RemoveFeatureConfig:
    type: str = "remove_feature"
    feature_names: list[str] | None = None


@dataclass
class ConvertToVideoConfig:
    type: str = "convert_to_video"
    output_dir: str | None = None
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    fast_decode: int = 0
    episode_indices: list[int] | None = None
    num_workers: int = 4


@dataclass
class EditDatasetConfig:
    repo_id: str
    operation: DeleteEpisodesConfig | SplitConfig | MergeConfig | RemoveFeatureConfig | ConvertToVideoConfig
    root: str | None = None
    new_repo_id: str | None = None
    push_to_hub: bool = False


def get_output_path(repo_id: str, new_repo_id: str | None, root: Path | None) -> tuple[str, Path]:
    if new_repo_id:
        output_repo_id = new_repo_id
        output_dir = root / new_repo_id if root else HF_LEROBOT_HOME / new_repo_id
    else:
        output_repo_id = repo_id
        dataset_path = root / repo_id if root else HF_LEROBOT_HOME / repo_id
        old_path = Path(str(dataset_path) + "_old")

        if dataset_path.exists():
            if old_path.exists():
                shutil.rmtree(old_path)
            shutil.move(str(dataset_path), str(old_path))

        output_dir = dataset_path

    return output_repo_id, output_dir


def handle_delete_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, DeleteEpisodesConfig):
        raise ValueError("Operation config must be DeleteEpisodesConfig")

    if not cfg.operation.episode_indices:
        raise ValueError("episode_indices must be specified for delete_episodes operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    logging.info(f"Deleting episodes {cfg.operation.episode_indices} from {cfg.repo_id}")
    new_dataset = delete_episodes(
        dataset,
        episode_indices=cfg.operation.episode_indices,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}, Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_split(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, SplitConfig):
        raise ValueError("Operation config must be SplitConfig")

    if not cfg.operation.splits:
        raise ValueError(
            "splits dict must be specified with split names as keys and fractions/episode lists as values"
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    logging.info(f"Splitting dataset {cfg.repo_id} with splits: {cfg.operation.splits}")
    split_datasets = split_dataset(dataset, splits=cfg.operation.splits)

    for split_name, split_ds in split_datasets.items():
        split_repo_id = f"{cfg.repo_id}_{split_name}"
        logging.info(
            f"{split_name}: {split_ds.meta.total_episodes} episodes, {split_ds.meta.total_frames} frames"
        )

        if cfg.push_to_hub:
            logging.info(f"Pushing {split_name} split to hub as {split_repo_id}")
            LeRobotDataset(split_ds.repo_id, root=split_ds.root).push_to_hub()


def handle_merge(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, MergeConfig):
        raise ValueError("Operation config must be MergeConfig")

    if not cfg.operation.repo_ids:
        raise ValueError("repo_ids must be specified for merge operation")

    if not cfg.repo_id:
        raise ValueError("repo_id must be specified as the output repository for merged dataset")

    logging.info(f"Loading {len(cfg.operation.repo_ids)} datasets to merge")
    datasets = [LeRobotDataset(repo_id, root=cfg.root) for repo_id in cfg.operation.repo_ids]

    output_dir = Path(cfg.root) / cfg.repo_id if cfg.root else HF_LEROBOT_HOME / cfg.repo_id

    logging.info(f"Merging datasets into {cfg.repo_id}")
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=cfg.repo_id,
        output_dir=output_dir,
    )

    logging.info(f"Merged dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {merged_dataset.meta.total_episodes}, Frames: {merged_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {cfg.repo_id}")
        LeRobotDataset(merged_dataset.repo_id, root=output_dir).push_to_hub()


def handle_remove_feature(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RemoveFeatureConfig):
        raise ValueError("Operation config must be RemoveFeatureConfig")

    if not cfg.operation.feature_names:
        raise ValueError("feature_names must be specified for remove_feature operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    logging.info(f"Removing features {cfg.operation.feature_names} from {cfg.repo_id}")
    new_dataset = remove_feature(
        dataset,
        feature_names=cfg.operation.feature_names,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Remaining features: {list(new_dataset.meta.features.keys())}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def save_episode_images_for_video(
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


def save_batch_episodes_images(
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
        
        # Define function to save a single image with global frame index
        def save_single_image(i_item_tuple):
            i, item = i_item_tuple
            img = item[img_key]
            # Use global frame index for naming
            img.save(str(imgs_dir / f"frame-{frame_idx + i:06d}.png"), quality=100)
            return i
        
        # Save images
        items = list(enumerate(episode_dataset))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(save_single_image, item) for item in items]
            for future in as_completed(futures):
                future.result()
        
        frame_idx += episode_length
    
    return episode_durations


def convert_dataset_to_videos(
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
) -> LeRobotDataset:
    """Convert image-based dataset to video-based dataset.

    Creates a new LeRobotDataset with videos instead of images, following the proper
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

    Returns:
        New LeRobotDataset with videos
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
    all_episode_metadata = []
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
            all_episode_metadata.append(ep_meta)
            cumulative_frame_idx += ep_length

        # Process each camera and batch encode multiple episodes together
        video_file_size_limit = new_meta.video_files_size_in_mb
        
        for img_key in tqdm(img_keys, desc="Processing cameras"):
            # Estimate size per frame based on first episode if available
            # Otherwise use conservative heuristic
            size_per_frame_mb = 0.01  # Conservative default
            if len(episode_indices) > 0:
                # Try to get better estimate from first episode's image size
                try:
                    first_ep_idx = episode_indices[0]
                    from_idx = dataset.meta.episodes["dataset_from_index"][first_ep_idx]
                    sample_img = dataset.hf_dataset.with_format(None)[from_idx][img_key]
                    # Rough estimate: compressed size is ~1-5% of uncompressed
                    # For RGB image: height * width * 3 bytes
                    img_size_mb = (sample_img.size[0] * sample_img.size[1] * 3) / (1024 * 1024)
                    size_per_frame_mb = img_size_mb * 0.03  # 3% of uncompressed size
                    logging.info(f"  Estimated {size_per_frame_mb:.4f} MB per frame for {img_key}")
                    
                except Exception:
                    logging.warning(f"  Could not estimate frame size for {img_key}, using default {size_per_frame_mb} MB")
            logging.info(f"Processing camera: {img_key}")
            chunk_idx, file_idx = 0, 0
            cumulative_timestamp = 0.0
            
            # Process episodes in batches to stay under size limit
            i = 0
            while i < len(episode_indices):
                # Determine batch of episodes to encode together
                batch_episodes = []
                estimated_size = 0.0
                
                while i < len(episode_indices):
                    ep_idx = episode_indices[i]
                    ep_length = dataset.meta.episodes["length"][ep_idx]
                    ep_estimated_size = ep_length * size_per_frame_mb
                    
                    # Check if adding this episode would exceed limit (leave 10% margin)
                    if estimated_size > 0 and estimated_size + ep_estimated_size >= video_file_size_limit * 0.9:
                        break
                    
                    batch_episodes.append(ep_idx)
                    estimated_size += ep_estimated_size
                    i += 1
                    
                    # Also cap at reasonable batch size to avoid memory issues
                    # Process at most 50 episodes or 10k frames at once
                    total_frames = sum(dataset.meta.episodes["length"][idx] for idx in batch_episodes)
                    # TODO (jadechoghari): Remove this once we have a better way to estimate the size of the video file
                    if len(batch_episodes) >= 50 or total_frames >= 10000:
                        break
                
                if not batch_episodes:
                    # Edge case: single episode is larger than limit, process it anyway
                    batch_episodes = [episode_indices[i]]
                    i += 1
                
                total_frames_in_batch = sum(dataset.meta.episodes["length"][idx] for idx in batch_episodes)
                logging.info(
                    f"  Encoding batch of {len(batch_episodes)} episodes "
                    f"({batch_episodes[0]}-{batch_episodes[-1]}) = {total_frames_in_batch} frames"
                )
                
                # Save images for all episodes in this batch
                imgs_dir = temp_dir / f"batch_{chunk_idx}_{file_idx}" / img_key
                episode_durations = save_batch_episodes_images(
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
                for ep_idx, duration in zip(batch_episodes, episode_durations, strict=False):
                    from_timestamp = cumulative_timestamp
                    to_timestamp = cumulative_timestamp + duration
                    cumulative_timestamp = to_timestamp
                    
                    # Find episode metadata entry and add video metadata
                    ep_meta = next(m for m in all_episode_metadata if m["episode_index"] == ep_idx)
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
        episodes_df = pd.DataFrame(all_episode_metadata)
        episodes_path = new_meta.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_path, index=False)

        # Update metadata info
        new_meta.info["total_episodes"] = len(episode_indices)
        new_meta.info["total_frames"] = sum(ep["length"] for ep in all_episode_metadata)
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

        from lerobot.datasets.utils import write_info

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

    logging.info(f"✓ Completed converting {dataset.repo_id} to video format")
    logging.info(f"New dataset saved to: {output_dir}")

    # Return new dataset
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


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


def handle_convert_to_video(cfg: EditDatasetConfig) -> None:
    # Note: Parser may create any config type with the right fields, so we access fields directly
    # instead of checking isinstance()
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    # Determine output directory and repo_id
    # Priority: 1) new_repo_id, 2) operation.output_dir, 3) auto-generated name
    output_dir_config = getattr(cfg.operation, "output_dir", None)

    if cfg.new_repo_id:
        # Use new_repo_id for both local storage and hub push
        output_repo_id = cfg.new_repo_id
        # Place new dataset as a sibling to the original dataset
        # Get the parent of the actual dataset root (not cfg.root which might be the lerobot cache dir)
        # Extract just the dataset name (after last slash) for the local directory
        local_dir_name = cfg.new_repo_id.split("/")[-1]
        output_dir = dataset.root.parent / local_dir_name
        logging.info(f"Saving to new dataset: {cfg.new_repo_id} at {output_dir}")
    elif output_dir_config:
        # Use custom output directory for local-only storage
        output_dir = Path(output_dir_config)
        # Extract repo name from output_dir for the dataset
        output_repo_id = output_dir.name
        logging.info(f"Saving to local directory: {output_dir}")
    else:
        # Auto-generate name: append "_video" to original repo_id
        output_repo_id = f"{cfg.repo_id}_video"
        # Place new dataset as a sibling to the original dataset
        # Extract just the dataset name (after last slash) for the local directory
        local_dir_name = output_repo_id.split("/")[-1]
        output_dir = dataset.root.parent / local_dir_name
        logging.info(f"Saving to auto-generated location: {output_dir}")

    logging.info(f"Converting dataset {cfg.repo_id} to video format")

    new_dataset = convert_dataset_to_videos(
        dataset=dataset,
        output_dir=output_dir,
        repo_id=output_repo_id,
        vcodec=getattr(cfg.operation, "vcodec", "libsvtav1"),
        pix_fmt=getattr(cfg.operation, "pix_fmt", "yuv420p"),
        g=getattr(cfg.operation, "g", 2),
        crf=getattr(cfg.operation, "crf", 30),
        fast_decode=getattr(cfg.operation, "fast_decode", 0),
        episode_indices=getattr(cfg.operation, "episode_indices", None),
        num_workers=getattr(cfg.operation, "num_workers", 4),
    )

    logging.info("Video dataset created successfully!")
    logging.info(f"Location: {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}")
    logging.info(f"Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}...")
        new_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")
    else:
        logging.info("Dataset saved locally (not pushed to hub)")


@parser.wrap()
def edit_dataset(cfg: EditDatasetConfig) -> None:
    operation_type = cfg.operation.type

    if operation_type == "delete_episodes":
        handle_delete_episodes(cfg)
    elif operation_type == "split":
        handle_split(cfg)
    elif operation_type == "merge":
        handle_merge(cfg)
    elif operation_type == "remove_feature":
        handle_remove_feature(cfg)
    elif operation_type == "convert_to_video":
        handle_convert_to_video(cfg)
    else:
        raise ValueError(
            f"Unknown operation type: {operation_type}\n"
            f"Available operations: delete_episodes, split, merge, remove_feature, convert_to_video"
        )


def main() -> None:
    init_logging()
    edit_dataset()


if __name__ == "__main__":
    main()
