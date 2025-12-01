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

Convert image dataset to video format:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot/pusht_image \
        --operation.type convert_to_video \
        --operation.output_dir outputs/converted_videos \
        --operation.vcodec libsvtav1 \
        --operation.crf 30

Using JSON config file:
    python -m lerobot.scripts.lerobot_edit_dataset \
        --config_path path/to/edit_config.json
"""

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from lerobot.configs import parser
from lerobot.datasets.dataset_tools import (
    delete_episodes,
    merge_datasets,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames
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
    output_dir: str = "outputs/converted_videos"
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    fast_decode: int = 0
    episode_indices: list[int] | None = None
    num_workers: int = 4
    overwrite: bool = False


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


def save_episode_images(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    episode_index: int = 0,
    overwrite: bool = False,
    num_workers: int = 4,
) -> None:
    """Save images from a specific episode to disk.
    
    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        episode_index: Index of the episode to save (default: 0)
        overwrite: Whether to overwrite existing images
        num_workers: Number of threads for parallel image saving (default: 4)
    """
    ep_num_images = dataset.meta.episodes["length"][episode_index]
    
    # Check if images already exist
    if not overwrite and imgs_dir.exists() and len(list(imgs_dir.glob("frame-*.png"))) == ep_num_images:
        logging.info(f"Images for episode {episode_index} already exist in {imgs_dir}. Skipping.")
        return
    
    # Create directory
    imgs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset without torch format for PIL image access
    hf_dataset = dataset.hf_dataset.with_format(None)
    
    # Get all image keys (for all cameras)
    img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]
    
    if len(img_keys) == 0:
        raise ValueError(f"No image keys found in dataset {dataset.repo_id}")
    
    # Use first camera only
    img_key = img_keys[0]
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
    # Use ThreadPoolExecutor for parallel processing
    items = list(enumerate(episode_dataset))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_image, item) for item in items]
        for future in tqdm(
            as_completed(futures),
            total=len(items),
            desc=f"Saving {dataset.repo_id} episode {episode_index} images",
            leave=False,
        ):
            future.result()  # This will raise any exceptions that occurred


def process_single_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    output_dir: Path,
    vcodec: str,
    pix_fmt: str,
    g: int | None,
    crf: int | None,
    fast_decode: int,
    fps: int,
    num_image_workers: int,
    overwrite: bool,
) -> str:
    """Process a single episode: save images and encode to video.
    
    Args:
        dataset: The LeRobot dataset
        episode_index: Index of the episode to process
        output_dir: Base directory for outputs
        vcodec: Video codec
        pix_fmt: Pixel format
        g: Group of pictures size
        crf: Constant rate factor
        fast_decode: Fast decode tuning
        fps: Frames per second
        num_image_workers: Number of threads for parallel image saving
        overwrite: Whether to overwrite existing files
        
    Returns:
        Status message for this episode
    """
    # Create paths
    imgs_dir = output_dir / "images" / dataset.repo_id.replace("/", "_") / f"episode_{episode_index:06d}"
    
    # Create video filename with encoding parameters
    video_filename = f"{dataset.repo_id.replace('/', '_')}_ep{episode_index:06d}_{vcodec}_{pix_fmt}_g{g}_crf{crf}.mp4"
    video_path = output_dir / "videos" / dataset.repo_id.replace("/", "_") / video_filename
    
    # Save episode images
    save_episode_images(dataset, imgs_dir, episode_index, overwrite, num_image_workers)
    
    # Encode to video
    if overwrite or not video_path.is_file():
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
        
        return f"✓ Video saved to {video_path}"
    else:
        return f"Video already exists: {video_path}. Skipping."


def convert_dataset_to_videos(
    dataset: LeRobotDataset,
    output_dir: Path,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fast_decode: int = 0,
    episode_indices: list[int] | None = None,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Convert dataset images to video files.
    
    Args:
        dataset: The LeRobot dataset
        output_dir: Base directory for outputs
        vcodec: Video codec (default: libsvtav1)
        pix_fmt: Pixel format (default: yuv420p)
        g: Group of pictures size (default: 2)
        crf: Constant rate factor (default: 30)
        fast_decode: Fast decode tuning (default: 0)
        episode_indices: List of episode indices to convert (None = all episodes)
        num_workers: Number of threads for parallel episode processing (default: 4)
        overwrite: Whether to overwrite existing files
    """
    # Check that it's an image dataset
    if len(dataset.meta.video_keys) > 0:
        raise ValueError(
            f"This operation is for image datasets only. Video dataset provided: {dataset.repo_id}"
        )
    
    fps = dataset.fps
    
    # Determine which episodes to process
    num_episodes = len(dataset.meta.episodes)
    if episode_indices is None:
        episode_indices = list(range(num_episodes))
    
    logging.info(f"Processing {len(episode_indices)} episodes from {dataset.repo_id} with {num_workers} workers")
    
    # Process episodes in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_single_episode,
                dataset=dataset,
                episode_index=episode_index,
                output_dir=output_dir,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
                g=g,
                crf=crf,
                fast_decode=fast_decode,
                fps=fps,
                num_image_workers=4,  # Use fixed workers for image saving within each episode
                overwrite=overwrite,
            )
            for episode_index in episode_indices
        ]
        
        for future in tqdm(
            as_completed(futures),
            total=len(episode_indices),
            desc="Episodes",
        ):
            result = future.result()  # This will raise any exceptions that occurred
            logging.info(result)
    
    logging.info(f"\n✓ Completed processing {dataset.repo_id}")


def handle_convert_to_video(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ConvertToVideoConfig):
        raise ValueError("Operation config must be ConvertToVideoConfig")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_dir = Path(cfg.operation.output_dir)

    logging.info(f"Converting dataset {cfg.repo_id} to video format")
    convert_dataset_to_videos(
        dataset=dataset,
        output_dir=output_dir,
        vcodec=cfg.operation.vcodec,
        pix_fmt=cfg.operation.pix_fmt,
        g=cfg.operation.g,
        crf=cfg.operation.crf,
        fast_decode=cfg.operation.fast_decode,
        episode_indices=cfg.operation.episode_indices,
        num_workers=cfg.operation.num_workers,
        overwrite=cfg.operation.overwrite,
    )


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
