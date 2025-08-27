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
Add ReWiND-style linear progress rewards to existing LeRobot datasets.

This script creates a complete copy of the dataset with rewards added to each frame.
It downloads the original dataset (including videos), adds rewards, and pushes everything to a new repository.

Usage:
    # Create full dataset copy with rewards
    python src/lerobot/scripts/annotate_dataset_rewards.py --input-repo IPEC-COMMUNITY/bc_z_lerobot --output-repo username/bc_z_with_rewards

    # Test with 1% of episodes
    python src/lerobot/scripts/annotate_dataset_rewards.py --input-repo IPEC-COMMUNITY/bc_z_lerobot --output-repo username/test_rewards --percentage 1
"""

import argparse
import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.constants import REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def compute_linear_progress_reward(episode_length: int) -> np.ndarray:
    """
    Compute linear progress rewards from 0 to 1.

    ReWiND-style: progress increases linearly from 0 at start to 1 at completion.

    Args:
        episode_length: Number of frames in the episode

    Returns:
        rewards: Array of shape (episode_length,) with values linearly from 0 to 1
    """
    return np.linspace(0, 1, episode_length, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Add linear progress rewards to LeRobot dataset and push to Hub"
    )
    parser.add_argument(
        "--input-repo",
        type=str,
        default="IPEC-COMMUNITY/bc_z_lerobot",
        help="Input dataset repository on HuggingFace Hub",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        required=True,
        help="Output dataset repository name (e.g., username/dataset_with_rewards)",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of episodes to process (useful for testing, e.g., 1 for 1%%)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the output repository private",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to save the modified dataset (defaults to ~/.cache/huggingface/lerobot/<output-repo>)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FULL DATASET COPY WITH REWARDS")
    print("This will download the entire dataset including videos,")
    print("add rewards, and push everything to a new repository.")
    print("=" * 60)

    # First, load just the metadata to get total episodes
    print(f"\nLoading metadata from Hub: {args.input_repo}")

    # Load metadata only first
    metadata = LeRobotDatasetMetadata(repo_id=args.input_repo)
    total_episodes = metadata.total_episodes

    # Calculate which episodes to process
    num_episodes_to_process = max(1, int(total_episodes * args.percentage / 100))
    episodes_to_load = list(range(num_episodes_to_process))  # Load only first N episodes

    print(f"Dataset has {total_episodes} episodes")
    print(f"Processing {num_episodes_to_process} episodes ({args.percentage}%)")

    # Determine local directory for the new dataset
    if args.local_dir:
        local_dir = Path(args.local_dir)
    else:
        from lerobot.constants import HF_LEROBOT_HOME

        local_dir = HF_LEROBOT_HOME / args.output_repo

    # Use a temporary directory for downloading source dataset
    temp_source_dir = Path(mkdtemp(prefix="lerobot_source_"))

    # Load the dataset with videos to temp directory
    print("Downloading dataset with videos to temp directory...")
    print(f"Temp directory: {temp_source_dir}")
    dataset = LeRobotDataset(
        repo_id=args.input_repo,
        root=temp_source_dir,  # Temporary location for source
        episodes=episodes_to_load if args.percentage < 100 else None,
        download_videos=True,  # Download videos
    )

    print(f"Downloaded {dataset.num_episodes} episodes with {dataset.num_frames} frames")

    # Create a new dataset with rewards
    print(f"\nCreating new dataset at: {local_dir}")

    # Clean up any existing directory from previous runs
    if local_dir.exists():
        print(f"⚠️  Directory already exists: {local_dir}")
        print("   Removing it to start fresh...")
        shutil.rmtree(local_dir)

    # Define features including reward
    # Simply copy all features from the original dataset
    new_features = dict(dataset.features)

    # Add reward feature
    new_features[REWARD] = {"shape": (1,), "dtype": "float32", "names": ["reward"]}

    # Determine which features are videos
    video_keys = dataset.meta.video_keys if hasattr(dataset.meta, "video_keys") else []
    image_keys = dataset.meta.image_keys if hasattr(dataset.meta, "image_keys") else []
    visual_keys = set(video_keys + image_keys)

    print(f"  Visual features to be handled as videos: {visual_keys}")

    # Check for language features
    language_keys = [
        k
        for k in dataset.features.keys()
        if any(lang in k.lower() for lang in ["language", "task", "instruction", "text"])
    ]
    if language_keys:
        print(f"  Language/task features found: {language_keys}")

    # Copy dataset structure to new location
    new_dataset = LeRobotDataset.create(
        repo_id=args.output_repo,
        root=local_dir,
        fps=dataset.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    # Process each episode
    print("\nAdding rewards to episodes...")

    episode_data_index = dataset.episode_data_index

    for ep_idx, episode_idx in enumerate(tqdm(episodes_to_load)):
        # Get episode boundaries
        ep_start = episode_data_index["from"][ep_idx].item()
        ep_end = episode_data_index["to"][ep_idx].item()
        episode_length = ep_end - ep_start

        # Compute linear progress rewards for this episode
        rewards = compute_linear_progress_reward(episode_length)

        # Get episode metadata
        episode_info = dataset.meta.episodes[episode_idx]
        tasks = episode_info.get("tasks", [])
        if not tasks:
            # Try to get task from first frame if not in episode metadata
            first_frame = dataset[ep_start]
            if "task" in first_frame:
                tasks = [first_frame["task"]]
            else:
                tasks = [""]

        # Process each frame in the episode
        for frame_idx in range(episode_length):
            global_idx = ep_start + frame_idx

            # Get original frame data
            frame_data = dataset[global_idx]

            # Create frame dict for the new dataset
            frame = {}
            for key in dataset.features:
                # Skip only auto-generated metadata fields
                # Keep task-related fields that contain language annotations
                if key in ["index", "episode_index", "frame_index", "timestamp"]:
                    continue

                # For visual features that are videos, extract the actual frame
                if key in visual_keys:
                    # Get the image data to save as temporary files
                    if key in frame_data:
                        img = frame_data[key]
                        # Convert to numpy if tensor
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        # Ensure channels-last format (H, W, C) for saving
                        if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:
                            img = np.transpose(img, (1, 2, 0))

                        # Resize to match expected shape if needed
                        expected_shape = new_features[key].get("shape")
                        if expected_shape and img.shape != tuple(expected_shape):
                            # Try to match the shape - handle both HWC and CHW formats
                            if len(expected_shape) == 3:
                                # Determine if expected is HWC or CHW
                                if expected_shape[-1] in [1, 3, 4]:  # Likely HWC
                                    target_h, target_w = expected_shape[0], expected_shape[1]
                                elif expected_shape[0] in [
                                    1,
                                    3,
                                    4,
                                ]:  # Likely CHW - shouldn't happen after transpose
                                    target_h, target_w = expected_shape[1], expected_shape[2]
                                else:
                                    # Assume HWC
                                    target_h, target_w = expected_shape[0], expected_shape[1]

                                # Resize using PIL for quality
                                if img.dtype != np.uint8:
                                    img = (img * 255).astype(np.uint8)
                                pil_img = Image.fromarray(img)
                                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                                img = np.array(pil_img)

                        frame[key] = img
                    continue

                if key in frame_data:
                    value = frame_data[key]

                    # Handle language/task fields specially
                    if key == "task" and isinstance(value, str):
                        # Skip string task - will be passed separately to add_frame
                        continue
                    elif key == "task_index":
                        # Skip task_index as it will be regenerated based on task
                        continue
                    elif key in ["observation.language", "language", "instruction"] and isinstance(
                        value, str
                    ):
                        # Keep language fields as-is
                        frame[key] = value
                        continue

                    # Regular field processing
                    # Convert tensors to numpy for saving
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()

                    # Ensure arrays are the right shape
                    if hasattr(value, "shape") and len(value.shape) == 0:
                        # Convert scalar to 1D array
                        value = np.array([value])

                    frame[key] = value

            # Add reward
            frame[REWARD] = np.array([rewards[frame_idx]], dtype=np.float32)

            # Get task for this specific frame (might vary within episode)
            if "task" in frame_data:
                task = frame_data["task"]
            else:
                task = tasks[0] if tasks else ""

            # Add frame to new dataset
            timestamp = frame_idx / dataset.fps
            new_dataset.add_frame(frame, task=task, timestamp=timestamp)

        # Save the episode (this will encode videos from the saved frames)
        new_dataset.save_episode()

    print(
        f"\n✓ Created new dataset with rewards: {new_dataset.num_episodes} episodes, {new_dataset.num_frames} frames"
    )

    # Push to Hub
    print(f"\nPushing to Hub: {args.output_repo}")
    new_dataset.push_to_hub(
        private=args.private,
        push_videos=True,
    )

    print(f"\n✓ Dataset pushed to: https://huggingface.co/datasets/{args.output_repo}")

    # Clean up temporary source directory
    if temp_source_dir.exists():
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_source_dir)

    # Print summary
    print("\n=== Summary ===")
    print(f"Input dataset: {args.input_repo}")
    print(f"Output dataset: {args.output_repo}")
    print(f"Episodes processed: {num_episodes_to_process}/{total_episodes} ({args.percentage}%)")
    print(f"Frames with rewards: {new_dataset.num_frames}")
    print("Reward type: Linear progress (0→1)")
    print("===============")


if __name__ == "__main__":
    main()
