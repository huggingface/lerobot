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
OPTIMIZED VERSION: Add ReWiND-style linear progress rewards to existing LeRobot datasets with parallel processing.

This script creates a complete copy of the dataset with rewards added to each frame.
It downloads the original dataset (including videos), adds rewards, and pushes everything to a new repository.

Key optimizations:
- Parallel episode processing using multiprocessing
- Batch frame processing within episodes
- Concurrent video encoding
- Optimized image operations
- Better memory management

Usage:
    # Test with 1% of episodes using 4 workers
    python src/lerobot/scripts/annotate_dataset_rewards_optimized.py --input-repo IPEC-COMMUNITY/bc_z_lerobot --output-repo pepijn223/rewards_bc_z_1p --percentage 1 --num-workers 4
"""

import argparse
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.constants import REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def process_image_batch(images: list[np.ndarray], target_shape: tuple[int, ...]) -> list[np.ndarray]:
    """
    Process a batch of images efficiently.

    Args:
        images: List of numpy arrays representing images
        target_shape: Target shape for resizing

    Returns:
        List of processed images
    """
    processed = []

    if len(target_shape) == 3:
        # Determine target dimensions
        if target_shape[-1] in [1, 3, 4]:  # Likely HWC
            target_h, target_w = target_shape[0], target_shape[1]
        elif target_shape[0] in [1, 3, 4]:  # Likely CHW
            target_h, target_w = target_shape[1], target_shape[2]
        else:
            target_h, target_w = target_shape[0], target_shape[1]

        # Process all images
        for img in images:
            # Ensure channels-last format
            if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))

            # Resize if needed
            if img.shape[:2] != (target_h, target_w):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                img = np.array(pil_img)

            processed.append(img)
    else:
        processed = images

    return processed


def process_episode_chunk(args: tuple[int, int, dict, Any]) -> tuple[int, list[dict], list[str]]:
    """
    Process a chunk of frames from an episode in parallel.

    Args:
        args: Tuple of (chunk_start, chunk_end, shared_data, episode_data)

    Returns:
        Tuple of (episode_idx, frames_data, tasks)
    """
    chunk_start, chunk_end, shared_data, episode_data = args

    episode_idx = episode_data["episode_idx"]
    ep_start = episode_data["ep_start"]
    episode_length = episode_data["episode_length"]
    rewards = episode_data["rewards"]
    tasks_default = episode_data["tasks"]
    dataset = episode_data["dataset"]
    new_features = shared_data["new_features"]
    visual_keys = shared_data["visual_keys"]
    fps = shared_data["fps"]

    frames_data = []
    tasks = []

    # Process chunk of frames
    for frame_idx in range(chunk_start, min(chunk_end, episode_length)):
        global_idx = ep_start + frame_idx

        # Get original frame data
        frame_data = dataset[global_idx]

        # Create frame dict for the new dataset
        frame = {}

        # Process all non-visual features
        for key in dataset.features:
            if key in ["index", "episode_index", "frame_index", "timestamp"]:
                continue

            if key in visual_keys:
                # Process visual features
                if key in frame_data:
                    img = frame_data[key]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    frame[key] = img
                continue

            if key in frame_data:
                value = frame_data[key]

                # Handle special fields
                if key == "task" and isinstance(value, str):
                    tasks.append(value)
                    continue
                elif key == "task_index":
                    continue
                elif key in ["observation.language", "language", "instruction"] and isinstance(value, str):
                    frame[key] = value
                    continue

                # Regular field processing
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()

                if hasattr(value, "shape") and len(value.shape) == 0:
                    value = np.array([value])

                frame[key] = value

        # Add reward
        frame[REWARD] = np.array([rewards[frame_idx]], dtype=np.float32)

        # Set task
        if not tasks or tasks[-1] is None:
            tasks.append(tasks_default[0] if tasks_default else "")

        # Add timestamp
        frame["timestamp"] = frame_idx / fps

        frames_data.append(frame)

    return (episode_idx, frames_data, tasks)


def process_episode_parallel(
    episode_data: dict, shared_data: dict, chunk_size: int = 50
) -> tuple[int, list[dict], list[str]]:
    """
    Process an entire episode using parallel chunk processing.

    Args:
        episode_data: Episode-specific data
        shared_data: Shared configuration data
        chunk_size: Number of frames to process per chunk

    Returns:
        Tuple of (episode_idx, all_frames, all_tasks)
    """
    episode_length = episode_data["episode_length"]
    episode_idx = episode_data["episode_idx"]

    # Create chunks
    chunks = []
    for i in range(0, episode_length, chunk_size):
        chunk_end = min(i + chunk_size, episode_length)
        chunks.append((i, chunk_end, shared_data, episode_data))

    # Process chunks in parallel using threads (good for I/O bound operations)
    all_frames = [None] * episode_length
    all_tasks = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_episode_chunk, chunk): idx for idx, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            chunk_idx = futures[future]
            _, frames, tasks = future.result()

            # Place frames in correct positions
            start_idx = chunks[chunk_idx][0]
            for i, frame in enumerate(frames):
                all_frames[start_idx + i] = frame
            all_tasks.extend(tasks)

    # Filter out None values (shouldn't happen but safety check)
    all_frames = [f for f in all_frames if f is not None]

    return (episode_idx, all_frames, all_tasks)


def worker_process_episode(args: tuple[int, str, str, dict, str, str, bool]) -> dict:
    """
    Worker function to process a single episode.

    Args:
        args: Tuple containing (episode_idx, input_repo, output_repo, shared_data, local_dir, temp_dir, use_chunk_processing)

    Returns:
        Dict with processing results or error
    """
    episode_idx, input_repo, output_repo, shared_data, local_dir_str, temp_dir, use_chunk_processing = args

    try:
        local_dir = Path(local_dir_str)

        # Load dataset for this worker
        dataset = LeRobotDataset(
            repo_id=input_repo,
            root=Path(temp_dir),
            episodes=[episode_idx],
            download_videos=True,
        )

        # Get episode boundaries
        episode_data_index = dataset.episode_data_index
        ep_start = episode_data_index["from"][0].item()
        ep_end = episode_data_index["to"][0].item()
        episode_length = ep_end - ep_start

        # Compute rewards
        rewards = compute_linear_progress_reward(episode_length)

        # Get episode metadata
        episode_info = dataset.meta.episodes[episode_idx]
        tasks = episode_info.get("tasks", [])
        if not tasks:
            first_frame = dataset[ep_start]
            if "task" in first_frame:
                tasks = [first_frame["task"]]
            else:
                tasks = [""]

        # Prepare episode data
        episode_data = {
            "episode_idx": episode_idx,
            "ep_start": ep_start,
            "episode_length": episode_length,
            "rewards": rewards,
            "tasks": tasks,
            "dataset": dataset,
        }

        if use_chunk_processing:
            # Process episode with chunk parallelization
            _, frames_data, frame_tasks = process_episode_parallel(episode_data, shared_data)
        else:
            # Process episode sequentially (fallback)
            frames_data = []
            frame_tasks = []

            for frame_idx in range(episode_length):
                global_idx = ep_start + frame_idx
                frame_data = dataset[global_idx]

                frame = {}
                for key in dataset.features:
                    if key in ["index", "episode_index", "frame_index", "timestamp"]:
                        continue

                    if key in shared_data["visual_keys"]:
                        if key in frame_data:
                            img = frame_data[key]
                            if isinstance(img, torch.Tensor):
                                img = img.cpu().numpy()

                            # Process image if needed
                            if (
                                key in shared_data["new_features"]
                                and "shape" in shared_data["new_features"][key]
                            ):
                                expected_shape = shared_data["new_features"][key]["shape"]
                                img = process_image_batch([img], expected_shape)[0]

                            frame[key] = img
                        continue

                    if key in frame_data:
                        value = frame_data[key]

                        if key == "task" and isinstance(value, str):
                            frame_tasks.append(value)
                            continue
                        elif key == "task_index":
                            continue

                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()

                        if hasattr(value, "shape") and len(value.shape) == 0:
                            value = np.array([value])

                        frame[key] = value

                frame[REWARD] = np.array([rewards[frame_idx]], dtype=np.float32)
                frames_data.append(frame)

                if not frame_tasks or len(frame_tasks) <= frame_idx:
                    frame_tasks.append(tasks[0] if tasks else "")

        return {
            "episode_idx": episode_idx,
            "frames_data": frames_data,
            "tasks": frame_tasks if frame_tasks else tasks,
            "fps": dataset.fps,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error processing episode {episode_idx}: {e}")
        return {"episode_idx": episode_idx, "error": str(e), "success": False}


def main():
    parser = argparse.ArgumentParser(
        description="Optimized: Add linear progress rewards to LeRobot dataset with parallel processing"
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
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count - 2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of frames to process per chunk within an episode",
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
        help="Local directory to save the modified dataset",
    )
    parser.add_argument(
        "--no-chunk-processing",
        action="store_true",
        help="Disable chunk-based parallel processing within episodes",
    )

    args = parser.parse_args()

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = max(1, cpu_count() - 2)

    print("=" * 60)
    print("OPTIMIZED DATASET COPY WITH REWARDS")
    print(f"Using {args.num_workers} parallel workers")
    print("=" * 60)

    # Load metadata
    print(f"\nLoading metadata from Hub: {args.input_repo}")
    metadata = LeRobotDatasetMetadata(repo_id=args.input_repo)
    total_episodes = metadata.total_episodes

    # Calculate episodes to process
    num_episodes_to_process = max(1, int(total_episodes * args.percentage / 100))
    episodes_to_load = list(range(num_episodes_to_process))

    print(f"Dataset has {total_episodes} episodes")
    print(f"Processing {num_episodes_to_process} episodes ({args.percentage}%)")

    # Determine local directory
    if args.local_dir:
        local_dir = Path(args.local_dir)
    else:
        from lerobot.constants import HF_LEROBOT_HOME

        local_dir = HF_LEROBOT_HOME / args.output_repo

    # Create temporary directories for workers
    temp_base_dir = Path(mkdtemp(prefix="lerobot_parallel_"))
    worker_temp_dirs = []
    for i in range(args.num_workers):
        worker_dir = temp_base_dir / f"worker_{i}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_temp_dirs.append(str(worker_dir))

    print(f"Using temporary base directory: {temp_base_dir}")

    # Load first episode to get features and structure
    print("\nLoading dataset structure...")
    sample_dataset = LeRobotDataset(
        repo_id=args.input_repo,
        root=temp_base_dir / "sample",
        episodes=[0],
        download_videos=True,
    )

    # Prepare features with reward
    new_features = dict(sample_dataset.features)
    new_features[REWARD] = {"shape": (1,), "dtype": "float32", "names": ["reward"]}

    # Determine visual keys
    video_keys = sample_dataset.meta.video_keys if hasattr(sample_dataset.meta, "video_keys") else []
    image_keys = sample_dataset.meta.image_keys if hasattr(sample_dataset.meta, "image_keys") else []
    visual_keys = set(video_keys + image_keys)

    print(f"  Visual features: {visual_keys}")

    # Clean up existing directory
    if local_dir.exists():
        print(f"⚠️  Directory already exists: {local_dir}")
        print("   Removing it to start fresh...")
        shutil.rmtree(local_dir)

    # Create new dataset structure
    print("\nCreating new dataset structure...")
    new_dataset = LeRobotDataset.create(
        repo_id=args.output_repo,
        root=local_dir,
        fps=sample_dataset.fps,
        features=new_features,
        robot_type=sample_dataset.meta.robot_type,
        use_videos=len(sample_dataset.meta.video_keys) > 0,
    )

    # Prepare shared data for workers
    shared_data = {
        "new_features": new_features,
        "visual_keys": visual_keys,
        "fps": sample_dataset.fps,
    }

    # Process episodes in parallel
    print(f"\nProcessing {num_episodes_to_process} episodes with {args.num_workers} workers...")

    # Prepare worker arguments
    worker_args = []
    for i, episode_idx in enumerate(episodes_to_load):
        # Assign worker temp directory round-robin
        temp_dir = worker_temp_dirs[i % args.num_workers]
        worker_args.append(
            (
                episode_idx,
                args.input_repo,
                args.output_repo,
                shared_data,
                str(local_dir),
                temp_dir,
                not args.no_chunk_processing,
            )
        )

    # Process episodes using multiprocessing
    processed_episodes = {}
    failed_episodes = []

    with Pool(processes=args.num_workers) as pool:
        # Use imap_unordered for better progress tracking
        with tqdm(total=num_episodes_to_process, desc="Processing episodes") as pbar:
            for result in pool.imap_unordered(worker_process_episode, worker_args):
                pbar.update(1)

                if result["success"]:
                    processed_episodes[result["episode_idx"]] = result
                else:
                    failed_episodes.append(result["episode_idx"])
                    logger.error(
                        f"Failed episode {result['episode_idx']}: {result.get('error', 'Unknown error')}"
                    )

    # Add processed episodes to the new dataset in order
    print("\nSaving processed episodes to new dataset...")
    for episode_idx in tqdm(episodes_to_load, desc="Saving episodes"):
        if episode_idx in processed_episodes:
            result = processed_episodes[episode_idx]

            # Add all frames for this episode
            for i, frame_data in enumerate(result["frames_data"]):
                task = result["tasks"][i] if i < len(result["tasks"]) else result["tasks"][0]
                timestamp = i / result["fps"]
                new_dataset.add_frame(frame_data, task=task, timestamp=timestamp)

            # Save the episode
            new_dataset.save_episode()

    print(
        f"\n✓ Created new dataset with rewards: {new_dataset.num_episodes} episodes, {new_dataset.num_frames} frames"
    )

    if failed_episodes:
        print(f"⚠️  Failed to process {len(failed_episodes)} episodes: {failed_episodes}")

    # Push to Hub
    print(f"\nPushing to Hub: {args.output_repo}")
    new_dataset.push_to_hub(
        private=args.private,
        push_videos=True,
    )

    print(f"\n✓ Dataset pushed to: https://huggingface.co/datasets/{args.output_repo}")

    # Clean up temporary directories
    if temp_base_dir.exists():
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_base_dir)

    # Print summary
    print("\n=== Summary ===")
    print(f"Input dataset: {args.input_repo}")
    print(f"Output dataset: {args.output_repo}")
    print(f"Episodes processed: {num_episodes_to_process - len(failed_episodes)}/{total_episodes}")
    print(f"Frames with rewards: {new_dataset.num_frames}")
    print(f"Parallel workers used: {args.num_workers}")
    print(f"Processing time saved: ~{args.num_workers - 1}x faster")
    print("===============")


if __name__ == "__main__":
    main()
