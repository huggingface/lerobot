#!/usr/bin/env python

"""
Convert a single BEHAVIOR-1K task from HDF5 to LeRobotDataset v3.0 format.

Usage examples:
# Convert a single task
python convert_to_lerobot_v3.py \
    --data-folder /path/to/data \
    --repo-id "username/behavior-1k-assembling-gift-baskets" \
    --task-id 0 \
    --push-to-hub

"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.utils.utils import init_logging

from .behavior_lerobot_dataset_v3 import BehaviorLeRobotDatasetV3
from .behaviour_1k_constants import BEHAVIOR_DATASET_FEATURES, FPS, ROBOT_TYPE, TASK_INDICES_TO_NAMES

init_logging()


def load_hdf5_episode(hdf5_path: str, episode_id: int = 0) -> dict:
    """
    Load episode data from HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file
        episode_id: Episode ID to load (default: 0)

    Returns:
        Dictionary containing episode data
    """
    episode_data = {}

    with h5py.File(hdf5_path, "r") as f:
        # Find the episode with most samples if episode_id not specified
        if episode_id == -1:
            num_samples = [f["data"][key].attrs["num_samples"] for key in f["data"]]
            episode_id = num_samples.index(max(num_samples))

        demo_key = f"demo_{episode_id}"
        if demo_key not in f["data"]:
            raise ValueError(f"Episode {episode_id} not found in {hdf5_path}")

        demo_data = f["data"][demo_key]

        # Load actions
        episode_data["action"] = np.array(demo_data["action"][:])

        # Load observations
        episode_data["obs"] = {}
        for key in demo_data["obs"]:
            episode_data["obs"][key] = np.array(demo_data["obs"][key][:])

        # Load attributes
        episode_data["attrs"] = {}
        for attr_name in demo_data.attrs:
            episode_data["attrs"][attr_name] = demo_data.attrs[attr_name]

        # Add global attributes
        for attr_name in f["data"].attrs:
            episode_data["attrs"][f"global_{attr_name}"] = f["data"].attrs[attr_name]

    return episode_data


def convert_episode(
    data_folder: str,
    task_id: int,
    demo_id: int,
    dataset: BehaviorLeRobotDatasetV3,
    include_videos: bool = True,
    include_segmentation: bool = True,
) -> None:
    """
    Convert a single episode from HDF5 to LeRobotDataset v3.0 format.

    Args:
        data_folder: Base data folder containing HDF5 files
        repo_id: Repository ID for the dataset
        task_id: Task ID
        demo_id: Demo ID (episode ID)
        dataset: BehaviorLeRobotDatasetV3 instance to add data to
        include_videos: Whether to include video data
        include_segmentation: Whether to include segmentation data
    """
    # Construct paths
    task_name = TASK_INDICES_TO_NAMES[task_id]
    hdf5_path = f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"

    if not os.path.exists(hdf5_path):
        logging.error(f"HDF5 file not found: {hdf5_path}")
        return

    logging.info(f"Converting episode {demo_id} from task {task_name}")

    # Load episode data
    episode_data = load_hdf5_episode(hdf5_path, episode_id=0)

    # Filter out segmentation if not requested
    if not include_segmentation:
        keys_to_remove = [k for k in episode_data["obs"] if "seg_instance_id" in k]
        for key in keys_to_remove:
            del episode_data["obs"][key]

    # Add episode to dataset
    dataset.add_episode_from_hdf5(
        hdf5_data=episode_data,
        task_id=task_id,
        episode_id=demo_id,
        include_videos=include_videos,
    )


def convert_task_to_dataset(
    data_folder: str,
    repo_id: str,
    task_id: int,
    push_to_hub: bool = False,
) -> None:
    """
    Convert a single BEHAVIOR-1K task from HDF5 to LeRobotDataset v3.0 format.

    Args:
        data_folder: Base folder containing HDF5 data
        repo_id: Repository ID (e.g., "username/behavior-1k-task-name")
        task_id: Task ID to convert
        push_to_hub: Whether to push to HuggingFace Hub
    """
    task_name = TASK_INDICES_TO_NAMES[task_id]
    task_folder = f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}"

    if not os.path.exists(task_folder):
        raise ValueError(f"Task folder not found: {task_folder}")

    # Create output directory
    output_dir = Path.home() / ".cache/huggingface/lerobot" / repo_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting task '{task_name}' (ID: {task_id}) to: {output_dir}")

    # Initialize dataset for this task
    dataset = BehaviorLeRobotDatasetV3.create(
        repo_id=repo_id,
        fps=FPS,
        features=BEHAVIOR_DATASET_FEATURES,
        robot_type=ROBOT_TYPE,
    )

    # Find all episodes in the task folder
    task_episode_ids = []
    for filename in os.listdir(task_folder):
        if filename.startswith("episode_") and filename.endswith(".hdf5"):
            eid = int(filename.split("_")[1].split(".")[0])
            task_episode_ids.append(eid)
    task_episode_ids.sort()

    logging.info(f"Processing {len(task_episode_ids)} episodes for task {task_name}")

    # Convert each episode
    episodes_converted = 0
    for demo_id in tqdm(task_episode_ids, desc="Converting episodes"):
        convert_episode(
            data_folder=data_folder,
            task_id=task_id,
            demo_id=demo_id,
            dataset=dataset,
            include_videos=True,
            include_segmentation=True,
        )
        episodes_converted += 1

    logging.info(f"Converted {episodes_converted} episodes for task {task_name}")

    # Finalize dataset
    logging.info(f"Finalizing dataset for task {task_name}...")
    dataset.finalize()

    # Push to hub if requested
    if push_to_hub:
        logging.info(f"Pushing task {task_name} dataset to HuggingFace Hub...")
        dataset.push_to_hub()

    logging.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert a single BEHAVIOR-1K task to LeRobotDataset v3.0")
    parser.add_argument("--data-folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Output repository ID (e.g., 'username/behavior-1k-assembling-gift-baskets')",
    )
    parser.add_argument(
        "--task-id", type=int, required=True, help="Task ID to convert (e.g., 0 for assembling_gift_baskets)"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub after conversion"
    )

    args = parser.parse_args()

    # Convert single task to dataset
    convert_task_to_dataset(
        data_folder=args.data_folder,
        repo_id=args.repo_id,
        task_id=args.task_id,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
