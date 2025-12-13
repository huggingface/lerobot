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
Convert Isaac Lab Arena HDF5 datasets to LeRobot format.

This script converts datasets from Isaac Lab Arena (GR1 humanoid robot)
stored in HDF5 format to the LeRobot dataset format for training robot
learning policies.

Usage:
    python convert_hdf5_lerobot.py \
        --hdf5-path /path/to/arena_gr1_manipulation_dataset.hdf5 \
        --repo-id username/arena_gr1_manipulation \
        --root /path/to/lerobot/datasets \
        --task-name "Open microwave door"
"""

import argparse
import logging
from pathlib import Path

import h5py
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Isaac Lab Arena dataset constants
ARENA_FPS = 50  # Isaac Lab typically runs at 50 Hz
ARENA_ROBOT_TYPE = "GR1"  # GR1 humanoid robot

# Define features structure based on the HDF5 dataset structure
ARENA_FEATURES = {
    # Actions (36-dimensional: includes body, arms, hands, head)
    "action": {
        "dtype": "float32",
        "shape": (36,),
        "names": None,
    },
    # Observations
    "observation.state": {
        "dtype": "float32",
        "shape": (54,),  # Robot joint positions
        "names": None,
    },
    "observation.images.robot_pov_cam": {
        "dtype": "video",
        "shape": (512, 512, 3),
        "names": ["height", "width", "channels"],
    },
    # Additional observation features
    "observation.hand_joint_state": {
        "dtype": "float32",
        "shape": (22,),
        "names": None,
    },
    "observation.head_joint_state": {
        "dtype": "float32",
        "shape": (3,),
        "names": None,
    },
    "observation.left_eef_pos": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"axes": ["x", "y", "z"]},
    },
    "observation.left_eef_quat": {
        "dtype": "float32",
        "shape": (4,),
        "names": {"axes": ["w", "x", "y", "z"]},
    },
    "observation.right_eef_pos": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"axes": ["x", "y", "z"]},
    },
    "observation.right_eef_quat": {
        "dtype": "float32",
        "shape": (4,),
        "names": {"axes": ["w", "x", "y", "z"]},
    },
    "observation.robot_root_pos": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"axes": ["x", "y", "z"]},
    },
    "observation.robot_root_rot": {
        "dtype": "float32",
        "shape": (4,),
        "names": {"axes": ["w", "x", "y", "z"]},
    },
}


def get_episodes_from_hdf5(hdf5_file: h5py.File) -> list[str]:
    """
    Extract episode names from HDF5 file.

    Args:
        hdf5_file: Open HDF5 file handle

    Returns:
        List of episode names (e.g., ['demo_0', 'demo_1', ...])
    """
    episodes = []
    data_group = hdf5_file.get("data")

    if data_group is None:
        raise ValueError("HDF5 file does not contain 'data' group")

    for key in data_group.keys():
        if key.startswith("demo_"):
            episodes.append(key)

    episodes.sort()  # Ensure consistent ordering
    logger.info(f"Found {len(episodes)} episodes in HDF5 file")
    return episodes


def load_episode_data(hdf5_file: h5py.File, episode_name: str) -> dict:
    """
    Load all data for a single episode from HDF5.

    Args:
        hdf5_file: Open HDF5 file handle
        episode_name: Name of episode (e.g., 'demo_0')

    Returns:
        Dictionary containing all episode data
    """
    episode_group = hdf5_file[f"data/{episode_name}"]

    episode_data = {
        "actions": episode_group["actions"][:],
        "obs": {
            "robot_joint_pos": episode_group["obs/robot_joint_pos"][:],
            "hand_joint_state": episode_group["obs/hand_joint_state"][:],
            "head_joint_state": episode_group["obs/head_joint_state"][:],
            "left_eef_pos": episode_group["obs/left_eef_pos"][:],
            "left_eef_quat": episode_group["obs/left_eef_quat"][:],
            "right_eef_pos": episode_group["obs/right_eef_pos"][:],
            "right_eef_quat": episode_group["obs/right_eef_quat"][:],
            "robot_root_pos": episode_group["obs/robot_root_pos"][:],
            "robot_root_rot": episode_group["obs/robot_root_rot"][:],
        },
        "camera_obs": {
            "robot_pov_cam_rgb": episode_group["camera_obs/robot_pov_cam_rgb"][:],
        },
    }

    num_frames = len(episode_data["actions"])
    logger.info(f"Loaded episode {episode_name} with {num_frames} frames")

    return episode_data


def convert_hdf5_to_lerobot(
    hdf5_path: str | Path,
    repo_id: str,
    root: str | Path | None = None,
    task_name: str = "Manipulation task",
    push_to_hub: bool = False,
    num_episodes: int | None = None,
    use_videos: bool = True,
) -> LeRobotDataset:
    """
    Convert Isaac Lab Arena HDF5 dataset to LeRobot format.

    Args:
        hdf5_path: Path to the HDF5 file
        repo_id: Repository ID for the LeRobot dataset
        root: Root directory to save the dataset
        task_name: Task description/name
        push_to_hub: Whether to push to Hugging Face Hub
        num_episodes: Number of episodes to convert (None for all)
        use_videos: Whether to encode images as videos

    Returns:
        LeRobotDataset object
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    logger.info(f"Converting HDF5 dataset from {hdf5_path}")
    logger.info(f"Creating LeRobot dataset: {repo_id}")

    # Open HDF5 file
    with h5py.File(hdf5_path, "r") as hdf5_file:
        # Get list of episodes
        episodes = get_episodes_from_hdf5(hdf5_file)

        if num_episodes is not None:
            episodes = episodes[:num_episodes]
            logger.info(f"Converting only first {num_episodes} episodes")

        # Create LeRobot dataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=ARENA_FPS,
            root=root,
            robot_type=ARENA_ROBOT_TYPE,
            features=ARENA_FEATURES,
            use_videos=use_videos,
        )

        # Start image writer for efficient image saving
        if use_videos:
            lerobot_dataset.start_image_writer(num_processes=0, num_threads=4)

        # Convert each episode
        for episode_idx, episode_name in enumerate(episodes):
            msg = (
                f"Converting episode {episode_idx + 1}/{len(episodes)}: "
                f"{episode_name}"
            )
            logger.info(msg)

            # Load episode data
            episode_data = load_episode_data(hdf5_file, episode_name)
            num_frames = len(episode_data["actions"])

            # Add frames to dataset
            for frame_idx in range(num_frames):
                # Prepare frame dictionary
                obs = episode_data["obs"]
                frame = {
                    "task": task_name,
                    "action": episode_data["actions"][frame_idx],
                    "observation.state": obs["robot_joint_pos"][frame_idx],
                    "observation.hand_joint_state": obs["hand_joint_state"][frame_idx],
                    "observation.head_joint_state": obs["head_joint_state"][frame_idx],
                    "observation.left_eef_pos": obs["left_eef_pos"][frame_idx],
                    "observation.left_eef_quat": obs["left_eef_quat"][frame_idx],
                    "observation.right_eef_pos": obs["right_eef_pos"][frame_idx],
                    "observation.right_eef_quat": obs["right_eef_quat"][frame_idx],
                    "observation.robot_root_pos": obs["robot_root_pos"][frame_idx],
                    "observation.robot_root_rot": obs["robot_root_rot"][frame_idx],
                }

                # Add camera observation
                if use_videos:
                    rgb_array = episode_data["camera_obs"]["robot_pov_cam_rgb"][
                        frame_idx
                    ]
                    # Convert to PIL Image (HWC format, uint8)
                    frame["observation.images.robot_pov_cam"] = Image.fromarray(
                        rgb_array
                    )

                # Add frame to dataset
                lerobot_dataset.add_frame(frame)

            # Save episode
            lerobot_dataset.save_episode()
            msg = f"Saved episode {episode_idx} with {num_frames} frames"
            logger.info(msg)

    # Stop image writer
    if use_videos:
        lerobot_dataset.stop_image_writer()

    # Finalize the dataset (close parquet writers to write footer metadata)
    lerobot_dataset.finalize()

    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing dataset to hub: {repo_id}")
        lerobot_dataset.push_to_hub()

    msg = f"Conversion complete! Dataset saved to {lerobot_dataset.root}"
    logger.info(msg)
    logger.info(f"Total episodes: {lerobot_dataset.num_episodes}")
    logger.info(f"Total frames: {len(lerobot_dataset)}")

    return lerobot_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab Arena HDF5 dataset to LeRobot"
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        required=True,
        help="Path to the HDF5 dataset file",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for LeRobot dataset (e.g., 'user/dataset')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root dir (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="Open microwave door",
        help="Task description/name",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub after conversion",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to convert (default: all)",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store images as individual files instead of videos",
    )

    args = parser.parse_args()

    convert_hdf5_to_lerobot(
        hdf5_path=args.hdf5_path,
        repo_id=args.repo_id,
        root=args.root,
        task_name=args.task_name,
        push_to_hub=args.push_to_hub,
        num_episodes=args.num_episodes,
        use_videos=not args.no_videos,
    )


if __name__ == "__main__":
    # Example usage when run directly
    default_path = (
        "/home/ksachdev/repos/collab-lerobot/src/lerobot/envs/data/"
        "arena_gr1_manipulation_dataset_generated.hdf5"
    )

    if Path(default_path).exists():
        logger.info(f"Converting default dataset: {default_path}")
        convert_hdf5_to_lerobot(
            hdf5_path=default_path,
            repo_id="arena/gr1_microwave_manipulation",
            task_name="Open microwave door",
            num_episodes=5,  # Convert first 5 episodes as a test
            use_videos=True,
        )
    else:
        msg = "No default HDF5 file found. Run with --hdf5-path."
        logger.info(msg)
        main()
