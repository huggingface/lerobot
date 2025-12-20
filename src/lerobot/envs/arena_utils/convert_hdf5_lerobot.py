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

This script converts datasets from Isaac Lab Arena (humanoid robots)
stored in HDF5 format to the LeRobot dataset format for training robot
learning policies. It automatically detects the dataset structure and
adapts to different robot types (G1, GR1, etc.).

Usage:
    python convert_hdf5_lerobot.py \
        --hdf5-path /path/to/arena_dataset.hdf5 \
        --repo-id username/arena_dataset \
        --root /path/to/lerobot/datasets \
        --task-instruction "Open microwave door"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.arena_utils.inspect_hdf5 import HDF5Inspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_hdf5_to_lerobot(
    hdf5_path: str | Path,
    repo_id: str,
    root: str | Path | None = None,
    task_instruction: str = "Perform the task",
    push_to_hub: bool = False,
    num_episodes: int | None = None,
    use_videos: bool = True,
    state_key: str | None = None,
    action_key: str | None = None,
    robot_type: str | None = None,
    fps: int | None = None,
) -> LeRobotDataset:
    """
    Convert Isaac Lab Arena HDF5 dataset to LeRobot format.

    This function automatically detects the dataset structure from the HDF5
    file and creates appropriate LeRobot features without hardcoding.

    Args:
        hdf5_path: Path to the HDF5 file
        repo_id: Repository ID for the LeRobot dataset
        root: Root directory to save the dataset
        task_instruction: Task description/name
        push_to_hub: Whether to push to Hugging Face Hub
        num_episodes: Number of episodes to convert (None for all)
        use_videos: Whether to encode images as videos
        state_key: Observation key to use as state (auto-detected if None)
        action_key: Action key to use (default: 'actions')
        robot_type: Override auto-detected robot type
        fps: Override auto-detected FPS

    Returns:
        LeRobotDataset object
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    logger.info(f"Converting HDF5 dataset from {hdf5_path}")

    # Use inspector to analyze the HDF5 file
    inspector = HDF5Inspector(hdf5_path)
    schema = inspector.get_schema()

    # Log detected schema info
    logger.info(f"Detected environment: {schema.env_name}")
    logger.info(f"Detected robot type: {schema.robot_type}")
    logger.info(f"Detected FPS: {schema.fps}")
    logger.info(f"Total episodes: {len(schema.episode_names)}")

    # Use provided overrides or auto-detected values
    final_robot_type = robot_type or schema.robot_type
    final_fps = fps or schema.fps
    final_action_key = action_key or schema.primary_action_key

    # Generate features from schema
    features = schema.get_lerobot_features(
        include_cameras=use_videos,
        state_key=state_key,
        action_key=final_action_key,
    )

    # Log the features being used
    logger.info("LeRobot features:")
    for name, feat in features.items():
        logger.info(f"  {name}: shape={feat['shape']}, dtype={feat['dtype']}")

    # Get camera names for loading
    camera_names = schema.get_camera_names() if use_videos else []
    logger.info(f"Cameras to include: {camera_names}")

    # Determine state key being used
    state_obs_key = state_key
    if state_obs_key is None:
        # Find which state key was selected
        for key in ["robot_joint_pos", "joint_position", "robot_state"]:
            if key in schema.observation_fields:
                state_obs_key = key
                break
        if state_obs_key is None and schema.observation_fields:
            state_obs_key = next(iter(schema.observation_fields.keys()))

    logger.info(f"Using state observation: {state_obs_key}")
    logger.info(f"Using action key: {final_action_key}")

    # Get episodes to convert
    episodes = schema.episode_names
    if num_episodes is not None:
        episodes = episodes[:num_episodes]
        logger.info(f"Converting only first {num_episodes} episodes")

    logger.info(f"Creating LeRobot dataset: {repo_id}")

    # Create LeRobot dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=final_fps,
        root=root,
        robot_type=final_robot_type,
        features=features,
        use_videos=use_videos,
    )

    # Start image writer for efficient image saving
    if use_videos and camera_names:
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=4)

    # Convert each episode
    for episode_idx, episode_name in enumerate(episodes):
        msg = f"Converting episode {episode_idx + 1}/{len(episodes)}: {episode_name}"
        logger.info(msg)

        # Load episode data using inspector
        episode_data = inspector.load_episode_data(
            episode_name,
            include_cameras=use_videos,
            camera_names=camera_names,
            action_key=final_action_key,
        )
        num_frames = episode_data["num_frames"]

        if num_frames == 0:
            logger.warning(f"Episode {episode_name} has no frames, skipping")
            continue

        # Add frames to dataset
        for frame_idx in range(num_frames):
            # Prepare frame dictionary
            frame = {
                "task": task_instruction,
            }

            # Add action
            if episode_data["actions"] is not None:
                frame["action"] = episode_data["actions"][frame_idx]

            # Add state observation
            if state_obs_key and state_obs_key in episode_data["obs"]:
                obs_data = episode_data["obs"][state_obs_key]
                frame["observation.state"] = obs_data[frame_idx]

            # Add camera observations
            if use_videos:
                for cam_name in camera_names:
                    if cam_name in episode_data["camera_obs"]:
                        cam_data = episode_data["camera_obs"][cam_name]
                        rgb_array = cam_data[frame_idx]
                        # Convert to PIL Image (HWC format, uint8)
                        key = f"observation.images.{cam_name}"
                        frame[key] = Image.fromarray(rgb_array)

            # Add frame to dataset
            lerobot_dataset.add_frame(frame)

        # Save episode
        lerobot_dataset.save_episode()
        msg = f"Saved episode {episode_idx} with {num_frames} frames"
        logger.info(msg)

    # Stop image writer
    if use_videos and camera_names:
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
    parser = argparse.ArgumentParser(description="Convert Isaac Lab Arena HDF5 dataset to LeRobot format")
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
        "--task-instruction",
        type=str,
        default="Perform the task",
        help="Task instruction/description",
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
    parser.add_argument(
        "--state-key",
        type=str,
        default=None,
        help="Observation key to use as state (auto-detected if not set)",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default=None,
        help="Action key to use (default: 'actions')",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type (auto-detected from env name if not set)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS (auto-detected from sim args if not set)",
    )

    args = parser.parse_args()

    convert_hdf5_to_lerobot(
        hdf5_path=args.hdf5_path,
        repo_id=args.repo_id,
        root=args.root,
        task_instruction=args.task_instruction,
        push_to_hub=args.push_to_hub,
        num_episodes=args.num_episodes,
        use_videos=not args.no_videos,
        state_key=args.state_key,
        action_key=args.action_key,
        robot_type=args.robot_type,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
