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

import argparse
import logging
import re
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

# Default FPS for datasets without specific config
DEFAULT_FPS = 10
DEFAULT_ROBOT_TYPE = "unknown"


def determine_dataset_info(raw_dir: Path):
    """Determine dataset name and version from directory structure."""
    last_part = raw_dir.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir.parent.name
        data_dir = raw_dir.parent.parent
    else:
        version = ""
        dataset_name = last_part
        data_dir = raw_dir.parent
    return dataset_name, version, data_dir


def generate_features_from_builder(builder, dataset_name: str) -> dict[str, Any]:
    """Generate LeRobot features schema from TFDS builder and dataset config."""

    # Generate state names based on encoding type
    state_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        state_encoding = OXE_DATASET_CONFIGS[dataset_name]["state_encoding"]
        if state_encoding == StateEncoding.POS_EULER:
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
            if "libero" in dataset_name:
                state_names = [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper",
                    "gripper",
                ]  # 2D gripper state
        elif state_encoding == StateEncoding.POS_QUAT:
            state_names = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        elif state_encoding == StateEncoding.JOINT:
            state_names = [f"motor_{i}" for i in range(7)] + ["gripper"]
            state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
            pad_count = state_obs_keys[:-1].count(None)
            state_names[-pad_count - 1 : -1] = ["pad"] * pad_count
            state_names[-1] = "pad" if state_obs_keys[-1] is None else state_names[-1]

    # Generate action names based on encoding type
    action_names = [f"motor_{i}" for i in range(8)]
    if dataset_name in OXE_DATASET_CONFIGS:
        action_encoding = OXE_DATASET_CONFIGS[dataset_name]["action_encoding"]
        if action_encoding == ActionEncoding.EEF_POS:
            action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        elif action_encoding == ActionEncoding.JOINT_POS:
            action_names = [f"motor_{i}" for i in range(7)] + ["gripper"]

    # Base features (state and action)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": {"axes": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"axes": action_names},
        },
    }

    # Add image features from TFDS builder info
    obs_features = builder.info.features["steps"]["observation"]
    for key, value in obs_features.items():
        # Skip depth images and non-image features
        if "depth" in key or not any(x in key for x in ["image", "rgb"]):
            continue

        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": tuple(value.shape),
            "names": ["height", "width", "channels"],
        }

    return features


def transform_raw_dataset(episode, dataset_name: str):
    """Apply OXE standardization transforms to raw TFDS episode."""
    # Batch all steps in the episode
    traj = next(iter(episode["steps"].batch(episode["steps"].cardinality())))

    # Apply dataset-specific transform if available
    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    # Create consolidated state vector
    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    # Build proprio (proprioceptive state) vector
    proprio_components = []
    for key in state_obs_keys:
        if key is None:
            # Add padding for missing state components
            component = tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)
        else:
            component = tf.cast(traj["observation"][key], tf.float32)
            # Ensure component has right shape (add dimension if needed)
            if len(component.shape) == 1:
                component = component[:, None]
        proprio_components.append(component)

    proprio = tf.concat(proprio_components, axis=1)

    # Update trajectory with standardized format
    traj.update(
        {
            "proprio": proprio,
            "task": traj.get("language_instruction", ""),
            "action": tf.cast(traj["action"], tf.float32),
        }
    )

    episode["steps"] = traj
    return episode


def generate_lerobot_frames(tf_episode):
    """Generate LeRobot frames from transformed TFDS episode."""
    traj = tf_episode["steps"]

    # Get the task/language instruction
    if isinstance(traj["task"], tf.Tensor):
        if traj["task"].dtype == tf.string:
            task = traj["task"][0].numpy().decode() if len(traj["task"]) > 0 else ""
        else:
            task = str(traj["task"][0].numpy()) if len(traj["task"]) > 0 else ""
    else:
        task = str(traj["task"]) if traj["task"] else ""

    # Iterate through each timestep
    num_steps = tf.shape(traj["action"])[0].numpy()
    for i in range(num_steps):
        frame = {}

        # Add observation state
        frame["observation.state"] = traj["proprio"][i].numpy()

        # Add action
        frame["action"] = traj["action"][i].numpy()

        # Add images
        for key, value in traj["observation"].items():
            if any(x in key for x in ["image", "rgb"]) and "depth" not in key:
                frame[f"observation.images.{key}"] = value[i].numpy()

        # Add task
        frame["task"] = task

        # Cast fp64 to fp32
        for key in frame:
            if isinstance(frame[key], np.ndarray) and frame[key].dtype == np.float64:
                frame[key] = frame[key].astype(np.float32)

        yield frame


def port_rlds(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
):
    """Port RLDS dataset to LeRobot format."""

    # Determine dataset info
    dataset_name, version, data_dir = determine_dataset_info(raw_dir)

    # Build TFDS dataset
    builder = tfds.builder(
        f"{dataset_name}/{version}" if version else dataset_name, data_dir=data_dir, version=version
    )

    # Handle sharding if specified
    if num_shards is not None and shard_index is not None:
        if shard_index >= num_shards:
            raise ValueError(f"Shard index {shard_index} >= num_shards {num_shards}")

        # Calculate shard splits
        total_episodes = builder.info.splits["train"].num_examples
        episodes_per_shard = total_episodes // num_shards
        start_idx = shard_index * episodes_per_shard
        if shard_index == num_shards - 1:
            # Last shard gets remaining episodes
            end_idx = total_episodes
        else:
            end_idx = start_idx + episodes_per_shard

        split_str = f"train[{start_idx}:{end_idx}]"
        raw_dataset = builder.as_dataset(split=split_str)
    else:
        raw_dataset = builder.as_dataset(split="train")

    # Apply filtering (e.g., success filter for kuka)
    if dataset_name == "kuka":
        raw_dataset = raw_dataset.filter(lambda e: e["success"])

    # Apply transformations
    raw_dataset = raw_dataset.map(partial(transform_raw_dataset, dataset_name=dataset_name))

    # Get dataset configuration
    fps = DEFAULT_FPS
    robot_type = DEFAULT_ROBOT_TYPE

    if dataset_name in OXE_DATASET_CONFIGS:
        config = OXE_DATASET_CONFIGS[dataset_name]
        fps = config.get("control_frequency", DEFAULT_FPS)
        robot_type = config.get("robot_type", DEFAULT_ROBOT_TYPE)
        robot_type = robot_type.lower().replace(" ", "_").replace("-", "_")

    # Generate features schema
    features = generate_features_from_builder(builder, dataset_name)

    # Create LeRobot dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=int(fps),
        features=features,
    )

    # Process episodes
    start_time = time.time()
    num_episodes = raw_dataset.cardinality().numpy().item()
    logging.info(f"Number of episodes: {num_episodes}")

    for episode_index, episode in enumerate(raw_dataset):
        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

        logging.info(
            f"{episode_index} / {num_episodes} episodes processed "
            f"(after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)"
        )

        # Generate and add frames
        for frame in generate_lerobot_frames(episode):
            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode()
        logging.info("Save_episode")

    # Push to hub if requested
    if push_to_hub:
        tags = ["openx", dataset_name]
        if robot_type != "unknown":
            tags.append(robot_type)

        lerobot_dataset.push_to_hub(
            tags=tags,
            private=False,
        )


def validate_dataset(repo_id):
    """Sanity check that ensures metadata can be loaded and all files are present."""
    meta = LeRobotDatasetMetadata(repo_id)

    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")

    for ep_idx in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(ep_idx)

        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")

        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Video file is missing in: {vid_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to hub.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to split the dataset into for parallel processing.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Index of the shard to process (0-indexed).",
    )

    args = parser.parse_args()

    port_rlds(**vars(args))


if __name__ == "__main__":
    main()
