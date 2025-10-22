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
import time
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

DROID_SHARDS = 2048
DROID_FPS = 15
DROID_ROBOT_TYPE = "Franka"

# Dataset schema slightly adapted from: https://droid-dataset.github.io/droid/the-droid-dataset.html#-dataset-schema
DROID_FEATURES = {
    # true on first step of the episode
    "is_first": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    # true on last step of the episode
    "is_last": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    # true on last step of the episode if it is a terminal step, True for demos
    "is_terminal": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    # language_instruction is also stored as "task" to follow LeRobot standard
    "language_instruction": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "language_instruction_2": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "language_instruction_3": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "observation.state.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": {
            "axes": ["gripper"],
        },
    },
    "observation.state.cartesian_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
    },
    "observation.state.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
    },
    # Add this new feature to follow LeRobot standard of using joint position + gripper
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
    },
    # Initially called wrist_image_left
    "observation.images.wrist_left": {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": [
            "height",
            "width",
            "channels",
        ],
    },
    # Initially called exterior_image_1_left
    "observation.images.exterior_1_left": {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": [
            "height",
            "width",
            "channels",
        ],
    },
    # Initially called exterior_image_2_left
    "observation.images.exterior_2_left": {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": [
            "height",
            "width",
            "channels",
        ],
    },
    "action.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": {
            "axes": ["gripper"],
        },
    },
    "action.gripper_velocity": {
        "dtype": "float32",
        "shape": (1,),
        "names": {
            "axes": ["gripper"],
        },
    },
    "action.cartesian_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
    },
    "action.cartesian_velocity": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
    },
    "action.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
    },
    "action.joint_velocity": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
    },
    # This feature was called "action" in RLDS dataset and consists of [6x joint velocities, 1x gripper position]
    "action.original": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
    },
    # Add this new feature to follow LeRobot standard of using joint position + gripper
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
    },
    "discount": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    # Meta data that are the same for all frames in the episode
    "task_category": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "building": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "collector_id": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "date": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "camera_extrinsics.wrist_left": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
    },
    "camera_extrinsics.exterior_1_left": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
    },
    "camera_extrinsics.exterior_2_left": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
    },
    "is_episode_successful": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}


def is_episode_successful(tf_episode_metadata):
    # Adapted from: https://github.com/droid-dataset/droid_policy_learning/blob/dd1020eb20d981f90b5ff07dc80d80d5c0cb108b/robomimic/utils/rlds_utils.py#L8
    return "/success/" in tf_episode_metadata["file_path"].numpy().decode()


def generate_lerobot_frames(tf_episode):
    m = tf_episode["episode_metadata"]
    frame_meta = {
        "task_category": m["building"].numpy().decode(),
        "building": m["building"].numpy().decode(),
        "collector_id": m["collector_id"].numpy().decode(),
        "date": m["date"].numpy().decode(),
        "camera_extrinsics.wrist_left": m["extrinsics_wrist_cam"].numpy(),
        "camera_extrinsics.exterior_1_left": m["extrinsics_exterior_cam_1"].numpy(),
        "camera_extrinsics.exterior_2_left": m["extrinsics_exterior_cam_2"].numpy(),
        "is_episode_successful": np.array([is_episode_successful(m)]),
    }
    for f in tf_episode["steps"]:
        # Dataset schema slightly adapted from: https://droid-dataset.github.io/droid/the-droid-dataset.html#-dataset-schema
        frame = {
            "is_first": np.array([f["is_first"].numpy()]),
            "is_last": np.array([f["is_last"].numpy()]),
            "is_terminal": np.array([f["is_terminal"].numpy()]),
            "language_instruction": f["language_instruction"].numpy().decode(),
            "language_instruction_2": f["language_instruction_2"].numpy().decode(),
            "language_instruction_3": f["language_instruction_3"].numpy().decode(),
            "observation.state.gripper_position": f["observation"]["gripper_position"].numpy(),
            "observation.state.cartesian_position": f["observation"]["cartesian_position"].numpy(),
            "observation.state.joint_position": f["observation"]["joint_position"].numpy(),
            "observation.images.wrist_left": f["observation"]["wrist_image_left"].numpy(),
            "observation.images.exterior_1_left": f["observation"]["exterior_image_1_left"].numpy(),
            "observation.images.exterior_2_left": f["observation"]["exterior_image_2_left"].numpy(),
            "action.gripper_position": f["action_dict"]["gripper_position"].numpy(),
            "action.gripper_velocity": f["action_dict"]["gripper_velocity"].numpy(),
            "action.cartesian_position": f["action_dict"]["cartesian_position"].numpy(),
            "action.cartesian_velocity": f["action_dict"]["cartesian_velocity"].numpy(),
            "action.joint_position": f["action_dict"]["joint_position"].numpy(),
            "action.joint_velocity": f["action_dict"]["joint_velocity"].numpy(),
            "discount": np.array([f["discount"].numpy()]),
            "reward": np.array([f["reward"].numpy()]),
            "action.original": f["action"].numpy(),
        }

        # language_instruction is also stored as "task" to follow LeRobot standard
        frame["task"] = frame["language_instruction"]

        # Add this new feature to follow LeRobot standard of using joint position + gripper
        frame["observation.state"] = np.concatenate(
            [frame["observation.state.joint_position"], frame["observation.state.gripper_position"]]
        )
        frame["action"] = np.concatenate([frame["action.joint_position"], frame["action.gripper_position"]])

        # Meta data that are the same for all frames in the episode
        frame.update(frame_meta)

        # Cast fp64 to fp32
        for key in frame:
            if isinstance(frame[key], np.ndarray) and frame[key].dtype == np.float64:
                frame[key] = frame[key].astype(np.float32)

        yield frame


def port_droid(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
):
    dataset_name = raw_dir.parent.name
    version = raw_dir.name
    data_dir = raw_dir.parent.parent

    builder = tfds.builder(f"{dataset_name}/{version}", data_dir=data_dir, version="")

    if num_shards is not None:
        tfds_num_shards = builder.info.splits["train"].num_shards
        if tfds_num_shards != DROID_SHARDS:
            raise ValueError(
                f"Number of shards of Droid dataset is expected to be {DROID_SHARDS} but is {tfds_num_shards}."
            )
        if num_shards != tfds_num_shards:
            raise ValueError(
                f"We only shard over the fixed number of shards provided by tensorflow dataset ({tfds_num_shards}), but {num_shards} shards provided instead."
            )
        if shard_index >= tfds_num_shards:
            raise ValueError(
                f"Shard index is greater than the num of shards ({shard_index} >= {num_shards})."
            )

        raw_dataset = builder.as_dataset(split=f"train[{shard_index}shard]")
    else:
        raw_dataset = builder.as_dataset(split="train")

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=DROID_ROBOT_TYPE,
        fps=DROID_FPS,
        features=DROID_FEATURES,
    )

    start_time = time.time()
    num_episodes = raw_dataset.cardinality().numpy().item()
    logging.info(f"Number of episodes {num_episodes}")

    for episode_index, episode in enumerate(raw_dataset):
        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

        logging.info(
            f"{episode_index} / {num_episodes} episodes processed (after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)"
        )

        for frame in generate_lerobot_frames(episode):
            lerobot_dataset.add_frame(frame)

        lerobot_dataset.save_episode()
        logging.info("Save_episode")

    lerobot_dataset.finalize()

    if push_to_hub:
        lerobot_dataset.push_to_hub(
            # Add openx tag, since it belongs to the openx collection of datasets
            tags=["openx"],
            private=False,
        )


def validate_dataset(repo_id):
    """Sanity check that ensure meta data can be loaded and all files are present."""
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
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
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
        help="Number of shards. Can be either None to load the full dataset, or 2048 to load one of the 2048 tensorflow dataset files.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Index of the shard. Can be either None to load the full dataset, or in [0,2047] to load one of the 2048 tensorflow dataset files.",
    )

    args = parser.parse_args()

    port_droid(**vars(args))


if __name__ == "__main__":
    main()
