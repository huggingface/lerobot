#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
For all datasets in the RLDS format.
For https://github.com/google-deepmind/open_x_embodiment (OPENX) datasets.

NOTE: You need to install tensorflow and tensorflow_datsets before running this script.

Example:
    python openx_rlds.py \
        --raw-dir /path/to/bridge_orig/1.0.0 \
        --local-dir /path/to/local_dir \
        --repo-id your_id \
        --use-videos \
        --push-to-hub
"""

import argparse
import os
import re
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
oxe_utils_dir = os.path.join(current_dir, "oxe_utils")
sys.path.append(oxe_utils_dir)

from oxe_utils.configs import OXE_DATASET_CONFIGS, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

np.set_printoptions(precision=2)


def transform_raw_dataset(episode, dataset_name):
    traj = next(iter(episode["steps"].batch(episode["steps"].cardinality())))

    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS:
        traj = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](traj)

    if dataset_name in OXE_DATASET_CONFIGS:
        state_obs_keys = OXE_DATASET_CONFIGS[dataset_name]["state_obs_keys"]
    else:
        state_obs_keys = [None for _ in range(8)]

    proprio = tf.concat(
        [
            (
                tf.zeros((tf.shape(traj["action"])[0], 1), dtype=tf.float32)  # padding
                if key is None
                else tf.cast(traj["observation"][key], tf.float32)
            )
            for key in state_obs_keys
        ],
        axis=1,
    )

    traj.update(
        {
            "proprio": proprio,
            "task": traj.pop("language_instruction"),
            "action": tf.cast(traj["action"], tf.float32),
        }
    )

    episode["steps"] = traj
    return episode


def generate_features_from_raw(builder: tfds.core.DatasetBuilder, use_videos: bool = True):
    dataset_name = builder.name

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

    DEFAULT_FEATURES = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"motors": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        },
    }

    obs = builder.info.features["steps"]["observation"]
    features = {
        f"observation.images.{key}": {
            "dtype": "video" if use_videos else "image",
            "shape": value.shape,
            "names": ["height", "width", "rgb"],
        }
        for key, value in obs.items()
        if "depth" not in key and any(x in key for x in ["image", "rgb"])
    }
    return {**features, **DEFAULT_FEATURES}


def save_as_lerobot_dataset(lerobot_dataset: LeRobotDataset, raw_dataset: tf.data.Dataset, **kwargs):
    for episode in raw_dataset.as_numpy_iterator():
        traj = episode["steps"]
        for i in range(traj["action"].shape[0]):
            image_dict = {
                f"observation.images.{key}": value[i]
                for key, value in traj["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }
            lerobot_dataset.add_frame(
                {
                    **image_dict,
                    "observation.state": traj["proprio"][i],
                    "action": traj["action"][i],
                }
            )
        lerobot_dataset.save_episode(task=traj["task"][0].decode())

    lerobot_dataset.consolidate(
        run_compute_stats=True,
        keep_image_files=kwargs["keep_images"],
        stat_kwargs={"batch_size": kwargs["batch_size"], "num_workers": kwargs["num_workers"]},
    )


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    local_dir: Path = None,
    push_to_hub: bool = False,
    fps: int = None,
    robot_type: str = None,
    use_videos: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    image_writer_process: int = 5,
    image_writer_threads: int = 10,
    keep_images: bool = True,
):
    last_part = raw_dir.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir.parent.name
        data_dir = raw_dir.parent.parent
    else:
        version = ""
        dataset_name = last_part
        data_dir = raw_dir.parent

    if local_dir is None:
        local_dir = Path(LEROBOT_HOME)
    local_dir /= f"{dataset_name}_{version}_lerobot"
    if local_dir.exists():
        shutil.rmtree(local_dir)

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, use_videos)
    raw_dataset = builder.as_dataset(split="train").map(
        partial(transform_raw_dataset, dataset_name=dataset_name)
    )

    if fps is None:
        if dataset_name in OXE_DATASET_CONFIGS:
            fps = OXE_DATASET_CONFIGS[dataset_name]["control_frequency"]
        else:
            fps = 10

    if robot_type is None:
        if dataset_name in OXE_DATASET_CONFIGS:
            robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"]
            robot_type = robot_type.lower().replace(" ", "_").replace("-", "_")
        else:
            robot_type = "unknown"

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        root=local_dir,
        fps=fps,
        use_videos=use_videos,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_process,
    )

    save_as_lerobot_dataset(
        lerobot_dataset, raw_dataset, keep_images=keep_images, batch_size=batch_size, num_workers=num_workers
    )

    if push_to_hub:
        assert repo_id is not None
        tags = ["LeRobot", dataset_name, "rlds"]
        if dataset_name in OXE_DATASET_CONFIGS:
            tags.append("openx")
        if robot_type != "unknown":
            tags.append(robot_type)
        lerobot_dataset.push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
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
        "--robot-type",
        type=str,
        default=None,
        help="Robot type of this dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate used to collect videos. Default fps equals to the control frequency of the robot.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--image-writer-process",
        type=int,
        default=5,
        help="Number of processes of image writer for saving images.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Number of threads per process of image writer for saving images.",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Whether to keep the cached images.",
    )

    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()
