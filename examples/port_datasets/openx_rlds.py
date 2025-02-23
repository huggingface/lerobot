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

NOTE: Install `tensorflow` and `tensorflow_datasets` before running this script.
```bash
pip install tensorflow
pip install tensorflow_datasets
```

Example:
```bash
python examples/port_datasets/openx_rlds.py \
    --raw-dir /fsx/mustafa_shukor/droid \
    --repo-id cadene/droid \
    --use-videos \
    --push-to-hub
```
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from examples.port_datasets.openx_utils.configs import OXE_DATASET_CONFIGS, StateEncoding
from examples.port_datasets.openx_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

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


def generate_features_from_raw(dataset_name: str, builder: tfds.core.DatasetBuilder, use_videos: bool = True):
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


def save_as_lerobot_dataset(
    dataset_name: str,
    lerobot_dataset: LeRobotDataset,
    raw_dataset: tf.data.Dataset,
    num_shards: int | None = None,
    shard_index: int | None = None,
):
    start_time = time.time()
    total_num_episodes = raw_dataset.cardinality().numpy().item()
    logging.info(f"Total number of episodes {total_num_episodes}")

    if num_shards is not None:
        sharded_dataset = raw_dataset.shard(num_shards=num_shards, index=shard_index)
        sharded_num_episodes = sharded_dataset.cardinality().numpy().item()
        logging.info(f"{sharded_num_episodes=}")
        num_episodes = sharded_num_episodes
        iter_ = iter(sharded_dataset)
    else:
        num_episodes = total_num_episodes
        iter_ = iter(raw_dataset)

    if num_episodes <= 0:
        raise ValueError(f"Number of episodes is {num_episodes}, but needs to be positive.")

    for episode_index in range(num_episodes):
        logging.info(f"{episode_index} / {num_episodes} episodes processed")

        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)
        logging.info(f"It has been {d} days, {h} hours, {m} minutes, {s:.3f} seconds")

        episode = next(iter_)
        logging.info("next")
        episode = transform_raw_dataset(episode, dataset_name)

        traj = episode["steps"]
        for i in range(traj["action"].shape[0]):
            image_dict = {
                f"observation.images.{key}": value[i].numpy()
                for key, value in traj["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }
            lerobot_dataset.add_frame(
                {
                    **image_dict,
                    "observation.state": traj["proprio"][i].numpy(),
                    "action": traj["action"][i].numpy(),
                    "task": traj["task"][i].numpy().decode(),
                }
            )

        lerobot_dataset.save_episode()
        logging.info("save_episode")


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    push_to_hub: bool = False,
    fps: int = None,
    robot_type: str = None,
    use_videos: bool = True,
    image_writer_process: int = 5,
    image_writer_threads: int = 10,
    num_shards: int | None = None,
    shard_index: int | None = None,
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

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(dataset_name, builder, use_videos)

    if num_shards is not None:
        if num_shards != builder.info.splits["train"].num_shards:
            raise ValueError()
        if shard_index >= builder.info.splits["train"].num_shards:
            raise ValueError()

        raw_dataset = builder.as_dataset(split=f"train[{shard_index}shard]")
    else:
        raw_dataset = builder.as_dataset(split="train")

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
        fps=fps,
        use_videos=use_videos,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_process,
    )

    save_as_lerobot_dataset(
        dataset_name,
        lerobot_dataset,
        raw_dataset,
    )

    if push_to_hub:
        assert repo_id is not None
        tags = []
        if dataset_name in OXE_DATASET_CONFIGS:
            tags.append("openx")
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
        "--image-writer-process",
        type=int,
        default=0,
        help="Number of processes of image writer for saving images.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="Number of threads per process of image writer for saving images.",
    )

    args = parser.parse_args()

    # droid_dir = Path("/fsx/remi_cadene/.cache/huggingface/lerobot/cadene/droid")
    # if droid_dir.exists():
    #     shutil.rmtree(droid_dir)

    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()
