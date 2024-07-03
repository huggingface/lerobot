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
Contains utilities to process raw data format of npz files from TRI sim environments.
"""

import os
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import tqdm
import yaml
from datasets import Dataset, Features, Image, Sequence, Value
from scipy.spatial.transform import Rotation as R  # noqa: N817

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

PREFIX = "processed"
SUFFIX = "npz"


def get_cameras(metadata: Dict):
    return metadata["camera_id_to_semantic_name"]


def process_actions(actions: np.ndarray) -> np.ndarray:
    translation_1 = actions[:, 0:3]
    rotation_1 = actions[:, 3:9]
    translation_2 = actions[:, 9:12]
    rotation_2 = actions[:, 12:18]
    gripper_1 = actions[:, 18:19]
    gripper_2 = actions[:, 19:20]

    # Now complete the rotation matrix
    def calculate_third_row(row_1, row_2):
        row_3 = np.cross(row_1, row_2)
        return row_3 / np.linalg.norm(row_3, axis=-1, keepdims=True)

    def calculate_rotation_matrix(row_1, row_2):
        row_3 = calculate_third_row(row_1, row_2)
        return np.stack([row_1, row_2, row_3], axis=-1)

    def calculate_axis_angle(rotation_matrix):
        axis_angle = R.from_matrix(rotation_matrix).as_rotvec()
        return axis_angle

    rotation_1_axis_angle = calculate_axis_angle(
        calculate_rotation_matrix(rotation_1[:, 0:3], rotation_1[:, 3:6])
    )
    rotation_2_axis_angle = calculate_axis_angle(
        calculate_rotation_matrix(rotation_2[:, 0:3], rotation_2[:, 3:6])
    )
    action_1 = np.concatenate([translation_1, rotation_1_axis_angle, gripper_1], axis=-1)
    action_2 = np.concatenate([translation_2, rotation_2_axis_angle, gripper_2], axis=-1)
    actions = np.concatenate([action_1, action_2], axis=-1)
    return actions


def check_format(raw_dir: Path) -> bool:
    # only frames from simulation are uncompressed
    num_episodes = len([x for x in os.listdir(raw_dir) if x.startswith("episode")])
    required_files = [
        "summary",
        "observations",
        "actions",
        "intrinsics",
        "extrinsics",
        "detailed_reward_traj",
        "detailed_task_predicate_traj",
    ]

    for ep_idx in range(num_episodes):
        ep_folder: Path = raw_dir / f"episode_{ep_idx}" / PREFIX
        assert ep_folder.exists()

        metadata_path = ep_folder / "metadata.yaml"
        assert metadata_path.exists()
        metadata = yaml.safe_load(metadata_path.read_text())
        cameras = get_cameras(metadata)

        assert len(cameras) > 0

        for camera in cameras:
            assert (ep_folder / f"images_{camera}").exists()
            assert (ep_folder / f"images_{camera}").is_dir()

        for file in required_files:
            assert (ep_folder / f"{file}.{SUFFIX}").exists()

    return True


def load_from_raw(
    raw_dir: Path, videos_dir: Path, fps: int = 30, video: bool = True, episodes: list[int] | None = None
):
    # only frames from simulation are uncompressed
    num_episodes = len([x for x in os.listdir(raw_dir) if x.startswith("episode")])

    # Only video form is supported for now
    assert video, "Only video form is supported for now."

    state_variables = [
        "robot__actual__poses__right::panda__xyz",
        "robot__actual__poses__right::panda__rot_6d",
        "robot__actual__poses__left::panda__xyz",
        "robot__actual__poses__left::panda__rot_6d",
        "robot__actual__grippers__right::panda_hand",
        "robot__actual__grippers__left::panda_hand",
    ]

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = raw_dir / f"episode_{ep_idx}" / PREFIX

        actions_before_processing = np.load(ep_path / "actions.npz")["actions"]
        num_frames = actions_before_processing.shape[0]
        actions = process_actions(actions_before_processing)

        ep_dict = {}

        metadata = yaml.safe_load((ep_path / "metadata.yaml").read_text())
        cameras = get_cameras(metadata)

        observations = np.load(ep_path / "observations.npz")
        for camera, canonical_name in cameras.items():
            img_key = f"observation.images.{canonical_name}"
            video_fname = f"{canonical_name}_episode_{ep_idx:06d}.mp4"

            if video and not (videos_dir / video_fname).exists():
                if not (ep_path / video_fname).exists():
                    print("Creating video for camera ", camera)
                    print(ep_path / video_fname)
                    # lazily load all images in RAM
                    imgs_array = observations[camera]

                    # save png images in temporary directory
                    tmp_imgs_dir = ep_path / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    encode_video_frames(tmp_imgs_dir, ep_path / video_fname, fps)

                    # delete tmp directory
                    shutil.rmtree(tmp_imgs_dir)

                # copy the video to the video directory
                shutil.copy2(ep_path / video_fname, videos_dir / video_fname)

            ep_dict[img_key] = [
                {"path": f"videos/{video_fname}", "timestamp": i / fps} for i in range(num_frames)
            ]

        ep_dict["action"] = torch.from_numpy(actions)
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

        ep_dict["observation.state"] = torch.cat(
            [torch.from_numpy(observations[state_vars]) for state_vars in state_variables], dim=-1
        )

        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True
        ep_dict["next.done"] = done

        assert isinstance(ep_idx, int)
        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video: bool) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
