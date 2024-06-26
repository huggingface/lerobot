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
"""Process BridgeData V2 files"""

import gc
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir) -> bool:
    traj_search_path = os.path.join("**", "traj_group*", "traj*")
    traj_folders = list(raw_dir.glob(traj_search_path))

    assert len(traj_folders) != 0
    for traj in traj_folders:
        # Data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
        date_time = datetime.strptime(str(traj).split("/")[-4], "%Y-%m-%d_%H-%M-%S")
        latency_shift = date_time < datetime(2021, 7, 23)
        is_in_scripted_raw = "scripted_raw" in str(traj)
        if len(list(traj.glob("*.pkl"))) == 2:  # ~30 trajectories dont have both files
            actions = load_actions(traj)
            num_frames = len(list(traj.glob(os.path.join("images0", "*.jpg"))))
            state, time_stamp = load_state_timestamp(traj)
            if is_in_scripted_raw:
                assert len(actions) == num_frames - 1
                assert len(state) == num_frames
                assert len(time_stamp) == num_frames
            else:
                if latency_shift:
                    assert len(actions) == num_frames - 1
                    assert len(state) == num_frames - 1
                    assert len(time_stamp) == num_frames - 1
                else:
                    assert len(actions) == num_frames - 1
                    assert len(state) == num_frames
                    assert len(time_stamp) == num_frames


def load_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return np.array(act_list)


def load_state_timestamp(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"], x["time_stamp"]


def load_from_raw(raw_dir: Path, videos_dir: Path, fps: int, video: bool, episodes: list[int] | None = None):
    traj_search_path = os.path.join("**", "traj_group*", "traj*")
    traj_folders = list(raw_dir.glob(traj_search_path))
    ep_dicts = []

    # remove paths from traj_folders that don't have actions
    traj_folders = [ep_path for ep_path in traj_folders if len(list(ep_path.glob("*.pkl"))) == 2]

    num_episodes = len(traj_folders)
    ep_ids = episodes if episodes else range(num_episodes)

    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = traj_folders[ep_idx]

        is_in_scripted_raw = "scripted_raw" in str(ep_path)

        if not is_in_scripted_raw:
            date_time = datetime.strptime(str(ep_path).split("/")[-4], "%Y-%m-%d_%H-%M-%S")
            latency_shift = date_time < datetime(2021, 7, 23)

        image_paths = list(ep_path.glob(os.path.join("images0", "*.jpg")))
        image_paths = sorted(
            image_paths, key=lambda name: int(str(name).split("im_")[1].split(".jpg")[0].zfill(5))
        )
        num_frames = len(image_paths)

        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        actions = load_actions(ep_path)
        state, time_stamp = load_state_timestamp(ep_path)
        ep_dict = {}
        img_key = "observation.image"
        imgs_array = [str(x) for x in image_paths]

        if is_in_scripted_raw:
            actions = np.insert(
                actions, 0, actions[0], axis=0
            )  # duplicate first action to compensate for extra frame
        else:
            if latency_shift:
                state = state[1:]
                actions = actions[1:]
                time_stamp = time_stamp[1:]

        if video:
            # save png images in temporary directory
            tmp_imgs_dir = videos_dir / "tmp_images"
            save_images_concurrently(imgs_array, tmp_imgs_dir)

            # encode images to a mp4 video
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            video_path = videos_dir / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps)

            # clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            ep_dict[img_key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[img_key] = imgs_array

        ep_dict["action"] = torch.from_numpy(actions)
        ep_dict["observation.state"] = torch.from_numpy(state)
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["next.done"] = done

        assert isinstance(ep_idx, int)
        ep_dicts.append(ep_dict)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}
    if video:
        features["observation.image"] = VideoFrame()
    else:
        features["observation.image"] = Image()
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
        fps = 5

    data_dir = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dir, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
