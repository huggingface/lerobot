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
"""Process pickle files formatted like in: https://github.com/fyhMer/fowm"""

import pickle
import shutil
from pathlib import Path

import einops
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir):
    keys = {"actions", "rewards", "dones"}
    nested_keys = {"observations": {"rgb", "state"}, "next_observations": {"rgb", "state"}}

    xarm_files = list(raw_dir.glob("*.pkl"))
    assert len(xarm_files) > 0

    with open(xarm_files[0], "rb") as f:
        dataset_dict = pickle.load(f)

    assert isinstance(dataset_dict, dict)
    assert all(k in dataset_dict for k in keys)

    # Check for consistent lengths in nested keys
    expected_len = len(dataset_dict["actions"])
    assert all(len(dataset_dict[key]) == expected_len for key in keys if key in dataset_dict)

    for key, subkeys in nested_keys.items():
        nested_dict = dataset_dict.get(key, {})
        assert all(len(nested_dict[subkey]) == expected_len for subkey in subkeys if subkey in nested_dict)


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    pkl_path = raw_dir / "buffer.pkl"

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    # load data indices from which each episode starts and ends
    from_ids, to_ids = [], []
    from_idx, to_idx = 0, 0
    for done in pkl_data["dones"]:
        to_idx += 1
        if not done:
            continue
        from_ids.append(from_idx)
        to_ids.append(to_idx)
        from_idx = to_idx

    num_episodes = len(from_ids)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx, selected_ep_idx in tqdm.tqdm(enumerate(ep_ids)):
        from_idx = from_ids[selected_ep_idx]
        to_idx = to_ids[selected_ep_idx]
        num_frames = to_idx - from_idx

        image = torch.tensor(pkl_data["observations"]["rgb"][from_idx:to_idx])
        image = einops.rearrange(image, "b c h w -> b h w c")
        state = torch.tensor(pkl_data["observations"]["state"][from_idx:to_idx])
        action = torch.tensor(pkl_data["actions"][from_idx:to_idx])
        # TODO(rcadene): we have a missing last frame which is the observation when the env is done
        # it is critical to have this frame for tdmpc to predict a "done observation/state"
        # next_image = torch.tensor(pkl_data["next_observations"]["rgb"][from_idx:to_idx])
        # next_state = torch.tensor(pkl_data["next_observations"]["state"][from_idx:to_idx])
        next_reward = torch.tensor(pkl_data["rewards"][from_idx:to_idx])
        next_done = torch.tensor(pkl_data["dones"][from_idx:to_idx])

        ep_dict = {}

        imgs_array = [x.numpy() for x in image]
        img_key = "observation.image"
        if video:
            # save png images in temporary directory
            tmp_imgs_dir = videos_dir / "tmp_images"
            save_images_concurrently(imgs_array, tmp_imgs_dir)

            # encode images to a mp4 video
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            video_path = videos_dir / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

            # clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            ep_dict[img_key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        ep_dict["observation.state"] = state
        ep_dict["action"] = action
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        # ep_dict["next.observation.image"] = next_image
        # ep_dict["next.observation.state"] = next_state
        ep_dict["next.reward"] = next_reward
        ep_dict["next.done"] = next_done
        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video):
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
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # TODO(rcadene): add success
    # features["next.success"] = Value(dtype='bool', id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 15

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info
