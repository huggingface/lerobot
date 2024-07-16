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
Contains utilities to process raw data format of png images files recorded with capture_camera_feed.py
"""

from pathlib import Path

import torch
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch
from lerobot.common.datasets.video_utils import VideoFrame


def check_format(raw_dir: Path) -> bool:
    image_paths = list(raw_dir.glob("frame_*.png"))
    if len(image_paths) == 0:
        raise ValueError


def load_from_raw(raw_dir: Path, fps: int, episodes: list[int] | None = None):
    if episodes is not None:
        # TODO(aliberts): add support for multi-episodes.
        raise NotImplementedError()

    ep_dict = {}
    ep_idx = 0

    image_paths = sorted(raw_dir.glob("frame_*.png"))
    num_frames = len(image_paths)

    ep_dict["observation.image"] = [PILImage.open(x) for x in image_paths]
    ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
    ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
    ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

    ep_dicts = [ep_dict]
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

    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
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
    encoding: dict | None = None,
):
    if video or episodes or encoding is not None:
        # TODO(aliberts): support this
        raise NotImplementedError

    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
