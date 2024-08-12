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
Contains utilities to process raw data format from dora-record
"""

import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, Features, Image, Sequence, Value

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame


def check_format(raw_dir) -> bool:
    assert raw_dir.exists()

    leader_file = list(raw_dir.glob("*.parquet"))
    if len(leader_file) == 0:
        raise ValueError(f"Missing parquet files in '{raw_dir}'")
    return True


def load_from_raw(raw_dir: Path, videos_dir: Path, fps: int, video: bool, episodes: list[int] | None = None):
    # Load data stream that will be used as reference for the timestamps synchronization
    reference_files = list(raw_dir.glob("observation.images.cam_*.parquet"))
    if len(reference_files) == 0:
        raise ValueError(f"Missing reference files for camera, starting with  in '{raw_dir}'")
    # select first camera in alphanumeric order
    reference_key = sorted(reference_files)[0].stem
    reference_df = pd.read_parquet(raw_dir / f"{reference_key}.parquet")
    reference_df = reference_df[["timestamp_utc", reference_key]]

    # Merge all data stream using nearest backward strategy
    df = reference_df
    for path in raw_dir.glob("*.parquet"):
        key = path.stem  # action or observation.state or ...
        if key == reference_key:
            continue
        if "failed_episode_index" in key:
            # TODO(rcadene): add support for removing episodes that are tagged as "failed"
            continue
        modality_df = pd.read_parquet(path)
        modality_df = modality_df[["timestamp_utc", key]]
        df = pd.merge_asof(
            df,
            modality_df,
            on="timestamp_utc",
            # "nearest" is the best option over "backward", since the latter can desynchronizes camera timestamps by
            # matching timestamps that are too far appart, in order to fit the backward constraints. It's not the case for "nearest".
            # However, note that "nearest" might synchronize the reference camera with other cameras on slightly future timestamps.
            # are too far appart.
            direction="nearest",
            tolerance=pd.Timedelta(f"{1/fps} seconds"),
        )
    # Remove rows with episode_index -1 which indicates data that correspond to in-between episodes
    df = df[df["episode_index"] != -1]

    image_keys = [key for key in df if "observation.images." in key]

    def get_episode_index(row):
        episode_index_per_cam = {}
        for key in image_keys:
            path = row[key][0]["path"]
            match = re.search(r"_(\d{6}).mp4", path)
            if not match:
                raise ValueError(path)
            episode_index = int(match.group(1))
            episode_index_per_cam[key] = episode_index
        if len(set(episode_index_per_cam.values())) != 1:
            raise ValueError(
                f"All cameras are expected to belong to the same episode, but getting {episode_index_per_cam}"
            )
        return episode_index

    df["episode_index"] = df.apply(get_episode_index, axis=1)

    # dora only use arrays, so single values are encapsulated into a list
    df["frame_index"] = df.groupby("episode_index").cumcount()
    df = df.reset_index()
    df["index"] = df.index

    # set 'next.done' to True for the last frame of each episode
    df["next.done"] = False
    df.loc[df.groupby("episode_index").tail(1).index, "next.done"] = True

    df["timestamp"] = df["timestamp_utc"].map(lambda x: x.timestamp())
    # each episode starts with timestamp 0 to match the ones from the video
    df["timestamp"] = df.groupby("episode_index")["timestamp"].transform(lambda x: x - x.iloc[0])

    del df["timestamp_utc"]

    # sanity check
    has_nan = df.isna().any().any()
    if has_nan:
        raise ValueError("Dataset contains Nan values.")

    # sanity check episode indices go from 0 to n-1
    ep_ids = [ep_idx for ep_idx, _ in df.groupby("episode_index")]
    expected_ep_ids = list(range(df["episode_index"].max() + 1))
    if ep_ids != expected_ep_ids:
        raise ValueError(f"Episodes indices go from {ep_ids} instead of {expected_ep_ids}")

    # Create symlink to raw videos directory (that needs to be absolute not relative)
    videos_dir.parent.mkdir(parents=True, exist_ok=True)
    videos_dir.symlink_to((raw_dir / "videos").absolute())

    # sanity check the video paths are well formated
    for key in df:
        if "observation.images." not in key:
            continue
        for ep_idx in ep_ids:
            video_path = videos_dir / f"{key}_episode_{ep_idx:06d}.mp4"
            if not video_path.exists():
                raise ValueError(f"Video file not found in {video_path}")

    data_dict = {}
    for key in df:
        # is video frame
        if "observation.images." in key:
            # we need `[0] because dora only use arrays, so single values are encapsulated into a list.
            # it is the case for video_frame dictionary = [{"path": ..., "timestamp": ...}]
            data_dict[key] = [video_frame[0] for video_frame in df[key].values]

            # sanity check the video path is well formated
            video_path = videos_dir.parent / data_dict[key][0]["path"]
            if not video_path.exists():
                raise ValueError(f"Video file not found in {video_path}")
        # is number
        elif df[key].iloc[0].ndim == 0 or df[key].iloc[0].shape[0] == 1:
            data_dict[key] = torch.from_numpy(df[key].values)
        # is vector
        elif df[key].iloc[0].shape[0] > 1:
            data_dict[key] = torch.stack([torch.from_numpy(x.copy()) for x in df[key].values])
        else:
            raise ValueError(key)

    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
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
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
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
    encoding: dict | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30
    else:
        raise NotImplementedError()

    if not video:
        raise NotImplementedError()

    if encoding is not None:
        warnings.warn(
            "Video encoding is currently done outside of LeRobot for the dora_parquet format.",
            stacklevel=1,
        )

    data_df = load_from_raw(raw_dir, videos_dir, fps, episodes)
    hf_dataset = to_hf_dataset(data_df, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = "unknown"

    return hf_dataset, episode_data_index, info
