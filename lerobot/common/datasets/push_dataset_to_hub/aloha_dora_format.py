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

import logging
import re
import shutil
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, Features, Image, Sequence, Value

from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame
from lerobot.common.utils.utils import init_logging


def check_format(raw_dir) -> bool:
    # TODO(rcadene): remove hardcoding
    raw_dir = raw_dir / "018f9c37-c092-72fd-bd83-6f5a5c1b59d2"
    assert raw_dir.exists()

    leader_file = list(raw_dir.glob("*.parquet"))
    if len(leader_file) == 0:
        raise ValueError(f"Missing parquet files in '{raw_dir}'")
    return True


def load_from_raw(raw_dir: Path, out_dir: Path):
    # TODO(rcadene): remove hardcoding
    raw_dir = raw_dir / "018f9c37-c092-72fd-bd83-6f5a5c1b59d2"

    # Load data stream that will be used as reference for the timestamps synchronization
    reference_key = "observation.images.cam_right_wrist"
    reference_df = pd.read_parquet(raw_dir / f"{reference_key}.parquet")
    reference_df = reference_df[["timestamp_utc", reference_key]]

    # Merge all data stream using nearest backward strategy
    data_df = reference_df
    for path in raw_dir.glob("*.parquet"):
        key = path.stem  # action or observation.state or ...
        if key == reference_key:
            continue
        df = pd.read_parquet(path)
        df = df[["timestamp_utc", key]]
        data_df = pd.merge_asof(
            data_df,
            df,
            on="timestamp_utc",
            direction="backward",
        )
    # dora only use arrays, so single values are encapsulated into a list
    data_df["episode_index"] = data_df["episode_index"].map(lambda x: x[0])
    data_df["frame_index"] = data_df.groupby("episode_index").cumcount()
    data_df["index"] = data_df.index

    # set 'next.done' to True for the last frame of each episode
    data_df["next.done"] = False
    data_df.loc[data_df.groupby("episode_index").tail(1).index, "next.done"] = True

    # Get the episode index containing for each unique episode index
    first_ep_index_df = data_df.groupby("episode_index").agg(start_index=("index", "first")).reset_index()
    from_ = first_ep_index_df["start_index"].tolist()
    to_ = from_[1:] + [len(data_df)]
    episode_data_index = {
        "from": from_,
        "to": to_,
    }

    data_df["timestamp"] = data_df["timestamp_utc"].map(lambda x: x.timestamp())
    # each episode starts with timestamp 0 to match the ones from the video
    data_df["timestamp"] = data_df.groupby("episode_index")["timestamp"].transform(lambda x: x - x.iloc[0])

    del data_df["timestamp_utc"]

    # Remove rows with a NaN in any column. It can happened during the first frames of an episode,
    # because some cameras didnt start recording yet.
    data_df = data_df.dropna(axis=1)

    # Create symlink to raw videos directory (that needs to be absolute not relative)
    # out_dir.mkdir(parents=True, exist_ok=True)
    # absolute_videos_dir = (raw_dir / "videos").absolute()
    # (out_dir / "videos").symlink_to(absolute_videos_dir)

    # TODO(rcadene): remove before merge
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    for from_path in (raw_dir / "videos").glob("*.mp4"):
        match = re.search(r"_(\d+)\.mp4$", from_path.name)
        if not match:
            raise ValueError(from_path.name)
        ep_idx = match.group(1)
        to_path = out_dir / "videos" / from_path.name.replace(ep_idx, f"{int(ep_idx):06d}")
        shutil.copy2(from_path, to_path)

    data_dict = {}
    for key in data_df:
        # is video frame
        if "observation.images." in key:
            # we need `[0] because dora only use arrays, so single values are encapsulated into a list.
            # it is the case for video_frame dictionary = [{"path": ..., "timestamp": ...}]
            data_dict[key] = [video_frame[0] for video_frame in data_df[key].values]

            # TODO(rcadene): remove before merge
            for item in data_dict[key]:
                path = item["path"]
                match = re.search(r"_(\d+)\.mp4$", path)
                if not match:
                    raise ValueError(path)
                ep_idx = match.group(1)
                item["path"] = path.replace(ep_idx, f"{int(ep_idx):06d}")
        # is number
        elif data_df[key].iloc[0].ndim == 0 or data_df[key].iloc[0].shape[0] == 1:
            data_dict[key] = torch.from_numpy(data_df[key].values)
        # is vector
        elif data_df[key].iloc[0].shape[0] > 1:
            data_dict[key] = torch.stack([torch.from_numpy(x.copy()) for x in data_df[key].values])
        else:
            raise ValueError(key)

    return data_dict, episode_data_index


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


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    init_logging()

    if debug:
        logging.warning("debug=True not implemented. Falling back to debug=False.")

    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30
    else:
        raise NotImplementedError()

    if not video:
        raise NotImplementedError()

    data_df, episode_data_index = load_from_raw(raw_dir, out_dir)
    hf_dataset = to_hf_dataset(data_df, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
