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
    raw_dir = raw_dir / "018f9fdc-6b7b-7432-a529-40d2cc718032"
    assert raw_dir.exists()

    leader_file = list(raw_dir.glob("*.parquet"))
    if len(leader_file) == 0:
        raise ValueError(f"Missing parquet files in '{raw_dir}'")
    return True


def load_from_raw(raw_dir: Path, out_dir: Path):
    # TODO(rcadene): remove hardcoding
    raw_dir = raw_dir / "018f9fdc-6b7b-7432-a529-40d2cc718032"

    # Load data stream that will be used as reference for the timestamps synchronization
    reference_key = "observation.images.cam_right_wrist"
    reference_df = pd.read_parquet(raw_dir / f"{reference_key}.parquet")
    reference_df = reference_df[["timestamp_utc", reference_key]]

    # Merge all data stream using nearest backward strategy
    df = reference_df
    for path in raw_dir.glob("*.parquet"):
        key = path.stem  # action or observation.state or ...
        if key == reference_key:
            continue
        modality_df = pd.read_parquet(path)
        modality_df = modality_df[["timestamp_utc", key]]
        df = pd.merge_asof(
            df,
            modality_df,
            on="timestamp_utc",
            direction="backward",
        )

    # Remove rows with a NaN in any column. It can happened during the first frames of an episode,
    # because some cameras didnt start recording yet.
    df = df.dropna(axis=0)

    # Remove rows with episode_index -1 which indicates a failed episode
    df = df[df["episode_index"] != -1]

    # dora only use arrays, so single values are encapsulated into a list
    df["episode_index"] = df["episode_index"].map(lambda x: x[0])
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

    # sanity check episode indices go from 0 to n-1
    ep_ids = [ep_idx for ep_idx, _ in df.groupby("episode_index")]
    expected_ep_ids = list(range(df["episode_index"].max()))
    assert ep_ids == expected_ep_ids, f"Episodes indices go from {ep_ids} instead of {expected_ep_ids}"

    # Create symlink to raw videos directory (that needs to be absolute not relative)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.symlink_to((raw_dir / "videos").absolute())

    # sanity check the video paths are well formated
    for key in df:
        if "observation.images." not in key:
            continue
        for ep_idx in ep_ids:
            video_path = videos_dir / f"{key}_episode_{ep_idx:06d}.mp4"
            assert video_path.exists(), f"Video file not found in {video_path}"

    data_dict = {}
    for key in df:
        # is video frame
        if "observation.images." in key:
            # we need `[0] because dora only use arrays, so single values are encapsulated into a list.
            # it is the case for video_frame dictionary = [{"path": ..., "timestamp": ...}]
            data_dict[key] = [video_frame[0] for video_frame in df[key].values]

            # sanity check the video path is well formated
            video_path = videos_dir.parent / data_dict[key][0]["path"]
            assert video_path.exists(), f"Video file not found in {video_path}"
        # is number
        elif df[key].iloc[0].ndim == 0 or df[key].iloc[0].shape[0] == 1:
            data_dict[key] = torch.from_numpy(df[key].values)
        # is vector
        elif df[key].iloc[0].shape[0] > 1:
            data_dict[key] = torch.stack([torch.from_numpy(x.copy()) for x in df[key].values])
        else:
            raise ValueError(key)

    # Get the episode index containing for each unique episode index
    first_ep_index_df = df.groupby("episode_index").agg(start_index=("index", "first")).reset_index()
    from_ = first_ep_index_df["start_index"].tolist()
    to_ = from_[1:] + [len(df)]
    episode_data_index = {
        "from": from_,
        "to": to_,
    }

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
