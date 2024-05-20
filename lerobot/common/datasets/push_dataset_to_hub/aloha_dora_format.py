#!/usr/bin/env python
"""
Contains utilities to process raw data format from dora-record
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset


def check_format(raw_dir) -> bool:
    leader_file = list(raw_dir.glob("*_leader.parquet"))

    if len(leader_file) != 1:
        raise ValueError(
            f"Issues with leader file in {raw_dir}. Make sure there is one and only one leader file"
        )
    return True


def load_from_raw(raw_dir: Path, out_dir=None, fps=30, video=True, debug=False):
    parquet_files = list(raw_dir.glob("*.parquet"))
    leader_file = list(raw_dir.glob("*_leader.parquet"))[0]

    # Remove leader file from parquet files
    parquet_files = [x for x in parquet_files if x != leader_file]

    ## Load leader data
    data_df = pd.read_parquet(leader_file)
    data_df = data_df[["timestamp_utc", leader_file.stem]]

    ## Merge all data using nearest backward strategy
    for data in parquet_files:
        df = pd.read_parquet(data)
        data_df = pd.merge_asof(
            data_df,
            df[["timestamp_utc", data.stem]],
            on="timestamp_utc",
            direction="backward",
        )
    data_df["episode_index"] = data_df["episode_index"].map(lambda x: x[0])

    # Get the episode index containing for each unique episode index
    episode_data_index = data_df["episode_index"].drop_duplicates().reset_index()
    episode_data_index["from"] = episode_data_index["index"]
    episode_data_index["to"] = episode_data_index["index"].shift(-1)

    # Remove column index
    episode_data_index = episode_data_index.drop(columns=["index"])

    # episode_data_index to dict
    episode_data_index = episode_data_index.to_dict(orient="list")

    return data_df, episode_data_index


def to_hf_dataset(df, video) -> Dataset:
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_df, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_df, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
