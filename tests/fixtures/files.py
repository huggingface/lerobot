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
import json
from pathlib import Path

import datasets
import jsonlines
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
)


@pytest.fixture(scope="session")
def info_path(info_factory):
    def _create_info_json_file(dir: Path, info: dict | None = None) -> Path:
        if not info:
            info = info_factory()
        fpath = dir / INFO_PATH
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        return fpath

    return _create_info_json_file


@pytest.fixture(scope="session")
def stats_path(stats_factory):
    def _create_stats_json_file(dir: Path, stats: dict | None = None) -> Path:
        if not stats:
            stats = stats_factory()
        fpath = dir / STATS_PATH
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        return fpath

    return _create_stats_json_file


@pytest.fixture(scope="session")
def episodes_stats_path(episodes_stats_factory):
    def _create_episodes_stats_jsonl_file(dir: Path, episodes_stats: list[dict] | None = None) -> Path:
        if not episodes_stats:
            episodes_stats = episodes_stats_factory()
        fpath = dir / EPISODES_STATS_PATH
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(fpath, "w") as writer:
            writer.write_all(episodes_stats.values())
        return fpath

    return _create_episodes_stats_jsonl_file


@pytest.fixture(scope="session")
def tasks_path(tasks_factory):
    def _create_tasks_jsonl_file(dir: Path, tasks: list | None = None) -> Path:
        if not tasks:
            tasks = tasks_factory()
        fpath = dir / TASKS_PATH
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(fpath, "w") as writer:
            writer.write_all(tasks.values())
        return fpath

    return _create_tasks_jsonl_file


@pytest.fixture(scope="session")
def episode_path(episodes_factory):
    def _create_episodes_jsonl_file(dir: Path, episodes: list | None = None) -> Path:
        if not episodes:
            episodes = episodes_factory()
        fpath = dir / EPISODES_PATH
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(fpath, "w") as writer:
            writer.write_all(episodes.values())
        return fpath

    return _create_episodes_jsonl_file


@pytest.fixture(scope="session")
def single_episode_parquet_path(hf_dataset_factory, info_factory):
    def _create_single_episode_parquet(
        dir: Path, ep_idx: int = 0, hf_dataset: datasets.Dataset | None = None, info: dict | None = None
    ) -> Path:
        if not info:
            info = info_factory()
        if hf_dataset is None:
            hf_dataset = hf_dataset_factory()

        data_path = info["data_path"]
        chunks_size = info["chunks_size"]
        ep_chunk = ep_idx // chunks_size
        fpath = dir / data_path.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        table = hf_dataset.data.table
        ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
        pq.write_table(ep_table, fpath)
        return fpath

    return _create_single_episode_parquet


@pytest.fixture(scope="session")
def multi_episode_parquet_path(hf_dataset_factory, info_factory):
    def _create_multi_episode_parquet(
        dir: Path, hf_dataset: datasets.Dataset | None = None, info: dict | None = None
    ) -> Path:
        if not info:
            info = info_factory()
        if hf_dataset is None:
            hf_dataset = hf_dataset_factory()

        data_path = info["data_path"]
        chunks_size = info["chunks_size"]
        total_episodes = info["total_episodes"]
        for ep_idx in range(total_episodes):
            ep_chunk = ep_idx // chunks_size
            fpath = dir / data_path.format(episode_chunk=ep_chunk, episode_index=ep_idx)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            table = hf_dataset.data.table
            ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
            pq.write_table(ep_table, fpath)
        return dir / "data"

    return _create_multi_episode_parquet
