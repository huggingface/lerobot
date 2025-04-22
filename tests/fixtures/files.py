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
from pathlib import Path

import datasets
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from lerobot.common.datasets.utils import (
    write_episodes,
    write_hf_dataset,
    write_info,
    write_stats,
    write_tasks,
)


@pytest.fixture(scope="session")
def create_info(info_factory):
    def _create_info(dir: Path, info: dict | None = None):
        if info is None:
            info = info_factory()
        write_info(info, dir)

    return _create_info


@pytest.fixture(scope="session")
def create_stats(stats_factory):
    def _create_stats(dir: Path, stats: dict | None = None):
        if stats is None:
            stats = stats_factory()
        write_stats(stats, dir)

    return _create_stats


@pytest.fixture(scope="session")
def create_tasks(tasks_factory):
    def _create_tasks(dir: Path, tasks: pd.DataFrame | None = None):
        if tasks is None:
            tasks = tasks_factory()
        write_tasks(tasks, dir)

    return _create_tasks


@pytest.fixture(scope="session")
def create_episodes(episodes_factory):
    def _create_episodes(dir: Path, episodes: datasets.Dataset | None = None):
        if episodes is None:
            # TODO(rcadene): add features, fps as arguments
            episodes = episodes_factory()
        write_episodes(episodes, dir)

    return _create_episodes


@pytest.fixture(scope="session")
def create_hf_dataset(hf_dataset_factory):
    def _create_hf_dataset(dir: Path, hf_dataset: datasets.Dataset | None = None):
        if hf_dataset is None:
            hf_dataset = hf_dataset_factory()
        write_hf_dataset(hf_dataset, dir)

    return _create_hf_dataset


@pytest.fixture(scope="session")
def single_episode_parquet_path(hf_dataset_factory, info_factory):
    def _create_single_episode_parquet(
        dir: Path, ep_idx: int = 0, hf_dataset: datasets.Dataset | None = None, info: dict | None = None
    ) -> Path:
        raise NotImplementedError()
        if info is None:
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
        raise NotImplementedError()
        if info is None:
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
