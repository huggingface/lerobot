import json
from pathlib import Path

import datasets
import jsonlines
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from datasets import Dataset

from lerobot.common.datasets.utils import (
    write_episodes,
    write_episodes_stats,
    write_hf_dataset,
    write_info,
    write_stats,
    write_tasks,
)


@pytest.fixture(scope="session")
def create_info(info_factory):
    def _create_info(dir: Path, info: dict | None = None):
        if not info:
            info = info_factory()
        write_info(info, dir)

    return _create_info


@pytest.fixture(scope="session")
def create_stats(stats_factory):
    def _create_stats(dir: Path, stats: dict | None = None):
        if not stats:
            stats = stats_factory()
        write_stats(stats, dir)

    return _create_stats


@pytest.fixture(scope="session")
def create_episodes_stats(episodes_stats_factory):
    def _create_episodes_stats(dir: Path, episodes_stats: Dataset | None = None):
        if not episodes_stats:
            episodes_stats = episodes_stats_factory()
        write_episodes_stats(episodes_stats, dir)

    return _create_episodes_stats


@pytest.fixture(scope="session")
def create_tasks(tasks_factory):
    def _create_tasks(dir: Path, tasks: Dataset | None = None):
        if not tasks:
            tasks = tasks_factory()
        write_tasks(tasks, dir)

    return _create_tasks


@pytest.fixture(scope="session")
def create_episodes(episodes_factory):
    def _create_episodes(dir: Path, episodes: Dataset | None = None):
        if not episodes:
            episodes = episodes_factory()
        write_episodes(episodes, dir)

    return _create_episodes

@pytest.fixture(scope="session")
def create_hf_dataset(hf_dataset_factory):
    def _create_hf_dataset(dir: Path, hf_dataset: Dataset | None = None):
        if not hf_dataset:
            hf_dataset = hf_dataset_factory()
        write_hf_dataset(hf_dataset, dir)

    return _create_hf_dataset


@pytest.fixture(scope="session")
def single_episode_parquet_path(hf_dataset_factory, info_factory):
    def _create_single_episode_parquet(
        dir: Path, ep_idx: int = 0, hf_dataset: datasets.Dataset | None = None, info: dict | None = None
    ) -> Path:
        raise NotImplementedError()
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
        raise NotImplementedError()
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
