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
import logging
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    get_hf_dataset_size_in_mb,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)


def write_hf_dataset(
    hf_dataset: Dataset,
    local_dir: Path,
    data_file_size_mb: float | None = None,
    chunk_size: int | None = None,
):
    """
    Writes a Hugging Face Dataset to one or more Parquet files in a structured directory format.

    If the dataset size is within `DEFAULT_DATA_FILE_SIZE_IN_MB`, it's saved as a single file.
    Otherwise, the dataset is split into multiple smaller Parquet files, each not exceeding the size limit.
    The file and chunk indices are managed to organize the output files in a hierarchical structure,
    e.g., `data/chunk-000/file-000.parquet`, `data/chunk-000/file-001.parquet`, etc.
    This function ensures that episodes are not split across multiple files.

    Args:
        hf_dataset (Dataset): The Hugging Face Dataset to be written to disk.
        local_dir (Path): The root directory where the dataset files will be stored.
        data_file_size_mb (float, optional): Maximal size for the parquet data file, in MB. Defaults to DEFAULT_DATA_FILE_SIZE_IN_MB.
        chunk_size (int, optional): Maximal number of files within a chunk folder before creating another one. Defaults to DEFAULT_CHUNK_SIZE.
    """
    if data_file_size_mb is None:
        data_file_size_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    dataset_size_in_mb = get_hf_dataset_size_in_mb(hf_dataset)

    if dataset_size_in_mb <= data_file_size_mb:
        # If the dataset is small enough, write it to a single file.
        path = local_dir / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        hf_dataset.to_parquet(path)
        return

    # If the dataset is too large, split it into smaller chunks, keeping episodes whole.
    episode_indices = np.array(hf_dataset["episode_index"])
    episode_boundaries = np.where(np.diff(episode_indices) != 0)[0] + 1
    episode_starts = np.concatenate(([0], episode_boundaries))
    episode_ends = np.concatenate((episode_boundaries, [len(hf_dataset)]))

    num_episodes = len(episode_starts)
    current_episode_idx = 0
    chunk_idx, file_idx = 0, 0

    while current_episode_idx < num_episodes:
        shard_start_row = episode_starts[current_episode_idx]
        shard_end_row = episode_ends[current_episode_idx]
        next_episode_to_try_idx = current_episode_idx + 1

        while next_episode_to_try_idx < num_episodes:
            potential_shard_end_row = episode_ends[next_episode_to_try_idx]
            dataset_shard_candidate = hf_dataset.select(range(shard_start_row, potential_shard_end_row))
            shard_size_mb = get_hf_dataset_size_in_mb(dataset_shard_candidate)

            if shard_size_mb > data_file_size_mb:
                break
            else:
                shard_end_row = potential_shard_end_row
                next_episode_to_try_idx += 1

        dataset_shard = hf_dataset.select(range(shard_start_row, shard_end_row))

        if (
            shard_start_row == episode_starts[current_episode_idx]
            and shard_end_row == episode_ends[current_episode_idx]
        ):
            shard_size_mb = get_hf_dataset_size_in_mb(dataset_shard)
            if shard_size_mb > data_file_size_mb:
                logging.warning(
                    f"Episode with index {hf_dataset[shard_start_row.item()]['episode_index']} has size {shard_size_mb:.2f}MB, "
                    f"which is larger than data_file_size_mb ({data_file_size_mb}MB). "
                    "Writing it to a separate shard anyway to preserve episode integrity."
                )

        # Define the path for the current shard and ensure the directory exists.
        path = local_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write the shard to a Parquet file.
        dataset_shard.to_parquet(path)

        # Update chunk and file indices for the next iteration.
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
        current_episode_idx = next_episode_to_try_idx


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
    def _create_hf_dataset(
        dir: Path,
        hf_dataset: datasets.Dataset | None = None,
        data_file_size_in_mb: float | None = None,
        chunk_size: int | None = None,
    ):
        if hf_dataset is None:
            hf_dataset = hf_dataset_factory()
        write_hf_dataset(hf_dataset, dir, data_file_size_in_mb, chunk_size)

    return _create_hf_dataset
