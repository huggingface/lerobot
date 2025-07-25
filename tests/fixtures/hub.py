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
import pytest
from huggingface_hub.utils import filter_repo_objects

from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    STATS_PATH,
)
from tests.fixtures.constants import LEROBOT_TEST_DIR


@pytest.fixture(scope="session")
def mock_snapshot_download_factory(
    info_factory,
    create_info,
    stats_factory,
    create_stats,
    tasks_factory,
    create_tasks,
    episodes_factory,
    create_episodes,
    hf_dataset_factory,
    create_hf_dataset,
    create_videos,
):
    """
    This factory allows to patch snapshot_download such that when called, it will create expected files rather
    than making calls to the hub api. Its design allows to pass explicitly files which you want to be created.
    """

    def _mock_snapshot_download_func(
        info: dict | None = None,
        stats: dict | None = None,
        tasks: pd.DataFrame | None = None,
        episodes: datasets.Dataset | None = None,
        hf_dataset: datasets.Dataset | None = None,
        data_files_size_in_mb: float = DEFAULT_DATA_FILE_SIZE_IN_MB,
        chunks_size: int = DEFAULT_CHUNK_SIZE,
    ):
        if info is None:
            info = info_factory(data_files_size_in_mb=data_files_size_in_mb, chunks_size=chunks_size)
        if stats is None:
            stats = stats_factory(features=info["features"])
        if tasks is None:
            tasks = tasks_factory(total_tasks=info["total_tasks"])
        if episodes is None:
            episodes = episodes_factory(
                features=info["features"],
                fps=info["fps"],
                total_episodes=info["total_episodes"],
                total_frames=info["total_frames"],
                tasks=tasks,
            )
        if hf_dataset is None:
            hf_dataset = hf_dataset_factory(tasks=tasks, episodes=episodes, fps=info["fps"])

        def _mock_snapshot_download(
            repo_id: str,  # TODO(rcadene): repo_id should be used no?
            local_dir: str | Path | None = None,
            allow_patterns: str | list[str] | None = None,
            ignore_patterns: str | list[str] | None = None,
            *args,
            **kwargs,
        ) -> str:
            if local_dir is None:
                local_dir = LEROBOT_TEST_DIR

            # List all possible files
            all_files = [
                INFO_PATH,
                STATS_PATH,
                # TODO(rcadene): remove naive chunk 0 file 0 ?
                DEFAULT_TASKS_PATH.format(chunk_index=0, file_index=0),
                DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0),
                DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0),
            ]

            video_keys = [key for key, feats in info["features"].items() if feats["dtype"] == "video"]
            for key in video_keys:
                all_files.append(DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0))

            allowed_files = filter_repo_objects(
                all_files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
            )

            request_info = False
            request_tasks = False
            request_episodes = False
            request_stats = False
            request_data = False
            request_videos = False
            for rel_path in allowed_files:
                if rel_path.startswith("meta/info.json"):
                    request_info = True
                elif rel_path.startswith("meta/stats"):
                    request_stats = True
                elif rel_path.startswith("meta/tasks"):
                    request_tasks = True
                elif rel_path.startswith("meta/episodes"):
                    request_episodes = True
                elif rel_path.startswith("data/"):
                    request_data = True
                elif rel_path.startswith("videos/"):
                    request_videos = True
                else:
                    raise ValueError(f"{rel_path} not supported.")

            if request_info:
                create_info(local_dir, info)
            if request_stats:
                create_stats(local_dir, stats)
            if request_tasks:
                create_tasks(local_dir, tasks)
            if request_episodes:
                create_episodes(local_dir, episodes)
            if request_data:
                create_hf_dataset(local_dir, hf_dataset, data_files_size_in_mb, chunks_size)
            if request_videos:
                create_videos(root=local_dir, info=info)

            return str(local_dir)

        return _mock_snapshot_download

    return _mock_snapshot_download_func
