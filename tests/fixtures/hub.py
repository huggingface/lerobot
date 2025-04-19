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

from lerobot.common.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_TASKS_PATH,
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
    ):
        if info is None:
            info = info_factory()
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

            allowed_files = filter_repo_objects(
                all_files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
            )

            has_info = False
            has_tasks = False
            has_episodes = False
            has_stats = False
            has_data = False
            for rel_path in allowed_files:
                if rel_path.startswith("meta/info.json"):
                    has_info = True
                elif rel_path.startswith("meta/stats"):
                    has_stats = True
                elif rel_path.startswith("meta/tasks"):
                    has_tasks = True
                elif rel_path.startswith("meta/episodes"):
                    has_episodes = True
                elif rel_path.startswith("data/"):
                    has_data = True
                else:
                    raise ValueError(f"{rel_path} not supported.")

            if has_info:
                create_info(local_dir, info)
            if has_stats:
                create_stats(local_dir, stats)
            if has_tasks:
                create_tasks(local_dir, tasks)
            if has_episodes:
                create_episodes(local_dir, episodes)
            # TODO(rcadene): create_videos?
            if has_data:
                create_hf_dataset(local_dir, hf_dataset)

            return str(local_dir)

        return _mock_snapshot_download

    return _mock_snapshot_download_func
