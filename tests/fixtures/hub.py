from pathlib import Path

import datasets
import pytest
from huggingface_hub.utils import filter_repo_objects

from lerobot.common.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_EPISODES_STATS_PATH,
    DEFAULT_TASKS_PATH,
    INFO_PATH,
    LEGACY_STATS_PATH,
)
from tests.fixtures.constants import LEROBOT_TEST_DIR


@pytest.fixture(scope="session")
def mock_snapshot_download_factory(
    info_factory,
    create_info,
    stats_factory,
    create_stats,
    episodes_stats_factory,
    create_episodes_stats,
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
        episodes_stats: datasets.Dataset | None = None,
        tasks: datasets.Dataset | None = None,
        episodes: datasets.Dataset | None = None,
        hf_dataset: datasets.Dataset | None = None,
    ):
        if not info:
            info = info_factory()
        if not stats:
            stats = stats_factory(features=info["features"])
        if not episodes_stats:
            episodes_stats = episodes_stats_factory(
                features=info["features"], total_episodes=info["total_episodes"]
            )
        if not tasks:
            tasks = tasks_factory(total_tasks=info["total_tasks"])
        if not episodes:
            episodes = episodes_factory(
                total_episodes=info["total_episodes"], total_frames=info["total_frames"], tasks=tasks
            )
        if not hf_dataset:
            hf_dataset = hf_dataset_factory(tasks=tasks, episodes=episodes, fps=info["fps"])

        def _mock_snapshot_download(
            repo_id: str,
            local_dir: str | Path | None = None,
            allow_patterns: str | list[str] | None = None,
            ignore_patterns: str | list[str] | None = None,
            *args,
            **kwargs,
        ) -> str:
            if not local_dir:
                local_dir = LEROBOT_TEST_DIR

            # List all possible files
            all_files = [
                INFO_PATH,
                LEGACY_STATS_PATH,
                # TODO(rcadene)
                DEFAULT_TASKS_PATH.format(chunk_index=0, file_index=0),
                DEFAULT_EPISODES_STATS_PATH.format(chunk_index=0, file_index=0),
                DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0),
                DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0),
            ]

            allowed_files = filter_repo_objects(
                all_files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
            )

            has_info = False
            has_tasks = False
            has_episodes = False
            has_episodes_stats = False
            has_stats = False
            has_data = False
            for rel_path in allowed_files:
                if rel_path.startswith("meta/info.json"):
                    has_info = True
                elif rel_path.startswith("meta/stats"):
                    has_stats = True
                elif rel_path.startswith("meta/tasks"):
                    has_tasks = True
                elif rel_path.startswith("meta/episodes_stats"):
                    has_episodes_stats = True
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
            if has_episodes_stats:
                create_episodes_stats(local_dir, episodes_stats)
            if has_data:
                create_hf_dataset(local_dir, hf_dataset)

            return str(local_dir)

        return _mock_snapshot_download

    return _mock_snapshot_download_func
