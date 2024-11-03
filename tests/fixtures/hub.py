from pathlib import Path

import pytest
from huggingface_hub.utils import filter_repo_objects

from lerobot.common.datasets.utils import EPISODES_PATH, INFO_PATH, STATS_PATH, TASKS_PATH
from tests.fixtures.defaults import LEROBOT_TEST_DIR


@pytest.fixture(scope="session")
def mock_snapshot_download_factory(
    info,
    info_path,
    stats,
    stats_path,
    tasks,
    tasks_path,
    episodes,
    episode_path,
    single_episode_parquet_path,
    hf_dataset,
):
    """
    This factory allows to patch snapshot_download such that when called, it will create expected files rather
    than making calls to the hub api. Its design allows to pass explicitly files which you want to be created.
    """

    def _mock_snapshot_download_func(
        info_dict=info, stats_dict=stats, task_dicts=tasks, episode_dicts=episodes, hf_ds=hf_dataset
    ):
        def _extract_episode_index_from_path(fpath: str) -> int:
            path = Path(fpath)
            if path.suffix == ".parquet" and path.stem.startswith("episode_"):
                episode_index = int(path.stem[len("episode_") :])  # 'episode_000000' -> 0
                return episode_index
            else:
                return None

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
            all_files = []
            meta_files = [INFO_PATH, STATS_PATH, TASKS_PATH, EPISODES_PATH]
            all_files.extend(meta_files)

            data_files = []
            for episode_dict in episode_dicts:
                ep_idx = episode_dict["episode_index"]
                ep_chunk = ep_idx // info_dict["chunks_size"]
                data_path = info_dict["data_path"].format(episode_chunk=ep_chunk, episode_index=ep_idx)
                data_files.append(data_path)
            all_files.extend(data_files)

            allowed_files = filter_repo_objects(
                all_files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
            )

            # Create allowed files
            for rel_path in allowed_files:
                if rel_path.startswith("data/"):
                    episode_index = _extract_episode_index_from_path(rel_path)
                    if episode_index is not None:
                        _ = single_episode_parquet_path(local_dir, hf_ds, ep_idx=episode_index)
                if rel_path == INFO_PATH:
                    _ = info_path(local_dir, info_dict)
                elif rel_path == STATS_PATH:
                    _ = stats_path(local_dir, stats_dict)
                elif rel_path == TASKS_PATH:
                    _ = tasks_path(local_dir, task_dicts)
                elif rel_path == EPISODES_PATH:
                    _ = episode_path(local_dir, episode_dicts)
                else:
                    pass
            return str(local_dir)

        return _mock_snapshot_download

    return _mock_snapshot_download_func
