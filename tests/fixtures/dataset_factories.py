import random
from functools import partial
from pathlib import Path
from typing import Protocol
from unittest.mock import patch

import datasets
import numpy as np
import PIL.Image
import pytest
import torch

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURES,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    get_hf_features_from_features,
    hf_transform_to_torch,
)
from tests.fixtures.constants import (
    DEFAULT_FPS,
    DUMMY_CAMERA_FEATURES,
    DUMMY_MOTOR_FEATURES,
    DUMMY_REPO_ID,
    DUMMY_ROBOT_TYPE,
    DUMMY_VIDEO_INFO,
)


class LeRobotDatasetFactory(Protocol):
    def __call__(self, *args, **kwargs) -> LeRobotDataset: ...


def get_task_index(task_dicts: dict, task: str) -> int:
    tasks = {d["task_index"]: d["task"] for d in task_dicts.values()}
    task_to_task_index = {task: task_idx for task_idx, task in tasks.items()}
    return task_to_task_index[task]


@pytest.fixture(scope="session")
def img_tensor_factory():
    def _create_img_tensor(height=100, width=100, channels=3, dtype=torch.float32) -> torch.Tensor:
        return torch.rand((channels, height, width), dtype=dtype)

    return _create_img_tensor


@pytest.fixture(scope="session")
def img_array_factory():
    def _create_img_array(height=100, width=100, channels=3, dtype=np.uint8) -> np.ndarray:
        if np.issubdtype(dtype, np.unsignedinteger):
            # Int array in [0, 255] range
            img_array = np.random.randint(0, 256, size=(height, width, channels), dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            # Float array in [0, 1] range
            img_array = np.random.rand(height, width, channels).astype(dtype)
        else:
            raise ValueError(dtype)
        return img_array

    return _create_img_array


@pytest.fixture(scope="session")
def img_factory(img_array_factory):
    def _create_img(height=100, width=100) -> PIL.Image.Image:
        img_array = img_array_factory(height=height, width=width)
        return PIL.Image.fromarray(img_array)

    return _create_img


@pytest.fixture(scope="session")
def features_factory():
    def _create_features(
        motor_features: dict = DUMMY_MOTOR_FEATURES,
        camera_features: dict = DUMMY_CAMERA_FEATURES,
        use_videos: bool = True,
    ) -> dict:
        if use_videos:
            camera_ft = {
                key: {"dtype": "video", **ft, **DUMMY_VIDEO_INFO} for key, ft in camera_features.items()
            }
        else:
            camera_ft = {key: {"dtype": "image", **ft} for key, ft in camera_features.items()}
        return {
            **motor_features,
            **camera_ft,
            **DEFAULT_FEATURES,
        }

    return _create_features


@pytest.fixture(scope="session")
def info_factory(features_factory):
    def _create_info(
        codebase_version: str = CODEBASE_VERSION,
        fps: int = DEFAULT_FPS,
        robot_type: str = DUMMY_ROBOT_TYPE,
        total_episodes: int = 0,
        total_frames: int = 0,
        total_tasks: int = 0,
        total_videos: int = 0,
        total_chunks: int = 0,
        chunks_size: int = DEFAULT_CHUNK_SIZE,
        data_path: str = DEFAULT_PARQUET_PATH,
        video_path: str = DEFAULT_VIDEO_PATH,
        motor_features: dict = DUMMY_MOTOR_FEATURES,
        camera_features: dict = DUMMY_CAMERA_FEATURES,
        use_videos: bool = True,
    ) -> dict:
        features = features_factory(motor_features, camera_features, use_videos)
        return {
            "codebase_version": codebase_version,
            "robot_type": robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_videos,
            "total_chunks": total_chunks,
            "chunks_size": chunks_size,
            "fps": fps,
            "splits": {},
            "data_path": data_path,
            "video_path": video_path if use_videos else None,
            "features": features,
        }

    return _create_info


@pytest.fixture(scope="session")
def stats_factory():
    def _create_stats(
        features: dict[str] | None = None,
    ) -> dict:
        stats = {}
        for key, ft in features.items():
            shape = ft["shape"]
            dtype = ft["dtype"]
            if dtype in ["image", "video"]:
                stats[key] = {
                    "max": np.full((3, 1, 1), 1, dtype=np.float32).tolist(),
                    "mean": np.full((3, 1, 1), 0.5, dtype=np.float32).tolist(),
                    "min": np.full((3, 1, 1), 0, dtype=np.float32).tolist(),
                    "std": np.full((3, 1, 1), 0.25, dtype=np.float32).tolist(),
                    "count": [10],
                }
            else:
                stats[key] = {
                    "max": np.full(shape, 1, dtype=dtype).tolist(),
                    "mean": np.full(shape, 0.5, dtype=dtype).tolist(),
                    "min": np.full(shape, 0, dtype=dtype).tolist(),
                    "std": np.full(shape, 0.25, dtype=dtype).tolist(),
                    "count": [10],
                }
        return stats

    return _create_stats


@pytest.fixture(scope="session")
def episodes_stats_factory(stats_factory):
    def _create_episodes_stats(
        features: dict[str],
        total_episodes: int = 3,
    ) -> dict:
        episodes_stats = {}
        for episode_index in range(total_episodes):
            episodes_stats[episode_index] = {
                "episode_index": episode_index,
                "stats": stats_factory(features),
            }
        return episodes_stats

    return _create_episodes_stats


@pytest.fixture(scope="session")
def tasks_factory():
    def _create_tasks(total_tasks: int = 3) -> int:
        tasks = {}
        for task_index in range(total_tasks):
            task_dict = {"task_index": task_index, "task": f"Perform action {task_index}."}
            tasks[task_index] = task_dict
        return tasks

    return _create_tasks


@pytest.fixture(scope="session")
def episodes_factory(tasks_factory):
    def _create_episodes(
        total_episodes: int = 3,
        total_frames: int = 400,
        tasks: dict | None = None,
        multi_task: bool = False,
    ):
        if total_episodes <= 0 or total_frames <= 0:
            raise ValueError("num_episodes and total_length must be positive integers.")
        if total_frames < total_episodes:
            raise ValueError("total_length must be greater than or equal to num_episodes.")

        if not tasks:
            min_tasks = 2 if multi_task else 1
            total_tasks = random.randint(min_tasks, total_episodes)
            tasks = tasks_factory(total_tasks)

        if total_episodes < len(tasks) and not multi_task:
            raise ValueError("The number of tasks should be less than the number of episodes.")

        # Generate random lengths that sum up to total_length
        lengths = np.random.multinomial(total_frames, [1 / total_episodes] * total_episodes).tolist()

        tasks_list = [task_dict["task"] for task_dict in tasks.values()]
        num_tasks_available = len(tasks_list)

        episodes = {}
        remaining_tasks = tasks_list.copy()
        for ep_idx in range(total_episodes):
            num_tasks_in_episode = random.randint(1, min(3, num_tasks_available)) if multi_task else 1
            tasks_to_sample = remaining_tasks if remaining_tasks else tasks_list
            episode_tasks = random.sample(tasks_to_sample, min(num_tasks_in_episode, len(tasks_to_sample)))
            if remaining_tasks:
                for task in episode_tasks:
                    remaining_tasks.remove(task)

            episodes[ep_idx] = {
                "episode_index": ep_idx,
                "tasks": episode_tasks,
                "length": lengths[ep_idx],
            }

        return episodes

    return _create_episodes


@pytest.fixture(scope="session")
def hf_dataset_factory(features_factory, tasks_factory, episodes_factory, img_array_factory):
    def _create_hf_dataset(
        features: dict | None = None,
        tasks: list[dict] | None = None,
        episodes: list[dict] | None = None,
        fps: int = DEFAULT_FPS,
    ) -> datasets.Dataset:
        if not tasks:
            tasks = tasks_factory()
        if not episodes:
            episodes = episodes_factory()
        if not features:
            features = features_factory()

        timestamp_col = np.array([], dtype=np.float32)
        frame_index_col = np.array([], dtype=np.int64)
        episode_index_col = np.array([], dtype=np.int64)
        task_index = np.array([], dtype=np.int64)
        for ep_dict in episodes.values():
            timestamp_col = np.concatenate((timestamp_col, np.arange(ep_dict["length"]) / fps))
            frame_index_col = np.concatenate((frame_index_col, np.arange(ep_dict["length"], dtype=int)))
            episode_index_col = np.concatenate(
                (episode_index_col, np.full(ep_dict["length"], ep_dict["episode_index"], dtype=int))
            )
            ep_task_index = get_task_index(tasks, ep_dict["tasks"][0])
            task_index = np.concatenate((task_index, np.full(ep_dict["length"], ep_task_index, dtype=int)))

        index_col = np.arange(len(episode_index_col))

        robot_cols = {}
        for key, ft in features.items():
            if ft["dtype"] == "image":
                robot_cols[key] = [
                    img_array_factory(height=ft["shapes"][1], width=ft["shapes"][0])
                    for _ in range(len(index_col))
                ]
            elif ft["shape"][0] > 1 and ft["dtype"] != "video":
                robot_cols[key] = np.random.random((len(index_col), ft["shape"][0])).astype(ft["dtype"])

        hf_features = get_hf_features_from_features(features)
        dataset = datasets.Dataset.from_dict(
            {
                **robot_cols,
                "timestamp": timestamp_col,
                "frame_index": frame_index_col,
                "episode_index": episode_index_col,
                "index": index_col,
                "task_index": task_index,
            },
            features=hf_features,
        )
        dataset.set_transform(hf_transform_to_torch)
        return dataset

    return _create_hf_dataset


@pytest.fixture(scope="session")
def lerobot_dataset_metadata_factory(
    info_factory,
    stats_factory,
    episodes_stats_factory,
    tasks_factory,
    episodes_factory,
    mock_snapshot_download_factory,
):
    def _create_lerobot_dataset_metadata(
        root: Path,
        repo_id: str = DUMMY_REPO_ID,
        info: dict | None = None,
        stats: dict | None = None,
        episodes_stats: list[dict] | None = None,
        tasks: list[dict] | None = None,
        episodes: list[dict] | None = None,
    ) -> LeRobotDatasetMetadata:
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

        mock_snapshot_download = mock_snapshot_download_factory(
            info=info,
            stats=stats,
            episodes_stats=episodes_stats,
            tasks=tasks,
            episodes=episodes,
        )
        with (
            patch("lerobot.common.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version_patch,
            patch(
                "lerobot.common.datasets.lerobot_dataset.snapshot_download"
            ) as mock_snapshot_download_patch,
        ):
            mock_get_safe_version_patch.side_effect = lambda repo_id, version: version
            mock_snapshot_download_patch.side_effect = mock_snapshot_download

            return LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    return _create_lerobot_dataset_metadata


@pytest.fixture(scope="session")
def lerobot_dataset_factory(
    info_factory,
    stats_factory,
    episodes_stats_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    mock_snapshot_download_factory,
    lerobot_dataset_metadata_factory,
) -> LeRobotDatasetFactory:
    def _create_lerobot_dataset(
        root: Path,
        repo_id: str = DUMMY_REPO_ID,
        total_episodes: int = 3,
        total_frames: int = 150,
        total_tasks: int = 1,
        multi_task: bool = False,
        info: dict | None = None,
        stats: dict | None = None,
        episodes_stats: list[dict] | None = None,
        tasks: list[dict] | None = None,
        episode_dicts: list[dict] | None = None,
        hf_dataset: datasets.Dataset | None = None,
        **kwargs,
    ) -> LeRobotDataset:
        if not info:
            info = info_factory(
                total_episodes=total_episodes, total_frames=total_frames, total_tasks=total_tasks
            )
        if not stats:
            stats = stats_factory(features=info["features"])
        if not episodes_stats:
            episodes_stats = episodes_stats_factory(features=info["features"], total_episodes=total_episodes)
        if not tasks:
            tasks = tasks_factory(total_tasks=info["total_tasks"])
        if not episode_dicts:
            episode_dicts = episodes_factory(
                total_episodes=info["total_episodes"],
                total_frames=info["total_frames"],
                tasks=tasks,
                multi_task=multi_task,
            )
        if not hf_dataset:
            hf_dataset = hf_dataset_factory(tasks=tasks, episodes=episode_dicts, fps=info["fps"])

        mock_snapshot_download = mock_snapshot_download_factory(
            info=info,
            stats=stats,
            episodes_stats=episodes_stats,
            tasks=tasks,
            episodes=episode_dicts,
            hf_dataset=hf_dataset,
        )
        mock_metadata = lerobot_dataset_metadata_factory(
            root=root,
            repo_id=repo_id,
            info=info,
            stats=stats,
            episodes_stats=episodes_stats,
            tasks=tasks,
            episodes=episode_dicts,
        )
        with (
            patch("lerobot.common.datasets.lerobot_dataset.LeRobotDatasetMetadata") as mock_metadata_patch,
            patch("lerobot.common.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version_patch,
            patch(
                "lerobot.common.datasets.lerobot_dataset.snapshot_download"
            ) as mock_snapshot_download_patch,
        ):
            mock_metadata_patch.return_value = mock_metadata
            mock_get_safe_version_patch.side_effect = lambda repo_id, version: version
            mock_snapshot_download_patch.side_effect = mock_snapshot_download

            return LeRobotDataset(repo_id=repo_id, root=root, **kwargs)

    return _create_lerobot_dataset


@pytest.fixture(scope="session")
def empty_lerobot_dataset_factory() -> LeRobotDatasetFactory:
    return partial(LeRobotDataset.create, repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS)
