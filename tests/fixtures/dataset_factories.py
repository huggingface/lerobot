import random
from pathlib import Path
from unittest.mock import patch

import datasets
import numpy as np
import PIL.Image
import pytest
import torch

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    hf_transform_to_torch,
)
from tests.fixtures.defaults import (
    DEFAULT_FPS,
    DUMMY_CAMERA_KEYS,
    DUMMY_KEYS,
    DUMMY_REPO_ID,
    DUMMY_ROBOT_TYPE,
)


def make_dummy_shapes(keys: list[str] | None = None, camera_keys: list[str] | None = None) -> dict:
    shapes = {}
    if keys:
        shapes.update({key: 10 for key in keys})
    if camera_keys:
        shapes.update({key: {"width": 100, "height": 70, "channels": 3} for key in camera_keys})
    return shapes


def get_task_index(tasks_dicts: dict, task: str) -> int:
    tasks = {d["task_index"]: d["task"] for d in tasks_dicts}
    task_to_task_index = {task: task_idx for task_idx, task in tasks.items()}
    return task_to_task_index[task]


@pytest.fixture(scope="session")
def img_tensor_factory():
    def _create_img_tensor(width=100, height=100, channels=3, dtype=torch.float32) -> torch.Tensor:
        return torch.rand((channels, height, width), dtype=dtype)

    return _create_img_tensor


@pytest.fixture(scope="session")
def img_array_factory():
    def _create_img_array(width=100, height=100, channels=3, dtype=np.uint8) -> np.ndarray:
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
    def _create_img(width=100, height=100) -> PIL.Image.Image:
        img_array = img_array_factory(width=width, height=height)
        return PIL.Image.fromarray(img_array)

    return _create_img


@pytest.fixture(scope="session")
def info_factory():
    def _create_info(
        codebase_version: str = CODEBASE_VERSION,
        fps: int = DEFAULT_FPS,
        robot_type: str = DUMMY_ROBOT_TYPE,
        keys: list[str] = DUMMY_KEYS,
        image_keys: list[str] | None = None,
        video_keys: list[str] = DUMMY_CAMERA_KEYS,
        shapes: dict | None = None,
        names: dict | None = None,
        total_episodes: int = 0,
        total_frames: int = 0,
        total_tasks: int = 0,
        total_videos: int = 0,
        total_chunks: int = 0,
        chunks_size: int = DEFAULT_CHUNK_SIZE,
        data_path: str = DEFAULT_PARQUET_PATH,
        videos_path: str = DEFAULT_VIDEO_PATH,
    ) -> dict:
        if not image_keys:
            image_keys = []
        if not shapes:
            shapes = make_dummy_shapes(keys=keys, camera_keys=[*image_keys, *video_keys])
        if not names:
            names = {key: [f"motor_{i}" for i in range(shapes[key])] for key in keys}

        video_info = {"videos_path": videos_path}
        for key in video_keys:
            video_info[key] = {
                "video.fps": fps,
                "video.width": shapes[key]["width"],
                "video.height": shapes[key]["height"],
                "video.channels": shapes[key]["channels"],
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
        return {
            "codebase_version": codebase_version,
            "data_path": data_path,
            "robot_type": robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_videos,
            "total_chunks": total_chunks,
            "chunks_size": chunks_size,
            "fps": fps,
            "splits": {},
            "keys": keys,
            "video_keys": video_keys,
            "image_keys": image_keys,
            "shapes": shapes,
            "names": names,
            "videos": video_info if len(video_keys) > 0 else None,
        }

    return _create_info


@pytest.fixture(scope="session")
def stats_factory():
    def _create_stats(
        keys: list[str] = DUMMY_KEYS,
        image_keys: list[str] | None = None,
        video_keys: list[str] = DUMMY_CAMERA_KEYS,
        shapes: dict | None = None,
    ) -> dict:
        if not image_keys:
            image_keys = []
        if not shapes:
            shapes = make_dummy_shapes(keys=keys, camera_keys=[*image_keys, *video_keys])
        stats = {}
        for key in keys:
            shape = shapes[key]
            stats[key] = {
                "max": np.full(shape, 1, dtype=np.float32).tolist(),
                "mean": np.full(shape, 0.5, dtype=np.float32).tolist(),
                "min": np.full(shape, 0, dtype=np.float32).tolist(),
                "std": np.full(shape, 0.25, dtype=np.float32).tolist(),
            }
        for key in [*image_keys, *video_keys]:
            shape = (3, 1, 1)
            stats[key] = {
                "max": np.full(shape, 1, dtype=np.float32).tolist(),
                "mean": np.full(shape, 0.5, dtype=np.float32).tolist(),
                "min": np.full(shape, 0, dtype=np.float32).tolist(),
                "std": np.full(shape, 0.25, dtype=np.float32).tolist(),
            }
        return stats

    return _create_stats


@pytest.fixture(scope="session")
def tasks_factory():
    def _create_tasks(total_tasks: int = 3) -> int:
        tasks_list = []
        for i in range(total_tasks):
            task_dict = {"task_index": i, "task": f"Perform action {i}."}
            tasks_list.append(task_dict)
        return tasks_list

    return _create_tasks


@pytest.fixture(scope="session")
def episodes_factory(tasks_factory):
    def _create_episodes(
        total_episodes: int = 3,
        total_frames: int = 400,
        task_dicts: dict | None = None,
        multi_task: bool = False,
    ):
        if total_episodes <= 0 or total_frames <= 0:
            raise ValueError("num_episodes and total_length must be positive integers.")
        if total_frames < total_episodes:
            raise ValueError("total_length must be greater than or equal to num_episodes.")

        if not task_dicts:
            min_tasks = 2 if multi_task else 1
            total_tasks = random.randint(min_tasks, total_episodes)
            task_dicts = tasks_factory(total_tasks)

        if total_episodes < len(task_dicts) and not multi_task:
            raise ValueError("The number of tasks should be less than the number of episodes.")

        # Generate random lengths that sum up to total_length
        lengths = np.random.multinomial(total_frames, [1 / total_episodes] * total_episodes).tolist()

        tasks_list = [task_dict["task"] for task_dict in task_dicts]
        num_tasks_available = len(tasks_list)

        episodes_list = []
        remaining_tasks = tasks_list.copy()
        for ep_idx in range(total_episodes):
            num_tasks_in_episode = random.randint(1, min(3, num_tasks_available)) if multi_task else 1
            tasks_to_sample = remaining_tasks if remaining_tasks else tasks_list
            episode_tasks = random.sample(tasks_to_sample, min(num_tasks_in_episode, len(tasks_to_sample)))
            if remaining_tasks:
                for task in episode_tasks:
                    remaining_tasks.remove(task)

            episodes_list.append(
                {
                    "episode_index": ep_idx,
                    "tasks": episode_tasks,
                    "length": lengths[ep_idx],
                }
            )

        return episodes_list

    return _create_episodes


@pytest.fixture(scope="session")
def hf_dataset_factory(img_array_factory, episodes, tasks):
    def _create_hf_dataset(
        episode_dicts: list[dict] = episodes,
        task_dicts: list[dict] = tasks,
        keys: list[str] = DUMMY_KEYS,
        image_keys: list[str] | None = None,
        shapes: dict | None = None,
        fps: int = DEFAULT_FPS,
    ) -> datasets.Dataset:
        if not image_keys:
            image_keys = []
        if not shapes:
            shapes = make_dummy_shapes(keys=keys, camera_keys=image_keys)
        key_features = {
            key: datasets.Sequence(length=shapes[key], feature=datasets.Value(dtype="float32"))
            for key in keys
        }
        image_features = {key: datasets.Image() for key in image_keys} if image_keys else {}
        common_features = {
            "episode_index": datasets.Value(dtype="int64"),
            "frame_index": datasets.Value(dtype="int64"),
            "timestamp": datasets.Value(dtype="float32"),
            "next.done": datasets.Value(dtype="bool"),
            "index": datasets.Value(dtype="int64"),
            "task_index": datasets.Value(dtype="int64"),
        }
        features = datasets.Features(
            {
                **key_features,
                **image_features,
                **common_features,
            }
        )

        episode_index_col = np.array([], dtype=np.int64)
        frame_index_col = np.array([], dtype=np.int64)
        timestamp_col = np.array([], dtype=np.float32)
        next_done_col = np.array([], dtype=bool)
        task_index = np.array([], dtype=np.int64)

        for ep_dict in episode_dicts:
            episode_index_col = np.concatenate(
                (episode_index_col, np.full(ep_dict["length"], ep_dict["episode_index"], dtype=int))
            )
            frame_index_col = np.concatenate((frame_index_col, np.arange(ep_dict["length"], dtype=int)))
            timestamp_col = np.concatenate((timestamp_col, np.arange(ep_dict["length"]) / fps))
            next_done_ep = np.full(ep_dict["length"], False, dtype=bool)
            next_done_ep[-1] = True
            next_done_col = np.concatenate((next_done_col, next_done_ep))
            ep_task_index = get_task_index(task_dicts, ep_dict["tasks"][0])
            task_index = np.concatenate((task_index, np.full(ep_dict["length"], ep_task_index, dtype=int)))

        index_col = np.arange(len(episode_index_col))
        key_cols = {key: np.random.random((len(index_col), shapes[key])).astype(np.float32) for key in keys}

        image_cols = {}
        if image_keys:
            for key in image_keys:
                image_cols[key] = [
                    img_array_factory(width=shapes[key]["width"], height=shapes[key]["height"])
                    for _ in range(len(index_col))
                ]

        dataset = datasets.Dataset.from_dict(
            {
                **key_cols,
                **image_cols,
                "episode_index": episode_index_col,
                "frame_index": frame_index_col,
                "timestamp": timestamp_col,
                "next.done": next_done_col,
                "index": index_col,
                "task_index": task_index,
            },
            features=features,
        )
        dataset.set_transform(hf_transform_to_torch)
        return dataset

    return _create_hf_dataset


@pytest.fixture(scope="session")
def lerobot_dataset_factory(
    info,
    stats,
    tasks,
    episodes,
    hf_dataset,
    mock_snapshot_download_factory,
):
    def _create_lerobot_dataset(
        root: Path,
        repo_id: str = DUMMY_REPO_ID,
        info_dict: dict = info,
        stats_dict: dict = stats,
        task_dicts: list[dict] = tasks,
        episode_dicts: list[dict] = episodes,
        hf_ds: datasets.Dataset = hf_dataset,
        **kwargs,
    ) -> LeRobotDataset:
        mock_snapshot_download = mock_snapshot_download_factory(
            info_dict=info_dict,
            stats_dict=stats_dict,
            tasks_dicts=task_dicts,
            episodes_dicts=episode_dicts,
            hf_ds=hf_ds,
        )
        with (
            patch(
                "lerobot.common.datasets.lerobot_dataset.get_hub_safe_version"
            ) as mock_get_hub_safe_version_patch,
            patch(
                "lerobot.common.datasets.lerobot_dataset.snapshot_download"
            ) as mock_snapshot_download_patch,
        ):
            mock_get_hub_safe_version_patch.side_effect = lambda repo_id, version, enforce_v2=True: version
            mock_snapshot_download_patch.side_effect = mock_snapshot_download

            return LeRobotDataset(repo_id=repo_id, root=root, **kwargs)

    return _create_lerobot_dataset


@pytest.fixture(scope="session")
def lerobot_dataset_from_episodes_factory(
    info_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    lerobot_dataset_factory,
):
    def _create_lerobot_dataset_total_episodes(
        root: Path,
        total_episodes: int = 3,
        total_frames: int = 150,
        total_tasks: int = 1,
        multi_task: bool = False,
        repo_id: str = DUMMY_REPO_ID,
        **kwargs,
    ):
        info_dict = info_factory(
            total_episodes=total_episodes, total_frames=total_frames, total_tasks=total_tasks
        )
        task_dicts = tasks_factory(total_tasks)
        episode_dicts = episodes_factory(
            total_episodes=total_episodes,
            total_frames=total_frames,
            task_dicts=task_dicts,
            multi_task=multi_task,
        )
        hf_dataset = hf_dataset_factory(episode_dicts=episode_dicts, task_dicts=task_dicts)
        return lerobot_dataset_factory(
            root=root,
            repo_id=repo_id,
            info_dict=info_dict,
            task_dicts=task_dicts,
            episode_dicts=episode_dicts,
            hf_ds=hf_dataset,
            **kwargs,
        )

    return _create_lerobot_dataset_total_episodes
