import datasets
import pytest

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_episode_data_index
from tests.fixtures.defaults import DUMMY_CAMERA_KEYS


@pytest.fixture(scope="session")
def empty_info(info_factory) -> dict:
    return info_factory(
        keys=[],
        image_keys=[],
        video_keys=[],
        shapes={},
        names={},
    )


@pytest.fixture(scope="session")
def info(info_factory) -> dict:
    return info_factory(
        total_episodes=4,
        total_frames=420,
        total_tasks=3,
        total_videos=8,
        total_chunks=1,
    )


@pytest.fixture(scope="session")
def stats(stats_factory) -> list:
    return stats_factory()


@pytest.fixture(scope="session")
def tasks() -> list:
    return [
        {"task_index": 0, "task": "Pick up the block."},
        {"task_index": 1, "task": "Open the box."},
        {"task_index": 2, "task": "Make paperclips."},
    ]


@pytest.fixture(scope="session")
def episodes() -> list:
    return [
        {"episode_index": 0, "tasks": ["Pick up the block."], "length": 100},
        {"episode_index": 1, "tasks": ["Open the box."], "length": 80},
        {"episode_index": 2, "tasks": ["Pick up the block."], "length": 90},
        {"episode_index": 3, "tasks": ["Make paperclips."], "length": 150},
    ]


@pytest.fixture(scope="session")
def episode_data_index(episodes) -> dict:
    return get_episode_data_index(episodes)


@pytest.fixture(scope="session")
def hf_dataset(hf_dataset_factory) -> datasets.Dataset:
    return hf_dataset_factory()


@pytest.fixture(scope="session")
def hf_dataset_image(hf_dataset_factory) -> datasets.Dataset:
    image_keys = DUMMY_CAMERA_KEYS
    return hf_dataset_factory(image_keys=image_keys)


@pytest.fixture(scope="session")
def lerobot_dataset(lerobot_dataset_factory, tmp_path_factory) -> LeRobotDataset:
    root = tmp_path_factory.getbasetemp()
    return lerobot_dataset_factory(root=root)
