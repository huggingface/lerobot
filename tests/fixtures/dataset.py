import datasets
import numpy as np
import pytest

from lerobot.common.datasets.utils import get_episode_data_index, hf_transform_to_torch


@pytest.fixture(scope="session")
def img_array_factory():
    def _create_img_array(width=100, height=100) -> np.ndarray:
        return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

    return _create_img_array


@pytest.fixture(scope="session")
def tasks():
    return [
        {"task_index": 0, "task": "Pick up the block."},
        {"task_index": 1, "task": "Open the box."},
        {"task_index": 2, "task": "Make paperclips."},
    ]


@pytest.fixture(scope="session")
def episode_dicts():
    return [
        {"episode_index": 0, "tasks": ["Pick up the block."], "length": 100},
        {"episode_index": 1, "tasks": ["Open the box."], "length": 80},
        {"episode_index": 2, "tasks": ["Pick up the block."], "length": 90},
        {"episode_index": 3, "tasks": ["Make paperclips."], "length": 150},
    ]


@pytest.fixture(scope="session")
def episode_data_index(episode_dicts):
    return get_episode_data_index(episode_dicts)


@pytest.fixture(scope="session")
def hf_dataset(hf_dataset_factory, episode_dicts, tasks):
    keys = ["state", "action"]
    shapes = {
        "state": 10,
        "action": 10,
    }
    return hf_dataset_factory(episode_dicts, tasks, keys, shapes)


@pytest.fixture(scope="session")
def hf_dataset_image(hf_dataset_factory, episode_dicts, tasks):
    keys = ["state", "action"]
    image_keys = ["image"]
    shapes = {
        "state": 10,
        "action": 10,
        "image": {
            "width": 100,
            "height": 70,
            "channels": 3,
        },
    }
    return hf_dataset_factory(episode_dicts, tasks, keys, shapes, image_keys=image_keys)


def get_task_index(tasks_dicts: dict, task: str) -> int:
    """
    Given a task in natural language, returns its task_index if the task already exists in the dataset,
    otherwise creates a new task_index.
    """
    tasks = {d["task_index"]: d["task"] for d in tasks_dicts}
    task_to_task_index = {task: task_idx for task_idx, task in tasks.items()}
    return task_to_task_index[task]


@pytest.fixture(scope="session")
def hf_dataset_factory(img_array_factory):
    def _create_hf_dataset(
        episode_dicts: list[dict],
        tasks: list[dict],
        keys: list[str],
        shapes: dict,
        fps: int = 30,
        image_keys: list[str] | None = None,
    ):
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
            ep_task_index = get_task_index(tasks, ep_dict["tasks"][0])
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
