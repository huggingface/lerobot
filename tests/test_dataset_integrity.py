import json

import pytest
from datasets import load_dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "7jep7/rook_to_d4_v4"
dataset = LeRobotDataset(repo_id=REPO_ID, force_cache_sync=True)


@pytest.fixture(scope="session")
def meta():
    return dataset.meta


def test_task_indices_match(meta):
    tasks_path = meta.root / "meta/tasks.jsonl"
    task_ids = set()
    with open(tasks_path) as f:
        for line in f:
            task = json.loads(line)
            task_ids.add(task["task_index"])

    for ep_idx in meta.episodes:
        parquet_path = meta.root / meta.get_data_file_path(ep_idx)
        ep_data = load_dataset("parquet", data_files=str(parquet_path), split="train")
        for idx in ep_data["task_index"]:
            assert idx in task_ids, f"Invalid task_index {idx} in episode {ep_idx}"


def test_episode_tasks_exist(meta):
    for ep in meta.episodes.values():
        for task in ep["tasks"]:
            assert task in meta.task_to_task_index, f"Missing task string: {task}"


def test_stats_exist_for_features(meta):
    float_keys = [k for k, v in meta.features.items() if v["dtype"] == "float"]
    for key in float_keys:
        assert key in meta.stats, f"Missing stats for: {key}"
        for field in ["mean", "std"]:
            assert field in meta.stats[key], f"Missing {field} for {key}"


def test_parquet_has_expected_features(meta):
    expected_keys = set(meta.features.keys()) | {
        "index",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    }
    for ep_idx in meta.episodes:
        parquet_path = meta.root / meta.get_data_file_path(ep_idx)
        ep_data = load_dataset("parquet", data_files=str(parquet_path), split="train")
        actual_keys = set(ep_data.column_names)
        missing = expected_keys - actual_keys
        assert not missing, f"Missing features {missing} in episode {ep_idx}"


def test_no_duplicate_tasks(meta):
    seen = {}
    for task, idx in meta.task_to_task_index.items():
        assert idx not in seen, f"Duplicate task_index: {idx} for {task}"
        seen[idx] = task


