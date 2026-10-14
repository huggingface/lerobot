# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset, make_train_eval_datasets
from tests.fixtures.dataset_factories import LeRobotDatasetFactory


@pytest.fixture
def split_config(tmp_path: Path, empty_lerobot_dataset_factory: LeRobotDatasetFactory) -> SimpleNamespace:
    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "dataset",
        use_videos=False,
        features={"action": {"dtype": "float32", "shape": (2,), "names": None}},
    )
    for ep in range(6):
        for frame in range(ep % 3 + 2):
            dataset.add_frame({"action": np.array([ep, frame], dtype=np.float32), "task": f"Task {ep % 2}"})
        dataset.save_episode()
    dataset.finalize()
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id=dataset.repo_id, root=dataset.root, eval_split=0.5, use_imagenet_stats=False
        ),
        trainable_config=SimpleNamespace(
            observation_delta_indices=None, action_delta_indices=None, reward_delta_indices=None
        ),
        rename_map={},
        tolerance_s=1e-4,
    )


@pytest.mark.parametrize(
    "episodes, expected", [(None, ([0, 1], [2, 4, 3, 5])), ([5, 1, 4, 2], ([5, 4], [1, 2]))]
)
def test_missing_episode_tasks_matches_metadata(
    split_config: SimpleNamespace,
    episodes: list[int] | None,
    expected: tuple[list[int], list[int]],
) -> None:
    split_config.dataset.episodes = episodes
    train, evaluation = make_train_eval_datasets(split_config)
    assert (train.episodes, evaluation.episodes) == expected
    for path in (split_config.dataset.root / "meta/episodes").rglob("*.parquet"):
        pq.write_table(pq.read_table(path).drop(["tasks"]), path)
    dataset = make_dataset(split_config)

    def task_only(batch: dict[str, Any]) -> dict[str, Any]:
        assert set(batch) == {"task_index"}
        return batch

    dataset.hf_dataset.set_transform(task_only)
    with patch("lerobot.datasets.factory.make_dataset", return_value=dataset):
        train, evaluation = make_train_eval_datasets(split_config)
    assert (train.episodes, evaluation.episodes) == expected


@pytest.mark.parametrize("tasks", [["shared", "ignored"], []])
def test_episode_tasks_take_precedence(split_config: SimpleNamespace, tasks: list[str]) -> None:
    for path in (split_config.dataset.root / "meta/episodes").rglob("*.parquet"):
        table = pq.read_table(path)
        values = [tasks + [f"episode {ep}"] if tasks else [] for ep in range(6)]
        table = table.set_column(table.column_names.index("tasks"), "tasks", pa.array(values))
        pq.write_table(table, path)
    train, evaluation = make_train_eval_datasets(split_config)
    assert (train.episodes, evaluation.episodes) == ([0, 1, 2], [3, 4, 5])


def test_fallback_uses_first_frame(split_config: SimpleNamespace) -> None:
    for path in (split_config.dataset.root / "meta/episodes").rglob("*.parquet"):
        pq.write_table(pq.read_table(path).drop(["tasks"]), path)
    for path in (split_config.dataset.root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        values = [
            row["task_index"] if row["frame_index"] == 0 else 1 - row["task_index"]
            for row in table.select(["task_index", "frame_index"]).to_pylist()
        ]
        pq.write_table(
            table.set_column(table.column_names.index("task_index"), "task_index", pa.array(values)), path
        )
    train, evaluation = make_train_eval_datasets(split_config)
    assert (train.episodes, evaluation.episodes) == ([0, 1], [2, 4, 3, 5])


def test_missing_both_task_sources(split_config: SimpleNamespace) -> None:
    root = split_config.dataset.root
    for folder, column in [("meta/episodes", "tasks"), ("data", "task_index")]:
        for path in (root / folder).rglob("*.parquet"):
            pq.write_table(pq.read_table(path).drop([column]), path)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    del info["features"]["task_index"]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="episode 'tasks'.*per-frame 'task_index'"):
        make_train_eval_datasets(split_config)
