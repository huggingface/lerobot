#!/usr/bin/env python

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

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_edit_dataset import EditDatasetConfig, ModifyTasksConfig, handle_modify_tasks


@pytest.fixture
def sample_dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test_dataset",
        features=features,
    )

    for ep_idx in range(5):
        for _ in range(10):
            frame = {
                "action": np.random.randn(6).astype(np.float32),
                "observation.state": np.random.randn(4).astype(np.float32),
                "observation.images.top": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                "task": f"task_{ep_idx % 2}",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def test_handle_modify_tasks_with_replacements(sample_dataset):
    cfg = EditDatasetConfig(
        repo_id=sample_dataset.repo_id,
        root=str(sample_dataset.root),
        operation=ModifyTasksConfig(
            task_replacements={
                "task_0": "Pick the cube",
                "task_1": "Place the cube",
            }
        ),
    )

    handle_modify_tasks(cfg)

    modified_dataset = LeRobotDataset(cfg.repo_id, root=sample_dataset.root)
    assert modified_dataset.meta.episodes[0]["tasks"][0] == "Pick the cube"
    assert modified_dataset.meta.episodes[1]["tasks"][0] == "Place the cube"
    assert len(modified_dataset.meta.tasks) == 2


def test_handle_modify_tasks_rejects_default_task_and_replacements(sample_dataset):
    cfg = EditDatasetConfig(
        repo_id=sample_dataset.repo_id,
        root=str(sample_dataset.root),
        operation=ModifyTasksConfig(
            new_task="Default task",
            task_replacements={"task_0": "Pick the cube"},
        ),
    )

    with pytest.raises(ValueError, match="Cannot combine new_task with task_replacements"):
        handle_modify_tasks(cfg)
