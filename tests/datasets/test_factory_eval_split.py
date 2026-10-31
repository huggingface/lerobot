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
"""make_train_eval_datasets() must never remove a task from training entirely.

Regression test: a task with few episodes could see ceil(len(eps) * eval_split)
round up to its whole episode count (ceil(1 * 0.1) == 1), silently sending 100%
of that task's data to eval although DatasetConfig requires eval_split < 1.0.
"""

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DEFAULT_FPS
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
}


def _build_dataset(root, episode_tasks: list[str]) -> None:
    """One 2-frame episode per entry of `episode_tasks`."""
    dataset = LeRobotDataset.create(
        repo_id="dummy/eval_split_test", fps=DEFAULT_FPS, features=FEATURES, root=root, use_videos=False
    )
    frame = {
        "observation.state": torch.zeros(2, dtype=torch.float32),
        "action": torch.zeros(2, dtype=torch.float32),
    }
    for task in episode_tasks:
        dataset.add_frame({**frame, "task": task})
        dataset.add_frame({**frame, "task": task})
        dataset.save_episode()
    dataset.finalize()


def _make_cfg(root, eval_split: float) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/eval_split_test", root=str(root), eval_split=eval_split),
        policy=DummyCheckpointConfig(device="cpu"),
    )


def test_single_episode_task_stays_in_train(tmp_path):
    """A 1-episode task must keep that episode in train; a 9-episode task still splits normally."""
    root = tmp_path / "ds"
    _build_dataset(root, ["task_A"] + ["task_B"] * 9)

    train_ds, eval_ds = make_train_eval_datasets(_make_cfg(root, eval_split=0.1))

    train_tasks = {train_ds.meta.episodes[i]["tasks"][0] for i in train_ds.episodes}
    assert "task_A" in train_tasks, "a task must never lose all its episodes to eval"
    # ceil(9 * 0.1) == 1 held out for task_B; task_A is capped at 0 (only 1 episode).
    assert (train_ds.num_episodes, eval_ds.num_episodes) == (9, 1)


def test_all_single_episode_tasks_raises_clear_error(tmp_path):
    """Every task capped to 0 eval episodes must fail fast with a clear message, instead of
    building an empty-episode eval dataset that crashes later with an opaque HF `datasets` error.
    """
    root = tmp_path / "ds"
    _build_dataset(root, ["task_A", "task_B", "task_C"])  # 1 episode each

    with pytest.raises(ValueError, match="eval_split=0.5 holds out no eval episodes"):
        make_train_eval_datasets(_make_cfg(root, eval_split=0.5))
