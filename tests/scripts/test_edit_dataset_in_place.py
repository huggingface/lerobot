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

import shutil

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_edit_dataset
from lerobot.scripts.lerobot_edit_dataset import (
    DeleteEpisodesConfig,
    EditDatasetConfig,
    RemoveFeatureConfig,
    edit_dataset,
)

pytestmark = pytest.mark.dataset


def _edit_in_place(tmp_path, operation):
    """Build a one-episode dataset, then run `operation` on it in place."""
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        "local/dataset",
        10,
        root=root,
        use_videos=False,
        features={"observation.state": {"dtype": "float32", "shape": (2,), "names": None}},
    )
    for _ in range(3):
        dataset.add_frame({"observation.state": np.zeros(2, np.float32), "task": "t"})
    dataset.save_episode()
    dataset.finalize()
    edit_dataset(
        EditDatasetConfig(repo_id="local/dataset", root=str(root), new_root=str(root), operation=operation)
    )


def _write_partial_output_then_fail(dataset, output_dir, **kwargs):
    (output_dir / "data").mkdir(parents=True)
    raise ValueError("failed halfway")


@pytest.mark.parametrize(
    "operation",
    [RemoveFeatureConfig(feature_names=["missing"]), DeleteEpisodesConfig(episode_indices=[7])],
)
def test_a_failed_in_place_edit_leaves_the_dataset_where_it_was(tmp_path, operation):
    with pytest.raises(ValueError):
        _edit_in_place(tmp_path, operation)
    assert not (tmp_path / "dataset_old").exists()
    assert LeRobotDataset("local/dataset", root=tmp_path / "dataset").meta.total_episodes == 1


def test_a_failed_in_place_edit_drops_its_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(lerobot_edit_dataset, "delete_episodes", _write_partial_output_then_fail)
    with pytest.raises(ValueError, match="failed halfway"):
        _edit_in_place(tmp_path, DeleteEpisodesConfig(episode_indices=[0]))
    assert not (tmp_path / "dataset_old").exists()
    assert LeRobotDataset("local/dataset", root=tmp_path / "dataset").meta.total_episodes == 1


def test_a_failed_restore_keeps_the_backup_and_the_original_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lerobot_edit_dataset, "delete_episodes", _write_partial_output_then_fail)
    real_rmtree = shutil.rmtree

    def refuse_output_dir(path, *args, **kwargs):
        if path == (tmp_path / "dataset").resolve():
            raise PermissionError("locked")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(lerobot_edit_dataset.shutil, "rmtree", refuse_output_dir)
    with pytest.raises(ValueError, match="failed halfway"):
        _edit_in_place(tmp_path, DeleteEpisodesConfig(episode_indices=[0]))
    assert LeRobotDataset("local/dataset", root=tmp_path / "dataset_old").meta.total_episodes == 1
    assert not (tmp_path / "dataset" / "dataset_old").exists()
