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
"""Offline CPU tests for filter_episodes and lerobot-edit-dataset CLI integration."""

import shutil
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset, filter_episodes
from lerobot.datasets.dataset_tools import _build_episode_filter_predicate
from lerobot.scripts.lerobot_edit_dataset import (
    EditDatasetConfig,
    FilterEpisodesConfig,
    OperationConfig,
    handle_filter_episodes,
)

pytestmark = pytest.mark.dataset

_TEST_FPS = 10
_EPISODE_SPECS = [
    ("task_a", 5),  # ep 0: 0.5s
    ("task_b", 10),  # ep 1: 1.0s
    ("task_a", 15),  # ep 2: 1.5s
    ("task_c", 20),  # ep 3: 2.0s
    ("task_b", 25),  # ep 4: 2.5s
]


@pytest.fixture(scope="module")
def multi_length_dataset_template(tmp_path_factory):
    """Build a synthetic multi-task, variable-length dataset once per module."""
    features = {
        "action": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (32, 32, 3), "names": None},
    }
    root = tmp_path_factory.mktemp("filter_episodes_template") / "src_dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/filter_test",
        fps=_TEST_FPS,
        root=root,
        features=features,
    )
    rng = np.random.default_rng(0)
    for task_name, num_frames in _EPISODE_SPECS:
        for _ in range(num_frames):
            dataset.add_frame(
                {
                    "action": rng.standard_normal(4).astype(np.float32),
                    "observation.state": rng.standard_normal(3).astype(np.float32),
                    "observation.images.top": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                    "task": task_name,
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


@pytest.fixture
def multi_length_dataset(tmp_path, multi_length_dataset_template):
    """Private per-test copy of the synthetic dataset."""
    root = tmp_path / "src_dataset"
    shutil.copytree(multi_length_dataset_template, root)
    return LeRobotDataset("local/filter_test", root=root)


def _offline_hub_patches(output_dir):
    return (
        patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
        patch("lerobot.datasets.dataset_metadata.snapshot_download", return_value=str(output_dir)),
    )


# ---------------------------------------------------------------------------
# Predicate building & validation (delegated to dataset.meta.filter_episodes)
# ---------------------------------------------------------------------------


def test_predicate_matches_tasks_and_exclude_tasks(multi_length_dataset):
    p_a = _build_episode_filter_predicate(fps=_TEST_FPS, tasks=["task_a"])
    assert multi_length_dataset.meta.filter_episodes(p_a) == [0, 2]

    p_ac = _build_episode_filter_predicate(fps=_TEST_FPS, tasks=["task_a", "task_c"])
    assert multi_length_dataset.meta.filter_episodes(p_ac) == [0, 2, 3]

    p_excl = _build_episode_filter_predicate(fps=_TEST_FPS, exclude_tasks=["task_b"])
    assert multi_length_dataset.meta.filter_episodes(p_excl) == [0, 2, 3]

    p_both = _build_episode_filter_predicate(
        fps=_TEST_FPS, tasks=["task_a", "task_b"], exclude_tasks=["task_b"]
    )
    assert multi_length_dataset.meta.filter_episodes(p_both) == [0, 2]

    p_none = _build_episode_filter_predicate(fps=_TEST_FPS, tasks=["nonexistent_task"])
    assert multi_length_dataset.meta.filter_episodes(p_none) == []


def test_predicate_matches_frames_duration_and_custom_callable(multi_length_dataset):
    p_frames = _build_episode_filter_predicate(fps=_TEST_FPS, min_frames=10, max_frames=20)
    assert multi_length_dataset.meta.filter_episodes(p_frames) == [1, 2, 3]

    p_dur = _build_episode_filter_predicate(fps=_TEST_FPS, min_duration_s=1.0, max_duration_s=2.0)
    assert multi_length_dataset.meta.filter_episodes(p_dur) == [1, 2, 3]

    p_combo = _build_episode_filter_predicate(fps=_TEST_FPS, min_frames=15, tasks=["task_b"])
    assert multi_length_dataset.meta.filter_episodes(p_combo) == [4]

    p_custom = _build_episode_filter_predicate(
        fps=_TEST_FPS,
        predicate=lambda ep: ep["episode_index"] % 2 == 0 and ep["length"] >= 15,
    )
    assert multi_length_dataset.meta.filter_episodes(p_custom) == [2, 4]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "Must specify at least one filter criterion"),
        ({"tasks": []}, "tasks must not be empty"),
        ({"min_frames": -1}, "min_frames must be >= 0"),
        ({"max_frames": -5}, "max_frames must be >= 0"),
        ({"min_frames": 20, "max_frames": 10}, "min_frames .* must be <= max_frames"),
        ({"min_duration_s": -0.1}, "min_duration_s must be finite and >= 0"),
        ({"max_duration_s": float("nan")}, "max_duration_s must be finite and >= 0"),
        ({"min_duration_s": float("inf")}, "min_duration_s must be finite and >= 0"),
        ({"min_duration_s": 3.0, "max_duration_s": 1.0}, "min_duration_s .* must be <= max_duration_s"),
    ],
)
def test_filter_episodes_invalid_args(multi_length_dataset, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        filter_episodes(multi_length_dataset, output_dir=tmp_path / "out", **kwargs)


# ---------------------------------------------------------------------------
# filter_episodes end-to-end materialization
# ---------------------------------------------------------------------------


def test_filter_episodes_by_tasks(multi_length_dataset, tmp_path):
    output_dir = tmp_path / "filtered_tasks"
    p1, p2 = _offline_hub_patches(output_dir)
    with p1, p2:
        filtered = filter_episodes(
            multi_length_dataset,
            tasks=["task_a"],
            output_dir=output_dir,
            repo_id="local/filter_test_task_a",
        )

    assert filtered.meta.total_episodes == 2
    assert filtered.meta.total_frames == 20  # 5 + 15
    assert len(filtered) == 20
    assert sorted(filtered.meta.tasks.index.tolist()) == ["task_a"]
    assert {int(idx.item()) for idx in filtered.hf_dataset["episode_index"]} == {0, 1}
    assert filtered.meta.stats is not None
    assert "action" in filtered.meta.stats
    assert filtered[0]["task"] == "task_a"


def test_filter_episodes_combined_duration_and_exclude(multi_length_dataset, tmp_path):
    output_dir = tmp_path / "filtered_combined"
    p1, p2 = _offline_hub_patches(output_dir)
    with p1, p2:
        filtered = filter_episodes(
            multi_length_dataset,
            exclude_tasks=["task_a"],
            min_duration_s=1.5,
            max_duration_s=2.5,
            output_dir=output_dir,
        )

    # Matches ep 3 (task_c, 2.0s) and ep 4 (task_b, 2.5s)
    assert filtered.meta.total_episodes == 2
    assert filtered.meta.total_frames == 45  # 20 + 25
    assert sorted(filtered.meta.tasks.index.tolist()) == ["task_b", "task_c"]


def test_filter_episodes_keeps_all_when_all_match(multi_length_dataset, tmp_path):
    output_dir = tmp_path / "filtered_all"
    p1, p2 = _offline_hub_patches(output_dir)
    with p1, p2:
        filtered = filter_episodes(
            multi_length_dataset,
            min_frames=1,
            output_dir=output_dir,
        )

    assert filtered.meta.total_episodes == 5
    assert filtered.meta.total_frames == 75


def test_filter_episodes_no_matches_raises(multi_length_dataset, tmp_path):
    with pytest.raises(ValueError, match="No episodes matched the filter criteria"):
        filter_episodes(
            multi_length_dataset,
            min_frames=100,
            output_dir=tmp_path / "empty_out",
        )


def test_filter_episodes_with_predicate(multi_length_dataset, tmp_path):
    output_dir = tmp_path / "filtered_pred"
    p1, p2 = _offline_hub_patches(output_dir)
    with p1, p2:
        filtered = filter_episodes(
            multi_length_dataset,
            predicate=lambda ep: ep["length"] in {5, 20},
            output_dir=output_dir,
        )

    assert filtered.meta.total_episodes == 2
    assert filtered.meta.total_frames == 25


# ---------------------------------------------------------------------------
# CLI integration (lerobot-edit-dataset)
# ---------------------------------------------------------------------------


def test_cli_operation_registered():
    assert "filter_episodes" in OperationConfig.get_known_choices()


def test_cli_handle_filter_episodes_dry_run(multi_length_dataset, tmp_path, caplog):
    output_dir = tmp_path / "should_not_exist"
    cfg = EditDatasetConfig(
        repo_id="local/filter_test",
        root=str(multi_length_dataset.root),
        new_repo_id="local/filter_test_out",
        new_root=str(output_dir),
        operation=FilterEpisodesConfig(tasks=["task_a"], dry_run=True),
    )

    with caplog.at_level("INFO"):
        handle_filter_episodes(cfg)

    assert not output_dir.exists()
    assert "Matched 2 / 5 episodes" in caplog.text


def test_cli_handle_filter_episodes_writes_new_dataset(multi_length_dataset, tmp_path):
    output_dir = tmp_path / "cli_filtered"
    cfg = EditDatasetConfig(
        repo_id="local/filter_test",
        root=str(multi_length_dataset.root),
        new_repo_id="local/filter_test_cli",
        new_root=str(output_dir),
        operation=FilterEpisodesConfig(min_frames=15, max_frames=20),
    )

    p1, p2 = _offline_hub_patches(output_dir)
    with p1, p2:
        handle_filter_episodes(cfg)

    assert output_dir.exists()
    loaded = LeRobotDataset("local/filter_test_cli", root=output_dir)
    assert loaded.meta.total_episodes == 2  # ep 2 (15) and ep 3 (20)
    assert loaded.meta.total_frames == 35
