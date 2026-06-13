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

import draccus
import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_edit_dataset import (
    ConvertImageToVideoConfig,
    DeleteEpisodesConfig,
    EditDatasetConfig,
    InfoConfig,
    MergeConfig,
    ModifyTasksConfig,
    OperationConfig,
    RemoveFeatureConfig,
    SplitConfig,
    _validate_config,
    handle_modify_tasks,
)


def parse_cfg(cli_args: list[str]) -> EditDatasetConfig:
    """Helper to parse CLI args into an EditDatasetConfig via draccus."""
    return draccus.parse(EditDatasetConfig, args=cli_args)


class TestOperationTypeParsing:
    """Test that --operation.type correctly selects the right config subclass."""

    @pytest.mark.parametrize(
        "type_name, expected_cls",
        [
            ("delete_episodes", DeleteEpisodesConfig),
            ("split", SplitConfig),
            ("merge", MergeConfig),
            ("remove_feature", RemoveFeatureConfig),
            ("modify_tasks", ModifyTasksConfig),
            ("convert_image_to_video", ConvertImageToVideoConfig),
            ("info", InfoConfig),
        ],
    )
    def test_operation_type_resolves_correct_class(self, type_name, expected_cls):
        cfg = parse_cfg(
            ["--repo_id", "test/repo", "--new_repo_id", "test/merged", "--operation.type", type_name]
        )
        assert isinstance(cfg.operation, expected_cls), (
            f"Expected {expected_cls.__name__}, got {type(cfg.operation).__name__}"
        )

    def test_merge_requires_new_repo_id(self):
        cfg = parse_cfg(["--operation.type", "merge"])
        with pytest.raises(ValueError, match="--new_repo_id is required for merge"):
            _validate_config(cfg)

    @pytest.mark.parametrize("flag", ["concatenate_videos", "concatenate_data"])
    def test_merge_concatenate_flag_defaults_true(self, flag):
        cfg = parse_cfg(["--new_repo_id", "test/merged", "--operation.type", "merge"])
        assert isinstance(cfg.operation, MergeConfig)
        assert getattr(cfg.operation, flag) is True

    @pytest.mark.parametrize("flag", ["concatenate_videos", "concatenate_data"])
    def test_merge_concatenate_flag_can_be_disabled(self, flag):
        cfg = parse_cfg(
            ["--new_repo_id", "test/merged", "--operation.type", "merge", f"--operation.{flag}", "false"]
        )
        assert isinstance(cfg.operation, MergeConfig)
        assert getattr(cfg.operation, flag) is False

    def test_non_merge_requires_repo_id(self):
        cfg = parse_cfg(["--operation.type", "delete_episodes"])
        with pytest.raises(ValueError, match="--repo_id is required for delete_episodes"):
            _validate_config(cfg)

    @pytest.mark.parametrize(
        "type_name, expected_cls",
        [
            ("delete_episodes", DeleteEpisodesConfig),
            ("split", SplitConfig),
            ("merge", MergeConfig),
            ("remove_feature", RemoveFeatureConfig),
            ("modify_tasks", ModifyTasksConfig),
            ("convert_image_to_video", ConvertImageToVideoConfig),
            ("info", InfoConfig),
        ],
    )
    def test_get_choice_name_roundtrips(self, type_name, expected_cls):
        cfg = parse_cfg(
            ["--repo_id", "test/repo", "--new_repo_id", "test/merged", "--operation.type", type_name]
        )
        resolved_name = OperationConfig.get_choice_name(type(cfg.operation))
        assert resolved_name == type_name

    def test_modify_tasks_parses_overwrite_flag(self):
        cfg = parse_cfg(
            [
                "--repo_id",
                "test/repo",
                "--operation.type",
                "modify_tasks",
                "--operation.new_task",
                "Pick up the cube",
                "--operation.overwrite",
                "true",
            ]
        )

        assert isinstance(cfg.operation, ModifyTasksConfig)
        assert cfg.operation.overwrite is True


def _create_two_episode_dataset(empty_lerobot_dataset_factory, root, repo_id="test/source"):
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=root, repo_id=repo_id, features=features, use_videos=False)

    for task in ["Original task A", "Original task B"]:
        for _ in range(2):
            dataset.add_frame(
                {
                    "action": np.array([1.0], dtype=np.float32),
                    "observation.state": np.array([0.0], dtype=np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()

    dataset.finalize()
    return dataset


def test_handle_modify_tasks_writes_new_dataset_without_changing_source(
    tmp_path, empty_lerobot_dataset_factory
):
    source_root = tmp_path / "source"
    output_root = tmp_path / "renamed"
    _create_two_episode_dataset(empty_lerobot_dataset_factory, source_root)

    cfg = EditDatasetConfig(
        repo_id="test/source",
        root=str(source_root),
        new_repo_id="test/renamed",
        new_root=str(output_root),
        operation=ModifyTasksConfig(new_task="Renamed task"),
    )

    handle_modify_tasks(cfg)

    source_dataset = LeRobotDataset("test/source", root=source_root)
    renamed_dataset = LeRobotDataset("test/renamed", root=output_root)

    assert set(source_dataset.meta.tasks.index) == {"Original task A", "Original task B"}
    assert set(renamed_dataset.meta.tasks.index) == {"Renamed task"}
    assert all(renamed_dataset[i]["task"] == "Renamed task" for i in range(len(renamed_dataset)))


def test_handle_modify_tasks_overwrite_true_modifies_source_in_place(tmp_path, empty_lerobot_dataset_factory):
    source_root = tmp_path / "source"
    _create_two_episode_dataset(empty_lerobot_dataset_factory, source_root)

    cfg = EditDatasetConfig(
        repo_id="test/source",
        root=str(source_root),
        operation=ModifyTasksConfig(new_task="Overwritten task", overwrite=True),
    )

    handle_modify_tasks(cfg)

    source_dataset = LeRobotDataset("test/source", root=source_root)

    assert set(source_dataset.meta.tasks.index) == {"Overwritten task"}
    assert all(source_dataset[i]["task"] == "Overwritten task" for i in range(len(source_dataset)))
