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

from types import SimpleNamespace
from unittest.mock import patch

import draccus
import httpx
import numpy as np
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_edit_dataset import (
    ConvertImageToVideoConfig,
    DeleteEpisodesConfig,
    EditDatasetConfig,
    InfoConfig,
    MergeConfig,
    ModifyTasksConfig,
    OperationConfig,
    ReencodeVideosConfig,
    RemoveFeatureConfig,
    SplitConfig,
    _resolve_private,
    _validate_config,
    handle_delete_episodes,
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

    def test_modify_tasks_replacements_args_parse(self):
        cfg = parse_cfg(
            [
                "--repo_id",
                "test/repo",
                "--operation.type",
                "modify_tasks",
                "--operation.task_replacements",
                '{"task_0": "pick cube", "task_1": "place cube"}',
            ]
        )
        assert isinstance(cfg.operation, ModifyTasksConfig)
        assert cfg.operation.task_replacements == {
            "task_0": "pick cube",
            "task_1": "place cube",
        }


class TestDepthEncoderParsing:
    """Test that the depth encoder is exposed and parsed for video operations."""

    def test_reencode_has_default_depth_encoder(self):
        cfg = parse_cfg(["--repo_id", "test/repo", "--operation.type", "reencode_videos"])
        assert isinstance(cfg.operation, ReencodeVideosConfig)
        # A depth encoder is configured by default so depth videos are re-encoded too.
        assert cfg.operation.depth_encoder is not None
        assert hasattr(cfg.operation.depth_encoder, "depth_min")

    def test_reencode_parses_depth_encoder_overrides(self):
        cfg = parse_cfg(
            [
                "--repo_id",
                "test/repo",
                "--operation.type",
                "reencode_videos",
                "--operation.depth_encoder.extra_options",
                '{"x265-params": "lossless=1"}',
                "--operation.depth_encoder.depth_max",
                "12.0",
                "--operation.depth_encoder.use_log",
                "false",
            ]
        )
        assert cfg.operation.depth_encoder.extra_options == {"x265-params": "lossless=1"}
        assert cfg.operation.depth_encoder.depth_max == 12.0
        assert cfg.operation.depth_encoder.use_log is False

    def test_convert_image_to_video_parses_depth_encoder_overrides(self):
        cfg = parse_cfg(
            [
                "--repo_id",
                "test/repo",
                "--operation.type",
                "convert_image_to_video",
                "--operation.depth_encoder.depth_min",
                "0.05",
            ]
        )
        assert isinstance(cfg.operation, ConvertImageToVideoConfig)
        assert cfg.operation.depth_encoder.depth_min == 0.05


class TestPushPrivate:
    """Test that --private reaches the Hub upload (issue #2603)."""

    def test_private_defaults_to_none(self):
        cfg = parse_cfg(["--repo_id", "test/repo", "--operation.type", "delete_episodes"])
        assert cfg.private is None

    def test_private_flag_parses(self):
        cfg = parse_cfg(
            ["--repo_id", "test/repo", "--operation.type", "delete_episodes", "--private", "true"]
        )
        assert cfg.private is True

    @pytest.mark.parametrize(
        "handler, operation_args",
        [
            (
                handle_delete_episodes,
                ["--operation.type", "delete_episodes", "--operation.episode_indices", "[0]"],
            ),
            (handle_modify_tasks, ["--operation.type", "modify_tasks", "--operation.new_task", "new task"]),
        ],
    )
    def test_push_to_hub_forwards_private(
        self, tmp_path, empty_lerobot_dataset_factory, handler, operation_args
    ):
        features = {"action": {"dtype": "float32", "shape": (2,), "names": None}}
        dataset = empty_lerobot_dataset_factory(root=tmp_path / "input", features=features, use_videos=False)
        for _ in range(2):
            for _ in range(3):
                dataset.add_frame({"action": np.zeros(2, dtype=np.float32), "task": "task"})
            dataset.save_episode()
        dataset.finalize()

        cfg = parse_cfg(
            [
                "--repo_id",
                dataset.repo_id,
                "--root",
                str(tmp_path / "input"),
                "--new_repo_id",
                "user/edited",
                "--new_root",
                str(tmp_path / "output"),
                "--push_to_hub",
                "true",
                "--private",
                "true",
                *operation_args,
            ]
        )
        with (
            patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
            patch("lerobot.datasets.dataset_metadata.snapshot_download"),
            patch.object(LeRobotDataset, "push_to_hub", autospec=True) as mock_push,
        ):
            handler(cfg)

        mock_push.assert_called_once()
        assert mock_push.call_args.kwargs["private"] is True

    def test_push_inherits_private_source(self, tmp_path, empty_lerobot_dataset_factory):
        features = {"action": {"dtype": "float32", "shape": (2,), "names": None}}
        dataset = empty_lerobot_dataset_factory(root=tmp_path / "input", features=features, use_videos=False)
        for _ in range(2):
            for _ in range(3):
                dataset.add_frame({"action": np.zeros(2, dtype=np.float32), "task": "task"})
            dataset.save_episode()
        dataset.finalize()

        cfg = parse_cfg(
            [
                "--repo_id",
                dataset.repo_id,
                "--root",
                str(tmp_path / "input"),
                "--new_repo_id",
                "user/edited",
                "--new_root",
                str(tmp_path / "output"),
                "--push_to_hub",
                "true",
                "--operation.type",
                "delete_episodes",
                "--operation.episode_indices",
                "[0]",
            ]
        )
        with (
            patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
            patch("lerobot.datasets.dataset_metadata.snapshot_download"),
            patch(
                "lerobot.scripts.lerobot_edit_dataset.HfApi.repo_info",
                return_value=SimpleNamespace(private=True),
            ) as repo_info,
            patch.object(LeRobotDataset, "push_to_hub", autospec=True) as mock_push,
        ):
            handle_delete_episodes(cfg)

        repo_info.assert_called_once_with(dataset.repo_id, repo_type="dataset")
        assert mock_push.call_args.kwargs["private"] is True


class TestResolvePrivate:
    """Test that editing a private Hub dataset never publishes the result by default."""

    @staticmethod
    def _cfg(*extra: str) -> EditDatasetConfig:
        return parse_cfg(["--repo_id", "user/source", "--operation.type", "delete_episodes", *extra])

    @staticmethod
    def _hub(visibility: dict[str, bool]):
        """Patch HfApi.repo_info with a fake Hub where only the given repos exist."""

        def repo_info(repo_id, repo_type=None):
            if repo_id not in visibility:
                request = httpx.Request("GET", f"https://huggingface.co/api/datasets/{repo_id}")
                raise RepositoryNotFoundError("not found", response=httpx.Response(404, request=request))
            return SimpleNamespace(private=visibility[repo_id])

        return patch("lerobot.scripts.lerobot_edit_dataset.HfApi.repo_info", side_effect=repo_info)

    def test_private_source_makes_output_private(self):
        with self._hub({"user/source": True}):
            assert _resolve_private(self._cfg(), ["user/source"]) is True

    def test_public_source_keeps_default(self):
        with self._hub({"user/source": False}):
            assert _resolve_private(self._cfg(), ["user/source"]) is None

    def test_source_not_on_hub_keeps_default(self):
        with self._hub({}):
            assert _resolve_private(self._cfg(), ["local/only"]) is None

    def test_any_private_source_makes_merge_private(self):
        with self._hub({"user/a": False, "user/b": True}):
            assert _resolve_private(self._cfg(), ["user/a", "user/b"]) is True

    @pytest.mark.parametrize("flag, expected", [("true", True), ("false", False)])
    def test_explicit_flag_wins_without_lookup(self, flag, expected):
        with self._hub({"user/source": True}) as repo_info:
            assert _resolve_private(self._cfg("--private", flag), ["user/source"]) is expected
        repo_info.assert_not_called()
