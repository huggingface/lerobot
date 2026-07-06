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
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

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
    _validate_config,
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
