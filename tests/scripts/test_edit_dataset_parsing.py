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
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

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


def fake_hub(visibility: dict[str, bool], status_code: int = 404):
    """Patch HfApi.repo_info with a fake Hub where only the given repos exist.

    Other repos raise RepositoryNotFoundError (404), or a generic HfHubHTTPError for another status code.
    """

    def repo_info(repo_id, repo_type=None):
        if repo_id in visibility:
            return SimpleNamespace(private=visibility[repo_id])
        request = httpx.Request("GET", f"https://huggingface.co/api/datasets/{repo_id}")
        response = httpx.Response(status_code, request=request)
        if status_code == 404:
            raise RepositoryNotFoundError("not found", response=response)
        raise HfHubHTTPError("server error", response=response)

    return patch("lerobot.scripts.lerobot_edit_dataset.HfApi.repo_info", side_effect=repo_info)


def make_tiny_dataset(factory, root):
    features = {"action": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = factory(root=root, features=features, use_videos=False)
    for _ in range(2):
        for _ in range(3):
            dataset.add_frame({"action": np.zeros(2, dtype=np.float32), "task": "task"})
        dataset.save_episode()
    dataset.finalize()
    return dataset


def delete_episodes_cfg(dataset, tmp_path, *extra: str) -> EditDatasetConfig:
    return parse_cfg(
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
            *extra,
        ]
    )


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
        dataset = make_tiny_dataset(empty_lerobot_dataset_factory, tmp_path / "input")

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
            fake_hub({}),
            patch.object(LeRobotDataset, "push_to_hub", autospec=True) as mock_push,
        ):
            handler(cfg)

        mock_push.assert_called_once()
        assert mock_push.call_args.kwargs["private"] is True

    def test_push_inherits_private_source(self, tmp_path, empty_lerobot_dataset_factory):
        dataset = make_tiny_dataset(empty_lerobot_dataset_factory, tmp_path / "input")
        cfg = delete_episodes_cfg(dataset, tmp_path)
        with (
            patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
            patch("lerobot.datasets.dataset_metadata.snapshot_download"),
            fake_hub({dataset.repo_id: True}),
            patch.object(LeRobotDataset, "push_to_hub", autospec=True) as mock_push,
        ):
            handle_delete_episodes(cfg)

        assert mock_push.call_args.kwargs["private"] is True

    @pytest.mark.parametrize(
        "hub, extra",
        [
            # The source is not found and no --private is given.
            ({}, []),
            # The source is private, but the output repo already exists and is public.
            ({"user/edited": False}, []),
            # --private true, but the output repo already exists and is public.
            ({"user/edited": False}, ["--private", "true"]),
        ],
    )
    def test_stops_before_any_work(self, tmp_path, empty_lerobot_dataset_factory, hub, extra):
        dataset = make_tiny_dataset(empty_lerobot_dataset_factory, tmp_path / "input")
        cfg = delete_episodes_cfg(dataset, tmp_path, *extra)
        if hub:
            hub = {dataset.repo_id: True, **hub}
        with (
            fake_hub(hub),
            patch.object(LeRobotDataset, "push_to_hub", autospec=True) as mock_push,
            pytest.raises(ValueError, match="--private"),
        ):
            handle_delete_episodes(cfg)

        mock_push.assert_not_called()
        assert not (tmp_path / "output").exists()


class TestResolvePrivate:
    """Test that editing a private Hub dataset never publishes the result by default."""

    @staticmethod
    def _cfg(*extra: str) -> EditDatasetConfig:
        return parse_cfg(["--repo_id", "user/source", "--operation.type", "delete_episodes", *extra])

    def test_private_source_makes_output_private(self):
        with fake_hub({"user/source": True}):
            assert _resolve_private(self._cfg(), ["user/source"], ["user/output"]) is True

    def test_public_source_keeps_default(self):
        with fake_hub({"user/source": False}):
            assert _resolve_private(self._cfg(), ["user/source"], ["user/output"]) is None

    def test_source_not_found_stops(self):
        # The Hub answers the same for a local-only dataset and a private one the token cannot see.
        with fake_hub({}), pytest.raises(ValueError, match="local/only.*--private true or --private false"):
            _resolve_private(self._cfg(), ["local/only"], ["user/output"])

    @pytest.mark.parametrize("flag, expected", [("true", True), ("false", False)])
    def test_source_not_found_with_explicit_flag(self, flag, expected):
        with fake_hub({}):
            assert _resolve_private(self._cfg("--private", flag), ["local/only"], ["user/output"]) is expected

    def test_other_hub_errors_stop(self):
        with fake_hub({}, status_code=500), pytest.raises(HfHubHTTPError):
            _resolve_private(self._cfg(), ["user/source"], ["user/output"])

    @pytest.mark.parametrize(
        "hub, expected",
        [
            # A private recording merged with a public one stays private.
            ({"user/a": False, "user/b": True}, True),
            ({"user/a": True, "user/b": False}, True),
            # Any private source is enough, even if another one is not found.
            ({"user/b": True}, True),
            ({"user/a": False, "user/b": False}, None),
        ],
    )
    def test_merge_with_mixed_visibility(self, hub, expected):
        with fake_hub(hub):
            assert _resolve_private(self._cfg(), ["user/a", "user/b"], ["user/merged"]) is expected

    def test_merge_with_public_and_not_found_source_stops(self):
        with fake_hub({"user/a": False}), pytest.raises(ValueError, match="user/b"):
            _resolve_private(self._cfg(), ["user/a", "user/b"], ["user/merged"])

    def test_explicit_false_skips_lookups(self):
        with fake_hub({"user/source": True}) as repo_info:
            assert (
                _resolve_private(self._cfg("--private", "false"), ["user/source"], ["user/output"]) is False
            )
        repo_info.assert_not_called()

    def test_explicit_true_checks_only_the_output(self):
        with fake_hub({"user/source": False}) as repo_info:
            assert _resolve_private(self._cfg("--private", "true"), ["user/source"], ["user/output"]) is True
        repo_info.assert_called_once_with("user/output", repo_type="dataset")

    @pytest.mark.parametrize("extra", [[], ["--private", "true"]])
    def test_existing_public_output_stops(self, extra):
        # The Hub keeps the visibility of an existing repo, so private data would land in a public repo.
        with (
            fake_hub({"user/source": True, "user/output": False}),
            pytest.raises(ValueError, match="user/output already exists on the Hub and is public"),
        ):
            _resolve_private(self._cfg(*extra), ["user/source"], ["user/output"])

    def test_existing_private_output_is_fine(self):
        with fake_hub({"user/source": True, "user/output": True}):
            assert _resolve_private(self._cfg(), ["user/source"], ["user/output"]) is True

    def test_in_place_edit_of_private_dataset(self):
        with fake_hub({"user/source": True}):
            assert _resolve_private(self._cfg(), ["user/source"], ["user/source"]) is True
