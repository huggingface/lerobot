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
"""Tests for the camera-view curation pipeline (stubbed VLM, mocked Hub)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import pandas as pd  # noqa: E402

from lerobot.annotations.camera_curation.config import CameraCurationConfig  # noqa: E402
from lerobot.annotations.camera_curation.curator import (  # noqa: E402
    CameraVerdict,
    build_name_mapping,
    curate_cameras,
    is_valid_view_label,
    rename_camera_keys_on_hub,
    write_report,
)
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient  # noqa: E402
from lerobot.datasets.io_utils import load_info, write_info  # noqa: E402
from lerobot.datasets.utils import DatasetInfo  # noqa: E402
from lerobot.utils.io_utils import load_json, write_json  # noqa: E402

VOCAB = ("top", "wrist", "front", "bottom", "left", "right")


def _queued_vlm(responses: list) -> StubVlmClient:
    """Stub VLM that returns queued responses in batch order."""
    state = {"i": 0}

    def responder(_messages):
        r = responses[state["i"]]
        state["i"] += 1
        return r

    return StubVlmClient(responder=responder)


def _tiny_image() -> PIL.Image.Image:
    return PIL.Image.new("RGB", (16, 12))


def _make_min_meta(root: Path, camera_key: str, dtype: str = "video") -> None:
    """Write a minimal ``meta/`` tree with one camera + one action feature."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    features = {
        camera_key: {
            "dtype": dtype,
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
            "info": {"video.fps": 10.0} if dtype == "video" else None,
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    write_info(DatasetInfo(codebase_version="v3.0", fps=10, features=features), root)
    write_json({camera_key: {"mean": [0.0]}, "action": {"mean": [0.0]}}, root / "meta" / "stats.json")
    df = pd.DataFrame(
        {
            "episode_index": [0],
            f"videos/{camera_key}/from_timestamp": [0.0],
            f"videos/{camera_key}/to_timestamp": [1.0],
            f"videos/{camera_key}/chunk_index": [0],
            f"videos/{camera_key}/file_index": [0],
            f"stats/{camera_key}/mean": [[0.0]],
        }
    )
    df.to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


# ------------------------------ pure logic ------------------------------


def test_is_valid_view_label():
    assert is_valid_view_label("top", VOCAB, allow_combos=True)
    assert is_valid_view_label("left_wrist", VOCAB, allow_combos=True)
    assert not is_valid_view_label("left_wrist", VOCAB, allow_combos=False)
    assert not is_valid_view_label("banana", VOCAB, allow_combos=True)
    assert not is_valid_view_label("left_left", VOCAB, allow_combos=True)  # duplicate token
    assert not is_valid_view_label("top_wrist_front", VOCAB, allow_combos=True)  # 3 tokens
    assert not is_valid_view_label("", VOCAB, allow_combos=True)


def test_curate_cameras_parses_and_validates(tmp_path):
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    frames = {
        "observation.images.a": [_tiny_image()],
        "observation.images.b": [_tiny_image()],
        "observation.images.c": [],  # no frames -> reported, not sent to the VLM
    }
    vlm = _queued_vlm(
        [
            {"usable": True, "blur_reason": None, "view_label": "Left Wrist", "confidence": 0.9},
            {"usable": False, "blur_reason": "out of focus", "view_label": "banana", "confidence": 0.2},
        ]
    )
    verdicts = {v.camera_key: v for v in curate_cameras(frames, cfg, vlm)}

    assert verdicts["observation.images.a"].view_label == "left_wrist"  # normalized
    assert verdicts["observation.images.a"].usable is True
    assert verdicts["observation.images.b"].usable is False
    assert verdicts["observation.images.b"].blur_reason == "out of focus"
    assert verdicts["observation.images.b"].view_label is None  # invalid label dropped
    assert verdicts["observation.images.c"].view_label is None  # no frames


def test_build_name_mapping_and_collision():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    existing = {"observation.images.cam_0": {}, "observation.images.cam_1": {}, "observation.images.top": {}}
    verdicts = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="left_wrist"),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="front"),
        # already canonical -> skipped by build_name_mapping
        CameraVerdict("observation.images.top", usable=True, view_label="top"),
    ]
    mapping = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {
        "observation.images.cam_0": "observation.images.left_wrist",
        "observation.images.cam_1": "observation.images.front",
    }
    # proposed_new_key stamped back onto the verdicts
    assert verdicts[0].proposed_new_key == "observation.images.left_wrist"

    # two cameras wanting the same label collide under the default policy
    clash = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="top"),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="top"),
    ]
    with pytest.raises(ValueError, match="collision"):
        build_name_mapping(clash, {"observation.images.cam_0": {}, "observation.images.cam_1": {}}, cfg)


def test_build_name_mapping_disambiguates_two_wrists_from_source_names():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, allow_combos=True)
    existing = {"observation.images.cam_left": {}, "observation.images.cam_right": {}}
    verdicts = [
        CameraVerdict("observation.images.cam_left", usable=True, view_label="wrist"),
        CameraVerdict("observation.images.cam_right", usable=True, view_label="wrist"),
    ]
    mapping = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {
        "observation.images.cam_left": "observation.images.left_wrist",
        "observation.images.cam_right": "observation.images.right_wrist",
    }


def test_build_name_mapping_conflict_without_hint_still_errors():
    # Source keys carry no directional vocab word -> cannot disambiguate -> error.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, allow_combos=True)
    existing = {"observation.images.cam_0": {}, "observation.images.cam_1": {}}
    verdicts = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="wrist"),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="wrist"),
    ]
    with pytest.raises(ValueError, match="collision"):
        build_name_mapping(verdicts, existing, cfg)


def test_write_report(tmp_path):
    _make_min_meta(tmp_path, "observation.images.cam_0", dtype="video")
    cfg = CameraCurationConfig(repo_id="user/ds", view_vocabulary=VOCAB)
    verdicts = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="left_wrist", confidence=0.9)
    ]
    mapping = {"observation.images.cam_0": "observation.images.left_wrist"}

    report_path = write_report(tmp_path, verdicts, mapping, cfg)

    report = load_json(report_path)
    cam = report["cameras"]["observation.images.cam_0"]
    assert cam["view_label"] == "left_wrist"
    assert cam["proposed_new_key"] == "observation.images.left_wrist"
    # verdict stamped into info.json so it travels with the dataset
    info = load_info(tmp_path)
    assert info.features["observation.images.cam_0"]["info"]["curation"]["view_label"] == "left_wrist"


# ------------------------- lightweight Hub rename -------------------------


def test_rename_camera_keys_on_hub_builds_ops(tmp_path):
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete

    camera_key = "observation.images.cam_0"
    new_key = "observation.images.left_wrist"
    _make_min_meta(tmp_path, camera_key, dtype="video")
    old_mp4 = f"videos/{camera_key}/chunk-000/file-000.mp4"

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [old_mp4, "meta/info.json", "data/chunk-000/file-000.parquet"]
    fake_api.create_commit.return_value = MagicMock(oid="deadbeef")

    with patch("huggingface_hub.HfApi", return_value=fake_api):
        rename_camera_keys_on_hub("user/ds", {camera_key: new_key}, tmp_path, branch="curated")

    kwargs = fake_api.create_commit.call_args.kwargs
    ops = kwargs["operations"]
    copies = [o for o in ops if isinstance(o, CommitOperationCopy)]
    deletes = [o for o in ops if isinstance(o, CommitOperationDelete)]
    adds = [o for o in ops if isinstance(o, CommitOperationAdd)]

    new_mp4 = f"videos/{new_key}/chunk-000/file-000.mp4"
    assert any(o.src_path_in_repo == old_mp4 and o.path_in_repo == new_mp4 for o in copies)
    assert any(o.path_in_repo == old_mp4 for o in deletes)
    assert any(o.path_in_repo == "meta/info.json" for o in adds)
    assert kwargs["revision"] == "curated"

    # meta on disk was actually remapped
    info = load_info(tmp_path)
    assert new_key in info.features and camera_key not in info.features


def test_rename_camera_keys_on_hub_rejects_image_keys(tmp_path):
    camera_key = "observation.images.cam_0"
    _make_min_meta(tmp_path, camera_key, dtype="image")
    with patch("huggingface_hub.HfApi", return_value=MagicMock()):
        with pytest.raises(NotImplementedError, match="image data"):
            rename_camera_keys_on_hub("user/ds", {camera_key: "observation.images.top"}, tmp_path)


def test_rename_camera_keys_on_hub_rejects_swaps(tmp_path):
    _make_min_meta(tmp_path, "observation.images.a", dtype="video")
    with patch("huggingface_hub.HfApi", return_value=MagicMock()):
        with pytest.raises(NotImplementedError, match="swap"):
            rename_camera_keys_on_hub(
                "user/ds",
                {
                    "observation.images.a": "observation.images.b",
                    "observation.images.b": "observation.images.a",
                },
                tmp_path,
            )
