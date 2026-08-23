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
    _reconcile_label_with_mount,
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

VOCAB = ("side", "top", "bottom", "wrist", "left", "right", "front", "rear")


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
    assert is_valid_view_label("top", VOCAB, allow_combos=True)  # position
    assert is_valid_view_label("side", VOCAB, allow_combos=True)  # position
    assert is_valid_view_label("left_wrist", VOCAB, allow_combos=True)  # qualifier + position
    assert is_valid_view_label("front_side", VOCAB, allow_combos=True)  # front is a suffix now
    assert is_valid_view_label("rear_side", VOCAB, allow_combos=True)
    assert not is_valid_view_label("left_wrist", VOCAB, allow_combos=False)
    assert not is_valid_view_label("front", VOCAB, allow_combos=True)  # bare qualifier
    assert not is_valid_view_label("left", VOCAB, allow_combos=True)  # bare qualifier
    assert not is_valid_view_label("front_rear", VOCAB, allow_combos=True)  # two qualifiers
    assert not is_valid_view_label("side_top", VOCAB, allow_combos=True)  # two positions
    assert not is_valid_view_label("banana", VOCAB, allow_combos=True)
    assert not is_valid_view_label("top_wrist_front", VOCAB, allow_combos=True)  # 3 tokens
    assert not is_valid_view_label("", VOCAB, allow_combos=True)


def test_reconcile_label_with_mount():
    # robot_mounted is authoritative -> always a wrist label.
    assert _reconcile_label_with_mount("top", "robot_mounted") == "wrist"
    assert _reconcile_label_with_mount(None, "robot_mounted") == "wrist"
    assert _reconcile_label_with_mount("left_wrist", "robot_mounted") == "left_wrist"  # keep handedness
    # fixed -> a wrist label contradicts the mount and is dropped; others kept.
    assert _reconcile_label_with_mount("wrist", "fixed") is None
    assert _reconcile_label_with_mount("right_wrist", "fixed") is None
    assert _reconcile_label_with_mount("top", "fixed") == "top"
    assert _reconcile_label_with_mount("front_side", "fixed") == "front_side"
    # unknown / no mount signal -> trust the label as-is.
    assert _reconcile_label_with_mount("top", "unknown") == "top"
    assert _reconcile_label_with_mount("wrist", None) == "wrist"
    assert _reconcile_label_with_mount(None, None) is None


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


def test_curate_cameras_canonicalizes_combo_order():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    frames = {"observation.images.a": [_tiny_image()], "observation.images.b": [_tiny_image()]}
    # VLM emits combos direction-last; we normalize to <direction>_<position>.
    vlm = _queued_vlm(
        [
            {"usable": True, "blur_reason": None, "view_label": "wrist_left"},
            {"usable": True, "blur_reason": None, "view_label": "side_right"},
        ]
    )
    verdicts = {v.camera_key: v for v in curate_cameras(frames, cfg, vlm)}
    assert verdicts["observation.images.a"].view_label == "left_wrist"
    assert verdicts["observation.images.b"].view_label == "right_side"


def test_curate_cameras_unknown_is_abstain():
    # "unknown" is an explicit abstain: the camera stays usable but unlabeled
    # (no rename), distinct from an invalid label that also drops to None.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    frames = {"observation.images.a": [_tiny_image()], "observation.images.b": [_tiny_image()]}
    vlm = _queued_vlm(
        [
            {"usable": True, "blur_reason": None, "view_label": "unknown", "confidence": 0.3},
            {"usable": True, "blur_reason": None, "view_label": "UNKNOWN", "confidence": 0.3},
        ]
    )
    verdicts = {v.camera_key: v for v in curate_cameras(frames, cfg, vlm)}
    assert verdicts["observation.images.a"].usable is True
    assert verdicts["observation.images.a"].view_label is None
    assert verdicts["observation.images.b"].view_label is None  # case-insensitive


def test_curate_cameras_mount_type_is_authoritative():
    # Mount type overrides the finer label deterministically (no VLM relabel):
    #  - robot_mounted + a fixed label  -> forced to plain "wrist"
    #  - robot_mounted + no label       -> "wrist" (mount alone is enough)
    #  - fixed + a wrist label          -> dropped (can't infer a fixed position)
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    frames = {
        "observation.images.a": [_tiny_image()],
        "observation.images.b": [_tiny_image()],
        "observation.images.c": [_tiny_image()],
    }
    vlm = _queued_vlm(
        [
            {"usable": True, "mount_type": "robot_mounted", "view_label": "top"},
            {"usable": True, "mount_type": "robot_mounted", "view_label": "unknown"},
            {"usable": True, "mount_type": "fixed", "view_label": "wrist"},
        ]
    )
    verdicts = {v.camera_key: v for v in curate_cameras(frames, cfg, vlm)}
    assert verdicts["observation.images.a"].view_label == "wrist"
    assert verdicts["observation.images.a"].mount_type == "robot_mounted"
    assert verdicts["observation.images.b"].view_label == "wrist"
    assert verdicts["observation.images.c"].view_label is None  # wrist on a fixed cam dropped


def test_curate_cameras_robot_mount_keeps_handedness():
    # A robot-mounted camera already labeled with handedness keeps it.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    frames = {"observation.images.a": [_tiny_image()]}
    vlm = _queued_vlm([{"usable": True, "mount_type": "robot_mounted", "view_label": "left_wrist"}])
    verdicts = {v.camera_key: v for v in curate_cameras(frames, cfg, vlm)}
    assert verdicts["observation.images.a"].view_label == "left_wrist"


def test_curate_cameras_joint_labeling_second_pass():
    # Pass 1 (per-camera) gives non-colliding but wrong labels; the joint second
    # pass re-decides both (mount type + label) by comparison. Quality stays from
    # pass 1.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, joint_labeling=True)
    frames = {"observation.images.a": [_tiny_image()], "observation.images.b": [_tiny_image()]}
    vlm = _queued_vlm(
        [
            {"usable": True, "mount_type": "fixed", "view_label": "side"},  # cam a (pass 1)
            {"usable": False, "blur_reason": "out of focus", "mount_type": "fixed", "view_label": "side"},
            {  # joint pass 2
                "cameras": [
                    {"mount_type": "fixed", "view_label": "top"},
                    {"mount_type": "fixed", "view_label": "left_side"},
                ]
            },
        ]
    )
    verdicts = {v.camera_key: v for v in curate_cameras(frames, cfg, vlm)}
    # labels overwritten by the joint pass
    assert verdicts["observation.images.a"].view_label == "top"
    assert verdicts["observation.images.b"].view_label == "left_side"
    # quality untouched by the joint (mount + label only) pass
    assert verdicts["observation.images.b"].usable is False
    assert verdicts["observation.images.b"].blur_reason == "out of focus"


def test_build_name_mapping_and_collision():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    existing = {"observation.images.cam_0": {}, "observation.images.cam_1": {}, "observation.images.top": {}}
    verdicts = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="left_wrist"),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="side"),
        # already canonical -> skipped by build_name_mapping
        CameraVerdict("observation.images.top", usable=True, view_label="top"),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {
        "observation.images.cam_0": "observation.images.left_wrist",
        "observation.images.cam_1": "observation.images.side",
    }
    assert skipped == {}
    # proposed_new_key stamped back onto the verdicts
    assert verdicts[0].proposed_new_key == "observation.images.left_wrist"


def test_build_name_mapping_partial_skip_keeps_most_confident():
    # Default policy "skip": rename the unambiguous camera; for a contested label,
    # keep the most confident contender and skip the rest.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    existing = {f"observation.images.cam_{i}": {} for i in range(3)}
    verdicts = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="side", confidence=0.9),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="side", confidence=0.5),
        CameraVerdict("observation.images.cam_2", usable=True, view_label="top", confidence=0.8),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    # cam_0 wins "side" (higher confidence), cam_2 is unique, cam_1 is skipped.
    assert mapping == {
        "observation.images.cam_0": "observation.images.side",
        "observation.images.cam_2": "observation.images.top",
    }
    assert set(skipped) == {"observation.images.cam_1"}


def test_build_name_mapping_skip_target_taken_by_existing_feature():
    # A label already used by an untouched feature can't be freed by confidence.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    existing = {"observation.images.cam_0": {}, "observation.images.side": {}}
    verdicts = [CameraVerdict("observation.images.cam_0", usable=True, view_label="side", confidence=1.0)]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {}
    assert set(skipped) == {"observation.images.cam_0"}


def test_build_name_mapping_error_mode_raises():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, on_collision="error")
    clash = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="top"),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="top"),
    ]
    with pytest.raises(ValueError, match="collision"):
        build_name_mapping(clash, {"observation.images.cam_0": {}, "observation.images.cam_1": {}}, cfg)


def test_build_name_mapping_ignore_key_names_forces_relabel():
    # With ignore_key_names, the key words are NOT used to break the collision;
    # both want "wrist" -> the higher-confidence camera wins, the other is skipped.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, ignore_key_names=True)
    existing = {"observation.images.cam_left": {}, "observation.images.cam_right": {}}
    verdicts = [
        CameraVerdict("observation.images.cam_left", usable=True, view_label="wrist", confidence=0.9),
        CameraVerdict("observation.images.cam_right", usable=True, view_label="wrist", confidence=0.5),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {"observation.images.cam_left": "observation.images.wrist"}
    assert set(skipped) == {"observation.images.cam_right"}


def test_build_name_mapping_disambiguates_two_wrists_from_source_names():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, allow_combos=True)
    existing = {"observation.images.cam_left": {}, "observation.images.cam_right": {}}
    verdicts = [
        CameraVerdict("observation.images.cam_left", usable=True, view_label="wrist"),
        CameraVerdict("observation.images.cam_right", usable=True, view_label="wrist"),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {
        "observation.images.cam_left": "observation.images.left_wrist",
        "observation.images.cam_right": "observation.images.right_wrist",
    }
    assert skipped == {}


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


def test_rename_camera_keys_on_hub_with_path_prefix(tmp_path):
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete

    camera_key = "observation.images.cam_0"
    new_key = "observation.images.left_wrist"
    prefix = "user/task1"
    _make_min_meta(tmp_path, camera_key, dtype="video")
    old_mp4 = f"{prefix}/videos/{camera_key}/chunk-000/file-000.mp4"

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [old_mp4, f"{prefix}/meta/info.json"]
    fake_api.create_commit.return_value = MagicMock(oid="cafe")

    with patch("huggingface_hub.HfApi", return_value=fake_api):
        rename_camera_keys_on_hub("user/collection", {camera_key: new_key}, tmp_path, path_prefix=prefix)

    ops = fake_api.create_commit.call_args.kwargs["operations"]
    copies = [o for o in ops if isinstance(o, CommitOperationCopy)]
    deletes = [o for o in ops if isinstance(o, CommitOperationDelete)]
    adds = [o for o in ops if isinstance(o, CommitOperationAdd)]
    new_mp4 = f"{prefix}/videos/{new_key}/chunk-000/file-000.mp4"
    assert any(o.src_path_in_repo == old_mp4 and o.path_in_repo == new_mp4 for o in copies)
    assert any(o.path_in_repo == old_mp4 for o in deletes)
    # meta files are committed under the sub-dataset prefix too
    assert any(o.path_in_repo == f"{prefix}/meta/info.json" for o in adds)


def test_rename_camera_keys_on_hub_rejects_image_keys(tmp_path):
    camera_key = "observation.images.cam_0"
    _make_min_meta(tmp_path, camera_key, dtype="image")
    with (
        patch("huggingface_hub.HfApi", return_value=MagicMock()),
        pytest.raises(NotImplementedError, match="image data"),
    ):
        rename_camera_keys_on_hub("user/ds", {camera_key: "observation.images.top"}, tmp_path)


def test_rename_camera_keys_on_hub_rejects_swaps(tmp_path):
    _make_min_meta(tmp_path, "observation.images.a", dtype="video")
    with (
        patch("huggingface_hub.HfApi", return_value=MagicMock()),
        pytest.raises(NotImplementedError, match="swap"),
    ):
        rename_camera_keys_on_hub(
            "user/ds",
            {
                "observation.images.a": "observation.images.b",
                "observation.images.b": "observation.images.a",
            },
            tmp_path,
        )
