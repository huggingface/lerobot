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


def _capturing_vlm(responses: list) -> tuple[StubVlmClient, list]:
    """Stub VLM that records every message batch it is asked to complete.

    Returns ``(client, seen)`` where ``seen`` accumulates each ``_messages``
    passed to the responder, so a test can assert what content reached the VLM.
    """
    state = {"i": 0}
    seen: list = []

    def responder(_messages):
        seen.append(_messages)
        r = responses[state["i"]]
        state["i"] += 1
        return r

    return StubVlmClient(responder=responder), seen


def _all_text_sent(seen: list) -> str:
    """Concatenate every text block across all captured message batches."""
    texts: list[str] = []
    for messages in seen:
        for message in messages:
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
    return "\n".join(texts)


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


def _make_min_meta_multi(root: Path, camera_keys: list[str], dtype: str = "video") -> None:
    """Like ``_make_min_meta`` but with several cameras (for chain/swap renames)."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    features: dict = {
        k: {
            "dtype": dtype,
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
            "info": {"video.fps": 10.0} if dtype == "video" else None,
        }
        for k in camera_keys
    }
    features["action"] = {"dtype": "float32", "shape": (2,), "names": None}
    write_info(DatasetInfo(codebase_version="v3.0", fps=10, features=features), root)
    stats = {k: {"mean": [0.0]} for k in camera_keys}
    stats["action"] = {"mean": [0.0]}
    write_json(stats, root / "meta" / "stats.json")
    cols: dict = {"episode_index": [0]}
    for k in camera_keys:
        cols[f"videos/{k}/from_timestamp"] = [0.0]
        cols[f"videos/{k}/chunk_index"] = [0]
        cols[f"videos/{k}/file_index"] = [0]
        cols[f"stats/{k}/mean"] = [[0.0]]
    pd.DataFrame(cols).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


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


def test_vlm_never_sees_camera_key():
    # The VLM must classify purely from pixels + neutral "Camera i" numbering; the
    # existing dataset key name (often unreliable/misleading) must never reach it,
    # in either the per-camera or the joint pass.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, joint_labeling=True)
    frames = {
        "observation.images.ZZSECRETLEFT": [_tiny_image()],
        "observation.images.ZZSECRETRIGHT": [_tiny_image()],
    }
    vlm, seen = _capturing_vlm(
        [
            {"usable": True, "mount_type": "fixed", "view_label": "top"},  # per-camera cam 1
            {"usable": True, "mount_type": "robot_mounted", "view_label": "wrist"},  # per-camera cam 2
            {  # joint pass
                "cameras": [
                    {"mount_type": "fixed", "view_label": "top"},
                    {"mount_type": "robot_mounted", "view_label": "wrist"},
                ]
            },
        ]
    )
    curate_cameras(frames, cfg, vlm)

    sent = _all_text_sent(seen)
    assert "ZZSECRET" not in sent  # no fragment of either camera key
    assert "observation.images" not in sent  # nor the canonical prefix
    # sanity: the neutral numbering and preliminary evidence DID reach the VLM
    assert "Camera 1:" in sent and "Camera 2:" in sent
    assert "preliminary" in sent


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


def test_curate_cameras_parses_candidates():
    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    frames = {"observation.images.a": [_tiny_image()]}
    vlm = _queued_vlm(
        [
            {
                "usable": True,
                "mount_type": "fixed",
                "view_label": "top",
                "confidence": 0.6,
                "candidates": [
                    {"view_label": "top", "confidence": 0.6},
                    {"view_label": "left_side", "confidence": 0.35},
                    {"view_label": "banana", "confidence": 0.1},  # invalid -> dropped
                ],
            }
        ]
    )
    v = {x.camera_key: x for x in curate_cameras(frames, cfg, vlm)}["observation.images.a"]
    assert v.candidates == [("top", 0.6), ("left_side", 0.35)]  # sorted best-first, invalid dropped


def test_candidate_fallback_resolves_collision():
    # cam_a/cam_b both -> "top"; cam_b (lower conf) has a strong "left_side" #2, so it
    # falls back instead of being skipped. Both get renamed, no collision.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, candidate_fallback=True, ignore_key_names=True)
    existing = {"observation.images.cam_a": {}, "observation.images.cam_b": {}}
    verdicts = [
        CameraVerdict(
            "observation.images.cam_a",
            usable=True,
            view_label="top",
            confidence=0.9,
            candidates=[("top", 0.9)],
        ),
        CameraVerdict(
            "observation.images.cam_b",
            usable=True,
            view_label="top",
            confidence=0.5,
            candidates=[("top", 0.5), ("left_side", 0.45)],
        ),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {
        "observation.images.cam_a": "observation.images.top",
        "observation.images.cam_b": "observation.images.left_side",
    }
    assert skipped == {}


def test_candidate_fallback_below_threshold_still_skips():
    # cam_b's alternative is below the confidence threshold -> no fallback -> normal
    # skip resolution (most confident keeps "top", cam_b skipped).
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, candidate_fallback=True, ignore_key_names=True)
    existing = {"observation.images.cam_a": {}, "observation.images.cam_b": {}}
    verdicts = [
        CameraVerdict(
            "observation.images.cam_a",
            usable=True,
            view_label="top",
            confidence=0.9,
            candidates=[("top", 0.9)],
        ),
        CameraVerdict(
            "observation.images.cam_b",
            usable=True,
            view_label="top",
            confidence=0.5,
            candidates=[("top", 0.5), ("left_side", 0.30)],
        ),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {"observation.images.cam_a": "observation.images.top"}
    assert set(skipped) == {"observation.images.cam_b"}


def test_candidate_fallback_off_by_default():
    # Without the flag, the strong #2 is ignored and cam_b is skipped on collision.
    cfg = CameraCurationConfig(
        view_vocabulary=VOCAB, ignore_key_names=True
    )  # candidate_fallback defaults False
    existing = {"observation.images.cam_a": {}, "observation.images.cam_b": {}}
    verdicts = [
        CameraVerdict(
            "observation.images.cam_a",
            usable=True,
            view_label="top",
            confidence=0.9,
            candidates=[("top", 0.9)],
        ),
        CameraVerdict(
            "observation.images.cam_b",
            usable=True,
            view_label="top",
            confidence=0.5,
            candidates=[("top", 0.5), ("left_side", 0.9)],
        ),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert mapping == {"observation.images.cam_a": "observation.images.top"}
    assert set(skipped) == {"observation.images.cam_b"}


def test_build_name_mapping_suffix_numbers_from_one():
    # on_collision="suffix" renames ALL cameras; a contended label is numbered
    # from _1 (top_1, top_2), a unique label stays bare.
    cfg = CameraCurationConfig(view_vocabulary=VOCAB, on_collision="suffix", ignore_key_names=True)
    existing = {
        "observation.images.cam_0": {},
        "observation.images.cam_1": {},
        "observation.images.cam_2": {},
    }
    verdicts = [
        CameraVerdict("observation.images.cam_0", usable=True, view_label="top", confidence=0.9),
        CameraVerdict("observation.images.cam_1", usable=True, view_label="top", confidence=0.8),
        CameraVerdict("observation.images.cam_2", usable=True, view_label="wrist", confidence=0.7),
    ]
    mapping, skipped = build_name_mapping(verdicts, existing, cfg)
    assert skipped == {}  # nothing skipped — every camera renamed
    assert mapping == {
        "observation.images.cam_0": "observation.images.top_1",
        "observation.images.cam_1": "observation.images.top_2",
        "observation.images.cam_2": "observation.images.wrist",
    }


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


def test_build_messages_injects_task_context():
    from lerobot.annotations.camera_curation.curator import _build_messages, _task_context

    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    marker = "the task performed in this dataset is"
    with_task = _build_messages([], cfg, task="shake hands with the person")[0]["content"][-1]["text"]
    assert marker in with_task and "shake hands with the person" in with_task
    # No/blank task -> no injected TASK CONTEXT block.
    assert marker not in _build_messages([], cfg, task=None)[0]["content"][-1]["text"]
    assert _task_context("") == "" and _task_context(None) == ""


def test_build_report_flags_unusable_and_collision():
    from lerobot.annotations.camera_curation.curator import build_report

    cfg = CameraCurationConfig(view_vocabulary=VOCAB)
    verdicts = [
        CameraVerdict("observation.images.a", usable=True, view_label="top"),
        CameraVerdict("observation.images.b", usable=True, view_label="top"),
        CameraVerdict("observation.images.c", usable=False, view_label=None, blur_reason="placeholder"),
    ]
    mapping = {  # a/b collided on "top" and were suffixed
        "observation.images.a": "observation.images.top_1",
        "observation.images.b": "observation.images.top_2",
    }
    report = build_report(verdicts, mapping, cfg)
    assert report["has_unusable"] is True
    assert report["unusable_cameras"] == ["observation.images.c"]
    # unusable views carry the reason
    assert report["unusable_views"] == {"observation.images.c": "placeholder"}
    assert report["has_name_collision"] is True
    assert report["suffixed_cameras"] == ["observation.images.a", "observation.images.b"]
    # conflicting views show which cameras share the label
    assert report["conflicting_views"] == {"top": ["observation.images.a", "observation.images.b"]}
    # the report shows the actual {old: new} renames
    assert report["renames"] == mapping

    # A clean dataset (all usable, no conflict) reports both flags False and empty detail.
    clean = build_report(
        [CameraVerdict("observation.images.a", usable=True, view_label="top")],
        {"observation.images.a": "observation.images.top"},
        cfg,
    )
    assert clean["has_unusable"] is False and clean["has_name_collision"] is False
    assert clean["unusable_views"] == {} and clean["conflicting_views"] == {}


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


def test_rename_camera_keys_on_hub_chain_uses_two_commits(tmp_path):
    # image -> top, top -> side: "top" is both a source and a target (a chain), so
    # it can't be one commit. It goes in two: stash to temp, then place finals.
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete

    img, top, side = (
        "observation.images.image",
        "observation.images.top",
        "observation.images.side",
    )
    _make_min_meta_multi(tmp_path, [img, top], dtype="video")
    img_mp4 = f"videos/{img}/chunk-000/file-000.mp4"
    top_mp4 = f"videos/{top}/chunk-000/file-000.mp4"

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [img_mp4, top_mp4, "meta/info.json"]
    fake_api.create_commit.return_value = MagicMock(oid="beef")

    with patch("huggingface_hub.HfApi", return_value=fake_api):
        rename_camera_keys_on_hub("user/ds", {img: top, top: side}, tmp_path)

    assert fake_api.create_commit.call_count == 2
    first, second = (c.kwargs["operations"] for c in fake_api.create_commit.call_args_list)

    tmp_seg = "videos/__lerobot_curate_tmp_"
    # Commit 1: every original copied to a temp segment, and every original deleted;
    # no meta yet, and nothing lands on a real final key.
    c1_copies = [o for o in first if isinstance(o, CommitOperationCopy)]
    assert {o.src_path_in_repo for o in c1_copies} == {img_mp4, top_mp4}
    assert all(tmp_seg in o.path_in_repo for o in c1_copies)
    assert {o.path_in_repo for o in first if isinstance(o, CommitOperationDelete)} == {img_mp4, top_mp4}
    assert not any(isinstance(o, CommitOperationAdd) for o in first)

    # Commit 2: temps copied to the final keys, temps deleted, meta added.
    c2_copies = [o for o in second if isinstance(o, CommitOperationCopy)]
    assert all(tmp_seg in o.src_path_in_repo for o in c2_copies)
    finals = {o.path_in_repo for o in c2_copies}
    assert finals == {
        f"videos/{top}/chunk-000/file-000.mp4",
        f"videos/{side}/chunk-000/file-000.mp4",
    }
    assert all(tmp_seg in o.path_in_repo for o in second if isinstance(o, CommitOperationDelete))
    assert any(o.path_in_repo == "meta/info.json" for o in second if isinstance(o, CommitOperationAdd))

    # meta on disk reflects the final keys
    info = load_info(tmp_path)
    assert top in info.features and side in info.features and img not in info.features


def test_rename_camera_keys_on_hub_swap_uses_two_commits(tmp_path):
    # A true 2-cycle a<->b also goes through the temp indirection in two commits.
    from huggingface_hub import CommitOperationCopy

    a, b = "observation.images.a", "observation.images.b"
    _make_min_meta_multi(tmp_path, [a, b], dtype="video")
    a_mp4 = f"videos/{a}/chunk-000/file-000.mp4"
    b_mp4 = f"videos/{b}/chunk-000/file-000.mp4"

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [a_mp4, b_mp4, "meta/info.json"]
    fake_api.create_commit.return_value = MagicMock(oid="cafe")

    with patch("huggingface_hub.HfApi", return_value=fake_api):
        rename_camera_keys_on_hub("user/ds", {a: b, b: a}, tmp_path)

    assert fake_api.create_commit.call_count == 2
    _, second = (c.kwargs["operations"] for c in fake_api.create_commit.call_args_list)
    finals = {o.path_in_repo for o in second if isinstance(o, CommitOperationCopy)}
    assert finals == {a_mp4, b_mp4}  # a and b swapped into each other's paths
    info = load_info(tmp_path)
    assert a in info.features and b in info.features
