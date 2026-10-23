# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from lerobot.annotations.hands.egohand.association import (
    Policy,
    annotations_from_candidates,
    assign_physical,
    proposals,
)


def c(x, y=100, w=80, h=80, score=0.9, label=1):
    return [x, y, w, h, score, label]


def records(frames):
    return [{"L": [v for v in row if v[5] == 1], "R": [v for v in row if v[5] == 2]} for row in frames]


def test_clean_two_hands_and_reversed_classes():
    frames = [[c(100 + t, label=2), c(400 + t, label=1)] for t in range(20)]
    tracks, _ = assign_physical(records(frames))
    assert tracks["left"][1].all() and tracks["right"][1].all()
    np.testing.assert_allclose(tracks["left"][0][:, 0], np.arange(100, 120))
    np.testing.assert_allclose(tracks["right"][0][:, 0], np.arange(400, 420))


@pytest.mark.parametrize("weak_score", [0.2, 0.65])
def test_duplicate_cannot_steal_departed_hand_or_flip_live_identity(weak_score):
    frames = [[c(100), c(400, label=2)] for _ in range(4)]
    frames += [
        [c(400, score=0.85, label=2), c(400.5, w=100, h=150, score=weak_score, label=1)] for _ in range(8)
    ]
    tracks, _ = assign_physical(records(frames))
    assert not tracks["left"][1][4:].any()
    assert tracks["right"][1][4:].all()


def test_backward_anchor_corrects_wrong_single_hand_bootstrap():
    frames = [[c(100, label=2)] for _ in range(10)]
    frames += [[c(100, label=2), c(400, label=1)] for _ in range(6)]
    tracks, _ = assign_physical(records(frames))
    assert tracks["left"][1].all()
    assert not tracks["right"][1][:10].any()


def test_single_hand_clip_remains_anatomically_unknown():
    tracks, stats = assign_physical(records([[c(100)]] * 10))
    assert stats["unknown_identity_frames"] == 10
    assert not any(v[1].any() for v in tracks.values())


def test_empty_frames_never_become_direct_observations():
    frames = [[c(100), c(400, label=2)]] * 4 + [[]] * 4 + [[c(100), c(400, label=2)]] * 4
    tracks, _ = assign_physical(records(frames))
    assert not tracks["left"][1][4:8].any()
    assert not tracks["right"][1][4:8].any()


def test_anatomical_identity_not_resorted_when_tracks_cross_screen_x():
    frames = [[c(100 + 10 * t, y=100), c(400 - 10 * t, y=300, label=2)] for t in range(40)]
    tracks, _ = assign_physical(records(frames))
    assert tracks["left"][1].all() and tracks["right"][1].all()
    assert tracks["left"][0][-1, 0] > tracks["right"][0][-1, 0]


def test_class_duplicate_is_one_proposal():
    r = records([[c(100), c(100, score=0.8, label=2)]])[0]
    assert len(proposals(r, Policy())) == 1


def test_no_input_mutation_and_repeatable():
    r = records([[c(100), c(400, label=2)]] * 10)
    before = deepcopy(r)
    a, _ = assign_physical(r)
    b, _ = assign_physical(r)
    assert r == before
    for side in a:
        np.testing.assert_array_equal(a[side][0], b[side][0])
        np.testing.assert_array_equal(a[side][1], b[side][1])


def test_distant_lone_detection_cannot_steal_expired_tracks():
    frames = [[c(100), c(400, label=2)]] * 4 + [[]] * 20 + [[c(1000)]] * 4
    tracks, _ = assign_physical(records(frames))
    assert not tracks["left"][1][-4:].any()
    assert not tracks["right"][1][-4:].any()


def test_one_frame_bimanual_false_positive_cannot_anchor():
    frames = [[c(100)]] * 5 + [[c(100), c(400, label=2)]] + [[c(100)]] * 5
    tracks, stats = assign_physical(records(frames))
    assert stats["anchor_frame"] == -1
    assert not tracks["right"][1].any()


def test_invalid_fps_rejected():
    with pytest.raises(ValueError):
        assign_physical([], fps=0)


@pytest.mark.parametrize("fail_tracking", [False, True])
def test_crop_centers_follow_detections_without_changing_identity_or_native_defaults(
    monkeypatch, fail_tracking
):
    """Exercise the adapter boundary; the frozen GPU runtime covers real stabilization."""
    tracking = ModuleType("gem.common.detectors.hand_tracking")
    original_assign = object()
    tracking._assign = original_assign
    tracking.MAX_CENTER_SPEED = 90.0
    centers = np.array([[100, 100, 614.4], [500, 100, 614.4]], dtype=np.float32)
    observed = np.array([True, False])
    tracks = SimpleNamespace(
        boxes_xys={side: centers.copy() for side in ("left", "right")},
        raw_present={side: observed.copy() for side in ("left", "right")},
        emitted={side: np.ones(2, dtype=bool) for side in ("left", "right")},
        stats={},
    )
    calls = []

    def track(detections, width, *, enforce_lr_order):
        assert np.isinf(tracking.MAX_CENTER_SPEED)
        assert tracking._assign.func is assign_physical
        assert tracking._assign.keywords == {"fps": 10}
        assert not enforce_lr_order
        assert width == 3840
        assert len(detections) == 2
        calls.append("track")
        if fail_tracking:
            raise RuntimeError("native tracking failed")
        return tracks

    def smooth(value, timestamps):
        assert value is tracks
        np.testing.assert_array_equal(timestamps, [0, 0.1])
        calls.append("smooth")
        return value

    tracking.track_hand_detections = track
    modules = {
        "gem": ModuleType("gem"),
        "gem.common": ModuleType("gem.common"),
        "gem.common.detectors": ModuleType("gem.common.detectors"),
        "gem.common.detectors.hand": ModuleType("gem.common.detectors.hand"),
        "gem.common.detectors.hand_boxes": ModuleType("gem.common.detectors.hand_boxes"),
        tracking.__name__: tracking,
    }
    modules["gem.common.detectors.hand"].HandDetection = lambda **kwargs: SimpleNamespace(**kwargs)
    modules["gem.common.detectors.hand_boxes"].smooth_hand_boxes = smooth
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    if fail_tracking:
        with pytest.raises(RuntimeError, match="native tracking failed"):
            annotations_from_candidates([[], []], np.eye(3), 3840, 10)
        assert calls == ["track"]
    else:
        annotations, diagnostic = annotations_from_candidates([[], []], np.eye(3), 3840, 10)
        assert calls == ["track", "smooth"]
        assert diagnostic["crop_center_motion_cap_px_per_frame"] is None
        assert [row["side"] for row in annotations[1]["instances"]] == ["left", "right"]
        assert not any(row["bbox_observed"] for row in annotations[1]["instances"])
        np.testing.assert_array_equal(annotations[1]["instances"][0]["bbox"], centers[1])
    assert tracking.MAX_CENTER_SPEED == 90.0
    assert tracking._assign is original_assign
