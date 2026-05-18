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

"""Tests for the SmolVLA2 runtime's interactive-VQA helpers.

Covers camera selection, VQA-answer parsing, and the bounding-box /
keypoint overlay drawing — the pure functions, no model load.
"""

import numpy as np
import pytest

from lerobot.policies.smolvla2.inference.vqa import (
    answer_has_overlay,
    available_cameras,
    camera_short_name,
    draw_vqa_overlay,
    observation_image_to_pil,
    parse_vqa_answer,
    prompt_camera_choice,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------------


def test_available_cameras_extracts_and_sorts_image_keys():
    observation = {
        "observation.images.wrist": object(),
        "observation.state": object(),
        "observation.images.top": object(),
        "task": "x",
    }
    assert available_cameras(observation) == [
        "observation.images.top",
        "observation.images.wrist",
    ]


def test_available_cameras_handles_none_and_empty():
    assert available_cameras(None) == []
    assert available_cameras({}) == []


def test_camera_short_name_strips_prefix():
    assert camera_short_name("observation.images.top") == "top"
    assert camera_short_name("top") == "top"


def test_prompt_camera_choice_single_camera_auto_selects():
    cams = ["observation.images.top"]
    # input_fn must never be called for a single-camera setup.
    chosen = prompt_camera_choice(cams, input_fn=_boom, print_fn=lambda *_: None)
    assert chosen == "observation.images.top"


def test_prompt_camera_choice_by_number():
    cams = ["observation.images.top", "observation.images.wrist"]
    chosen = prompt_camera_choice(cams, input_fn=lambda _: "2", print_fn=lambda *_: None)
    assert chosen == "observation.images.wrist"


def test_prompt_camera_choice_by_name():
    cams = ["observation.images.top", "observation.images.wrist"]
    chosen = prompt_camera_choice(cams, input_fn=lambda _: "top", print_fn=lambda *_: None)
    assert chosen == "observation.images.top"


def test_prompt_camera_choice_invalid_returns_none():
    cams = ["observation.images.top", "observation.images.wrist"]
    assert prompt_camera_choice(cams, input_fn=lambda _: "99", print_fn=lambda *_: None) is None


def _boom(*_args, **_kwargs):
    raise AssertionError("input_fn should not be called")


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def test_parse_bbox_answer():
    answer = '{"detections": [{"label": "cube", "bbox_format": "xyxy", "bbox": [10, 20, 50, 80]}]}'
    parsed = parse_vqa_answer(answer)
    assert parsed["kind"] == "bbox"
    assert answer_has_overlay(parsed)


def test_parse_keypoint_answer():
    answer = '{"label": "blue cube", "point_format": "xy", "point": [120, 90]}'
    parsed = parse_vqa_answer(answer)
    assert parsed["kind"] == "keypoint"
    assert answer_has_overlay(parsed)


def test_parse_count_answer_is_not_an_overlay():
    parsed = parse_vqa_answer('{"label": "cubes", "count": 2}')
    assert parsed["kind"] == "count"
    assert not answer_has_overlay(parsed)


def test_parse_invalid_json_returns_none():
    assert parse_vqa_answer("not json at all") is None
    assert parse_vqa_answer("") is None
    # A JSON array is valid JSON but not a VQA answer object.
    assert parse_vqa_answer("[1, 2, 3]") is None


def test_parse_unknown_shape():
    parsed = parse_vqa_answer('{"weird": "payload"}')
    assert parsed["kind"] == "unknown"
    assert not answer_has_overlay(parsed)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------


def _blank(size=(160, 120)):
    return Image.new("RGB", size, (0, 0, 0))


def test_draw_bbox_overlay_changes_pixels_and_preserves_size():
    img = _blank()
    parsed = parse_vqa_answer(
        '{"detections": [{"label": "cube", "bbox_format": "xyxy", "bbox": [10, 20, 50, 80]}]}'
    )
    out = draw_vqa_overlay(img, parsed)
    assert out.size == img.size
    assert out.tobytes() != img.tobytes()


def test_draw_keypoint_overlay_changes_pixels():
    img = _blank()
    parsed = parse_vqa_answer('{"label": "cube", "point_format": "xy", "point": [80, 60]}')
    out = draw_vqa_overlay(img, parsed)
    assert out.size == img.size
    assert out.tobytes() != img.tobytes()


def test_draw_overlay_non_spatial_leaves_image_unchanged():
    img = _blank()
    parsed = parse_vqa_answer('{"label": "cubes", "count": 2}')
    out = draw_vqa_overlay(img, parsed)
    assert out.tobytes() == img.tobytes()


def test_draw_overlay_tolerates_malformed_coordinates():
    img = _blank()
    # bbox with the wrong arity must not raise.
    out = draw_vqa_overlay(img, {"kind": "bbox", "payload": {"detections": [{"bbox": [1, 2]}]}})
    assert out.size == img.size


def test_observation_image_to_pil_from_batched_float_array():
    # (1, C, H, W) float array in [0, 1], the runtime observation shape.
    arr = np.zeros((1, 3, 24, 32), dtype=np.float32)
    pil = observation_image_to_pil(arr)
    assert pil.size == (32, 24)
    assert pil.mode == "RGB"
