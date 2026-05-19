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

"""Training-side conversion of VQA answers to PaliGemma ``<loc>`` text.

PI052 trains spatial VQA answers (``bbox`` / ``keypoint``) in
PaliGemma's native ``<locNNNN>`` detection vocabulary so the LM head
reuses the detection prior instead of fighting it (the ``<loc>``-salad
bug). The dataset stores Qwen2.5-VL's grounding output — **0–1000
normalized** coordinates, *not* pixels. (Verified empirically on the
published datasets: x and y both span 0..1000 with ~30% of values
exceeding the camera's pixel dimensions.) The conversion is therefore
camera-resolution-independent. The dataset stays backbone-agnostic
JSON; the conversion lives in PI052's tokenizer. These tests pin the
JSON → ``<loc>`` rewrite.
"""

import pytest

pytest.importorskip("transformers")

from lerobot.policies.pi052.text_processor_pi052 import (  # noqa: E402
    _loc_token,
    _messages_vqa_to_loc,
    _vqa_answer_to_loc,
)


def test_loc_token_normalizes_and_clamps():
    # Default scale is the 0–1000 Qwen convention.
    assert _loc_token(0) == "<loc0000>"
    assert _loc_token(1000) == "<loc1023>"
    assert _loc_token(500) == f"<loc{round(500 / 1000 * 1023):04d}>"
    # out-of-range coordinates clamp into [0, 1023]
    assert _loc_token(9999) == "<loc1023>"
    assert _loc_token(-5) == "<loc0000>"


def test_vqa_answer_to_loc_keypoint_normalized():
    # Qwen 0–1000 normalized coordinates → camera-independent <loc>.
    answer = {"label": "blue cube", "point_format": "xy", "point": [500, 500]}
    assert _vqa_answer_to_loc(answer) == "<loc0512><loc0512> blue cube"


def test_vqa_answer_to_loc_bbox_normalized():
    answer = {
        "detections": [{"label": "cube", "bbox_format": "xyxy", "bbox": [0, 0, 1000, 1000]}]
    }
    assert _vqa_answer_to_loc(answer) == "<loc0000><loc0000><loc1023><loc1023> cube"


def test_vqa_answer_to_loc_returns_none_for_non_spatial():
    assert _vqa_answer_to_loc({"label": "cubes", "count": 2}) is None
    assert _vqa_answer_to_loc({"weird": "payload"}) is None


def test_messages_vqa_to_loc_rewrites_target_turn():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "where is the cube?"}]},
        {
            "role": "assistant",
            "content": '{"label": "cube", "point_format": "xy", "point": [500, 500]}',
        },
    ]
    out = _messages_vqa_to_loc(messages, target_indices=[1])
    assert out[1]["content"] == "<loc0512><loc0512> cube"
    # input messages are not mutated
    assert messages[1]["content"].startswith("{")


def test_messages_vqa_to_loc_leaves_plain_text_targets_untouched():
    messages = [
        {"role": "user", "content": "pick the cube"},
        {"role": "assistant", "content": "pick up the cube"},
    ]
    out = _messages_vqa_to_loc(messages, target_indices=[1])
    assert out[1]["content"] == "pick up the cube"


def test_messages_vqa_to_loc_noop_without_target_indices():
    messages = [
        {"role": "assistant", "content": '{"label": "c", "point_format": "xy", "point": [1, 2]}'}
    ]
    assert _messages_vqa_to_loc(messages, []) is messages


# ---------------------------------------------------------------------------
# Round-trip: training-side JSON -> <loc> -> runtime-side parse back
#
# Pins that the conversion preserves coordinate *order* (JSON is x-first,
# PaliGemma <loc> is y-first) and the 0–1000 → [0, 1023] scaling. The
# only loss is quantization to the 1024-bucket <loc> grid, so a coord
# survives within half a bucket (~1000/2046 ≈ 0.49 on the 0–1000 scale).
# ---------------------------------------------------------------------------


def test_loc_round_trip_keypoint_preserves_normalized_coords():
    from lerobot.policies.smolvla2.inference.vqa import parse_vqa_answer

    answer = {"label": "blue cube", "point_format": "xy", "point": [640, 480]}
    loc = _vqa_answer_to_loc(answer)
    parsed = parse_vqa_answer(loc)
    nx, ny = parsed["payload"]["point"]
    # parse_vqa_answer returns [0, 1] normalized; rescale back to 0–1000.
    assert abs(nx * 1000.0 - 640) <= 1000.0 / 2046 + 1e-6
    assert abs(ny * 1000.0 - 480) <= 1000.0 / 2046 + 1e-6
    assert parsed["payload"]["label"] == "blue cube"


def test_loc_round_trip_bbox_preserves_order_and_scale():
    from lerobot.policies.smolvla2.inference.vqa import parse_vqa_answer

    answer = {
        "detections": [{"label": "cube", "bbox_format": "xyxy", "bbox": [100, 200, 800, 900]}]
    }
    loc = _vqa_answer_to_loc(answer)
    parsed = parse_vqa_answer(loc)
    x1, y1, x2, y2 = parsed["payload"]["detections"][0]["bbox"]
    for got, want in ((x1, 100), (y1, 200), (x2, 800), (y2, 900)):
        assert abs(got * 1000.0 - want) <= 1000.0 / 2046 + 1e-6
