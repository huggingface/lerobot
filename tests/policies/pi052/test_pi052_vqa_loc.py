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
bug). The dataset stays backbone-agnostic JSON; the conversion lives in
PI052's tokenizer. These tests pin the JSON → ``<loc>`` rewrite.
"""

import pytest

pytest.importorskip("transformers")

from lerobot.policies.pi052.text_processor_pi052 import (  # noqa: E402
    _camera_image_shapes,
    _loc_token,
    _messages_vqa_to_loc,
    _vqa_answer_to_loc,
)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def test_camera_image_shapes_extracts_hw_from_image_keys():
    obs = {
        "observation.images.top": _FakeTensor((1, 3, 240, 320)),
        "observation.images.wrist": _FakeTensor((3, 480, 640)),
        "observation.state": _FakeTensor((1, 7)),
        "task": "x",
    }
    assert _camera_image_shapes(obs) == {
        "observation.images.top": (240, 320),
        "observation.images.wrist": (480, 640),
    }


def test_camera_image_shapes_handles_empty():
    assert _camera_image_shapes({}) == {}
    assert _camera_image_shapes(None) == {}


def test_loc_token_normalizes_and_clamps():
    assert _loc_token(0, 100) == "<loc0000>"
    assert _loc_token(100, 100) == "<loc1023>"
    assert _loc_token(50, 100) == f"<loc{round(50 / 100 * 1023):04d}>"
    # out-of-range coordinates clamp into [0, 1023]
    assert _loc_token(999, 100) == "<loc1023>"
    assert _loc_token(-5, 100) == "<loc0000>"


def test_vqa_answer_to_loc_keypoint():
    answer = {"label": "blue cube", "point_format": "xy", "point": [160, 120]}
    # height=240, width=320 → y=120/240=0.5, x=160/320=0.5
    out = _vqa_answer_to_loc(answer, height=240, width=320)
    assert out == "<loc0512><loc0512> blue cube"


def test_vqa_answer_to_loc_bbox():
    answer = {
        "detections": [
            {"label": "cube", "bbox_format": "xyxy", "bbox": [0, 0, 320, 240]},
        ]
    }
    out = _vqa_answer_to_loc(answer, height=240, width=320)
    assert out == "<loc0000><loc0000><loc1023><loc1023> cube"


def test_vqa_answer_to_loc_returns_none_for_non_spatial():
    assert _vqa_answer_to_loc({"label": "cubes", "count": 2}, 240, 320) is None
    assert _vqa_answer_to_loc({"weird": "payload"}, 240, 320) is None


def test_messages_vqa_to_loc_rewrites_target_turn():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "feature": "observation.images.top"},
                {"type": "text", "text": "where is the cube?"},
            ],
        },
        {"role": "assistant", "content": '{"label": "cube", "point_format": "xy", "point": [160, 120]}'},
    ]
    shapes = {"observation.images.top": (240, 320)}
    out = _messages_vqa_to_loc(messages, target_indices=[1], image_shapes=shapes)
    assert out[1]["content"] == "<loc0512><loc0512> cube"
    # input messages are not mutated
    assert messages[1]["content"].startswith("{")


def test_messages_vqa_to_loc_leaves_plain_text_targets_untouched():
    messages = [
        {"role": "user", "content": [{"type": "image", "feature": "observation.images.top"}]},
        {"role": "assistant", "content": "pick up the cube"},
    ]
    shapes = {"observation.images.top": (240, 320)}
    out = _messages_vqa_to_loc(messages, target_indices=[1], image_shapes=shapes)
    assert out[1]["content"] == "pick up the cube"


def test_messages_vqa_to_loc_noop_without_shapes():
    messages = [{"role": "assistant", "content": '{"label": "c", "point_format": "xy", "point": [1, 2]}'}]
    assert _messages_vqa_to_loc(messages, [0], None) is messages
    assert _messages_vqa_to_loc(messages, [0], {}) is messages
