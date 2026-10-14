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
"""Helpers shared across annotation-pipeline tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import PIL.Image

from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient


def make_canned_responder(
    responses_by_marker: dict[str, Any],
    default: Any = None,
) -> StubVlmClient:
    """Return a stub that picks a response by inspecting the user prompt.

    For each call the responder examines the last user-message text and
    returns the response keyed by the first marker substring it contains.
    Falls back to ``default`` if no marker matches.
    """

    def responder(messages: list[dict[str, Any]]) -> Any:
        last_user_text = ""
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                last_user_text = content
            elif isinstance(content, list):
                last_user_text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        for marker, response in responses_by_marker.items():
            if marker in last_user_text:
                return response
        return default

    return StubVlmClient(responder=responder)


def make_language_client() -> StubVlmClient:
    return make_canned_responder(
        {
            '"subtasks"': {
                "subtasks": [
                    {"text": "grasp bottle", "start": 0.0, "end": 0.5},
                    {"text": "pour water", "start": 0.5, "end": 2.9},
                ]
            },
            "compressed semantic memory": {"memory": "poured once"},
            '"rephrasings"': {"rephrasings": ["Pour water carefully."]},
            "acknowledgement the robot": {"text": "Sure."},
            "compact interjection": {"interjection": "use less water", "speech": "Using less water."},
            "frame-grounded visual question": {
                "question": "How many cups?",
                "answer": {"label": "cup", "count": 1},
            },
        }
    )


@dataclass
class StubFrameProvider:
    """Tiny images and call records for real annotation modules, without video I/O."""

    sentinel: Any = field(default_factory=lambda: PIL.Image.new("RGB", (32, 24)))
    cameras: tuple[str, ...] = ("observation.images.top",)
    calls: list[tuple[int, tuple[float, ...], str | None]] = field(default_factory=list)
    video_calls: list[tuple[int, int, str | None]] = field(default_factory=list)

    @property
    def camera_keys(self) -> list[str]:
        return list(self.cameras)

    def frames_at(self, record, timestamps, camera_key=None):
        self.calls.append((record.episode_index, tuple(timestamps), camera_key))
        return [self.sentinel] * len(timestamps)

    def video_for_episode(self, record, max_frames, camera_key=None):
        self.video_calls.append((record.episode_index, max_frames, camera_key))
        return [self.sentinel] * min(max_frames, len(record.frame_timestamps))


def persistent_atom(content: str, style: str = "subtask", timestamp: float = 0.0) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "style": style,
        "timestamp": timestamp,
        "tool_calls": None,
    }


def encode_vqa_answer(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)
