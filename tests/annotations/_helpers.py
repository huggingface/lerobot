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
from typing import Any

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
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_user_text = block.get("text", "")
        for marker, response in responses_by_marker.items():
            if marker in last_user_text:
                return response
        return default

    return StubVlmClient(responder=responder)


def encode_vqa_answer(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)
