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

"""Tests for SmolVLA2's chat-tokenizer ``tool_calls`` flattening.

``_split_plan_and_say`` (inference) expects the model to emit a textual
``<say>...</say>`` marker. ``_flatten_say_tool_calls`` is the training-time
serializer that produces it: it rewrites an assistant turn's structured
``say`` tool call into that marker *inside the content text*, before
``apply_chat_template`` runs — so the chat template only tokenizes plain
text and the supervised target span trains the model to emit the marker
the runtime parses back. These tests pin the round-trip.
"""

from lerobot.policies.smolvla2.chat_processor_smolvla2 import flatten_say_tool_calls
from lerobot.policies.smolvla2.inference.steps import _split_plan_and_say


def _say_call(text):
    return {"type": "function", "function": {"name": "say", "arguments": {"text": text}}}


def test_flatten_appends_say_marker_and_drops_tool_calls():
    msg = {"role": "assistant", "content": "Pick up the blue cube.", "tool_calls": [_say_call("On it!")]}
    out = flatten_say_tool_calls(msg)
    assert "tool_calls" not in out
    assert out["content"] == "Pick up the blue cube.\n<say>On it!</say>"


def test_flatten_roundtrips_through_inference_parser():
    """The marker the serializer writes must be exactly what the inference
    parser reads back — this is the train/inference contract."""
    msg = {"role": "assistant", "content": "Move toward the cube.", "tool_calls": [_say_call("Working on it")]}
    flat = flatten_say_tool_calls(msg)["content"]
    plan, speech = _split_plan_and_say(flat)
    assert plan == "Move toward the cube."
    assert speech == "Working on it"


def test_flatten_accepts_json_string_arguments():
    """``arguments`` may arrive as a JSON string rather than a dict."""
    call = {"type": "function", "function": {"name": "say", "arguments": '{"text": "hello there"}'}}
    out = flatten_say_tool_calls({"role": "assistant", "content": "p", "tool_calls": [call]})
    assert out["content"] == "p\n<say>hello there</say>"


def test_flatten_leaves_messages_without_tool_calls_untouched():
    msg = {"role": "assistant", "content": "just a plan"}
    assert flatten_say_tool_calls(msg) == msg


def test_flatten_drops_empty_or_non_say_tool_calls():
    """A non-``say`` call (or empty text) leaves content alone but still
    strips the structured calls so the template renders no JSON block."""
    weather = {"type": "function", "function": {"name": "check_weather", "arguments": {}}}
    out = flatten_say_tool_calls({"role": "assistant", "content": "plan only", "tool_calls": [weather]})
    assert out["content"] == "plan only"
    assert "tool_calls" not in out


def test_flatten_marker_only_when_content_empty():
    msg = {"role": "assistant", "content": "", "tool_calls": [_say_call("hi")]}
    out = flatten_say_tool_calls(msg)
    assert out["content"] == "<say>hi</say>"
