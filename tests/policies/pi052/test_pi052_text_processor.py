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

"""Tests for PI052's text-tokenizer ``say`` tool-call flattening.

PaliGemma's flat prompt has no structured tool calls, so an assistant
``say`` tool call must be serialized into a ``<say>...</say>`` text
marker — otherwise the spoken reply is dropped and never supervised.
"""

from lerobot.policies.pi052.text_processor_pi052 import _flatten_say_tool_calls


def _say_call(text):
    return {"type": "function", "function": {"name": "say", "arguments": {"text": text}}}


def test_flatten_appends_say_marker_and_drops_tool_calls():
    msg = {"role": "assistant", "content": "Heading to the cube.", "tool_calls": [_say_call("On it!")]}
    out = _flatten_say_tool_calls(msg)
    assert "tool_calls" not in out
    assert out["content"] == "Heading to the cube.\n<say>On it!</say>"


def test_flatten_marker_only_when_content_empty_or_none():
    out = _flatten_say_tool_calls({"role": "assistant", "tool_calls": [_say_call("hi")]})
    assert out["content"] == "<say>hi</say>"


def test_flatten_accepts_json_string_arguments():
    call = {"type": "function", "function": {"name": "say", "arguments": '{"text": "hello there"}'}}
    out = _flatten_say_tool_calls({"role": "assistant", "content": "p", "tool_calls": [call]})
    assert out["content"] == "p\n<say>hello there</say>"


def test_flatten_leaves_messages_without_tool_calls_untouched():
    msg = {"role": "assistant", "content": "just a plan"}
    assert _flatten_say_tool_calls(msg) == msg


def test_flatten_drops_non_say_tool_calls_but_keeps_content():
    weather = {"type": "function", "function": {"name": "check_weather", "arguments": {}}}
    out = _flatten_say_tool_calls(
        {"role": "assistant", "content": "plan only", "tool_calls": [weather]}
    )
    assert out["content"] == "plan only"
    assert "tool_calls" not in out
