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

"""Tests for PI052's text tokenizer.

Covers ``say`` tool-call flattening (PaliGemma's flat prompt has no
structured tool calls, so a ``say`` call must be serialized into a
``<say>...</say>`` text marker) and EOS-termination supervision (the
supervised target span must end with an EOS token so the LM head learns
to stop instead of rambling to ``max_length`` at inference).
"""

import torch

from lerobot.policies.pi052.text_processor_pi052 import (
    PI052TextTokenizerStep,
    _flatten_say_tool_calls,
    _format_messages,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


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


# ---------------------------------------------------------------------------
# EOS-termination supervision
# ---------------------------------------------------------------------------


def test_format_messages_appends_eos_to_target_turns_only():
    msgs = [
        {"role": "user", "content": "pick cube"},
        {"role": "assistant", "content": "move to cube"},
    ]
    prompt, spans = _format_messages(msgs, target_indices=[1], eos_token="<eos>")
    # EOS is appended to the supervised target (assistant) turn only.
    assert prompt == "User: pick cube\nAssistant: move to cube<eos>\n"
    # The user span is unchanged; the target span covers content + EOS.
    assert prompt[spans[0][0] : spans[0][1]] == "pick cube"
    assert prompt[spans[1][0] : spans[1][1]] == "move to cube<eos>"


def test_format_messages_without_eos_args_is_unchanged():
    """Inference callers omit target_indices / eos_token — no EOS baked in."""
    prompt, spans = _format_messages([{"role": "user", "content": "hi"}])
    assert prompt == "User: hi\n"
    assert prompt[spans[0][0] : spans[0][1]] == "hi"


def _eos_char_id() -> int:
    """Token id _CharTokenizer assigns to its 1-char EOS."""
    return ord("\x1f") % 251 + 1


def test_pi052_text_tokenizer_supervises_eos_at_target_end():
    """The appended EOS is the last supervised label on a target turn —
    that's the signal that teaches the LM head to stop. The trailing
    newline right after it stays unsupervised (-100)."""
    step = PI052TextTokenizerStep(max_length=64)
    step._tokenizer = _CharTokenizer()
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "messages": [
                {"role": "user", "content": "pick cube"},
                {"role": "assistant", "content": "move to cube"},
            ],
            "target_message_indices": [1],
            "message_streams": ["high_level", "high_level"],
            "index": torch.tensor(10),
        },
    }
    out = step(transition)
    ids = out[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS][0]
    labels = out[TransitionKey.COMPLEMENTARY_DATA]["text_labels"][0]

    supervised = (labels != -100).nonzero().flatten().tolist()
    assert supervised, "target turn produced no supervised labels"
    last = supervised[-1]
    # The last supervised token is the appended EOS.
    assert int(ids[last]) == _eos_char_id()
    assert int(labels[last]) == _eos_char_id()
    # The token right after the EOS (the trailing newline) is NOT supervised.
    assert int(labels[last + 1]) == -100


class _CharTokenizer:
    pad_token_id = 0
    eos_token = "\x1f"  # unit separator — a 1-char "EOS" for testing

    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
        return_offsets_mapping,
        padding_side,
    ):
        ids = [ord(c) % 251 + 1 for c in text[:max_length]]
        offsets = [(i, i + 1) for i in range(len(ids))]
        attention = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pad = max_length - len(ids)
            ids += [self.pad_token_id] * pad
            offsets += [(0, 0)] * pad
            attention += [0] * pad
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
            "offset_mapping": torch.tensor([offsets], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(max(int(i) - 1, 0)) for i in token_ids if int(i) != self.pad_token_id)


def test_pi052_text_tokenizer_handles_batched_rendered_messages():
    step = PI052TextTokenizerStep(max_length=64)
    step._tokenizer = _CharTokenizer()

    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            "messages": [
                [
                    {"role": "user", "content": "pick cube"},
                    {"role": "assistant", "content": "move to cube"},
                ],
                [{"role": "user", "content": "open drawer"}],
            ],
            "target_message_indices": [[1], []],
            "message_streams": [["high_level", "high_level"], ["low_level"]],
            "index": torch.tensor([10, 11]),
        },
    }

    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    comp = out[TransitionKey.COMPLEMENTARY_DATA]

    assert obs[OBS_LANGUAGE_TOKENS].shape == (2, 64)
    assert obs[OBS_LANGUAGE_ATTENTION_MASK].shape == (2, 64)
    assert comp["text_labels"].shape == (2, 64)
    assert comp["predict_actions"].tolist() == [False, True]
    assert (comp["text_labels"][0] != -100).any()
    assert not (comp["text_labels"][1] != -100).any()
