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

from pathlib import Path

import pytest
import torch

from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.datasets.language_render import render_sample
from lerobot.policies.pi052.text_processor_pi052 import (
    PI052TextTokenizerStep,
    _flatten_say_tool_calls,
    _format_messages,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


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
    out = _flatten_say_tool_calls({"role": "assistant", "content": "plan only", "tool_calls": [weather]})
    assert out["content"] == "plan only"
    assert "tool_calls" not in out


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


def test_pi052_steps_roundtrip_through_standard_pipeline_loader(tmp_path):
    recipe = TrainingRecipe(messages=[MessageTurn(role="user", content="${task}", stream="low_level")])
    pipeline = PolicyProcessorPipeline(
        steps=[
            RenderMessagesStep(recipe),
            PI052TextTokenizerStep(
                tokenizer_name="custom-tokenizer",
                max_length=77,
                plan_dropout_prob=0.2,
                dropout_seed=3,
            ),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    pipeline.save_pretrained(tmp_path)

    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )

    assert loaded.steps[0].recipe == recipe
    assert loaded.steps[1].tokenizer_name == "custom-tokenizer"
    assert loaded.steps[1].max_length == 77
    assert loaded.steps[1].plan_dropout_prob == 0.2
    assert loaded.steps[1].dropout_seed == 3


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


@pytest.mark.parametrize("recipe_name", ["subtask_mem", "subtask_mem_vqa_speech"])
@pytest.mark.parametrize("timestamp", [0.0, 1.5])
def test_memory_recipe_supervises_one_combined_response_without_future_context(recipe_name, timestamp):
    recipe = TrainingRecipe.from_yaml(Path(f"src/lerobot/configs/recipes/{recipe_name}.yaml"))
    assert "memory_update" not in recipe.blend
    assert sum(branch.weight for branch in recipe.blend.values()) == pytest.approx(1.0)
    high_level = recipe.blend["high_level_memory_subtask"]
    rows = [
        {"role": "assistant", "style": "memory", "content": "cup in box", "timestamp": 0.0},
        {"role": "assistant", "style": "subtask", "content": "pick plate", "timestamp": 0.0},
        {"role": "assistant", "style": "memory", "content": "cup and plate in box", "timestamp": 1.0},
        {"role": "assistant", "style": "subtask", "content": "pick spoon", "timestamp": 1.0},
        {"role": "assistant", "style": "memory", "content": "future success", "timestamp": 2.0},
    ]
    rendered = render_sample(
        recipe=high_level, persistent=rows, events=[], t=timestamp, sample_idx=0, task="clear table"
    )
    assert rendered is not None
    assert rendered["target_message_indices"] == [len(rendered["messages"]) - 1]
    target = rendered["messages"][-1]["content"]
    if timestamp == 0:
        assert target == "Memory: cup in box\nSubtask: pick plate"
        assert not any("Previous memory:" in m["content"] for m in rendered["messages"])
    else:
        assert rendered["messages"][1]["content"] == "Previous memory: cup in box"
        assert target == "Memory: cup and plate in box\nSubtask: pick spoon"
    assert "future success" not in str(rendered)
    assert target not in str(rendered["messages"][:-1])

    step = PI052TextTokenizerStep(max_length=1024)
    step._tokenizer = _CharTokenizer()
    output = step(
        {
            TransitionKey.OBSERVATION: {},
            TransitionKey.COMPLEMENTARY_DATA: {**rendered, "index": torch.tensor(0)},
        }
    )
    labels = output[TransitionKey.COMPLEMENTARY_DATA]["text_labels"][0]
    supervised = labels[labels != -100]
    # Both fields and EOS are one target; the prior memory remains unsupervised context.
    assert step._tokenizer.decode(supervised) == target + step._tokenizer.eos_token
    assert not output[TransitionKey.COMPLEMENTARY_DATA]["predict_actions"].item()

    low_level = render_sample(
        recipe=recipe.blend["low_level_execution"],
        persistent=rows,
        events=[],
        t=timestamp,
        sample_idx=0,
        task="clear table",
    )
    assert low_level["messages"] == [
        {"role": "user", "content": "pick plate" if timestamp == 0 else "pick spoon"}
    ]
    assert low_level["target_message_indices"] == []
    assert low_level["message_streams"] == ["low_level"]


@pytest.mark.parametrize("recipe_name", ["subtask_mem", "subtask_mem_vqa_speech"])
def test_memory_recipe_skips_joint_target_without_memory_annotations(recipe_name):
    recipe = TrainingRecipe.from_yaml(Path(f"src/lerobot/configs/recipes/{recipe_name}.yaml"))
    rendered = render_sample(
        recipe=recipe.blend["high_level_memory_subtask"],
        persistent=[{"role": "assistant", "style": "subtask", "content": "pick cup", "timestamp": 0.0}],
        events=[],
        t=0.0,
        sample_idx=0,
        task="clear table",
    )
    assert rendered is None


def test_memory_recipe_variants_share_the_same_high_level_contract():
    plain = TrainingRecipe.from_yaml(Path("src/lerobot/configs/recipes/subtask_mem.yaml"))
    extended = TrainingRecipe.from_yaml(Path("src/lerobot/configs/recipes/subtask_mem_vqa_speech.yaml"))
    assert (
        plain.blend["high_level_memory_subtask"].messages
        == extended.blend["high_level_memory_subtask"].messages
    )
    assert (
        plain.blend["high_level_memory_subtask"].bindings
        == extended.blend["high_level_memory_subtask"].bindings
    )


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
