#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import pytest

from lerobot.policies.being_h05.inference.being_h05_adapter import BeingH05PolicyAdapter
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import GenerationConfig
from lerobot.runtime.registry import get_language_adapter_factory


class FakeBeingH05Policy:
    config = SimpleNamespace(type="being_h05", author_model_id="BeingBeyond/Being-H05-2B")

    def __init__(self):
        self.text_calls = []

    def predict_action_chunk(self, observation):
        return ("action", observation)

    def generate_text(self, observation, prompt, **kwargs):
        self.text_calls.append((observation, prompt, kwargs))
        return ["grounded answer"]


def test_being_h05_adapter_routes_actions_and_is_registered():
    policy = FakeBeingH05Policy()
    adapter = BeingH05PolicyAdapter(policy)
    observation = {"being_h05.pixel_values": "pixels"}

    assert adapter.select_action(observation, RuntimeState(task="pick")) == ("action", observation)
    assert get_language_adapter_factory("being_h05") is BeingH05PolicyAdapter


def test_being_h05_adapter_passes_vqa_question_verbatim():
    policy = FakeBeingH05Policy()
    adapter = BeingH05PolicyAdapter(
        policy,
        GenerationConfig(min_new_tokens=2, temperature=0.4, top_p=0.9),
    )
    observation = {"being_h05.pixel_values": "pixels"}

    answer = adapter.generate_text(
        "vqa",
        observation,
        RuntimeState(task="tidy the counter"),
        user_text="What is beside the mug?",
    )

    assert answer == "grounded answer"
    _, prompt, kwargs = policy.text_calls[0]
    assert prompt == "What is beside the mug?"
    assert kwargs["min_new_tokens"] == 2
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 0.9


def test_being_h05_adapter_generates_runtime_context():
    policy = FakeBeingH05Policy()
    adapter = BeingH05PolicyAdapter(policy)
    state = RuntimeState(task="put the bowl in the sink")
    observation = {"being_h05.pixel_values": "pixels"}

    adapter.update_language_state(observation, state)

    assert state.language_context == {
        "subtask": "grounded answer",
        "memory": "grounded answer",
    }
    assert "put the bowl in the sink" in policy.text_calls[0][1]


def test_robocasa_adapter_keeps_direct_action_runtime_and_rejects_vqa():
    policy = FakeBeingH05Policy()
    policy.config = SimpleNamespace(
        type="being_h05",
        author_model_id="BeingBeyond/Being-H05-2B_robocasa",
    )
    adapter = BeingH05PolicyAdapter(policy)
    state = RuntimeState(task="open the drawer")
    observation = {"being_h05.pixel_values": "pixels"}

    adapter.update_language_state(observation, state)

    assert not policy.text_calls
    assert state.language_context == {}
    with pytest.raises(RuntimeError, match="does not retain usable text generation"):
        adapter.generate_text("vqa", observation, state, user_text="What is open?")
