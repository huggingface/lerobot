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

from lerobot.policies.eo1.inference.eo1_adapter import EO1PolicyAdapter
from lerobot.runtime import LanguageConditionedRuntime, RuntimeState
from lerobot.runtime.registry import get_language_adapter_factory


class FakeEO1Policy:
    def __init__(self):
        self.action_prompts = []
        self.text_calls = []

    def prepare_runtime_action_batch(self, observation, task):
        self.action_prompts.append(task)
        return {**observation, "runtime_task": task}

    def predict_action_chunk(self, observation):
        assert observation["runtime_task"] == "grasp the cup"
        return ["a0", "a1"]

    def generate_text(self, observation, **kwargs):
        self.text_calls.append((observation, kwargs))
        return ["grasp the cup" if kwargs["kind"] == "subtask" else "the cup is on the left"]


def test_registry_lazily_resolves_eo1_adapter():
    assert get_language_adapter_factory("eo1") is EO1PolicyAdapter


def test_runtime_generates_eo1_subtask_then_rebuilds_action_prompt():
    policy = FakeEO1Policy()
    runtime = LanguageConditionedRuntime(
        policy_adapter=EO1PolicyAdapter(policy),
        observation_provider=lambda: {"observation.state": "state"},
        action_executor=lambda action: None,
    )
    runtime.set_task("clear the table")

    runtime.step_once()

    assert runtime.state.language_context["subtask"] == "grasp the cup"
    assert policy.text_calls[0][0]["task"] == "clear the table"
    assert policy.action_prompts == ["grasp the cup"]


def test_eo1_adapter_routes_vqa_and_generation_settings():
    policy = FakeEO1Policy()
    adapter = EO1PolicyAdapter(policy)

    answer = adapter.generate_text(
        "vqa",
        {"observation.state": "state"},
        RuntimeState(task="clear the table"),
        user_text="Where is the cup?",
    )

    assert answer == "the cup is on the left"
    _, kwargs = policy.text_calls[0]
    assert kwargs["user_text"] == "Where is the cup?"
    assert kwargs["temperature"] == 0.0
