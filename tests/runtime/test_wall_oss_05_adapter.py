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

from lerobot.policies.wall_oss_05.inference.wall_oss_05_adapter import WallOSS05PolicyAdapter
from lerobot.runtime import LanguageConditionedRuntime, RuntimeState
from lerobot.runtime.registry import get_language_adapter_factory


class FakeWallPolicy:
    def __init__(self):
        self.action_calls = []
        self.text_calls = []

    def predict_action_chunk(self, observation):
        self.action_calls.append(observation)
        return ["a0", "a1"]

    def generate_text(self, observation, **kwargs):
        self.text_calls.append((observation, kwargs))
        if kwargs["kind"] == "subtask":
            return ["grasp the cup"]
        return ["the cup is on the left"]


def test_registry_lazily_resolves_wall_adapter():
    assert get_language_adapter_factory("wall_oss_05") is WallOSS05PolicyAdapter


def test_runtime_generates_wall_subtask_then_conditions_action_on_it():
    policy = FakeWallPolicy()
    runtime = LanguageConditionedRuntime(
        policy_adapter=WallOSS05PolicyAdapter(policy),
        observation_provider=lambda: {"task": "stale", "observation.state": "state"},
        action_executor=lambda action: None,
    )
    runtime.set_task("clear the table")

    runtime.step_once()

    assert runtime.state.language_context["subtask"] == "grasp the cup"
    assert policy.text_calls[0][0]["task"] == "clear the table"
    assert policy.action_calls[0]["task"] == "grasp the cup"


def test_wall_adapter_answers_vqa_with_runtime_generation_settings():
    policy = FakeWallPolicy()
    adapter = WallOSS05PolicyAdapter(policy)
    state = RuntimeState(task="clear the table")

    answer = adapter.generate_text(
        "vqa",
        {"task": "stale", "observation.state": "state"},
        state,
        user_text="Where is the cup?",
    )

    assert answer == "the cup is on the left"
    observation, kwargs = policy.text_calls[0]
    assert observation["task"] == "clear the table"
    assert kwargs["user_text"] == "Where is the cup?"
    assert kwargs["temperature"] == 0.0


def test_wall_adapter_does_not_generate_untrained_memory_stream():
    policy = FakeWallPolicy()
    adapter = WallOSS05PolicyAdapter(policy)

    assert adapter.generate_text("memory", {}, RuntimeState(task="clean")) == ""
    assert policy.text_calls == []
