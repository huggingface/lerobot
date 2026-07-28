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

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import (
    BaseLanguageAdapter,
    DirectTaskPolicyAdapter,
    GenerationConfig,
)


class ScriptedAdapter(BaseLanguageAdapter):
    """Base adapter whose text generation returns queued strings per kind."""

    def __init__(self, scripts, gen=None):
        super().__init__(policy=object(), gen=gen)
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls = []

    def select_action(self, observation, state):
        return None

    def generate_text(self, kind, observation, state, user_text=None):
        self.calls.append(kind)
        queue = self.scripts.get(kind, [])
        return queue.pop(0) if queue else ""


def test_cascade_sets_subtask_then_memory():
    adapter = ScriptedAdapter({"subtask": ["pick the red cup"], "memory": ["the cup is grasped"]})
    state = RuntimeState(task="clean")

    adapter.update_language_state(None, state)

    assert state.language_context["subtask"] == "pick the red cup"
    assert state.language_context["memory"] == "the cup is grasped"
    assert adapter.calls == ["subtask", "memory"]


def test_nonempty_generation_is_used_verbatim():
    adapter = ScriptedAdapter({"subtask": [":::: ::"], "memory": ["memory"]})
    state = RuntimeState(task="clean")

    adapter.update_language_state(None, state)

    assert state.language_context["subtask"] == ":::: ::"
    assert state.language_context["memory"] == "memory"
    assert adapter.calls == ["subtask", "memory"]


def test_throttle_regenerates_every_n_chunks():
    adapter = ScriptedAdapter(
        {
            "subtask": ["pick the first cup", "pick the second cup"],
            "memory": ["memory one two three", "memory four five six"],
        },
        gen=GenerationConfig(chunks_per_regen=2),
    )
    state = RuntimeState(task="clean")

    adapter.update_language_state(None, state)  # generates
    assert state.language_context["subtask"] == "pick the first cup"
    adapter.update_language_state(None, state)  # throttled — no generation
    assert state.language_context["subtask"] == "pick the first cup"
    adapter.update_language_state(None, state)  # generates again
    assert state.language_context["subtask"] == "pick the second cup"


def test_handle_interjection_sets_plan_and_strips_say():
    adapter = ScriptedAdapter({"interjection": ["turn to the left now <say>heading left</say>"]})
    state = RuntimeState(task="clean")

    adapter.handle_interjection("turn", None, state)

    assert state.language_context["plan"] == "turn to the left now"


def test_direct_task_adapter_delegates_action_chunk():
    class Policy:
        def predict_action_chunk(self, observation):
            return ("chunk", observation)

    observation = {"task": "pick up the cube"}
    adapter = DirectTaskPolicyAdapter(Policy())

    assert adapter.select_action(observation, RuntimeState()) == ("chunk", observation)
    assert adapter.generate_text("subtask", observation, RuntimeState()) == ""


def test_flat_policy_registry_reuses_direct_task_adapter():
    from lerobot.runtime.registry import get_language_adapter_factory

    assert get_language_adapter_factory("pi05") is DirectTaskPolicyAdapter
    assert get_language_adapter_factory("molmoact2") is DirectTaskPolicyAdapter
