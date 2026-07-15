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

from lerobot.runtime import (
    LanguageConditionedRuntime,
    RuntimeState,
)


class FakeAdapter:
    def __init__(self):
        self.updated = False
        self.interjections = []

    def select_action(self, observation, state):
        assert observation == {"observation.state": 1}
        assert state.task == "clean"
        return ["a0", "a1"]

    def update_language_state(self, observation, state):
        self.updated = True
        state.set_context("subtask", "pick cup", label="subtask")

    def handle_interjection(self, user_text, observation, state):
        self.interjections.append(user_text)
        state.set_context("plan", "new plan", label="plan")


def test_runtime_tick_updates_language_enqueues_and_dispatches_action():
    adapter = FakeAdapter()
    executed = []
    runtime = LanguageConditionedRuntime(
        policy_adapter=adapter,
        observation_provider=lambda: {"observation.state": 1},
        action_executor=executed.append,
    )
    runtime.set_task("clean")

    logs = runtime.step_once()

    assert adapter.updated
    assert runtime.state.language_context["subtask"] == "pick cup"
    assert executed == ["a0"]
    assert list(runtime.state.action_queue) == ["a1"]
    assert "  subtask: pick cup" in logs


def test_runtime_handles_user_interjection():
    adapter = FakeAdapter()
    runtime = LanguageConditionedRuntime(
        policy_adapter=adapter,
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("clean")
    runtime.state.extra["recent_interjection"] = "please say ok"
    runtime.state.emit("user_interjection")

    runtime.step_once()

    assert "please say ok" in adapter.interjections
    assert runtime.state.language_context["plan"] == "new plan"


def test_runtime_state_aliases_legacy_keys_to_language_context():
    state = RuntimeState()
    state["current_subtask"] = "open drawer"
    state["current_memory"] = "drawer open"

    assert state.get("current_subtask") == "open drawer"
    assert state.language_context == {"subtask": "open drawer", "memory": "drawer open"}
