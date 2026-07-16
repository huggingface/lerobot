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

import threading
import time

from lerobot.runtime import LanguageConditionedRuntime, Tick


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


def test_prompt_change_discards_in_flight_action_chunk():
    started = threading.Event()
    release = threading.Event()

    class BlockingAdapter(FakeAdapter):
        def select_action(self, observation, state):
            started.set()
            assert release.wait(timeout=2)
            return ["stale"]

    runtime = LanguageConditionedRuntime(
        policy_adapter=BlockingAdapter(),
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("old task")
    runtime.state.tick = Tick(index=1, monotonic_seconds=time.monotonic())
    inference = threading.Thread(target=runtime.maybe_enqueue_action_chunk, kwargs={"force": True})
    inference.start()
    assert started.wait(timeout=2)

    runtime.set_task("new task")
    release.set()
    inference.join(timeout=2)

    assert not inference.is_alive()
    assert list(runtime.state.action_queue) == []
