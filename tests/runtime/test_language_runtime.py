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

from lerobot.configs import ActionChunkPrediction, TextKind
from lerobot.runtime import (
    LanguageConditionedRuntime,
    RuntimeState,
    SubtaskController,
    Tick,
    build_language_batch,
)


class FakePolicy:
    """Minimal stand-in for the `PreTrainedPolicy` surface the runtime drives."""

    def __init__(self, texts=None, chunk_text=None):
        self.texts = list(texts) if texts is not None else None
        self.chunk_text = chunk_text
        self.batches = []

    def supports_text_generation(self):
        return self.texts is not None

    def generate_text(self, batch, *, kind=TextKind.SUBTASK, user_text=None):
        self.batches.append((kind, batch))
        return self.texts.pop(0) if self.texts else ""

    def predict_action_chunk(self, batch):
        assert batch == {"observation.state": 1}
        return ["a0", "a1"]

    def predict_action_chunk_with_text(self, batch):
        return ActionChunkPrediction(action=self.predict_action_chunk(batch), text=self.chunk_text)


def test_runtime_tick_generates_subtask_enqueues_and_dispatches_action():
    policy = FakePolicy(["pick cup"])
    executed = []
    runtime = LanguageConditionedRuntime(
        policy=policy,
        observation_provider=lambda: {"observation.state": 1},
        action_executor=executed.append,
    )
    runtime.set_task("clean")

    logs = runtime.step_once()

    assert runtime.state.language_context["subtask"] == "pick cup"
    assert executed == ["a0"]
    assert list(runtime.state.action_queue) == ["a1"]
    assert "  subtask: pick cup" in logs
    assert policy.batches[0][0] is TextKind.SUBTASK


def test_policy_without_text_head_keeps_the_operator_instruction():
    policy = FakePolicy(texts=None)
    runtime = LanguageConditionedRuntime(
        policy=policy,
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("clean")

    runtime.step_once()

    assert not runtime.subtask.enabled
    assert "subtask" not in runtime.state.language_context
    assert policy.batches == []


def test_generation_batch_carries_task_and_active_subtask():
    state = RuntimeState(task="clean the table")
    state.set_context("subtask", "pick the red cup")

    batch = build_language_batch({"observation.state": 1}, state)

    assert batch == {
        "observation.state": 1,
        "task": "clean the table",
        "subtask": "pick the red cup",
    }


def test_subtask_throttled_to_one_generation_per_n_chunks():
    policy = FakePolicy(["pick the first cup", "pick the second cup"])
    controller = SubtaskController(policy, chunks_per_regen=2)
    state = RuntimeState(task="clean")

    controller.update(None, state)  # generates
    assert state.language_context["subtask"] == "pick the first cup"
    controller.update(None, state)  # throttled
    assert state.language_context["subtask"] == "pick the first cup"
    controller.update(None, state)  # generates again
    assert state.language_context["subtask"] == "pick the second cup"


def test_rearm_regenerates_before_the_throttle_elapses():
    policy = FakePolicy(["first", "second"])
    controller = SubtaskController(policy, chunks_per_regen=5)
    state = RuntimeState(task="clean")

    controller.update(None, state)
    controller.rearm()
    controller.update(None, state)

    assert state.language_context["subtask"] == "second"


def test_empty_generation_leaves_the_previous_subtask_in_place():
    policy = FakePolicy(["pick the cup", ""])
    controller = SubtaskController(policy)
    state = RuntimeState(task="clean")

    controller.update(None, state)
    controller.update(None, state)

    assert state.language_context["subtask"] == "pick the cup"
    assert controller.diagnostics.empty == 1
    assert any("subtask gen returned empty" in line for line in state.log_lines)


def test_repeated_generation_counts_as_a_diagnostic():
    policy = FakePolicy(["pick the cup", "pick the cup"])
    controller = SubtaskController(policy)
    state = RuntimeState(task="clean")

    controller.update(None, state)
    controller.update(None, state)

    assert controller.diagnostics.repeat == 1
    assert controller.diagnostics.last_raw == "pick the cup"


def test_prompt_change_discards_in_flight_action_chunk():
    started = threading.Event()
    release = threading.Event()

    class BlockingPolicy(FakePolicy):
        def predict_action_chunk(self, batch):
            started.set()
            assert release.wait(timeout=2)
            return ["stale"]

    runtime = LanguageConditionedRuntime(
        policy=BlockingPolicy(chunk_text="stale reasoning"),
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
    # The discarded chunk takes its text with it, so the panel cannot show reasoning
    # belonging to a chunk that was never executed.
    assert runtime.state.last_prediction is None


def test_accepted_chunk_publishes_its_text_for_display_only():
    runtime = LanguageConditionedRuntime(
        policy=FakePolicy(chunk_text="reach for the cup"),
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("clean")

    runtime.step_once()

    assert runtime.state.last_prediction is not None
    assert runtime.state.last_prediction.text == "reach for the cup"
    assert runtime.state.last_prediction.action == ["a0", "a1"]
    # Never routed through the channel that is fed back to the policy as a command.
    assert "subtask" not in runtime.state.language_context


def test_policy_without_a_text_head_reports_no_chunk_text():
    from lerobot.policies.pretrained import PreTrainedPolicy

    class Chunker:
        """Any policy that only implements `predict_action_chunk`."""

        def predict_action_chunk(self, batch, **kwargs):
            return ["a0"]

    prediction = PreTrainedPolicy.predict_action_chunk_with_text(Chunker(), {})
    assert prediction.action == ["a0"]
    assert prediction.text is None


def test_action_chunk_prediction_is_frozen():
    import pytest

    prediction = ActionChunkPrediction(action=["a0"], text="reach")
    with pytest.raises(AttributeError):
        prediction.text = "something else"


def test_text_kind_members_compare_equal_to_their_wire_strings():
    # Policies that dispatch on strings internally keep working unchanged.
    assert TextKind.SUBTASK == "subtask"
    assert TextKind.VQA == "vqa"
    assert TextKind.VQA in {"vqa", "caption", "grounding"}
