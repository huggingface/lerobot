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

    # A policy overriding the template is what the runtime must respect, so the fake
    # deliberately does not use the base default.
    def build_prompt(self, kind, **values):
        assert kind == "subtask"
        return f"predict next subtask, given this high level goal: {values['task']}"

    def generate_text(self, batch, prompt):
        self.batches.append((prompt, batch))
        return self.texts.pop(0) if self.texts else ""

    def predict_action_chunk(self, batch, *, with_text=False):
        assert batch == {"observation.state": 1}
        chunk = ["a0", "a1"]
        return (chunk, self.chunk_text) if with_text else chunk


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
    # The runtime filled the policy's own template with the operator's goal.
    assert policy.batches[0][0] == "predict next subtask, given this high level goal: clean"


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
        def predict_action_chunk(self, batch, *, with_text=False):
            started.set()
            assert release.wait(timeout=2)
            return (["stale"], self.chunk_text) if with_text else ["stale"]

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
    assert runtime.state.last_chunk_text is None


def test_accepted_chunk_publishes_its_text_for_display_only():
    runtime = LanguageConditionedRuntime(
        policy=FakePolicy(chunk_text="reach for the cup"),
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("clean")

    runtime.step_once()

    assert runtime.state.last_chunk_text == "reach for the cup"
    # Never routed through the channel that is fed back to the policy as a command.
    assert "subtask" not in runtime.state.language_context


def test_policy_ignoring_with_text_still_drives_the_runtime():
    """Only an in-stream policy honours the flag; the rest return a bare chunk."""

    class Chunker(FakePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            return ["a0", "a1"]

    runtime = LanguageConditionedRuntime(
        policy=Chunker(),
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("clean")

    runtime.step_once()

    assert list(runtime.state.action_queue) == ["a1"]
    assert runtime.state.last_chunk_text is None


def test_default_subtask_template_is_the_trained_wall_oss_wording():
    """Pinned on purpose: editing it silently changes what every inheriting policy asks."""
    from lerobot.policies.pretrained import PreTrainedPolicy

    assert PreTrainedPolicy.PROMPT_TEMPLATES["subtask"] == ("{task}\nPredict the next action in language.")


def test_prompt_registry_merges_a_policy_dict_over_the_base_defaults():
    import pytest

    from lerobot.policies.pretrained import PreTrainedPolicy

    class Custom:
        PROMPT_TEMPLATES = {"memory": "summarise progress toward {task}"}
        prompt_template = PreTrainedPolicy.prompt_template
        build_prompt = PreTrainedPolicy.build_prompt

    policy = Custom()
    # A registered kind the policy declared.
    assert policy.build_prompt("memory", task="clear the table") == (
        "summarise progress toward clear the table"
    )
    # A kind it did not declare still resolves through the base defaults.
    assert policy.build_prompt("subtask", task="clear the table") == (
        "clear the table\nPredict the next action in language."
    )
    with pytest.raises(ValueError, match="no 'nope' prompt template"):
        policy.build_prompt("nope")


def test_build_prompt_tolerates_braces_in_the_substituted_value():
    from lerobot.policies.pretrained import PreTrainedPolicy

    class Policy:
        PROMPT_TEMPLATES: dict[str, str] = {}
        prompt_template = PreTrainedPolicy.prompt_template
        build_prompt = PreTrainedPolicy.build_prompt

    # `str.format` would raise on this task; `str.replace` does not.
    assert (
        Policy()
        .build_prompt("subtask", task="put the {red} block down")
        .startswith("put the {red} block down")
    )
