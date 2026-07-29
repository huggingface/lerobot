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

from types import SimpleNamespace

import pytest

from lerobot.runtime import LanguageConditionedRuntime, RuntimeState
from lerobot.runtime.adapter import GenerationConfig


class FakeG05Policy:
    def __init__(self, *, predict_cot=False, discrete_action=True, continuous_action=False):
        self.config = SimpleNamespace(
            predict_cot=predict_cot,
            discrete_action=discrete_action,
            continuous_action=continuous_action,
            runtime_system="system2" if predict_cot else "system1",
        )
        self.calls = []

    def predict_action_chunk(self, observation):
        self.calls.append(observation)
        return ["a0", "a1"]


def test_registry_lazily_resolves_g05_adapter():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter
    from lerobot.runtime.registry import get_language_adapter_factory

    assert get_language_adapter_factory("g05") is G05PolicyAdapter


def test_system1_passes_exact_runtime_task_without_mutating_observation():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    policy = FakeG05Policy()
    adapter = G05PolicyAdapter(policy)
    original = {"task": "stale generated subtask", "observation.state": "state"}
    raw_task = "  把 red cup 放到左边\nexactly as written  "

    chunk = adapter.select_action(original, RuntimeState(task=raw_task))

    assert chunk == ["a0", "a1"]
    assert policy.calls[0]["task"] == raw_task
    assert original["task"] == "stale generated subtask"


def test_system2_surfaces_same_pass_cot_and_action():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    class ReasoningPolicy(FakeG05Policy):
        def __init__(self):
            super().__init__(predict_cot=True, continuous_action=True)

        def predict_action_chunk_with_runtime(self, observation, *, task, system_mode=None):
            self.calls.append((observation, task, system_mode))
            return {
                "action_chunk": ["fm0", "fm1"],
                "cot_text": "BBox: cup [1,2,3,4]|\nSubtask: grasp the cup|Updated Memory: cup located",
            }

    policy = ReasoningPolicy()
    adapter = G05PolicyAdapter(policy)
    state = RuntimeState(task="  clear the table  ")

    chunk = adapter.select_action({"task": "wrong"}, state)

    assert chunk == ["fm0", "fm1"]
    assert policy.calls[0][1] == "  clear the table  "
    assert policy.calls[0][0]["task"] == "  clear the table  "
    assert policy.calls[0][2] == "system2"
    assert (
        state.language_context["cot_text"]
        == "BBox: cup [1,2,3,4]|\nSubtask: grasp the cup|Updated Memory: cup located"
    )
    assert state.extra["g05_subtask"] == "grasp the cup"
    assert state.language_context["memory"] == "cup located"


def test_system2_accepts_batch_safe_tuple_metadata():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    class ReasoningPolicy(FakeG05Policy):
        def __init__(self):
            super().__init__(predict_cot=True)

        def predict_action_chunk_with_runtime(self, observation, *, task, system_mode=None):
            return ("chunk", {"cot_text": ["Subtask: move left"], "plan": "first move left"})

    state = RuntimeState(task="move")
    chunk = G05PolicyAdapter(ReasoningPolicy()).select_action({}, state)

    assert chunk == "chunk"
    assert state.extra["g05_subtask"] == "move left"
    assert state.language_context["plan"] == "first move left"


def test_system2_reasoning_does_not_invalidate_same_pass_action_chunk():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    class ReasoningPolicy(FakeG05Policy):
        def __init__(self):
            super().__init__(predict_cot=True)

        def predict_action_chunk_with_runtime(self, observation, *, task, system_mode=None):
            return (["a0", "a1"], {"cot_text": "Subtask: pick cup"})

    executed = []
    runtime = LanguageConditionedRuntime(
        policy_adapter=G05PolicyAdapter(ReasoningPolicy()),
        observation_provider=lambda: {"task": "pick"},
        action_executor=executed.append,
    )
    runtime.set_task("pick")

    runtime.step_once()

    assert executed == ["a0"]
    assert list(runtime.state.action_queue) == ["a1"]
    assert runtime.state.extra["g05_subtask"] == "pick cup"


def test_system2_rejects_checkpoint_without_predict_cot():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    with pytest.raises(ValueError, match="predict_cot=True"):
        G05PolicyAdapter(FakeG05Policy(predict_cot=False), system_mode="system2")


def test_system1_rejects_checkpoint_without_an_action_head():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    with pytest.raises(ValueError, match="both discrete_action=False and continuous_action=False"):
        G05PolicyAdapter(FakeG05Policy(predict_cot=True, discrete_action=False, continuous_action=False))


def test_system2_requires_structured_single_pass_hook():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    adapter = G05PolicyAdapter(FakeG05Policy(predict_cot=True))
    with pytest.raises(RuntimeError, match="predict_action_chunk_with_runtime"):
        adapter.select_action({}, RuntimeState(task="pick"))


def test_direct_subtask_selects_system1_on_system2_checkpoint():
    from lerobot.policies.g05.inference.g05_adapter import G05PolicyAdapter

    class SwitchablePolicy(FakeG05Policy):
        def __init__(self):
            super().__init__(predict_cot=True, continuous_action=True)

        def predict_action_chunk_with_runtime(self, observation, *, task, system_mode=None):
            self.calls.append(system_mode)
            return ("chunk", {"cot_text": "Subtask: should not be generated"})

    policy = SwitchablePolicy()
    adapter = G05PolicyAdapter(policy, GenerationConfig(enable_subtask=False))
    state = RuntimeState(task="pick")

    chunk = adapter.select_action({}, state)

    assert adapter.system_mode == "system1"
    assert policy.calls == ["system1"]
    assert chunk == "chunk"
    assert "cot_text" not in state.language_context
