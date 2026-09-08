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

from lerobot.configs.recipe import load_recipe
from lerobot.datasets.language_render import render_sample
from lerobot.policies.pi052.inference.pi052_adapter import PI052PolicyAdapter
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import split_plan_and_say


def _scratchpad_adapter():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            memory_scratchpad=True,
            recipe_path="recipes/subtask_mem.yaml",
            joint_subtask_conditioning=False,
        )
    )
    return PI052PolicyAdapter(policy=policy)


@pytest.mark.parametrize("now", [0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 8.0])
def test_scratchpad_runtime_prompt_matches_training_one_second_history(monkeypatch, now):
    from lerobot.policies.pi052.inference import pi052_adapter as module

    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    adapter = _scratchpad_adapter()
    updates = [(0.0, "cup in box", "pick plate"), (5.0, "plate in box", "pick spoon")]
    state = RuntimeState(
        task="clear table",
        language_context={"subtask": "pick plate"},
        extra={
            "pi052_scratchpad": {"task": "clear table", "updates": updates},
        },
    )
    rows = []
    for timestamp, memory, subtask in updates:
        rows.extend(
            [
                {"role": "assistant", "style": "memory", "content": memory, "timestamp": timestamp},
                {"role": "assistant", "style": "subtask", "content": subtask, "timestamp": timestamp},
            ]
        )
    recipe = load_recipe("src/lerobot/configs/recipes/subtask_mem.yaml").blend["high_level_memory_subtask"]
    training = render_sample(recipe=recipe, persistent=rows, events=[], t=now, sample_idx=0, task=state.task)
    assert adapter.build_messages("subtask", state) == training["messages"][:-1]


def test_scratchpad_combined_update_repeat_and_reset(monkeypatch):
    from lerobot.policies.pi052.inference import pi052_adapter as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    adapter = _scratchpad_adapter()
    state = RuntimeState(task="clear table")
    prompts = []

    def generate(kind, observation, runtime_state):
        assert kind == "subtask"  # no second memory-generation call
        prompts.append(adapter.build_messages(kind, runtime_state))
        return "Memory: Cup in box.\nSubtask: Pick up the plate."

    adapter.generate_text = generate
    adapter._regenerate_context({}, state)
    assert state.language_context == {"memory": "Cup in box.", "subtask": "Pick up the plate."}
    assert "Current subtask: \n" in prompts[0][0]["content"]
    now[0] = 12.0
    adapter._regenerate_context({}, state)
    assert "Current subtask: Pick up the plate.\n" in prompts[1][0]["content"]
    assert adapter.diag.repeat == 1
    state.set_context("subtask", None)  # existing scene-reset signal
    now[0] = 14.0
    adapter._regenerate_context({}, state)
    assert "Memory: \nCurrent subtask: \n" in prompts[2][0]["content"]
    assert state.extra["pi052_scratchpad"]["updates"] == [(14.0, "Cup in box.", "Pick up the plate.")]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "Subtask: pick cup",
        "Memory: \nSubtask: pick cup",
        "Memory: ok\nSubtask: ",
        "Memory: ok\nSubtask: pick cup\nSubtask: pick plate",
    ],
)
def test_scratchpad_invalid_response_does_not_partially_update_or_start_actions(response):
    adapter = _scratchpad_adapter()
    state = RuntimeState(task="clear table")
    adapter.generate_text = lambda *args: response
    with pytest.raises(ValueError, match="Scratchpad"):
        adapter._regenerate_context({}, state)
    assert state.language_context == {}
    with pytest.raises(ValueError, match="valid scratchpad"):
        adapter.select_action({}, state)


def test_scratchpad_reset_during_generation_discards_response():
    adapter = _scratchpad_adapter()
    state = RuntimeState(task="clear table")

    def generate(*args):
        state.set_context("subtask", "operator instruction")
        return "Memory: stale memory\nSubtask: stale instruction"

    adapter.generate_text = generate
    adapter._regenerate_context({}, state)
    assert state.language_context == {"subtask": "operator instruction"}
    assert not state.extra["pi052_scratchpad"]["updates"]


def test_scratchpad_actions_receive_only_validated_subtask(monkeypatch):
    from lerobot.policies.pi052.inference import pi052_adapter as module

    adapter = _scratchpad_adapter()
    adapter.generate_text = lambda *args: "Memory: Cup in box.\nSubtask: Pick up the plate."
    state = RuntimeState(task="clear table")
    adapter._regenerate_context({}, state)
    prompts = []

    def build(policy, messages, **kwargs):
        prompts.append(messages)
        return {"lang_tokens": "tokens", "lang_masks": "mask"}

    monkeypatch.setattr(module, "_build_text_batch", build)
    adapter.policy.predict_action_chunk = lambda batch: "actions"
    assert adapter.select_action({}, state) == "actions"
    assert prompts == [[{"role": "user", "content": "Pick up the plate."}]]


def test_pi052_adapter_builds_recipe_prompts_from_runtime_state():
    adapter = PI052PolicyAdapter(policy=object())
    state = RuntimeState(
        task="clean the kitchen",
        language_context={"memory": "cup moved", "plan": "pick then place"},
        extra={"prior_subtask": "pick the cup"},
    )

    assert adapter.build_messages("subtask", state) == [{"role": "user", "content": "clean the kitchen"}]
    assert adapter.build_messages("memory", state) == [
        {"role": "user", "content": "clean the kitchen"},
        {"role": "assistant", "content": "Previous memory: cup moved"},
        {"role": "user", "content": "Completed subtask: pick the cup"},
    ]
    assert adapter.build_messages("interjection", state, user_text="wait") == [
        {"role": "user", "content": "clean the kitchen"},
        {"role": "assistant", "content": "Previous plan:\npick then place"},
        {"role": "user", "content": "wait"},
    ]


def test_pi052_adapter_strips_say_markers_from_plan_text():
    adapter = PI052PolicyAdapter(policy=object())
    text = "Move to the sink. <say>heading to the sink</say>"

    assert split_plan_and_say(text) == ("Move to the sink.", "heading to the sink")
    assert adapter.plan_from_text(text) == "Move to the sink."


def test_rollout_language_cli_smoke_does_not_load_model(monkeypatch):
    """lerobot-rollout dispatches language flags to the adapter-based runtime."""
    from lerobot.runtime import cli
    from lerobot.scripts import lerobot_rollout

    fake_policy = SimpleNamespace(config=SimpleNamespace(device="cpu", type="pi052"))

    monkeypatch.setattr(
        cli,
        "_load_policy_and_preprocessor",
        lambda policy_path, **kwargs: (fake_policy, None, None),
    )
    monkeypatch.setattr(cli, "_run_repl", lambda runtime, **kwargs: 0)

    assert lerobot_rollout.main(["--policy.path=fake", "--no_robot", "--task=clean", "--max_ticks=0"]) == 0


def test_rollout_language_dispatch_preserves_standard_molmoact2_path(monkeypatch):
    """MolmoAct2 only opts into open prompting when a language flag is present."""
    from lerobot.scripts import lerobot_rollout

    standard = [
        "--policy.path=lerobot/MolmoAct2-SO100_101-LeRobot",
        "--robot.type=so101_follower",
        "--task=pick up the cube",
    ]
    assert not lerobot_rollout._uses_language_runtime(standard)
    assert lerobot_rollout._uses_language_runtime([*standard, "--direct_subtask"])
    assert lerobot_rollout._uses_language_runtime(["--policy.path=lerobot/pi052_robocasa", "--sim"])

    standard_calls = []
    monkeypatch.setattr(lerobot_rollout, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_rollout, "rollout", lambda: standard_calls.append(True))
    lerobot_rollout.main(standard)
    assert standard_calls == [True]
