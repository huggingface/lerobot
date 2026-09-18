# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import threading
import time

import pytest

from lerobot.rollout.agent.control import ControlConfig, HybridRuntime


def until(runtime, predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        runtime.step_once()
        time.sleep(0.001)


@pytest.fixture
def make_runtime():
    runtimes = []

    def make(predict=None, **kwargs):
        positions = {"joint.pos": 0.0, "gripper.pos": 0.0}
        executed, records, events = [], [], []

        def execute(action):
            positions.update(action)
            executed.append(dict(action))
            return dict(action)

        runtime = HybridRuntime(
            predict=predict or (lambda request: [{"joint.pos": 4.0, "gripper.pos": 0.0}] * 4),
            observe=lambda: dict(positions),
            execute=execute,
            config=ControlConfig(fps=10, joint_limits=dict.fromkeys(positions, (-10, 10))),
            record=lambda *args: records.append(args),
            event=events.append,
            **kwargs,
        )
        runtime.set_task("pick cup")
        runtime.state.mode = "action"
        runtimes.append(runtime)
        return runtime, executed, records, events

    yield make
    for runtime in runtimes:
        runtime.stop()
        runtime._on_shutdown()


def test_network_delay_does_not_block_actions(make_runtime):
    release = threading.Event()

    def supervisor(snapshot):
        release.wait(2)
        return []

    runtime, executed, _, _ = make_runtime(supervisor=supervisor)
    started = time.monotonic()
    until(runtime, lambda: len(executed) >= 3)
    assert time.monotonic() - started < 1
    release.set()


def test_takeover_discards_queued_policy_actions_and_requires_handback(make_runtime):
    runtime, executed, records, events = make_runtime()
    until(runtime, lambda: len(executed) == 1)
    command = runtime.submit("move_joints", {"targets": {"joint.pos": 1.0}, "duration_s": 0.1})
    runtime.step_once()
    assert "error" not in command["result"]
    assert executed[-1]["joint.pos"] == 1.0
    assert runtime.state.mode == "paused"
    count = len(executed)
    runtime.step_once()
    assert len(executed) == count
    assert records[-1][3] == "hold"
    runtime.submit("resume_policy", {})
    until(runtime, lambda: len(executed) > count)
    assert any(event["kind"] == "motion_finished" for event in events)


@pytest.mark.parametrize("raises", [False, True])
def test_stale_inference_cannot_override_tool_takeover(make_runtime, raises):
    started, release = threading.Event(), threading.Event()

    def predict(snapshot):
        started.set()
        release.wait(2)
        if raises:
            raise RuntimeError("old inference failed")
        return [{"joint.pos": 9.0, "gripper.pos": 0.0}]

    runtime, executed, _, events = make_runtime(predict)
    runtime.step_once()
    assert started.wait(1)
    runtime.submit("move_joints", {"targets": {"joint.pos": 1.0}, "duration_s": 0.5})
    runtime.step_once()
    release.set()
    until(runtime, lambda: any(event["kind"] == "proposal_discarded" for event in events))
    assert runtime.source == "tool"
    assert all(action["joint.pos"] <= 1 for action in executed)
    assert runtime.status()["last_error"] is None


def test_operator_instruction_invalidates_late_decision(make_runtime):
    runtime, _, _, _ = make_runtime()
    runtime.step_once()
    snapshot = runtime.snapshot()
    runtime.set_task("pick screwdriver")
    command = runtime.submit("steer", {"instruction": "pick cup"}, revision=snapshot["revision"])
    runtime.step_once()
    assert "Stale decision" in command["result"]["error"]
    assert runtime.state.task == "pick screwdriver"


def test_old_observation_rejects_motion_but_allows_pause(make_runtime):
    runtime, _, _, _ = make_runtime()
    command = runtime.submit(
        "move_joints", {"targets": {"joint.pos": 1.0}, "duration_s": 1.0}, observed_at=time.monotonic() - 20
    )
    runtime.step_once()
    assert "Stale decision" in command["result"]["error"]
    pause = runtime.submit("pause", {}, revision=-1, observed_at=0)
    runtime.step_once()
    assert "error" not in pause["result"]
    assert runtime.state.mode == "paused"


@pytest.mark.parametrize("target", [float("nan"), float("inf"), 100.0])
def test_invalid_motion_does_not_change_controller(make_runtime, target):
    runtime, _, _, _ = make_runtime()
    command = runtime.submit("move_joints", {"targets": {"joint.pos": target}, "duration_s": 1})
    runtime.step_once()
    assert "error" in command["result"]
    assert runtime.source == "policy"


def test_recording_contains_actual_bounded_command_and_original_proposal(make_runtime):
    runtime, executed, records, _ = make_runtime()
    until(runtime, lambda: bool(executed))
    assert records[0][1]["joint.pos"] == 4
    assert records[0][2]["joint.pos"] == 1.5
    assert executed[0] == records[0][2]


def test_observation_failure_pauses_and_flushes_queue(make_runtime):
    runtime, executed, _, _ = make_runtime()
    until(runtime, lambda: bool(executed))
    runtime.observation_provider = lambda: None
    runtime.step_once()
    assert runtime.state.mode == "paused"
    assert not runtime.state.action_queue
    assert runtime.snapshot() is None


def test_episode_labels_require_evidence_and_reset_invalidates_results(make_runtime):
    outcomes = []
    runtime, executed, _, _ = make_runtime(finish=outcomes.append)
    until(runtime, lambda: bool(executed))
    bad = runtime.submit("finish_episode", {"outcome": "success"})
    runtime.step_once()
    assert "evidence" in bad["result"]["error"]
    runtime.submit("finish_episode", {"outcome": "failure", "evidence": "cup remained outside bin"})
    runtime.step_once()
    assert outcomes[0]["outcome"] == "failure"
    assert runtime.state.mode == "paused"
    assert not runtime.state.action_queue
    assert runtime.status()["episodes"] == 1


def test_steering_preserves_overall_goal_for_recipe_training(make_runtime):
    runtime, executed, records, _ = make_runtime()
    runtime.set_task("put all objects in the bin")
    command = runtime.submit("steer", {"instruction": "grasp the tape roll"})
    runtime.step_once()
    assert "error" not in command["result"]
    assert runtime.state.task == "put all objects in the bin"
    assert runtime.instruction() == "grasp the tape roll"
    until(runtime, lambda: bool(executed))
    assert records[0][4] == "grasp the tape roll"
    assert records[0][0]["task"] == "put all objects in the bin"
