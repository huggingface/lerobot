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

"""Local asynchronous compatibility and worker/hold transition regressions."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
import torch

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from lerobot.inference import ObservationSnapshot, RTCInferenceConfig, rtc
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout import RolloutConfig, context
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from tests.inference.rtc_helpers import ControlledPolicy, make_engine, observe, stop
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.utils import wait_for


@pytest.mark.parametrize("temporal_contract", [False, True])
def test_unsupported_local_policy_explains_sync_fallback_before_hardware(monkeypatch, temporal_contract):
    policy = ControlledPolicy()
    if temporal_contract:
        spec = replace(policy.chunk_inference_spec(), current_observation_only=False)
        monkeypatch.setattr(policy, "chunk_inference_spec", lambda: spec)
        reason = "temporal observation sampling"
    else:
        policy.config.n_obs_steps = 2
        reason = "n_obs_steps=1"
    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        policy=ACTConfig(device="cpu", pretrained_path=Path("unused")),
        inference=RTCInferenceConfig(rtc=RTCConfig(enabled=False)),
        device="cpu",
    )
    monkeypatch.setattr(context, "_load_pretrained_policy", lambda _: policy)

    def unexpected_hardware(*args, **kwargs):
        pytest.fail("Policy compatibility must be checked before constructing hardware")

    monkeypatch.setattr(context, "make_robot_from_config", unexpected_hardware)
    for construct in (
        lambda: context.build_rollout_context(cfg, Event()),
        lambda: make_engine(policy=policy),
    ):
        with pytest.raises(ValueError, match="--inference.type=sync") as raised:
            construct()
        assert reason in str(raised.value)
        assert isinstance(raised.value.__cause__, ValueError)
        assert reason in str(raised.value.__cause__)


def test_local_rtc_without_hold_runs_actions_but_rejects_language(caplog):
    robot = MockRobot(MockRobotConfig(n_motors=2))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)
    engine = make_engine(robot=wrapper)
    try:
        assert not wrapper.supports_hold
        assert not engine.supports_text_queries
        assert not engine.ask("what do you see?")
        assert "supported robot position hold" in caplog.text
        with pytest.raises(ValueError, match="supported position hold"):
            engine.start_autosteer("pick", 1)
        engine.start()
        engine.resume()
        observe(engine)
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        assert engine.get_action(None).shape == (2,)
        assert not engine.failed
    finally:
        stop(engine)
        robot.disconnect()


def test_task_change_during_compile_warmup_needs_no_hold_acknowledgment():
    engine = make_engine(use_torch_compile=True, compile_warmup_inferences=1)
    engine.start()
    try:
        engine.resume()
        observe(engine)
        assert wait_for(lambda: len(engine._policy.calls) == 1)
        assert not engine.ready
        assert engine._runtime.pending is not None
        assert engine.set_task("new task")
        assert not engine._hold.requested
        assert engine._hold.deadline is None
        # The control thread only waits for readiness here; it does not acknowledge holds.
        engine._policy.release.release(3)
        assert wait_for(lambda: engine.ready)
        assert wait_for(lambda: len(engine._policy.calls) == 4)
        assert engine.action_queue.empty(), "warmup results must never become motion"
        assert not engine.dispatch_allowed()
        assert engine._policy.calls[0] == (None, 0)
        assert [delay for _, delay in engine._policy.calls[1:3]] == [0, 1]
        assert all(prefix.shape == (4, 2) for prefix, _ in engine._policy.calls[1:3])
        assert engine._runtime.pending.observation.task == "new task"
        assert not engine.failed
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        assert engine.get_action(None) is not None
        assert engine.dispatched_task == "new task"
    finally:
        stop(engine)


@pytest.mark.parametrize(
    "acknowledge_first,autosteer,action_timeout,language_timeout",
    [(False, False, 2, 8), (True, True, 8, 2)],
)
def test_instruction_hold_language_upgrade_preserves_ack_and_bounded_budget(
    acknowledge_first, autosteer, action_timeout, language_timeout
):
    engine = make_engine(action_timeout_s=action_timeout, language_timeout_s=language_timeout)
    now = [100.0]
    engine._runtime.clock = lambda: now[0]
    engine.resume()
    assert engine._runtime.begin(ObservationSnapshot({}, now[0], "pick", 0)) is not None
    assert engine.set_task("new task")
    generation, started = engine._runtime.generation, now[0]
    if acknowledge_first:
        engine.acknowledge_hold()
    now[0] += 0.5
    if autosteer:
        engine.start_autosteer("pick", 1)
        engine.pump_query({})
    else:
        assert engine.ask("what?")
    assert engine._hold.reason == "Language request"
    assert engine._hold.acknowledged == acknowledge_first
    assert engine._runtime.generation == generation
    deadline = started + max(action_timeout, language_timeout)
    assert engine._hold.deadline == deadline
    now[0] = deadline - 0.1
    for _ in range(3):
        assert not engine.dispatch_allowed()
        engine.acknowledge_hold()
        assert engine._hold.deadline == deadline
    assert not engine.failed
    now[0] = deadline + 0.1
    assert not engine.dispatch_allowed()
    assert engine.failed
    assert "Language request deadline exceeded" in engine.failure_traceback


def test_upgraded_language_hold_requires_fresh_query_and_action_observations():
    engine = make_engine()
    generating, finish = Event(), Event()
    text_observations = []

    def text(batch):
        text_observations.append(batch["observation.state"].clone())
        generating.set()
        assert finish.wait(3)
        return "answer"

    engine._policy.generate_text = text
    engine.start()
    try:
        engine.resume()
        observe(engine, 1)
        assert wait_for(lambda: len(engine._policy.calls) == 1)
        assert engine.set_task("new task")
        generation = engine._runtime.generation
        assert engine.ask("what?")
        engine._policy.release.release()
        assert not generating.wait(0.02), "language cannot precede the control-thread hold"
        engine.acknowledge_hold()
        assert engine._runtime.generation == generation
        assert not generating.wait(0.02), "language cannot reuse the pre-hold capture"
        observe(engine, 2)
        assert generating.wait(3)
        assert engine.action_queue.empty(), "old-task inference cannot restore motion"
        torch.testing.assert_close(text_observations[0], torch.full((1, 2), 2.0))
        finish.set()
        assert wait_for(lambda: not engine._hold.requested)
        assert len(engine._policy.calls) == 1, "actions need a post-query capture"
        observe(engine, 3)
        # The worker clones the observation after publishing its call metadata.
        # Wait for the value inspected below, not the earlier metadata append.
        assert wait_for(lambda: len(engine._policy.action_observations) == 2)
        torch.testing.assert_close(engine._policy.action_observations[1], torch.full((1, 2), 3.0))
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        assert engine.get_action(None) is not None
        assert engine.dispatched_task == "new task"
        assert not engine.failed
    finally:
        finish.set()
        stop(engine)


@pytest.mark.parametrize("transition", ["pause_resume", "active_reset"])
def test_transition_discards_observation_already_copied_by_worker(monkeypatch, transition):
    engine = make_engine()
    preparing, release, policy_reset = Event(), Event(), Event()
    original = rtc.build_dataset_frame

    def build_frame(*args, **kwargs):
        if not preparing.is_set():
            preparing.set()
            assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(rtc, "build_dataset_frame", build_frame)
    monkeypatch.setattr(engine._policy, "reset", policy_reset.set)
    engine.start()
    try:
        engine.resume()
        observe(engine, 1)
        assert preparing.wait(3)
        old_generation = engine._runtime.generation
        if transition == "active_reset":
            engine.reset()
        else:
            engine.pause()
            engine.resume()
        assert engine._runtime.active
        assert engine._policy_active.is_set()
        assert engine._runtime.generation != old_generation
        observe(engine, 2)
        release.set()
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        if transition == "active_reset":
            assert policy_reset.is_set()
        assert engine._policy.action_observations
        for observation in engine._policy.action_observations:
            torch.testing.assert_close(observation, torch.full((1, 2), 2.0))
        assert not engine.failed
    finally:
        release.set()
        stop(engine)


def test_trained_rtc_tolerates_one_overlap_miss_but_reports_persistent_delay():
    engine = make_engine(rtc_config=RTCConfig(mode="trained", execution_horizon=4))
    now = [time.monotonic()]
    engine._runtime.clock = lambda: now[0]
    engine._robot.observation_time = now[0]
    engine.start()
    try:
        engine.resume()
        observe(engine)
        engine._policy.release.release()
        assert wait_for(lambda: len(engine._policy.calls) >= 2)
        assert engine.action_queue.qsize() == 20
        for miss in range(5):
            assert engine._policy.calls[-1][1] == 4
            now[0] += 0.2  # Six action intervals exceed the four-step checkpoint prefix.
            engine._robot.observation_time = now[0]
            observe(engine)
            engine._policy.release.release()
            if miss < 4:
                assert wait_for(lambda miss=miss: len(engine._policy.calls) >= miss + 3)
                assert not engine.failed, "a transient overlap miss should remain retryable"
        assert wait_for(lambda: engine.failed)
        assert "repeatedly exceeded the conditioned/checkpoint overlap" in engine.failure_traceback
        assert not engine.dispatch_allowed()
    finally:
        stop(engine)
