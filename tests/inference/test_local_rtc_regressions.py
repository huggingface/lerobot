# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Local asynchronous compatibility and worker/hold transition regressions."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Semaphore
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference.contracts import ObservationSnapshot
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rollout import RolloutConfig, context
from lerobot.rollout.inference import rtc
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from tests.mocks.mock_robot import MockRobot, MockRobotConfig


class Pipeline:
    steps: tuple = ()

    def __init__(self, width: int | None = None):
        self.width = width

    def __call__(self, value):
        return value if self.width is None else value[..., : self.width]

    def reset(self):
        pass


class ControlledPolicy:
    chunk_inference_spec = PreTrainedPolicy.chunk_inference_spec
    drop_queued_actions = PreTrainedPolicy.drop_queued_actions
    _action_queue_attrs = PreTrainedPolicy._action_queue_attrs

    def __init__(self, width: int = 2):
        self.width = width
        self.config = SimpleNamespace(
            n_obs_steps=1,
            chunk_size=20,
            n_action_steps=20,
            rtc_training_max_delay=4,
            action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        )
        self.release = Semaphore(0)
        self.calls: list[tuple[torch.Tensor | None, int]] = []
        self.action_observations: list[torch.Tensor] = []
        self.text_observations: list[torch.Tensor] = []

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None):
        self.calls.append((prev_chunk_left_over, inference_delay))
        self.action_observations.append(batch["observation.state"].clone())
        assert self.release.acquire(timeout=3), "test must release the model call"
        return torch.ones(1, 20, self.width)

    def generate_text(self, batch):
        self.text_observations.append(batch["observation.state"].clone())
        return "answer"

    def reset(self):
        pass

    def supports_rtc(self):
        return True

    def supports_text_generation(self):
        return True


def make_engine(policy=None, robot=None, postprocessor=None, **kwargs):
    policy = policy or ControlledPolicy()
    robot = robot or SimpleNamespace(robot_type="mock", action_features={}, supports_hold=True)
    return RTCInferenceEngine(
        policy=policy,
        preprocessor=Pipeline(),
        postprocessor=postprocessor or Pipeline(width=2),
        robot_wrapper=robot,
        rtc_config=kwargs.pop("rtc_config", RTCConfig(enabled=True, execution_horizon=4)),
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        task="pick",
        fps=30,
        device="cpu",
        **kwargs,
    )


def wait_for(predicate: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def observe(engine, value=0.0):
    engine.notify_observation({"j1.pos": value, "j2.pos": value})


def stop(engine):
    engine._policy.release.release(20)
    engine.stop()


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


def test_local_rtc_preserves_padded_model_prefix_and_canonical_actions():
    engine = make_engine(policy=ControlledPolicy(width=5))
    engine.start()
    try:
        engine.resume()
        observe(engine)
        engine._policy.release.release()
        assert wait_for(lambda: len(engine._policy.calls) >= 2)
        prefix, _ = engine._policy.calls[1]
        assert prefix.shape == (4, 5)
        assert engine.get_action(None).shape == (2,)
        assert not engine.failed
    finally:
        stop(engine)


@pytest.mark.parametrize("wrong_canonical_width", [True, False])
def test_local_rtc_rejects_invalid_canonical_or_changing_model_width(wrong_canonical_width):
    policy = ControlledPolicy(width=5)
    engine = make_engine(policy=policy, postprocessor=Pipeline(width=1 if wrong_canonical_width else 2))
    engine.start()
    try:
        engine.resume()
        observe(engine)
        policy.release.release()
        if not wrong_canonical_width:
            assert wait_for(lambda: len(policy.calls) >= 2)
            policy.width = 6
            policy.release.release()
        assert wait_for(lambda: engine.failed)
        expected_error = "must return action shape" if wrong_canonical_width else "width changed"
        assert expected_error in engine.failure_traceback
        traceback = engine.failure_traceback
        assert "Traceback" in traceback
        assert not engine.dispatch_allowed()
        assert engine.get_action(None) is None
        assert engine.failure_traceback == traceback, "motor checks must preserve worker traceback"
    finally:
        stop(engine)


def test_compile_warmup_exercises_prefix_and_delay_without_queueing_motion():
    # PyTorch 2.11's CUDA build without a GPU registers a worker-owned fake CUDA
    # guard globally. Exiting the compiling thread leaves a dangling guard for
    # later tests (c10/core/impl/DeviceGuardImplInterface.cpp). Keep real threaded
    # compilation and its assertions, but contain that upstream state in a child.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.inference.test_local_rtc_regressions import _check_compile_warmup; "
            "_check_compile_warmup()",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_compile_warmup():
    torch._dynamo.reset()
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    def predict(batch, *, inference_delay=0, prev_chunk_left_over=None):
        state = batch["observation.state"]
        output = state.new_zeros(1, 20, 2) + state.mean()
        if prev_chunk_left_over is not None:
            output[:, :4] += prev_chunk_left_over
        if inference_delay:
            output = output + inference_delay
        return output

    compiled = torch.compile(predict, backend=backend)
    policy = ControlledPolicy()
    waiting = Event()

    def call(batch, *, inference_delay=0, prev_chunk_left_over=None):
        policy.calls.append((prev_chunk_left_over, inference_delay))
        if len(policy.calls) > 3:
            waiting.set()
            assert policy.release.acquire(timeout=3)
        return compiled(batch, inference_delay=inference_delay, prev_chunk_left_over=prev_chunk_left_over)

    policy.predict_action_chunk = call
    engine = make_engine(policy=policy, use_torch_compile=True, compile_warmup_inferences=1)
    engine.start()
    try:
        engine.resume()
        observe(engine)
        assert wait_for(lambda: engine.ready)
        assert waiting.wait(3)
        assert not engine.failed
        assert engine.action_queue.empty()
        assert not engine.dispatch_allowed()
        assert len(policy.calls) == 4
        assert policy.calls[0] == (None, 0)
        assert policy.calls[1][0].shape == (4, 2)
        assert policy.calls[1][1] == 0
        assert policy.calls[2][0].shape == (4, 2)
        assert policy.calls[2][1] == 1
        graph_count = len(graphs)
        with torch.inference_mode(False), torch.no_grad():
            for prefix, delay in policy.calls[:3]:
                compiled(
                    {"observation.state": torch.zeros(1, 2)},
                    inference_delay=delay,
                    prev_chunk_left_over=prefix,
                )
        assert len(graphs) == graph_count, "representative graphs should already be warm"
    finally:
        stop(engine)
        torch._dynamo.reset()


def test_language_deadline_uses_runtime_clock_and_latches_until_restart():
    shutdown = Event()
    engine = make_engine(language_timeout_s=2, shutdown_event=shutdown)
    now = [100.0]
    engine._runtime.clock = lambda: now[0]
    engine.resume()
    assert engine.ask("what?")
    assert not engine.dispatch_allowed()
    now[0] += 2.1
    assert not engine.dispatch_allowed()
    assert engine.failed
    assert "Language request deadline exceeded" in engine.failure_traceback
    assert not shutdown.is_set()
    engine.acknowledge_hold()
    assert shutdown.is_set()
    engine.resume()
    assert not engine._runtime.active


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
        assert not engine._hold_requested
        assert engine._language_deadline is None
        # The control thread only waits for readiness here; it does not acknowledge holds.
        engine._policy.release.release(3)
        assert wait_for(lambda: engine.ready)
        assert wait_for(lambda: len(engine._policy.calls) == 4)
        assert engine.action_queue.empty(), "warmup results must never become motion"
        assert engine._runtime.pending.observation.task == "new task"
        assert not engine.failed
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        assert engine.get_action(None) is not None
        assert engine.dispatched_task == "new task"
    finally:
        stop(engine)


@pytest.mark.parametrize("acknowledge_first", [False, True])
@pytest.mark.parametrize("autosteer", [False, True])
@pytest.mark.parametrize("action_timeout,language_timeout", [(2, 8), (8, 2)])
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
    assert engine._hold_reason == "Language request"
    assert engine._hold_acknowledged.is_set() == acknowledge_first
    assert engine._runtime.generation == generation
    deadline = started + max(action_timeout, language_timeout)
    assert engine._language_deadline == deadline
    now[0] = deadline - 0.1
    for _ in range(3):
        assert not engine.dispatch_allowed()
        engine.acknowledge_hold()
        assert engine._language_deadline == deadline
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
        assert wait_for(lambda: not engine._hold_requested)
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


def test_pause_cancels_pending_language_hold_and_its_deadline():
    engine = make_engine(language_timeout_s=2)
    now = [100.0]
    engine._runtime.clock = lambda: now[0]
    engine.resume()
    assert engine.ask("what?")
    assert engine._hold_requested
    engine.pause()
    now[0] += 10
    engine.resume()
    assert not engine.has_pending_query
    assert not engine._hold_requested
    assert engine._language_deadline is None
    assert not engine.dispatch_allowed()
    assert not engine.failed, "the cancelled hold must not fault a newly resumed run"


def test_pause_resume_discards_observation_already_copied_by_worker(monkeypatch):
    engine = make_engine()
    preparing, release = Event(), Event()
    original = rtc.build_dataset_frame

    def build_frame(*args, **kwargs):
        if not preparing.is_set():
            preparing.set()
            assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(rtc, "build_dataset_frame", build_frame)
    engine.start()
    try:
        engine.resume()
        observe(engine, 1)
        assert preparing.wait(3)
        engine.pause()
        engine.resume()
        observe(engine, 2)
        release.set()
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        torch.testing.assert_close(engine._policy.action_observations[0], torch.full((1, 2), 2.0))
        assert not engine.failed
    finally:
        release.set()
        stop(engine)


def test_active_reset_discards_observation_copied_before_reset(monkeypatch):
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
        # Episodic rollout may reset while active: there is no pause for begin()
        # to reject the observation already copied by the worker.
        engine.reset()
        assert engine._runtime.active
        assert engine._policy_active.is_set()
        assert engine._runtime.generation != old_generation
        observe(engine, 2)
        release.set()
        engine._policy.release.release()
        assert wait_for(lambda: engine.action_queue.qsize() > 0)
        assert policy_reset.is_set()
        assert engine._policy.action_observations
        for observation in engine._policy.action_observations:
            torch.testing.assert_close(observation, torch.full((1, 2), 2.0))
        assert not engine.failed
    finally:
        release.set()
        stop(engine)


@pytest.mark.parametrize("pause_first", [False, True])
def test_reset_during_text_reports_cancellation_and_requires_fresh_actions(pause_first):
    engine = make_engine(rtc_queue_threshold=-1)
    generating, finish = Event(), Event()

    def text(batch):
        generating.set()
        assert finish.wait(3)
        return "old answer"

    engine._policy.generate_text = text
    engine.start()
    try:
        engine.resume()
        assert engine.ask("what?")
        engine.acknowledge_hold()
        observe(engine)
        assert generating.wait(3)
        if pause_first:
            engine.pause()
        engine.reset()
        engine.resume()
        finish.set()
        assert wait_for(lambda: not engine._query_in_flight)
        delivered = []
        engine.set_answer_observer(delivered.append)
        engine.pump_query()
        engine.pump_query()
        assert len(delivered) == 1
        assert delivered[0].answer is None
        assert "cancelled" in delivered[0].error
        assert not engine.dispatch_allowed()
        assert not engine.failed
    finally:
        finish.set()
        stop(engine)


def test_language_waits_for_observation_after_hold_acknowledgment():
    engine = make_engine(rtc_queue_threshold=-1)
    engine.start()
    try:
        engine.resume()
        observe(engine, 1)
        assert engine.ask("what?")
        engine.acknowledge_hold()
        with engine._obs_lock:
            assert engine._obs_holder["obs"] is None
        observe(engine, 2)
        assert wait_for(lambda: bool(engine._ready_answers))
        assert len(engine._policy.text_observations) == 1
        torch.testing.assert_close(engine._policy.text_observations[0], torch.full((1, 2), 2.0))
        assert not engine.dispatch_allowed()
    finally:
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


def test_engine_constructor_failure_disconnects_already_connected_robot(monkeypatch):
    robot = MockRobot(MockRobotConfig())
    cfg = RolloutConfig(
        robot=robot.config,
        policy=ACTConfig(device="cpu", pretrained_path=Path("unused")),
        device="cpu",
    )
    monkeypatch.setattr(context, "_load_pretrained_policy", lambda _: torch.nn.Linear(3, 3))
    monkeypatch.setattr(context, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(
        context,
        "make_pre_post_processors",
        lambda **_: (PolicyProcessorPipeline([]), PolicyProcessorPipeline([])),
    )

    def fail(*args, **kwargs):
        assert robot.is_connected
        raise ValueError("invalid engine configuration")

    monkeypatch.setattr(context, "create_inference_engine", fail)
    with pytest.raises(ValueError, match="invalid engine configuration"):
        context.build_rollout_context(cfg, Event())
    assert not robot.is_connected
