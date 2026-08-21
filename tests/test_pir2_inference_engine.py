#!/usr/bin/env python

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

"""Tests for the piR2 inference engine, driven by a fake policy and robot.

The engine's contract is that a background thread keeps one buffer alive and streams finished
actions to the control loop, so these tests check the loop's bookkeeping (warm start once,
delay estimation, emission counts, guards) rather than anything about denoising quality.
"""

import logging
import time
from collections import deque
from threading import Event

import pytest
import torch

from lerobot.rollout.inference.pir2 import PiR2InferenceEngine, estimate_pir2_delay

CHUNK_SIZE = 16
ACTION_DIM = 6


class _FakeConfig:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.rtc_training_schedule = "staircase"


class _FakeSlowChannel:
    def __init__(self, captured_at):
        self.captured_at = captured_at


class _FakePolicy:
    """Stands in for a staircase-trained pi0.5: records calls, returns recognizable actions."""

    def __init__(self):
        self.config = _FakeConfig()
        self.warm_starts = 0
        self.substep_delays: list[int] = []
        self.slow_channel_calls = 0
        self.seen_prefixes: list[object] = []

    def reset(self):
        pass

    def encode_slow_channel(self, batch):
        self.slow_channel_calls += 1
        return _FakeSlowChannel(captured_at=time.perf_counter())

    def warm_start_realtime_buffer(self, slow, delay):
        self.warm_starts += 1
        return torch.zeros(1, CHUNK_SIZE, ACTION_DIM)

    def realtime_substep(self, slow, buffer, delay):
        self.substep_delays.append(delay)
        self.seen_prefixes.append(slow)
        # Tag every emitted action with the call index so the test can spot duplicates.
        emitted = torch.full((1, delay, ACTION_DIM), float(len(self.substep_delays)))
        return emitted, buffer


class _IdentityProcessor:
    def __init__(self):
        self.steps = []

    def __call__(self, batch):
        return batch

    def reset(self):
        pass


class _FakeRobot:
    robot_type = "fake"
    action_features = {f"joint_{i}.pos": float for i in range(ACTION_DIM)}


def _make_engine(policy=None, **kwargs):
    return PiR2InferenceEngine(
        policy=policy or _FakePolicy(),
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        hw_features={},
        task="do the thing",
        fps=30.0,
        device="cpu",
        shutdown_event=Event(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Delay estimation
# ---------------------------------------------------------------------------


def test_delay_estimate_is_one_before_any_measurement():
    assert estimate_pir2_delay(deque(), 1 / 30, 8) == 1


@pytest.mark.parametrize(
    ("latency_s", "expected"),
    [
        (0.003, 1),  # Faster than a control tick still emits one action per call.
        (0.033, 1),
        (0.070, 2),
        (0.100, 3),
    ],
)
def test_delay_estimate_rounds_latency_to_control_steps(latency_s, expected):
    assert estimate_pir2_delay(deque([latency_s] * 5), 1 / 30, 8) == expected


def test_delay_estimate_is_clamped_to_the_schedule_limit():
    assert estimate_pir2_delay(deque([10.0]), 1 / 30, 8) == 8


def test_delay_estimate_uses_the_mean_not_the_max():
    # One slow call among fast ones should not permanently inflate d.
    window = deque([0.003] * 9 + [0.3])
    assert estimate_pir2_delay(window, 1 / 30, 25) == 1


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_engine_rejects_a_policy_without_the_pir2_entry_points():
    class _Bare:
        config = _FakeConfig()

    with pytest.raises(NotImplementedError, match="does not support piR2"):
        _make_engine(policy=_Bare())


def test_engine_rejects_a_prefix_trained_checkpoint():
    policy = _FakePolicy()
    policy.config.rtc_training_schedule = "prefix"
    with pytest.raises(ValueError, match="rtc_training_schedule=staircase"):
        _make_engine(policy=policy)


def test_max_delay_never_exceeds_half_the_chunk():
    engine = _make_engine(max_delay=CHUNK_SIZE)
    assert engine._max_delay == CHUNK_SIZE // 2  # noqa: SLF001


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


def _run_iterations(engine, policy, iterations):
    """Drive the loop body directly, avoiding thread-timing flakiness in tests."""
    engine._obs_holder = {"obs": {}, "robot_type": "fake"}  # noqa: SLF001
    engine.notify_observation({})
    engine.resume()
    if engine._slow is None:  # noqa: SLF001
        # Stand in for the VLM thread, which the loop cannot run without.
        engine._slow = policy.encode_slow_channel({})  # noqa: SLF001
    for _ in range(iterations):
        engine._shutdown_event.clear()  # noqa: SLF001
        _single_iteration(engine)


def _single_iteration(engine):
    """One pass of ``_denoise_loop``, stopped after a single substep."""
    stop_after_one = Event()
    original = engine._policy.realtime_substep  # noqa: SLF001

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        stop_after_one.set()
        engine._shutdown_event.set()  # noqa: SLF001
        return result

    engine._policy.realtime_substep = wrapped  # noqa: SLF001
    engine._denoise_loop()  # noqa: SLF001
    engine._policy.realtime_substep = original  # noqa: SLF001
    assert stop_after_one.is_set()


def test_buffer_is_warm_started_once_and_then_carried_across_calls():
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    _run_iterations(engine, policy, 3)

    assert policy.warm_starts == 1
    assert len(policy.substep_delays) == 3


def test_substeps_reuse_one_cached_prefix_instead_of_re_encoding():
    # The whole point of the slow channel: the backbone runs on its own thread, and the denoising
    # loop never pays for it.
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    engine._slow = policy.encode_slow_channel({})  # noqa: SLF001

    _run_iterations(engine, policy, 3)

    assert policy.slow_channel_calls == 1
    assert len(policy.substep_delays) == 3
    assert all(prefix is engine._slow for prefix in policy.seen_prefixes)  # noqa: SLF001


def test_denoise_loop_waits_for_the_first_cache():
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    engine._obs_holder = {"obs": {}, "robot_type": "fake"}  # noqa: SLF001
    engine.resume()

    # No cache published: the loop must not invent one by calling the backbone itself.
    engine._shutdown_event.set()  # noqa: SLF001
    engine._denoise_loop()  # noqa: SLF001

    assert policy.slow_channel_calls == 0
    assert policy.substep_delays == []


def test_a_badly_stale_prefix_is_reported(caplog):
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    # Older than a whole chunk at 30 fps, so the plan behind the clean front is out of date.
    engine._slow = _FakeSlowChannel(captured_at=time.perf_counter() - 2 * CHUNK_SIZE / 30.0)  # noqa: SLF001

    with caplog.at_level(logging.WARNING):
        _run_iterations(engine, policy, 1)

    assert "control steps old" in caplog.text
    # Stale is degraded, not fatal: the substep still runs.
    assert len(policy.substep_delays) == 1


def test_a_fresh_prefix_is_not_reported_as_stale(caplog):
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)

    with caplog.at_level(logging.WARNING):
        _run_iterations(engine, policy, 1)

    assert "control steps old" not in caplog.text


def test_every_substep_emits_exactly_delay_actions():
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    _run_iterations(engine, policy, 2)

    assert engine.pending_actions() == sum(policy.substep_delays)
    action = engine.get_action(None)
    assert action.shape == (ACTION_DIM,)


def test_emitted_actions_are_handed_out_in_order_without_duplicates():
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    _run_iterations(engine, policy, 3)

    # Each fake substep tags its actions with its call index, so the tags must be non-decreasing.
    tags = []
    while (action := engine.get_action(None)) is not None:
        tags.append(action[0].item())
    assert tags == sorted(tags)
    assert len(tags) == sum(policy.substep_delays)


def test_get_action_returns_none_when_nothing_has_been_emitted():
    engine = _make_engine()
    assert engine.get_action(None) is None


def test_reset_drops_the_buffer_so_the_next_episode_warm_starts_again():
    policy = _FakePolicy()
    engine = _make_engine(policy=policy)
    _run_iterations(engine, policy, 1)
    assert engine.ready

    engine.reset()
    assert not engine.ready
    assert engine.get_action(None) is None

    _run_iterations(engine, policy, 1)
    assert policy.warm_starts == 2
