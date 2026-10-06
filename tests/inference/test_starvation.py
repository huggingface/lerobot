# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic bounds for waiting without reusing exhausted motion."""

import pytest

from lerobot.inference import ExecutionMode
from tests.inference.test_execution import chunk, observation, runtime


def exhaust(rt, clock):
    request = rt.begin(rt.anchor_observation(observation(clock)))
    assert request is not None
    assert rt.accept(request, chunk(request), task_version=0)
    for _ in range(4):
        assert rt.pop() is not None
    assert rt.pop() is None


@pytest.mark.parametrize("mode", list(ExecutionMode))
def test_starvation_requires_hold_ack_and_fresh_capture_before_recovery(mode):
    rt, clock = runtime(mode)
    rt.starvation_grace = 1.0
    stale = observation(clock)
    exhaust(rt, clock)
    deadline, generation = rt.starvation_deadline, rt.generation
    assert rt.held and not rt.failure
    assert not rt.dispatch_allowed()
    assert rt.begin(stale) is None
    clock.now += 0.1
    rt.acknowledge_starvation_hold()
    rt.complete_starvation_invalidation(generation)
    assert rt.release_hold(generation)
    assert rt.begin(stale) is None
    fresh = rt.begin(rt.anchor_observation(observation(clock)))
    assert fresh is not None and fresh.continuation.model_actions is None
    clock.now += 0.2
    assert rt.accept(fresh, chunk(fresh, offset=10), task_version=0)
    assert rt.starvation_deadline is None
    assert rt.pop()[0].item() == 10
    assert clock.now < deadline


def test_grace_includes_fresh_prediction_and_rejects_late_result():
    rt, clock = runtime()
    rt.starvation_grace = 1.0
    exhaust(rt, clock)
    rt.acknowledge_starvation_hold()
    rt.release_hold(rt.generation)
    fresh = rt.begin(observation(clock))
    clock.now += 1.01
    assert not rt.accept(fresh, chunk(fresh), task_version=0)
    assert "starvation grace expired" in rt.failure
    assert rt.queue.empty()
    assert not rt.activate()


def test_original_request_deadline_remains_absolute_during_wait():
    rt, clock = runtime()
    rt.starvation_grace = 2.0
    first = rt.begin(observation(clock))
    assert rt.accept(first, chunk(first), task_version=0)
    rt.pop()
    old = rt.begin(observation(clock))
    clock.now += 2.9
    for _ in range(3):
        rt.pop()
    rt.pop()
    assert rt.starvation_deadline is not None
    assert not rt.accept(old, chunk(old), task_version=0)
    clock.now += 0.2
    rt.check_deadlines()
    assert "Action request deadline exceeded" in rt.failure


def test_completed_invalidation_retires_old_request_deadline_not_grace():
    rt, clock = runtime()
    rt.starvation_grace = 2.0
    first = rt.begin(observation(clock))
    rt.accept(first, chunk(first), task_version=0)
    rt.pop()
    rt.begin(observation(clock))
    clock.now += 2.9
    for _ in range(4):
        rt.pop()
    deadline = rt.starvation_deadline
    rt.complete_starvation_invalidation(rt.generation)
    clock.now += 0.2
    rt.check_deadlines()
    assert rt.failure is None
    assert rt.starvation_deadline == deadline
    clock.now = deadline + 0.01
    rt.check_deadlines()
    assert "starvation grace expired" in rt.failure


@pytest.mark.parametrize("transition", ["deactivate", "invalidate"])
def test_operator_transition_cancels_wait_without_accepting_old_results(transition):
    rt, clock = runtime()
    rt.starvation_grace = 1.0
    exhaust(rt, clock)
    rt.acknowledge_starvation_hold()
    rt.release_hold(rt.generation)
    fresh = rt.begin(observation(clock))
    getattr(rt, transition)()
    assert rt.starvation_deadline is None
    clock.now += 2.0
    assert not rt.accept(fresh, chunk(fresh), task_version=0)
    assert rt.failure is None
    assert rt.queue.empty()


def test_expired_action_source_faults_without_entering_starvation_grace():
    rt, clock = runtime(max_age=0.5)
    rt.starvation_grace = 1.0
    request = rt.begin(observation(clock))
    rt.accept(request, chunk(request), task_version=0)
    rt.pop()
    clock.now += 0.6
    assert not rt.dispatch_allowed()
    assert rt.starvation_deadline is None
    assert "observation is too old" in rt.failure
