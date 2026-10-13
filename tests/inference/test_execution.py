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

"""Deterministic timing/continuation tests shared by local and remote execution."""

import torch

from lerobot.inference import (
    ActionProvenance,
    ExecutionMode,
    HoldState,
)
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from tests.inference.execution_helpers import chunk, observation, runtime


def test_atomic_queue_snapshot_copies_both_spaces_and_provenance():
    queue = ActionQueue(RTCConfig())
    canonical = torch.arange(4.0).reshape(-1, 1)
    queue.merge(canonical * 2, canonical, 0, task="old", provenance="request1")
    queue.get()
    snap = queue.snapshot()
    assert snap.cursor == snap.index == 1
    assert snap.provenance == ("request1",) * 3
    torch.testing.assert_close(snap.model_actions, 2 * snap.canonical_actions)
    queue.get()
    queue.clear()
    assert not queue.merge(canonical, canonical, 0, snapshot=snap)
    assert queue.empty()
    assert len(snap.canonical_actions) == 3


def test_elapsed_startup_without_consumption_never_skips_first_action():
    rt, clock = runtime(ExecutionMode.RTC_GUIDED)
    request = rt.begin(observation(clock))
    clock.now += 0.7
    assert rt.accept(request, chunk(request), task_version=0)
    assert rt.pop()[0].item() == 0


def test_guided_merge_uses_committed_endpoint_cursor_not_wall_time_alone():
    rt, clock = runtime(ExecutionMode.RTC_GUIDED)
    first = rt.begin(observation(clock))
    rt.accept(first, chunk(first), task_version=0)
    rt.pop()  # endpoint committed to interpolator; robot need not have reached it
    next_request = rt.begin(observation(clock))
    rt.pop()
    clock.now += 0.25
    assert rt.accept(next_request, chunk(next_request, offset=10), task_version=0)
    assert rt.pop()[0].item() == 11  # exactly one commit during inference, not three


def test_plain_successor_slot_is_bounded_and_preserves_old_action_age():
    rt, clock = runtime(max_age=1)
    first = rt.begin(observation(clock))
    rt.accept(first, chunk(first), task_version=0)
    rt.pop()
    clock.now += 0.5
    successor = rt.begin(observation(clock, "b", 1))
    assert successor is not None
    assert rt.accept(successor, chunk(successor, offset=10), task_version=1)
    assert rt.begin(observation(clock, "b", 1)) is None
    old = rt.pop()
    assert old[1].task == "a"
    clock.now += 0.6
    assert not rt.dispatch_allowed()  # fresh successor did not refresh the old endpoint
    assert rt.failure is not None


def test_reset_and_timeout_make_late_results_ineligible():
    rt, clock = runtime()
    old = rt.begin(observation(clock))
    rt.invalidate()
    assert not rt.accept(old, chunk(old), task_version=0)
    request = rt.begin(observation(clock))
    clock.now += 4
    assert not rt.accept(request, chunk(request), task_version=0)
    assert rt.failure == "Action request deadline exceeded"
    assert rt.begin(observation(clock)) is None


def test_trained_rtc_rejects_unconditioned_actual_progress():
    rt, clock = runtime(ExecutionMode.RTC_TRAINED)
    initial = rt.begin(observation(clock))
    rt.accept(initial, chunk(initial, length=5), task_version=0)
    rt.pop()
    rt.pop()
    request = rt.begin(observation(clock))
    assert request.delay == 2
    rt.pop()
    rt.pop()
    rt.pop()
    # Even if the clock estimate is shorter, three committed actions cannot be
    # covered by a two-step trained prefix.
    clock.now += 0.1
    assert not rt.accept(request, chunk(request), task_version=0)


def test_planned_hold_suppresses_starvation_but_not_resume_startup_deadline():
    rt, clock = runtime()
    rt.invalidate(held=True)
    clock.now += 30
    assert not rt.dispatch_allowed()
    assert rt.failure is None
    rt.held = False
    rt.started_at = clock.now
    clock.now += 6
    assert not rt.dispatch_allowed()
    assert rt.failure == "Initial action deadline exceeded"


def test_invalid_action_cannot_grant_dispatch_permission():
    rt, clock = runtime()
    request = rt.begin(observation(clock))
    result = chunk(request)
    result.canonical_actions[0] = float("nan")
    assert not rt.accept(request, result, task_version=0)
    assert rt.failure == "Invalid action chunk"


def test_release_hold_rearms_only_current_healthy_active_generation():
    rt, clock = runtime()
    rt.activate(held=True)
    generation = rt.generation
    clock.now += 30
    rt.check_deadlines()
    assert rt.failure is None
    assert rt.release_hold(generation)
    rt.check_deadlines()
    assert rt.failure is None
    assert rt.started_at == clock.now
    rt.deactivate()
    assert not rt.release_hold(generation)
    assert not rt.active
    rt.fault("terminal")
    assert not rt.activate()
    assert not rt.release_hold(rt.generation)


def test_append_provenance_stays_with_actions_after_partial_consumption():
    queue = ActionQueue(RTCConfig(enabled=False))
    a, b = ActionProvenance(1.0, "A"), ActionProvenance(2.0, "B")
    values = torch.zeros(3, 2)
    queue.merge(values, values, 0, task="A", provenance=a)
    queue.get_with_provenance()
    queue.get_with_provenance()
    queue.merge(values + 1, values + 1, 0, task="B", provenance=b)
    assert [queue.get_with_provenance()[2] for _ in range(4)] == [a, b, b, b]


def test_replace_future_requires_current_cursor_and_generation():
    queue = ActionQueue(RTCConfig(enabled=False))
    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    source = ActionProvenance(1.0, "original")
    queue.merge(values, values, 0, task=source.task, provenance=source)
    replacement = torch.full((2, 2), 99.0)
    stale = queue.snapshot()
    queue.clear()
    queue.merge(values, values, 0, task=source.task, provenance=source)
    assert not queue.replace_future(replacement, [source] * 2, snapshot=stale)
    stale = queue.snapshot()
    queue.get_with_provenance()
    fresh = queue.snapshot()
    assert not queue.replace_future(replacement, [source] * 2, snapshot=stale)
    torch.testing.assert_close(queue.snapshot().canonical_actions, values[1:])
    sources = [ActionProvenance(2.0, "first"), ActionProvenance(3.0, "second")]
    assert queue.replace_future(replacement, sources, snapshot=fresh)
    after = queue.snapshot()
    assert after.index == 0
    assert after.cursor == fresh.cursor
    assert after.generation == fresh.generation
    assert after.provenance == tuple(sources)
    torch.testing.assert_close(after.model_actions, replacement)
    torch.testing.assert_close(after.canonical_actions, replacement)
    for expected_source in sources:
        action, task, provenance = queue.get_with_provenance()
        torch.testing.assert_close(action, torch.full((2,), 99.0))
        assert task == expected_source.task
        assert provenance == expected_source
    assert queue.empty()
    assert queue.snapshot().cursor == fresh.cursor + 2


def test_hold_state_upgrades_once_without_renewing_per_tick():
    hold = HoldState()
    assert hold.request("instruction", 10.0, 3.0)
    assert hold.requested and not hold.ready and hold.deadline == 13.0
    assert not hold.request("instruction", 11.0, 3.0)
    assert (hold.started_at, hold.deadline) == (10.0, 13.0)
    assert hold.acknowledge() and not hold.acknowledge()
    # Upgrading keeps the anchor and acknowledgment and widens the budget once.
    assert not hold.request("language", 12.0, 5.0, upgrade=True)
    assert hold.reason == "language" and hold.ready and hold.deadline == 15.0
    # Re-requesting the current reason, as every motor tick does, never renews the deadline.
    hold.deadline = 11.5
    assert not hold.request("language", 12.0, 5.0, upgrade=True)
    assert hold.deadline == 11.5 and hold.expired(12.0)
    hold.clear()
    assert not hold.requested and hold.deadline is None and not hold.acknowledged
    # A shorter upgrade cannot shrink an existing budget; starvation has no planned deadline.
    assert hold.request("instruction", 20.0, 3.0)
    assert not hold.request("language", 21.0, 1.0, upgrade=True)
    assert hold.deadline == 23.0
    hold.clear()
    assert hold.request("starvation", 30.0, None)
    assert not hold.expired(1e9)
    assert not hold.request("language", 31.0, 2.0, upgrade=True)
    assert hold.deadline == 32.0
