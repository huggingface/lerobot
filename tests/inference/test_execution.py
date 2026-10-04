"""Deterministic timing/continuation tests shared by local and remote execution."""

import numpy as np
import torch

from lerobot.inference import ActionChunk, ActionProvenance, ChunkRuntime, ExecutionMode, ObservationSnapshot
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def runtime(mode=ExecutionMode.CHUNK, max_age=10, training_max_delay=2):
    clock = Clock()
    runtime = ChunkRuntime(
        mode=mode,
        action_interval=0.1,
        refill_seconds=0.3,
        max_observation_age_s=max_age,
        action_timeout_s=3,
        startup_timeout_s=5,
        training_max_delay=training_max_delay,
        clock=clock,
    )
    runtime.active = True
    return runtime, clock


def observation(clock, task="a", version=0):
    return ObservationSnapshot({"state": np.zeros(1, dtype=np.float32)}, clock(), task, version, task)


def chunk(request, offset=0, length=4):
    actions = torch.arange(offset, offset + length, dtype=torch.float32).reshape(-1, 1)
    return ActionChunk(
        actions, actions, ActionProvenance(request.observation.capture_time, request.observation.task), length
    )


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


def test_old_inflight_task_is_discarded_even_after_same_text_reuse():
    rt, clock = runtime()
    request = rt.begin(observation(clock, "a", 0))
    assert not rt.accept(request, chunk(request), task_version=2)
    assert rt.queue.empty()
    assert rt.pending is None


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
