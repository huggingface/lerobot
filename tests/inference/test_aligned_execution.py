"""Plain alignment, canonical blending and conservative contributor age."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from lerobot.inference.contracts import ActionChunk, ActionProvenance, ExecutionMode, ObservationSnapshot
from lerobot.inference.execution import ChunkRuntime
from tests.inference.test_execution import Clock
from tests.inference.test_policy_runner import ConformingPolicy, observation, runner_for, tiny_config


def runtime(*, refill_seconds=0.6, **kwargs):
    # Merge-focused tests keep the six-action horizon eligible. Scheduling tests
    # below select a smaller threshold to exercise waiting for playback.
    clock = Clock()
    rt = ChunkRuntime(
        mode=ExecutionMode.CHUNK,
        action_interval=0.1,
        refill_seconds=refill_seconds,
        max_observation_age_s=10,
        action_timeout_s=3,
        startup_timeout_s=5,
        chunk_merge="aligned",
        clock=clock,
        **kwargs,
    )
    rt.active = True
    return rt, clock


def sample(rt, clock, *, task="a", version=0):
    source = ObservationSnapshot(
        {"state": np.zeros(2, dtype=np.float32)}, clock(), task, version, f"obs-{clock()}"
    )
    return rt.anchor_observation(source)


def result(request, offset=0, length=6, execution_steps=None, session="session"):
    values = torch.arange(offset, offset + length, dtype=torch.float32)
    actions = torch.stack((values, values + 100), dim=1)
    return ActionChunk(
        None,
        actions,
        ActionProvenance(request.observation.capture_time, request.observation.task, session_id=session),
        length if execution_steps is None else execution_steps,
    )


def initial(rt, clock):
    request = rt.begin(sample(rt, clock))
    assert request is not None
    assert rt.accept(request, result(request), task_version=0)
    return request


def test_observation_anchor_accounts_for_progress_before_and_during_request():
    rt, clock = runtime()
    initial(rt, clock)
    rt.pop()  # committed endpoint 0, still potentially interpolating toward it
    clock.now += 0.1
    source = sample(rt, clock)  # action zero assigned to endpoint 1
    rt.pop()  # endpoint 1 committed before request submission
    request = rt.begin(source)
    assert request.continuation.cursor == 2
    rt.pop()  # endpoint 2 committed during inference
    clock.now += 0.001  # wall-time estimate is irrelevant to aligned trimming
    assert rt.accept(request, result(request, 10), task_version=0)
    assert rt.last_accept["trimmed_actions"] == 2
    assert rt.queue.qsize() == 4
    assert rt.pop()[0].tolist() == [12, 112]


def test_startup_delay_does_not_trim_and_replanning_requires_fresh_advanced_sample():
    rt, clock = runtime()
    source = sample(rt, clock)
    request = rt.begin(source)
    clock.now += 0.8
    assert rt.accept(request, result(request), task_version=0)
    assert rt.queue.qsize() == 6
    assert rt.begin(source) is None
    assert rt.begin(sample(rt, clock)) is None  # new capture, same commitment step
    assert rt.pop()[0].tolist() == [0, 100]
    assert rt.begin(source) is None  # advanced execution, old capture/anchor
    request = rt.begin(sample(rt, clock))
    assert request is not None  # playback is below the explicitly configured threshold
    assert rt.begin(sample(rt, clock)) is None  # still one in flight


@pytest.mark.parametrize("blend_steps", [0, 2])
@pytest.mark.parametrize("committed, eligible", [(2, False), (3, True), (4, True)])
def test_aligned_refill_gates_above_at_and_below_playback_threshold(blend_steps, committed, eligible):
    rt, clock = runtime(
        refill_seconds=0.3, blend_steps=blend_steps, blend_indices=(0,) if blend_steps else ()
    )
    initial(rt, clock)
    for _ in range(committed):
        assert rt.pop() is not None
    clock.now += committed * rt.interval
    assert rt.should_request() is eligible
    request = rt.begin(sample(rt, clock))
    assert (request is not None) is eligible
    if request is not None:
        assert request.playback_at_submission == pytest.approx((6 - committed) * rt.interval)


def test_waiting_for_playback_does_not_reserve_an_old_capture():
    rt, clock = runtime(refill_seconds=0.3)
    first = initial(rt, clock)
    rt.pop()
    clock.now += 0.1
    waiting = sample(rt, clock)
    assert rt.begin(waiting) is None
    assert rt.pending is None
    rt.pop()
    clock.now += 0.1
    assert rt.begin(sample(rt, clock)) is None
    rt.pop()
    clock.now += 0.1
    latest = sample(rt, clock)
    request = rt.begin(latest)
    assert request is not None
    assert request.observation is latest
    assert request.observation.capture_time > waiting.capture_time > first.observation.capture_time
    assert request.observation.action_cursor == 3
    rt.pop()
    assert rt.accept(request, result(request, 10), task_version=0)
    assert rt.last_accept["trimmed_actions"] == 1
    assert rt.pop()[0].tolist() == [11, 111]


def test_aligned_refill_floor_seeds_from_startup_then_uses_steady_turnaround():
    rt, clock = runtime(refill_seconds=0.01)
    first = rt.begin(sample(rt, clock))
    clock.now += 0.8
    assert rt.accept(first, result(first), task_version=0)
    assert rt.effective_refill == pytest.approx(0.9)
    for _ in range(5):
        rt.pop()
    request = rt.begin(sample(rt, clock))
    assert request is not None
    clock.now += 0.25
    assert rt.accept(request, result(request, 10), task_version=0)
    assert rt.turnaround == pytest.approx(0.25)
    assert rt.effective_refill == pytest.approx(0.35)
    for _ in range(2):
        rt.pop()
    assert not rt.should_request()  # 0.4 seconds remain
    rt.pop()
    assert rt.should_request()  # 0.3 seconds remain: the measured floor dominates
    assert rt.begin(sample(rt, clock)) is not None


@pytest.mark.parametrize("floor_dominates", [False, True])
def test_threshold_covering_horizon_allows_frequent_requests_without_expanding_slice(floor_dominates):
    rt, clock = runtime(refill_seconds=0.01 if floor_dominates else 0.6)
    if floor_dominates:
        rt.turnarounds.append(0.5)
    first = rt.begin(sample(rt, clock))
    assert rt.accept(first, result(first, execution_steps=3), task_version=0)
    for offset in (10, 20, 30):
        assert rt.queue.qsize() == 3  # the six predicted actions never expand the execution slice
        assert rt.begin(sample(rt, clock)) is None  # still requires a fresh advanced capture
        rt.pop()
        clock.now += 0.1
        request = rt.begin(sample(rt, clock))
        assert request is not None
        assert rt.accept(request, result(request, offset, execution_steps=3), task_version=0)


def test_aligned_replacement_does_not_inherit_append_successor_slot_restriction():
    rt, clock = runtime()
    first = initial(rt, clock)
    rt.pop()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    rt.pop()  # commit old trajectory during inference
    clock.now += 0.1
    assert rt.accept(request, result(request, 10), task_version=0)
    assert rt.current.request_id == first.request_id
    assert {source.request_id for source in rt.queue.snapshot().provenance} == {request.request_id}
    assert rt.begin(sample(rt, clock)) is not None


def test_fresh_task_change_bypasses_playback_gate_at_same_cursor():
    rt, clock = runtime(refill_seconds=0.01)
    initial(rt, clock)
    assert not rt.should_request()
    clock.now += 0.1
    assert rt.begin(sample(rt, clock)) is None  # same task and cursor, full playback buffer
    request = rt.begin(sample(rt, clock, task="new", version=1))
    assert request is not None
    assert request.observation.action_cursor == 0
    assert request.playback_at_submission == pytest.approx(0.6)
    assert rt.accept(request, result(request, 10), task_version=1)
    assert rt.pop()[1].task == "new"


@pytest.mark.parametrize(
    "blocked_by", ["inactive", "held", "fault", "pending", "stale", "generation", "capture"]
)
def test_task_change_never_bypasses_permission_inflight_or_freshness(blocked_by):
    rt, clock = runtime(refill_seconds=0.01)
    initial(rt, clock)
    clock.now += 0.1
    source = sample(rt, clock, task="new", version=1)
    if blocked_by == "inactive":
        rt.active = False
    elif blocked_by == "held":
        rt.held = True
    elif blocked_by == "fault":
        rt.fault("operator fault")
    elif blocked_by == "pending":
        assert rt.begin(source) is not None
        source = replace(source, task_version=2, capture_time=clock() + 0.01)
        clock.now += 0.01
    elif blocked_by == "stale":
        rt.max_age = 0.05
        clock.now += 0.1
    elif blocked_by == "generation":
        source = replace(source, execution_generation=rt.generation + 1)
    elif blocked_by == "capture":
        source = replace(source, capture_time=rt.started_at)  # same capture as the previous request
    assert rt.begin(source) is None


@pytest.mark.parametrize("held", [False, True])
def test_reset_and_hold_resume_start_promptly_from_fresh_generation(held):
    rt, clock = runtime(refill_seconds=0.01)
    initial(rt, clock)
    clock.now += 0.1
    before_reset = sample(rt, clock)
    rt.invalidate(held=held)
    assert rt.begin(before_reset) is None
    if held:
        clock.now += 1
        assert rt.begin(sample(rt, clock)) is None
        rt.held = False
        rt.started_at = clock()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    assert request is not None
    assert request.playback_at_submission == 0
    assert rt.accept(request, result(request), task_version=0)
    assert rt.pop()[0].tolist() == [0, 100]


def test_execution_slice_is_honored_and_empty_suffix_does_not_destroy_eligible_buffer():
    rt, clock = runtime()
    initial(rt, clock)
    rt.pop()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    rt.pop()
    rt.pop()
    assert not rt.accept(request, result(request, 10, execution_steps=2), task_version=0)
    assert rt.failure is None
    assert rt.queue.qsize() == 3
    assert rt.last_accept["rejection"] == "no usable aligned suffix"
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    rt.pop()
    assert rt.accept(request, result(request, 20, execution_steps=3), task_version=0)
    assert rt.queue.qsize() == 2
    assert rt.pop()[0].tolist() == [21, 121]


def test_empty_suffix_eventually_uses_existing_starvation_fault():
    rt, clock = runtime()
    initial(rt, clock)
    rt.pop()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    for _ in range(5):
        rt.pop()
    assert not rt.accept(request, result(request, execution_steps=5), task_version=0)
    assert rt.pop() is None
    assert rt.failure == "Active motion buffer exhausted"


def test_weighted_blend_matches_future_steps_and_keeps_gripper_incoming():
    rt, clock = runtime(blend_steps=2, blend_weight=0.25, blend_indices=(0,))
    initial(rt, clock)
    committed = rt.pop()[0].clone()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    assert rt.accept(request, result(request, 10), task_version=0)
    assert committed.tolist() == [0, 100]
    values = rt.queue.snapshot().canonical_actions
    torch.testing.assert_close(values[:3], torch.tensor([[3.25, 110], [4.25, 111], [12, 112]]))
    assert rt.last_accept["blended_steps"] == 2
    assert rt.last_accept["overlap_steps"] == 5
    assert rt.queue.qsize() == 6  # remaining old horizon does not accumulate
    sources = rt.queue.snapshot().provenance
    assert sources[0].capture_time == 10
    assert sources[0].contributor_count == 2
    assert sources[0].oldest_contributor.request_id
    assert sources[2].capture_time == 10.1
    assert sources[2].contributor_count == 1


@pytest.mark.parametrize("changed", ["task", "session", "expired"])
def test_ineligible_old_targets_are_replaced_without_blending(changed):
    rt, clock = runtime(blend_steps=2, blend_weight=0.5, blend_indices=(0,))
    initial(rt, clock)
    rt.pop()
    clock.now += 0.1
    version = 1 if changed == "task" else 0
    if changed == "expired":
        clock.now += 11
        # Hold has not dispatched in the intervening interval; an incoming result
        # cannot refresh an expired currently committed endpoint either.
    request = rt.begin(sample(rt, clock, version=version))
    assert request is not None
    chunk = result(request, 10, session="other" if changed == "session" else "session")
    assert rt.accept(request, chunk, task_version=version)
    assert rt.last_accept["blended_steps"] == 0
    assert rt.queue.snapshot().canonical_actions[0].tolist() == [10, 110]
    if changed == "expired":
        assert not rt.dispatch_allowed()
        assert rt.failure == "Dispatched action source observation is too old"


def test_repeated_blends_keep_bounded_history_and_oldest_age():
    rt, clock = runtime(blend_steps=6, blend_weight=0.5, blend_indices=(0,))
    initial(rt, clock)
    # Replacements extend the horizon but every next endpoint retains original
    # contributors until that part of the old horizon has been committed.
    for index in range(1, 5):
        rt.pop()
        clock.now += 0.1
        request = rt.begin(sample(rt, clock))
        assert rt.accept(request, result(request, index * 10), task_version=0)
        provenance = rt.queue.snapshot().provenance[0]
        assert provenance.capture_time == 10
        assert provenance.contributor_count == index + 1
        assert len(provenance.contributor_digest) == 64
        assert not any(isinstance(value, (list, tuple)) for value in asdict(provenance).values())
    rt.max_age = 0.45
    rt.pop()
    clock.now += 0.1
    assert not rt.dispatch_allowed()
    assert rt.failure == "Dispatched action source observation is too old"


def test_weight_one_and_no_overlap_are_exact_replacements():
    rt, clock = runtime(blend_steps=3, blend_weight=1, blend_indices=(0,))
    initial(rt, clock)
    for _ in range(6):
        rt.pop()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    assert rt.accept(request, result(request, 10), task_version=0)
    assert rt.last_accept["blended_steps"] == rt.last_accept["overlap_steps"] == 0
    assert rt.pop()[1].contributor_count == 1
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    assert rt.accept(request, result(request, 20), task_version=0)
    assert rt.last_accept["overlap_steps"] == 5
    assert rt.last_accept["blended_steps"] == 0
    assert rt.pop()[0].tolist() == [20, 120]


def test_reset_hold_and_task_changes_cannot_reuse_observation_or_result():
    rt, clock = runtime()
    source = sample(rt, clock)
    request = rt.begin(source)
    rt.invalidate(held=True)
    assert not rt.accept(request, result(request), task_version=0)
    assert rt.begin(source) is None
    rt.held = False
    assert rt.begin(source) is None  # old generation, even if cursor matches
    clock.now += 0.1
    request = rt.begin(sample(rt, clock))
    assert not rt.accept(request, result(request), task_version=2)
    assert rt.queue.empty()
    clock.now += 0.1
    request = rt.begin(sample(rt, clock, task="new", version=2))
    assert request is not None  # a discarded startup result cannot stall retargeting
    assert rt.accept(request, result(request), task_version=2)
    assert rt.pop()[1].task == "new"


def test_relative_model_outputs_are_blended_only_after_absolute_postprocessing():
    rt, clock = runtime(blend_steps=2, blend_weight=0.5, blend_indices=(0, 1))
    runner = runner_for(ConformingPolicy(tiny_config()), relative=True)
    source = rt.anchor_observation(observation(1))
    request = rt.begin(source)
    first = runner.predict(source)
    assert rt.accept(request, first, task_version=4)
    rt.pop()
    clock.now += 0.1
    source = rt.anchor_observation(replace(observation(10), capture_time=clock()))
    request = rt.begin(source)
    incoming = runner.predict(source)
    previous = rt.queue.snapshot().canonical_actions.clone()
    assert rt.accept(request, incoming, task_version=4)
    expected = incoming.canonical_actions.clone()
    expected[:2, :2] = 0.5 * (previous[:2, :2] + expected[:2, :2])
    torch.testing.assert_close(rt.queue.snapshot().canonical_actions, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blend_steps": 2},
        {"blend_steps": -1},
        {"blend_weight": 0},
        {"blend_weight": float("nan")},
        {"blend_steps": 2, "blend_indices": (0, 0)},
        {"blend_steps": 2, "blend_indices": (-1,)},
        {"blend_indices": (0,)},
    ],
)
def test_invalid_blending_is_rejected_before_execution(kwargs):
    with pytest.raises(ValueError):
        runtime(**kwargs)
