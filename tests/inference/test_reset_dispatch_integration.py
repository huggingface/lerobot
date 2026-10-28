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

"""Control-thread resets across prediction, queue and observation-copy boundaries."""

from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="reset dispatch integration requires the dataset extra")

from tests.inference.reset_race_helpers import (
    ACTIONS,
    FRESH_INPUT,
    OLD_INPUT,
    BufferWitness,
    FreshnessWitness,
    InterpolationWitness,
    ResetDispatchRig,
    check_interpolation_trace,
    consumed_input,
    vector,
)

pytestmark = pytest.mark.dataset


@pytest.fixture
def reset_rig(monkeypatch):
    rig = ResetDispatchRig(monkeypatch)
    try:
        yield rig
    finally:
        rig.close()


def test_inflight_result_is_not_dispatched_after_reset(reset_rig):
    rig = reset_rig
    rig.start()
    old_capture = rig.capture(OLD_INPUT)
    old = rig.next_call()
    assert consumed_input(old) == old_capture, "SETUP: old invocation did not consume old capture"
    assert not old.released.is_set() and not old.returned.is_set(), "SETUP: prediction not blocked"

    rig.engine.reset()  # Control thread, between dispatch ticks, with the old invocation still held.
    boundary = len(rig.driver.commands)
    rig.events.append({"event": "reset_returned", "old_call": old.number, "boundary": boundary})
    assert not old.released.is_set() and not old.returned.is_set(), "SETUP: reset released prediction"
    fresh_capture = rig.capture(FRESH_INPUT)
    assert fresh_capture != old_capture, "SETUP: captures must differ"
    rig.dispatch()
    assert len(rig.driver.commands) == boundary + 1, "SETUP: missing blocked-prediction hold"
    assert rig.driver.commands[-1].kind == "hold", "SETUP: expected an actual public hold operation"
    rig.assert_trace(boundary, fresh_capture, require_recovery=False)

    rig.release(old)
    rig.wait_handled(old)
    rig.dispatch()
    # A stale accepted result can refill the queue. Inspect it before waiting for any successor.
    rig.assert_trace(boundary, fresh_capture, require_recovery=False)
    rig.events.append({"event": "stale_opportunity_checked"})

    fresh = rig.next_call(after=old)
    assert consumed_input(fresh) == fresh_capture, (
        f"OBSERVATION_FRESHNESS: call {fresh.number} consumed {consumed_input(fresh)}, "
        f"expected capture {fresh_capture}"
    )
    rig.release(fresh)
    rig.wait_handled(fresh)
    rig.dispatch()
    rig.assert_trace(boundary, fresh_capture, require_recovery=True)


def test_copied_observation_is_not_consumed_after_reset(reset_rig):
    rig = reset_rig
    gate = rig.arm_copy_gate()
    rig.start()
    old_capture = rig.capture(OLD_INPUT)
    old_capture_id = len(rig.driver.captures) - 1
    assert gate.entered.wait(5), "SETUP: worker copy was not acknowledged"
    rig.assert_healthy()
    assert vector(gate.observation) == old_capture, "SETUP: copied input differs from old capture"
    assert not gate.released.is_set(), "SETUP: copy gate already released"
    assert not rig.admissions and not rig.policy.entries, "SETUP: work admitted before copy release"
    rig.events.append(
        {"event": "copy_acknowledged", "gate": gate.identity, "capture": old_capture_id, "input": old_capture}
    )

    rig.engine.reset()  # The frame gate holds no engine lock; dispatch is between ticks.
    boundary = len(rig.driver.commands)
    rig.events.append({"event": "reset_returned", "boundary": boundary, "gate": gate.identity})
    assert not gate.released.is_set(), "SETUP: reset released the copy gate"
    assert not rig.admissions and not rig.policy.entries, "SETUP: work admitted while copy held"
    fresh_capture = rig.capture(FRESH_INPUT)
    fresh_capture_id = len(rig.driver.captures) - 1
    assert fresh_capture != old_capture, "SETUP: captures must differ"
    gate.release()

    fresh = rig.next_call()  # Never filter for a fresh input or infer freshness from call number.
    actual = consumed_input(fresh)
    # Frozen values survive cleanup releasing this prediction after an assertion failure.
    rig.freshness_witness = FreshnessWitness(
        consumed=actual,
        expected=fresh_capture,
        old_capture_id=old_capture_id,
        fresh_capture_id=fresh_capture_id,
        invocation=fresh.number,
        gate=gate.identity,
        ordering=tuple((i, event["event"]) for i, event in enumerate(rig.events)),
        prediction_released=fresh.released.is_set(),
        prediction_returned=fresh.returned.is_set(),
    )
    assert actual == fresh_capture, (
        f"OBSERVATION_FRESHNESS: call {fresh.number} consumed {actual}, expected capture {fresh_capture}"
    )
    assert not fresh.released.is_set() and not fresh.returned.is_set(), "SETUP: prediction was not held"
    rig.release(fresh)
    rig.wait_handled(fresh)
    rig.dispatch()
    rig.assert_trace(boundary, fresh_capture, require_recovery=True)


def test_populated_queue_is_not_dispatched_after_reset(reset_rig):
    rig = reset_rig
    rig.start()
    old_capture = rig.capture(OLD_INPUT)
    old = rig.next_call()
    assert consumed_input(old) == old_capture, "SETUP: initial prediction consumed the wrong capture"
    rig.release(old)
    rig.wait_handled(old)
    queued = rig.engine.action_queue.qsize()
    assert queued >= 2, "SETUP: real prediction did not populate at least two queued actions"
    rig.events.append({"event": "queue_populated", "call": old.number, "queued": queued})

    rig.dispatch()
    assert len(rig.driver.commands) == 1, "SETUP: initial dispatch did not reach the recorder"
    first = rig.driver.commands[-1]
    assert first.kind == "policy", "SETUP: initial dispatch must be an ordinary policy action"
    assert vector(first.requested) == ACTIONS[old_capture][0], "PAYLOAD_INTEGRITY: initial vector"
    applied = vector(first.applied)
    boundary = len(rig.driver.commands)
    rig.assert_trace(boundary, FRESH_INPUT, require_recovery=False)

    remaining = rig.engine.action_queue.qsize()  # No competing dispatch consumer.
    assert remaining > 0, "SETUP: no old queued action remains immediately before reset"
    rig.events.append(
        {
            "event": "queue_before_reset",
            "queued": remaining,
            "old_call": old.number,
            "command_sequence": boundary - 1,
            "applied": applied,
        }
    )
    rig.engine.reset()
    # Successful begin observers run under the caller's observation lock, before reset can return.
    # No new capture has been published yet: these are all pre-boundary admissions, including
    # a possible eager successor that has not reached predict_action_chunk yet.
    admitted_before_reset = {entry["id"] for entry in rig.admissions}
    rig.events.append(
        {"event": "reset_returned", "boundary": boundary, "admissions": sorted(admitted_before_reset)}
    )
    rig.dispatch()  # No subsequent policy result has been released.
    rig.assert_trace(boundary, FRESH_INPUT, require_recovery=False)  # A retained queue must fail here first.
    rig.events.append({"event": "queued_stale_opportunity_checked"})
    assert len(rig.driver.commands) == boundary + 1, "SETUP: missing post-reset command"
    hold = rig.driver.commands[-1]
    assert hold.kind == "hold", "SETUP: expected an actual public hold operation"
    assert vector(hold.requested) == applied, "HOLD_TARGET: previous driver-applied target was lost"

    fresh_capture = rig.capture(FRESH_INPUT)
    assert fresh_capture != old_capture, "SETUP: captures must differ"
    outstanding = admitted_before_reset - {old.admission_id}
    assert len(outstanding) <= 1, "SETUP: more than one unreleased serial successor admitted"
    previous = old
    if outstanding:
        successor = rig.next_call(after=old)
        assert successor.admission_id in outstanding, "SETUP: admitted old successor was not accounted for"
        assert consumed_input(successor) == old_capture, "SETUP: old successor input differs from capture"
        assert not successor.released.is_set(), "SETUP: successor result released before safety check"
        rig.events.append(
            {
                "event": "old_successor_identified",
                "call": successor.number,
                "admission": successor.admission_id,
            }
        )
        rig.release(successor)
        rig.wait_handled(successor)
        rig.dispatch()
        rig.assert_trace(boundary, fresh_capture, require_recovery=False)
        rig.events.append({"event": "old_successor_stale_opportunity_checked", "call": successor.number})
        previous = successor

    fresh = rig.next_call(after=previous)  # Inspect the actual next call; never filter out old inputs.
    assert fresh.admission_id not in admitted_before_reset, "OBSERVATION_FRESHNESS: old admission at recovery"
    assert consumed_input(fresh) == fresh_capture, (
        f"OBSERVATION_FRESHNESS: call {fresh.number} consumed {consumed_input(fresh)}, "
        f"expected capture {fresh_capture}"
    )
    rig.release(fresh)
    rig.wait_handled(fresh)
    rig.dispatch()
    rig.assert_trace(boundary, fresh_capture, require_recovery=True)


def test_control_reset_discards_pending_interpolation(reset_rig):
    rig = reset_rig
    owner = rig.configure_interpolation_owner()
    rig.start()
    old_capture = rig.capture(OLD_INPUT)
    owner._process_observation_and_notify(
        SimpleNamespace(robot_observation_processor=lambda observation: observation), rig.observation
    )
    assert owner._cached_obs_processed, "SETUP: owner cache was not populated"
    old = rig.next_call()
    assert consumed_input(old) == old_capture, "SETUP: initial input differs from capture"
    rig.release(old)
    rig.wait_handled(old)
    queued = rig.engine.action_queue.qsize()
    assert queued >= 2, "SETUP: real old prediction did not populate queue"
    rig.events.append({"event": "interpolation_queue_populated", "call": old.number, "queued": queued})

    rig.dispatch()
    assert len(rig.driver.commands) == 1 and rig.driver.commands[0].kind == "policy", "SETUP: missing A0"
    assert vector(rig.driver.commands[0].requested) == (4100, 4124), "PAYLOAD_INTEGRITY: expected A0"
    rig.dispatch()
    assert len(rig.driver.commands) == 2 and rig.driver.commands[1].kind == "policy", "SETUP: missing Q1"
    assert vector(rig.driver.commands[1].requested) == (4116, 4140), "PAYLOAD_INTEGRITY: expected Q1"
    boundary = len(rig.driver.commands)
    check_interpolation_trace(
        rig.driver.commands,
        rig.policy.entries,
        boundary=boundary,
        fresh_input=FRESH_INPUT,
        require_recovery=False,
    )
    interp = rig.interpolator
    rig.buffer_witness = BufferWitness(
        multiplier=interp.multiplier,
        index=interp._idx,
        remaining=tuple(tuple(value.tolist()) for value in interp._buffer[interp._idx :]),
        needs_new_action=interp.needs_new_action(),
        old_call=old.number,
        old_admission=old.admission_id,
        commands=boundary,
        dispatches=rig.dispatch_count,
        cached_observation=tuple(owner._cached_obs_processed.items()),
    )
    before = rig.buffer_witness
    assert before.multiplier == 4 and not before.needs_new_action, "SETUP: interpolation not buffered"
    assert len(before.remaining) == 3, "SETUP: expected three remaining old buffered outputs"
    rig.events.append(
        {
            "event": "interpolation_before_reset",
            "old_call": old.number,
            "buffer_index": before.index,
            "remaining": len(before.remaining),
            "commands": boundary,
            "dispatches": rig.dispatch_count,
        }
    )
    epoch = rig.engine._reset_epoch
    owner.reset_control_state()  # Actual inherited owner operation, with dispatch stopped.
    admitted_before_reset = {entry["id"] for entry in rig.admissions}
    cache_cleared = owner._cached_obs_processed is None
    rig.events.append(
        {
            "event": "owner_reset_returned",
            "epoch_before": epoch,
            "epoch_after": rig.engine._reset_epoch,
            "cache_cleared": cache_cleared,
            "commands": len(rig.driver.commands),
            "dispatches": rig.dispatch_count,
            "admissions": sorted(admitted_before_reset),
        }
    )
    assert cache_cleared, "OWNER_STATE: cached processed observation was not cleared"

    fresh_capture = rig.capture(FRESH_INPUT)
    assert fresh_capture != old_capture, "SETUP: captures must differ"
    outstanding = admitted_before_reset - {old.admission_id}
    assert len(outstanding) <= 1, "SETUP: more than one admitted serial successor"
    previous = old
    if outstanding:
        successor = rig.next_call(after=old)
        assert successor.admission_id in outstanding, "SETUP: unaccounted old admission"
        assert consumed_input(successor) == old_capture, "SETUP: old successor input mismatch"
        rig.events.append(
            {
                "event": "old_successor_identified",
                "call": successor.number,
                "admission": successor.admission_id,
            }
        )
        rig.release(successor)
        rig.wait_handled(successor)
        previous = successor  # No dispatch here: denial would independently clear interpolation.

    fresh = rig.next_call(after=previous)  # Inspect the actual next invocation, without filtering.
    assert fresh.admission_id not in admitted_before_reset, "OBSERVATION_FRESHNESS: old admission at recovery"
    assert consumed_input(fresh) == fresh_capture, "OBSERVATION_FRESHNESS: recovery consumed old input"
    rig.release(fresh)
    rig.wait_handled(fresh)
    queued = rig.engine.action_queue.qsize()
    assert fresh.returned.is_set() and fresh.accepted and queued > 0, "SETUP: fresh result not queue-ready"
    rig.events.append(
        {
            "event": "fresh_queue_ready",
            "call": fresh.number,
            "admission": fresh.admission_id,
            "input": consumed_input(fresh),
            "queued": queued,
        }
    )
    rig.interpolation_witness = InterpolationWitness(
        before_reset=before,
        pre_boundary_admissions=tuple(sorted(admitted_before_reset)),
        fresh_call=fresh.number,
        fresh_admission=fresh.admission_id,
        consumed=consumed_input(fresh),
        expected=fresh_capture,
        returned=fresh.returned.is_set(),
        handled=fresh.handled.is_set(),
        accepted_diagnostic=fresh.accepted,
        queued=queued,
        dispatches=rig.dispatch_count,
        commands=len(rig.driver.commands),
        permissions=len(rig.permissions),
        cached_observation_cleared=cache_cleared,
        ordering=tuple((i, event["event"]) for i, event in enumerate(rig.events)),
    )
    assert rig.dispatch_count == before.dispatches, "SETUP: dispatch occurred before fresh readiness"
    assert len(rig.driver.commands) == boundary, "SETUP: command occurred before fresh readiness"
    permission_boundary = len(rig.permissions)
    rig.dispatch()  # Exactly the first resumed tick; never drain an incorrect output.
    check_interpolation_trace(
        rig.driver.commands,
        rig.policy.entries,
        boundary=boundary,
        fresh_input=fresh_capture,
        require_recovery=True,
    )
    assert len(rig.driver.commands) == boundary + 1, "RECOVERY: expected one resumed ordinary command"
    assert rig.permissions[permission_boundary:] and all(
        item["allowed"] for item in rig.permissions[permission_boundary:]
    ), "SETUP: resumed dispatch was denied and could conceal retained interpolation"
