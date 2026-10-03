"""Regression checks for playback budgets and atomic lifecycle transitions."""

import json

import numpy as np
import pytest
import torch

from lerobot.inference.contracts import ActionChunk, ActionProvenance, ExecutionMode, ObservationSnapshot
from lerobot.inference.events import EventWriter
from lerobot.inference.execution import ChunkRuntime
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@pytest.mark.parametrize("merge", ["append", "aligned"])
@pytest.mark.parametrize("steps", [50, 100])
def test_default_budget_plays_multiple_stock_length_chunks(merge, steps):
    now = [100.0]
    runtime = ChunkRuntime(
        mode=ExecutionMode.CHUNK,
        action_interval=1 / 30,
        refill_seconds=0.5,
        max_observation_age_s=5,
        action_timeout_s=5,
        startup_timeout_s=10,
        chunk_merge=merge,
        clock=lambda: now[0],
    )
    runtime.activate()
    pending = None
    accepted = 0
    for tick in range(600):
        now[0] = 100 + tick / 30
        if pending is not None and now[0] >= pending.submitted_at + 0.05:
            values = torch.zeros(steps, 2)
            accepted += runtime.accept(
                pending,
                ActionChunk(None, values, ActionProvenance(pending.observation.capture_time, "task"), steps),
                task_version=0,
            )
            pending = None
        obs = ObservationSnapshot({"state": np.zeros(2)}, now[0], "task")
        obs = runtime.anchor_observation(obs)
        if pending is None:
            pending = runtime.begin(obs)
        runtime.pop()
        assert runtime.failure is None
    assert accepted >= 5


def test_release_hold_rearms_only_current_healthy_active_generation():
    now = [100.0]
    runtime = ChunkRuntime(
        mode=ExecutionMode.CHUNK,
        action_interval=0.1,
        refill_seconds=0.2,
        max_observation_age_s=5,
        action_timeout_s=1,
        startup_timeout_s=2,
        clock=lambda: now[0],
    )
    runtime.activate(held=True)
    generation = runtime.generation
    now[0] += 30
    runtime.check_deadlines()
    assert runtime.failure is None
    assert runtime.release_hold(generation)
    runtime.check_deadlines()
    assert runtime.failure is None
    assert runtime.started_at == now[0]
    runtime.deactivate()
    assert not runtime.release_hold(generation)
    assert not runtime.active
    runtime.fault("terminal")
    assert not runtime.activate()
    assert not runtime.release_hold(runtime.generation)


def test_append_provenance_stays_with_actions_after_partial_consumption():
    queue = ActionQueue(RTCConfig(enabled=False))
    a, b = ActionProvenance(1.0, "A"), ActionProvenance(2.0, "B")
    values = torch.zeros(3, 2)
    queue.merge(values, values, 0, task="A", provenance=a)
    queue.get_with_provenance()
    queue.get_with_provenance()
    queue.merge(values + 1, values + 1, 0, task="B", provenance=b)
    assert [queue.get_with_provenance()[2] for _ in range(4)] == [a, b, b, b]


def test_invalid_event_does_not_stop_subsequent_telemetry(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.write({"bad": np.float32(1)})
    writer.write({"bad": float("nan")})
    writer.write({"event": "later"})
    writer.close()
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"event": "later"},
        {"event": "writer_closed", "dropped_events": 2},
    ]


def test_misaligned_provenance_is_rejected_before_dispatch():
    queue = ActionQueue(RTCConfig(enabled=False))
    values = torch.zeros(3, 2)
    queue.merge(values, values, 0, provenance=ActionProvenance(1.0, "A"))
    queue._provenance_queue.append(ActionProvenance(2.0, "B"))
    with pytest.raises(RuntimeError, match="provenance"):
        queue.get_with_provenance()


@pytest.mark.parametrize("transition", ["commit", "clear"])
def test_replace_future_rejects_stale_snapshot_without_mutating_queue(transition):
    queue = ActionQueue(RTCConfig(enabled=False))
    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    source = ActionProvenance(1.0, "original")
    queue.merge(values, values, 0, task=source.task, provenance=source)
    stale = queue.snapshot()
    if transition == "commit":
        queue.get_with_provenance()
    else:
        queue.clear()
        queue.merge(values + 10, values + 10, 0, task=source.task, provenance=source)
    before = queue.snapshot()
    if transition == "commit":
        assert before.generation == stale.generation
        assert before.cursor != stale.cursor
    else:
        assert before.generation != stale.generation
        assert before.cursor == stale.cursor

    replacement = torch.full((2, 2), 99.0)
    replacement_source = ActionProvenance(2.0, "replacement")
    assert not queue.replace_future(replacement, [replacement_source] * 2, snapshot=stale)
    after = queue.snapshot()
    assert (after.generation, after.cursor, after.index, after.provenance) == (
        before.generation,
        before.cursor,
        before.index,
        before.provenance,
    )
    torch.testing.assert_close(after.model_actions, before.model_actions)
    torch.testing.assert_close(after.canonical_actions, before.canonical_actions)


def test_replace_future_accepts_fresh_snapshot_and_resets_index_without_rewinding_cursor():
    queue = ActionQueue(RTCConfig(enabled=False))
    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    source = ActionProvenance(1.0, "original")
    queue.merge(values, values, 0, task=source.task, provenance=source)
    queue.get_with_provenance()
    fresh = queue.snapshot()
    assert fresh.index == fresh.cursor == 1
    replacement = torch.full((2, 2), 99.0)
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
