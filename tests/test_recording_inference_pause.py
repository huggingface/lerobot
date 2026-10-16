# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Known blocking recording work invalidates async motion before it becomes stale."""

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.rollout.configs import DAggerStrategyConfig, HighlightStrategyConfig
from lerobot.rollout.ring_buffer import RolloutRingBuffer
from lerobot.rollout.strategies.dagger import DAggerEvents, DAggerPhase, DAggerStrategy
from lerobot.rollout.strategies.highlight import HighlightStrategy
from lerobot.utils.action_interpolator import ActionInterpolator
from tests.inference.test_aligned_execution import initial, result, runtime, sample
from tests.test_rollout import _make_loop_ctx, _make_sentry


def setup_strategy(*, sync=False):
    ctx, dataset = _make_loop_ctx(fps=200, multiplier=2, num_ticks=8)
    engine = ctx.policy.inference
    engine.control_thread_owns_policy = sync
    engine.failed = False
    ctx.hardware.robot_wrapper.supports_hold = True
    ctx.hardware.robot_wrapper.hardware_failure = None
    return _make_sentry(ctx, 2), ctx, dataset


def test_save_longer_than_freshness_budget_resumes_only_fresh_motion():
    strategy, ctx, _ = setup_strategy()
    rt, clock = runtime()
    initial(rt, clock)
    rt.pop()
    clock.now += 0.1
    pending = rt.begin(sample(rt, clock))
    assert pending is not None
    engine = ctx.policy.inference
    engine.pause.side_effect = rt.deactivate
    engine.resume.side_effect = rt.activate
    strategy._interpolator.add(torch.ones(2))
    strategy._cached_obs_processed = {"old": 1}

    with strategy._pause_for_recording(ctx):
        assert not rt.active
        assert rt.pending is None
        assert rt.queue.qsize() == 0
        assert strategy._cached_obs_processed is None
        assert strategy._interpolator.get() is None
        ctx.hardware.robot_wrapper.hold.assert_called_once()
        clock.now += rt.max_age * 10
        rt.check_deadlines()
        assert rt.failure is None

    assert rt.active
    assert rt.failure is None
    assert not rt.accept(pending, result(pending), task_version=0)
    fresh = rt.begin(sample(rt, clock))
    assert fresh is not None
    assert rt.accept(fresh, result(fresh), task_version=0)


@pytest.mark.parametrize("outcome", ["shutdown", "fault", "pause", "takeover", "stop", "queued_pause"])
def test_recording_save_never_overrides_a_stop_or_operator_transition(outcome):
    strategy, ctx, _ = setup_strategy()
    engine = ctx.policy.inference
    events = DAggerEvents()
    with strategy._pause_for_recording(ctx, resume_allowed=events.permits_autonomous_motion):
        if outcome == "shutdown":
            ctx.runtime.shutdown_event.set()
        elif outcome == "fault":
            engine.failed = True
        elif outcome == "pause":
            events.phase = DAggerPhase.PAUSED
        elif outcome == "takeover":
            events.phase = DAggerPhase.CORRECTING
        elif outcome == "stop":
            events.stop_recording.set()
        else:
            events.request_transition("pause_resume")
    engine.resume.assert_not_called()


def test_recording_save_does_not_resume_an_already_paused_engine():
    strategy, ctx, _ = setup_strategy()
    events = DAggerEvents()
    events.phase = DAggerPhase.PAUSED
    with strategy._pause_for_recording(ctx, resume_allowed=events.permits_autonomous_motion):
        events.phase = DAggerPhase.AUTONOMOUS
    ctx.policy.inference.resume.assert_not_called()


def test_save_error_keeps_async_inference_paused():
    strategy, ctx, _ = setup_strategy()
    with pytest.raises(OSError, match="disk full"), strategy._pause_for_recording(ctx):
        raise OSError("disk full")
    ctx.policy.inference.pause.assert_called_once()
    ctx.policy.inference.resume.assert_not_called()


def test_sync_recording_save_keeps_existing_behavior():
    strategy, ctx, _ = setup_strategy(sync=True)
    with strategy._pause_for_recording(ctx):
        pass
    ctx.policy.inference.pause.assert_not_called()
    ctx.policy.inference.resume.assert_not_called()
    ctx.hardware.robot_wrapper.hold.assert_not_called()


@pytest.mark.parametrize("kind", ["sentry", "dagger"])
def test_continuous_rotation_applies_pause_before_saving(kind):
    strategy, ctx, dataset = setup_strategy()
    if kind == "dagger":
        strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True))
        strategy._engine = ctx.policy.inference
        strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._episode_duration_s = 0
    saves = []

    def save():
        if not saves:
            ctx.policy.inference.pause.assert_called_once()
            ctx.hardware.robot_wrapper.hold.assert_called_once()
            assert strategy._cached_obs_processed is None
            ctx.runtime.shutdown_event.set()
        saves.append(True)

    dataset.save_episode.side_effect = save
    if kind == "sentry":
        strategy.run(ctx)
    else:
        strategy._run_continuous(ctx)
    assert saves
    # The initial start is the only resume: shutdown during save wins.
    ctx.policy.inference.resume.assert_called_once()


@pytest.mark.parametrize("operation", ["drain", "save"])
@pytest.mark.parametrize("outcome", ["complete", "shutdown", "error"])
def test_highlight_toggle_invalidates_motion_before_blocking_recording_work(operation, outcome):
    strategy = HighlightStrategy(HighlightStrategyConfig())

    def on_tick(tick):
        if tick == 3:
            strategy._save_requested.set()

    ctx, dataset = _make_loop_ctx(fps=200, multiplier=2, num_ticks=8, on_tick=on_tick)
    engine = ctx.policy.inference
    engine.control_thread_owns_policy = False
    engine.failed = False
    ctx.hardware.robot_wrapper.supports_hold = True
    ctx.hardware.robot_wrapper.hardware_failure = None
    strategy._engine = engine
    strategy._interpolator = ActionInterpolator(multiplier=2)
    strategy._ring = RolloutRingBuffer(max_seconds=10, max_memory_mb=1, fps=200)
    if operation == "save":
        strategy._recording_live.set()
    blocked = False
    notifications_at_pause = 0

    def block_once(*_args):
        nonlocal blocked, notifications_at_pause
        if blocked:
            return
        blocked = True
        engine.pause.assert_called_once()
        ctx.hardware.robot_wrapper.hold.assert_called_once()
        assert strategy._cached_obs_processed is None
        assert strategy._interpolator.needs_new_action()
        notifications_at_pause = engine.notify_observation.call_count
        if outcome == "shutdown":
            ctx.runtime.shutdown_event.set()
        elif outcome == "error":
            raise OSError("disk full")

    if operation == "drain":
        dataset.add_frame.side_effect = block_once
    else:
        dataset.save_episode.side_effect = block_once

    if outcome == "error":
        with pytest.raises(OSError, match="disk full"):
            strategy.run(ctx)
    else:
        strategy.run(ctx)

    assert blocked
    if outcome == "complete":
        assert engine.resume.call_count == 2
        fresh = engine.notify_observation.call_args_list[notifications_at_pause].args[0]
        assert fresh["m.pos"] > 3
    else:
        # An error or shutdown during the operation must not restart the worker.
        engine.resume.assert_called_once()
