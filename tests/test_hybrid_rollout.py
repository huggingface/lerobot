# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
from lerobot.rollout.hybrid import HybridConfig, HybridPlanner, InterventionLimit, PlannerDecision
from lerobot.rollout.inference import PolicyQuery, QueryKind
from lerobot.rollout.inference.hybrid import HybridInferenceEngine
from lerobot.rollout.planner import PlannerConfig


def decision(mode="policy", **kwargs):
    return PlannerDecision.parse(
        {
            "mode": mode,
            "scene": "A block is outside the bin",
            "reason": "Pick the visible block",
            "instruction": "Pick up the red block" if mode == "policy" else "",
            "targets": {},
            "duration_s": 0,
            **kwargs,
        }
    )


@pytest.fixture
def rig(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("lerobot.rollout.inference.hybrid.time.perf_counter", lambda: clock.now)
    delegate = MagicMock(task="Put all cubes in bin", ready=True, failed=False)
    delegate.get_action.return_value = torch.tensor([0.4, 0.5])
    delegate.dispatched_task = "Pick up the red block"
    cfg = HybridConfig(
        limits={
            "joint.pos": InterventionLimit(-1, 1, 0.2, 0.2, 0.02),
            "gripper.pos": InterventionLimit(0, 1, 1, 1, 0.03),
        }
    )
    engine = HybridInferenceEngine(delegate, cfg, list(cfg.limits), 2)
    engine.external_text = lambda *args, **kwargs: decision()
    engine.resume()
    engine.notify_observation({"joint.pos": 0.0, "gripper.pos": 0.5})
    engine.get_action({})
    return engine, delegate, clock


def accept(engine, value):
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal)
    engine._resolve_query(query, engine._obs, lambda *args: value, epoch=engine._query_epoch)
    return engine.get_action({})


def test_review_holds_and_clears_policy_queue(rig):
    engine, delegate, _ = rig
    assert engine.get_action({}).tolist() == [0, 0.5]
    delegate.get_action.assert_not_called()
    delegate.pause.assert_called()
    delegate.discard_actions.assert_called()


def test_policy_window_returns_to_review_and_preserves_interpolation(rig):
    engine, delegate, clock = rig
    assert accept(engine, decision()).tolist() == pytest.approx([0.4, 0.5])
    engine.get_action({})  # first policy action primes interpolation
    engine.get_action({})
    assert delegate.get_action.call_count == 2  # 2x command interpolation
    clock.now += engine.config.policy_window_s
    engine.notify_observation({"joint.pos": 0.3, "gripper.pos": 0.5})
    assert engine.get_action({}).tolist() == pytest.approx([0.3, 0.5])
    assert engine._mode == "review"


def test_intervention_bypasses_policy_and_reobserves_before_resuming(rig):
    engine, delegate, clock = rig
    proposal = decision("intervention", targets={"gripper.pos": 1.0}, duration_s=1)
    assert accept(engine, proposal).tolist() == [0, 0.5]
    clock.now += 0.5
    engine.notify_observation({"joint.pos": 0.0, "gripper.pos": 0.7})
    engine.get_action({})  # interpolate from the held starting pose
    assert engine.get_action({}).tolist() == [0, 0.75]
    clock.now += 0.5
    engine.notify_observation({"joint.pos": 0.0, "gripper.pos": 1.0})
    engine.get_action({})
    assert engine._mode == "review"
    delegate.get_action.assert_not_called()
    assert engine._consecutive == 1


@pytest.mark.parametrize(
    "targets,duration",
    [
        ({"joint.pos": 2}, 1),
        ({"joint.pos": 0.21}, 2),
        ({"joint.pos": 0.2}, 0.1),
        ({"wrong.pos": 0.1}, 1),
        ({"gripper.pos": 0.6, "joint.pos": 2}, 1),
        ({"joint.pos": 0.1}, 3),
    ],
)
def test_rejects_entire_invalid_intervention(rig, targets, duration):
    engine, delegate, _ = rig
    assert accept(engine, decision("intervention", targets=targets, duration_s=duration)) is None
    assert engine.terminal
    assert engine._target is None
    delegate.get_action.assert_not_called()


@pytest.mark.parametrize("mode", ["done", "intervention", "policy"])
def test_late_reply_after_takeover_cannot_act_or_complete(rig, mode):
    engine, delegate, _ = rig
    entered, release = Event(), Event()
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal)
    kwargs = {"targets": {"gripper.pos": 1}, "duration_s": 1} if mode == "intervention" else {}

    def generate(*args):
        entered.set()
        assert release.wait(2)
        return decision(mode, **kwargs)

    worker = Thread(
        target=engine._resolve_query,
        args=(query, engine._obs, generate),
        kwargs={"epoch": engine._query_epoch},
    )
    worker.start()
    assert entered.wait(2)
    engine.stop_autosteer()
    engine.set_task("Manual recovery")
    release.set()
    worker.join(2)
    assert engine._pending_decision is None
    assert not engine.terminal
    engine.get_action({})
    delegate.set_task.assert_called_with("Manual recovery")


def test_late_completion_after_reset_same_goal_is_discarded(rig):
    engine, _, _ = rig
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal)
    old_epoch = engine._query_epoch
    engine.reset()
    engine.resume()
    engine._resolve_query(query, {}, lambda *args: decision("done"), epoch=old_epoch)
    assert engine._pending_decision is None
    assert not engine.terminal


def test_completion_is_status_and_never_a_policy_task(rig):
    engine, delegate, _ = rig
    accept(engine, decision("done"))
    assert engine.terminal
    answer = engine._ready_answers.pop()
    assert answer.completed and answer.ok
    delegate.set_task.assert_not_called()


def test_deadline_and_drift_reject_proposal(rig):
    engine, delegate, clock = rig
    clock.now += engine.config.review_timeout_s + 1
    engine.notify_observation({"joint.pos": 0, "gripper.pos": 0.5})
    accept(engine, decision())
    assert engine.terminal
    delegate.resume.assert_not_called()


def test_drift_while_model_thinks_rejects_intervention(rig):
    engine, delegate, _ = rig
    engine.notify_observation({"joint.pos": 0.1, "gripper.pos": 0.5})
    assert accept(engine, decision("intervention", targets={"gripper.pos": 1}, duration_s=1)) is None
    assert engine.terminal
    delegate.resume.assert_not_called()


def test_no_hold_or_motion_from_stale_feedback(rig):
    engine, _, clock = rig
    clock.now += engine.config.max_observation_age_s + 0.1
    assert engine.get_action({}) is None
    assert engine.terminal


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
def test_non_numeric_targets_rejected(bad):
    with pytest.raises(ValueError):
        decision("intervention", targets={"joint.pos": bad}, duration_s=1)


def test_hybrid_prompt_and_logging(tmp_path):
    cfg = HybridConfig(limits={"gripper.pos": InterventionLimit(0, 1, 1, 1, 0.03, "0 closed, 1 open")})
    planner = HybridPlanner(
        PlannerConfig(instructions=["Pick up the red block"], log_path=str(tmp_path / "planner.jsonl")),
        "test_robot",
        hybrid=cfg,
        client=StubVlmClient(lambda _: asdict(decision())),
    )
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, "Put all cubes in bin")
    result = planner({"gripper.pos": 0.5}, query, "Put all cubes in bin")
    assert result.mode == "policy"
    assert "Overall goal" in planner.request_text(query, "previous")
    assert "0 closed, 1 open" in planner.request_text(query, "previous")
    assert (tmp_path / "planner.jsonl").exists()


def test_rtc_discard_invalidates_inflight_epoch():
    from threading import Lock

    from lerobot.rollout.inference.rtc import RTCInferenceEngine

    engine = object.__new__(RTCInferenceEngine)
    engine._obs_lock = Lock()
    engine._reset_epoch = 8
    engine._obs_holder = {"obs": {"old": 1}}
    engine._action_queue = MagicMock()
    engine.discard_actions()
    assert engine._reset_epoch == 9
    assert engine._obs_holder["obs"] is None
    engine._action_queue.clear.assert_called_once()


def test_rtc_handover_discards_a_chunk_already_generating(caplog):
    import logging

    from tests.test_interactive_rollout import _RTC_OBS, _running_rtc_engine, _wait_for

    with _running_rtc_engine() as (engine, policy):
        assert _wait_for(policy.in_inference.is_set)
        engine.pause()
        engine.discard_actions()
        with caplog.at_level(logging.INFO, logger="lerobot.rollout.inference.rtc"):
            policy.allow_one_inference()
            assert _wait_for(lambda: any("Discarding action chunk" in r.getMessage() for r in caplog.records))
        assert engine.action_queue.qsize() == 0
        engine.set_task("Fresh task after intervention")
        engine.discard_actions()
        engine.notify_observation(dict(_RTC_OBS))
        engine.resume()
        assert _wait_for(policy.in_inference.is_set)
        policy.allow_one_inference()
        assert _wait_for(lambda: engine.get_action(None) is not None)
        assert engine.dispatched_task == "Fresh task after intervention"


def test_hybrid_holds_are_recorded_with_two_times_interpolation():
    from tests.test_rollout import _make_loop_ctx, _make_sentry

    ctx, dataset = _make_loop_ctx(fps=200, multiplier=2, num_ticks=8)
    ctx.hardware.robot_wrapper.get_observation.side_effect = None
    tick = 0

    def observe():
        nonlocal tick
        tick += 1
        if tick == 8:
            ctx.runtime.shutdown_event.set()
        return {"m.pos": 0.2}

    ctx.hardware.robot_wrapper.get_observation.side_effect = observe
    delegate = MagicMock(task="goal", ready=True, failed=False)
    cfg = HybridConfig(limits={"m.pos": InterventionLimit(-1, 1, 0.1, 0.1, 0.02)}, settle_s=10)
    engine = HybridInferenceEngine(delegate, cfg, ["m.pos"], 2)
    ctx.policy.inference = engine
    strategy = _make_sentry(ctx, 2)
    strategy._interpolator = engine.control_interpolator
    strategy.run(ctx)
    assert dataset.add_frame.call_count == 4
    assert ctx.hardware.robot_wrapper.send_action.call_count == 8
    delegate.get_action.assert_not_called()


def test_partial_contract_rejected():
    delegate = MagicMock(task="goal")
    with pytest.raises(ValueError, match="exactly"):
        HybridInferenceEngine(delegate, HybridConfig(), ["missing.pos"], 1)


def test_unreached_intervention_stops_instead_of_resuming(rig):
    engine, delegate, clock = rig
    accept(engine, decision("intervention", targets={"gripper.pos": 1}, duration_s=1))
    clock.now += 1.6
    engine.notify_observation({"joint.pos": 0, "gripper.pos": 0.5})
    assert engine.get_action({}) is None
    assert engine.terminal
    delegate.resume.assert_not_called()


def test_timeout_does_not_release_a_still_running_http_request(rig):
    engine, _, clock = rig
    entered, release = Event(), Event()

    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return decision("done")

    engine.external_text = generate
    clock.now += engine.config.settle_s
    engine.notify_observation({"joint.pos": 0, "gripper.pos": 0.5})
    engine.pump_query(engine._obs)
    assert entered.wait(2)
    try:
        clock.now += engine.config.review_timeout_s + 1
        engine.notify_observation({"joint.pos": 0, "gripper.pos": 0.5})
        engine.get_action({})
        assert engine.terminal and engine._query_in_flight
        engine.reset()
        engine.resume()
        assert engine._query_in_flight  # cannot overlap requests after restarting
    finally:
        release.set()
