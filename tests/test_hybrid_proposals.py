# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
from lerobot.rollout.hybrid import HybridConfig, HybridPlanner, InterventionLimit, PlannerDecision
from lerobot.rollout.inference.base import ActionProposal, PolicyQuery, QueryKind
from lerobot.rollout.inference.hybrid import HybridInferenceEngine
from lerobot.rollout.planner import PlannerConfig
from tests.test_interactive_rollout import _RTC_OBS, _make_rtc_engine, _wait_for


def review_decision(mode="accept", **kwargs):
    return PlannerDecision(
        mode,
        "Block still outside bin",
        "Previous grasp pending; next approach aligned",
        "",
        {},
        0,
        execution_status="uncertain",
        intent_status="aligned",
        **kwargs,
    )


@pytest.fixture
def proposal_rig(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("lerobot.rollout.inference.hybrid.time.perf_counter", lambda: clock.now)
    delegate = MagicMock(task="Pick red block", ready=True, failed=False, supports_action_proposals=True)
    delegate.take_action_proposal.return_value = None
    cfg = HybridConfig(
        limits={k: InterventionLimit(-1, 1, 0.2, 0.2, 0.04) for k in ("a.pos", "b.pos")},
        review_policy_chunks=True,
        proposal_execution_steps=2,
    )
    engine = HybridInferenceEngine(delegate, cfg, list(cfg.limits), 2)
    pose = {"a.pos": 0.0, "b.pos": 0.0}
    engine.resume()
    engine.notify_observation(pose)
    engine.get_action({})
    assert engine._mode == "settling"
    clock.now += cfg.settle_s
    engine.notify_observation(pose)
    engine.get_action({})
    delegate.request_action_proposal.assert_called_once_with(pose, "Pick red block")
    proposal = ActionProposal(
        12, torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.8, 0.9]]), pose, "Pick red block", 30
    )
    delegate.take_action_proposal.return_value = proposal
    engine.get_action({})
    assert engine._mode == "review"
    return engine, delegate, clock, proposal


def apply_review(engine, proposal, decision):
    observation = dict(proposal.observation) | {"_hybrid_proposal": {"id": proposal.proposal_id}}
    engine._resolve_query(
        PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal),
        observation,
        lambda *args: decision,
        epoch=engine._query_epoch,
    )
    return engine.get_action({})


def test_only_approved_prefix_executes_and_suffix_is_discarded(proposal_rig):
    engine, delegate, _, proposal = proposal_rig
    assert engine.get_action({}).tolist() == [0, 0]
    output = [apply_review(engine, proposal, review_decision()).tolist()]
    output += [engine.get_action({}).tolist() for _ in range(3)]
    torch.testing.assert_close(
        torch.tensor(output), torch.tensor([[0.1, 0.15], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
    )
    engine.get_action({})
    assert engine._mode == "settling"
    assert engine._proposal is None and engine._approved_actions is None
    delegate.get_action.assert_not_called()
    delegate.resume.assert_not_called()


def test_changed_instruction_requires_a_new_review(proposal_rig):
    engine, delegate, _, proposal = proposal_rig
    d = PlannerDecision("policy", "Red complete", "Select blue", "Pick blue block", {}, 0)
    assert apply_review(engine, proposal, d).tolist() == [0, 0]
    assert engine._mode == "settling"
    assert engine.task == "Pick blue block"
    delegate.get_action.assert_not_called()
    delegate.resume.assert_not_called()


@pytest.mark.parametrize("takeover", ["pause", "reset", "manual"])
def test_late_accept_after_takeover_never_executes(proposal_rig, takeover):
    engine, delegate, _, proposal = proposal_rig
    epoch = engine._query_epoch
    if takeover == "manual":
        engine.set_task("Operator instruction")
    else:
        getattr(engine, takeover)()
    engine._resolve_query(
        PolicyQuery(QueryKind.NEXT_SUBTASK, engine.autosteer_goal),
        proposal.observation,
        lambda *args: review_decision(),
        epoch=epoch,
    )
    assert engine._pending_decision is None
    assert engine._approved_actions is None


def test_moved_robot_rejects_reviewed_proposal(proposal_rig):
    engine, _, _, proposal = proposal_rig
    engine.notify_observation({"a.pos": 0.1, "b.pos": 0.0})
    assert apply_review(engine, proposal, review_decision()) is None
    assert engine.terminal


def test_proposal_id_mismatch_rejects_accept(proposal_rig):
    engine, _, _, proposal = proposal_rig
    wrong = ActionProposal(99, proposal.actions, proposal.observation, proposal.task, 30)
    assert apply_review(engine, wrong, review_decision()) is None
    assert engine.terminal


def test_review_receives_prediction_and_exact_inference_snapshot(proposal_rig):
    engine, _, _, proposal = proposal_rig
    captured = []
    engine.external_text = lambda obs, query, task: captured.append(obs) or review_decision()
    engine.pump_query({"a.pos": 0.001, "b.pos": 0.002})
    assert _wait_for(lambda: bool(captured))
    assert captured[0]["a.pos"] == 0
    assert captured[0]["_hybrid_proposal"]["actions"] == proposal.actions.tolist()
    assert captured[0]["_hybrid_proposal"]["execute_steps"] == 2
    assert captured[0]["_hybrid_proposal"]["task"] == proposal.task


def test_review_requires_failure_or_misalignment_for_correction(proposal_rig):
    engine, _, _, proposal = proposal_rig
    correction = PlannerDecision(
        "intervention",
        "Uncertain",
        "Guess",
        "",
        {"a.pos": 0.1},
        1,
        execution_status="uncertain",
        intent_status="uncertain",
    )
    assert apply_review(engine, proposal, correction) is None
    assert engine.terminal


def test_proposal_request_times_out_while_holding(proposal_rig):
    engine, delegate, clock, proposal = proposal_rig
    engine._begin_review()
    clock.now += engine.config.settle_s
    engine.notify_observation(proposal.observation)
    delegate.take_action_proposal.return_value = None
    engine.get_action({})
    clock.now += engine.config.proposal_timeout_s + 1
    engine.notify_observation(proposal.observation)
    assert engine.get_action({}) is None
    assert engine.terminal
    delegate.get_action.assert_not_called()


def test_rtc_proposal_is_postprocessed_isolated_and_not_queued():
    engine, policy = _make_rtc_engine()
    engine._postprocessor = lambda actions: actions + 0.25
    engine.start()
    try:
        source = dict(_RTC_OBS)
        engine.request_action_proposal(source, "New task")
        source["j1.pos"] = 0.9
        assert _wait_for(policy.in_inference.is_set)
        policy.allow_one_inference()
        assert _wait_for(lambda: not engine._policy_active.is_set())
        proposal = engine.take_action_proposal()
        assert proposal.task == "New task"
        assert torch.all(proposal.actions == 0.25)
        assert proposal.observation["j1.pos"] == 0
        assert engine.get_action(None) is None
        assert engine.action_queue.qsize() == 0
        assert engine.take_action_proposal() is None
        assert len(policy.predicted_tasks) == 1
    finally:
        policy.unblock()
        engine.stop()


def test_rtc_proposal_inflight_at_reset_is_discarded():
    engine, policy = _make_rtc_engine()
    engine.start()
    try:
        engine.request_action_proposal(dict(_RTC_OBS), "Old task")
        assert _wait_for(policy.in_inference.is_set)
        engine.pause()
        engine.discard_actions()
        policy.allow_one_inference()
        assert _wait_for(lambda: len(policy.predicted_tasks) == 1)
        assert engine.take_action_proposal() is None
        assert engine.get_action(None) is None
        engine.request_action_proposal(dict(_RTC_OBS), "Fresh task")
        policy.allow_one_inference()
        assert _wait_for(lambda: not engine._policy_active.is_set())
        assert engine.take_action_proposal().task == "Fresh task"
    finally:
        policy.unblock()
        engine.stop()


def test_planner_requires_structured_assessment_and_logs_proposal(tmp_path):
    cfg = HybridConfig(review_policy_chunks=True, limits={"a.pos": InterventionLimit(-1, 1, 0.1, 0.1, 0.02)})
    planner = HybridPlanner(
        PlannerConfig(log_path=str(tmp_path / "planner.jsonl")),
        "mock",
        hybrid=cfg,
        client=StubVlmClient(lambda _: asdict(review_decision())),
    )
    obs = {
        "a.pos": 0,
        "_hybrid_proposal": {
            "id": 1,
            "action_keys": ["a.pos"],
            "actions": [[0.1]],
            "task": "Pick",
            "fps": 30,
            "execute_steps": 1,
        },
    }
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, "Pick")
    assert planner(obs, query, "Pick").mode == "accept"
    assert '"event": "proposal_review"' in (tmp_path / "planner.jsonl").read_text()
    with pytest.raises(ValueError, match="requires execution_status"):
        planner.parse_reply(
            asdict(PlannerDecision("policy", "Scene", "Reason", "Pick", {}, 0)), query, "Pick"
        )
