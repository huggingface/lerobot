# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.rollout.hybrid import HybridConfig, InterventionLimit, PlannerDecision
from lerobot.rollout.inference import PolicyQuery, QueryKind
from lerobot.rollout.inference.hybrid import HybridInferenceEngine
from lerobot.rollout.planner import PlannerConfig
from lerobot.rollout.vlm_agent import NoPolicyEngine, VlmAgentPlanner


@pytest.fixture
def agent(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("lerobot.rollout.inference.hybrid.time.perf_counter", lambda: clock.now)
    cfg = HybridConfig(vlm_only=True, limits={"gripper.pos": InterventionLimit(0, 1, 1, 1, 0.03)})
    engine = HybridInferenceEngine(NoPolicyEngine("Sort cubes"), cfg, list(cfg.limits), 1)
    engine.external_text = MagicMock()
    engine.start()
    engine.resume()
    engine.notify_observation({"gripper.pos": 0.5})
    engine.get_action({})
    return engine, clock


def apply(engine, mode, targets=None):
    value = PlannerDecision(mode, "Cube visible", "Open fingers", "", targets or {}, 2 if targets else 0)
    engine._pending_decision = (value, dict(engine._obs), engine._query_epoch, None)
    return engine.get_action({})


def test_direct_agent_can_move_more_than_three_times_and_report_measured_result(agent):
    engine, clock = agent
    for _ in range(5):
        apply(engine, "intervention", {"gripper.pos": 1})
        clock.now += 2.1
        engine.notify_observation({"gripper.pos": 1.0})
        engine.get_action({})
        assert not engine.terminal
        assert engine._mode == "review"
        assert engine._vlm_feedback["status"] == "reached"
        assert engine._last_execution["commanded_endpoint"] == {"gripper.pos": 1}


def test_nonarrival_is_feedback_and_holds_measured_pose(agent):
    engine, clock = agent
    apply(engine, "intervention", {"gripper.pos": 1})
    clock.now += 2.6
    engine.notify_observation({"gripper.pos": 0.6})
    assert engine.get_action({}).tolist() == pytest.approx([0.6])
    assert engine._vlm_feedback["status"] == "not_reached"
    assert engine._vlm_feedback["residual_robot_units"] == pytest.approx({"gripper.pos": 0.4})
    assert engine._mode == "review" and not engine.terminal


def test_bad_target_rejected_atomically_and_returned_to_agent(agent):
    engine, _ = agent
    assert apply(engine, "intervention", {"gripper.pos": 2}).tolist() == [0.5]
    assert engine._target is None and not engine.terminal
    assert engine._vlm_feedback["status"] == "rejected"
    assert not engine._vlm_feedback["executed"]


def test_manual_goal_still_uses_agent_not_vla_and_invalidates_late_reply(agent):
    engine, _ = agent
    old_epoch = engine._query_epoch
    engine.set_task("Open the right gripper")
    engine.get_action({})
    assert engine._mode == "review"
    assert engine.autosteer_goal == "Open the right gripper"
    engine._pending_decision = (PlannerDecision("done", "s", "r", "", {}, 0), {}, old_epoch, None)
    engine.get_action({})
    assert not engine.terminal


def test_observe_done_reset_and_stale_feedback(agent):
    engine, clock = agent
    apply(engine, "observe")
    assert not engine.terminal
    apply(engine, "done")
    assert engine.terminal
    engine.reset()
    assert not engine.terminal and engine._vlm_feedback is None
    engine.resume()
    clock.now += 1
    assert engine.get_action({}) is None
    assert engine.terminal  # hardware freshness is never bypassed


def test_tool_contract_controller_owns_duration_and_forbids_policy(agent):
    engine, _ = agent
    planner = VlmAgentPlanner(PlannerConfig(), "test", hybrid=engine.config, client=MagicMock())
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, "Sort cubes")
    reply = {
        "tool": "move_gripper",
        "scene": "Cube visible",
        "note": "Open",
        "ee_targets": {},
        "targets": {"gripper.pos": 1},
    }
    decision = planner.parse_reply(reply, query, "Sort cubes")
    assert decision.duration_s == engine.config.max_intervention_s
    with pytest.raises(ValueError):
        planner.parse_reply(reply | {"duration_s": 0.01}, query, "Sort cubes")
    with pytest.raises(ValueError):
        planner.parse_reply(reply | {"tool": "policy"}, query, "Sort cubes")
    assert "There is no VLA" in planner.request_text(query, "Sort cubes")


def test_vlm_only_context_never_loads_checkpoint_or_normalizer(monkeypatch):
    from lerobot.rollout import context
    from lerobot.rollout.configs import RolloutConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig())
    limits = {key: InterventionLimit(-100, 100, 1, 1, 0.1) for key in robot.action_features}
    monkeypatch.setattr(context, "make_robot_from_config", lambda _: robot)
    forbidden = MagicMock(side_effect=AssertionError("VLA code must not run"))
    for name in (
        "_load_pretrained_policy",
        "make_pre_post_processors",
        "create_inference_engine",
        "training_vocabulary",
    ):
        monkeypatch.setattr(context, name, forbidden)
    monkeypatch.setattr(context, "VlmAgentPlanner", MagicMock())
    cfg = RolloutConfig(
        robot=MockRobotConfig(), hybrid=HybridConfig(vlm_only=True, limits=limits), planner=PlannerConfig()
    )
    ctx = context.build_rollout_context(cfg, Event())
    try:
        assert ctx.policy.policy is None
        assert isinstance(ctx.policy.inference.delegate, NoPolicyEngine)
        assert ctx.policy.preprocessor.steps == []
        forbidden.assert_not_called()
    finally:
        robot.disconnect()


def test_malformed_tool_returns_feedback_but_transport_failure_stops(agent):
    engine, _ = agent
    engine._pending_decision = (None, dict(engine._obs), engine._query_epoch, "AgentToolError: unknown tool")
    engine.get_action({})
    assert not engine.terminal and engine._vlm_feedback["status"] == "rejected"
    engine._pending_decision = (None, dict(engine._obs), engine._query_epoch, "TimeoutError: API timeout")
    engine.get_action({})
    assert engine.terminal


def test_vlm_only_rejects_checkpoint_and_rtc(monkeypatch):
    from lerobot.rollout import configs
    from lerobot.rollout.inference import RTCInferenceConfig
    from tests.mocks.mock_robot import MockRobotConfig

    cfg = HybridConfig(vlm_only=True, limits={"x.pos": InterventionLimit(0, 1, 1, 1, 0.1)})
    with pytest.raises(ValueError, match="no RTC/VLA"):
        configs.RolloutConfig(
            robot=MockRobotConfig(), hybrid=cfg, planner=PlannerConfig(), inference=RTCInferenceConfig()
        )
    monkeypatch.setattr(configs.parser, "get_path_arg", lambda _: "model/checkpoint")
    forbidden = MagicMock(side_effect=AssertionError("Must reject before reading checkpoint"))
    monkeypatch.setattr(configs.PreTrainedConfig, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="remove --policy"):
        configs.RolloutConfig(robot=MockRobotConfig(), hybrid=cfg, planner=PlannerConfig())
    forbidden.assert_not_called()
