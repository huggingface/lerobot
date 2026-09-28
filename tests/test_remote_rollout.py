# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Control-thread boundaries shared by asynchronous inference backends."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.rollout.configs import BaseStrategyConfig, RolloutConfig
from lerobot.rollout.inference import InferenceEngine, RemoteInferenceConfig
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator


class PositionRobot:
    supports_position_hold = True
    action_features = {"joint.pos": float}
    robot_type = "test_position_robot"
    is_connected = True

    def __init__(self):
        self.position = 7.0
        self.reads = 0
        self.sent = []

    def get_observation(self):
        self.reads += 1
        return {"joint.pos": self.position}

    def send_action(self, action):
        self.sent.append(action.copy())
        return action

    def disconnect(self):
        self.is_connected = False


class GateEngine(InferenceEngine):
    control_thread_owns_policy = True
    supports_text_queries = True
    failed = False

    def __init__(self):
        super().__init__("original task")
        self.allowed = True
        self.pulls = 0
        self.holds = 0
        self.stop_during_pull = False

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def dispatch_allowed(self):
        return self.allowed

    def acknowledge_hold(self):
        self.holds += 1

    def get_action(self, obs_frame):
        self.pulls += 1
        if self.stop_during_pull:
            self.allowed = False
        return torch.tensor([10.0 * self.pulls])


def make_dispatch_context(engine):
    robot = PositionRobot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    wrapper.get_observation()
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(robot_wrapper=wrapper, initial_position={"joint.pos": 0.0}, teleop=None),
        data=SimpleNamespace(
            dataset_features={
                "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}
            },
            ordered_action_keys=["joint.pos"],
        ),
        processors=SimpleNamespace(robot_action_processor=lambda pair: pair[0]),
    )
    return ctx, robot


def test_hold_clears_interpolation_on_a_tick_without_a_queue_pull():
    engine = GateEngine()
    ctx, robot = make_dispatch_context(engine)
    interpolator = ActionInterpolator(multiplier=3)
    obs = {"joint.pos": 7.0}
    send_next_action(obs, obs, ctx, interpolator)
    send_next_action(obs, obs, ctx, interpolator)
    assert not interpolator.needs_new_action()
    assert engine.pulls == 2

    engine.allowed = False
    assert send_next_action(obs, obs, ctx, interpolator) is None
    assert engine.pulls == 2
    assert interpolator.needs_new_action()
    assert robot.sent[-1] == {"joint.pos": 7.0}
    assert robot.reads == 1
    assert engine.holds == 1


def test_permission_revoked_during_pull_cannot_dispatch_returned_action():
    engine = GateEngine()
    engine.stop_during_pull = True
    ctx, robot = make_dispatch_context(engine)
    assert send_next_action({"joint.pos": 7.0}, {}, ctx, ActionInterpolator()) is None
    assert robot.sent == [{"joint.pos": 7.0}]


def test_terminal_fault_teardown_holds_and_does_not_home():
    engine = GateEngine()
    engine.failed = True
    ctx, robot = make_dispatch_context(engine)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._teardown_hardware(ctx.hardware, return_to_initial_position=True)
    assert robot.sent == [{"joint.pos": 7.0}]
    assert not robot.is_connected


def test_hold_freezes_first_measured_pose_and_rejects_velocity_modes():
    wrapper = ThreadSafeRobot(PositionRobot())
    wrapper.configure_position_hold()
    wrapper.get_observation()
    wrapper.hold()
    wrapper.inner.position = 9.0
    wrapper.get_observation()
    wrapper.hold()
    assert wrapper.inner.sent == [{"joint.pos": 7.0}, {"joint.pos": 7.0}]
    wrapper.inner.action_features = {"joint.pos": float, "base.vel": float}
    with pytest.raises(ValueError, match="position-hold"):
        wrapper.configure_position_hold()


def test_same_text_autosteer_restart_discards_previous_intent():
    engine = GateEngine()
    engine.start_autosteer("same goal", 0)

    def restart_during_query(obs, query):
        engine.start_autosteer("same goal", 0)
        assert not engine.ask("second query")
        return "obsolete answer"

    engine._generate_text = restart_during_query
    engine.pump_query({"state": 1})
    assert engine.task == "original task"
    assert not engine._query_in_flight


def test_vqa_after_stopping_autosteer_captures_current_intent():
    engine = GateEngine()
    engine.start_autosteer("goal", 0)
    engine.stop_autosteer()
    answers = []
    engine.set_answer_observer(answers.append)
    engine._generate_text = lambda obs, query: "visible cube"
    engine._query_context_valid = lambda query: query.intent_generation == engine.query_intent_generation
    assert engine.ask("what is visible?")
    engine.pump_query({"state": 1})
    assert len(answers) == 1
    assert answers[0].answer == "visible cube"


def test_remote_rollout_config_requires_no_local_policy(monkeypatch):
    monkeypatch.setattr("lerobot.rollout.configs.parser.get_path_arg", lambda _: None)
    remote = RemoteInferenceConfig(deployment="test", semantics="so101-degrees-v1", hold_mode="position")
    cfg = RolloutConfig(robot=SimpleNamespace(), inference=remote)
    assert cfg.policy is None
    assert cfg.device is None
    with pytest.raises(ValueError, match="policy/device"):
        RolloutConfig(robot=SimpleNamespace(), inference=remote, device="cpu")


@pytest.mark.parametrize("field", ["refill_seconds", "action_timeout_s", "max_observation_age_s"])
def test_remote_budgets_reject_nonfinite_values(field):
    with pytest.raises(ValueError, match="finite and positive"):
        RemoteInferenceConfig(
            deployment="test", semantics="profile", hold_mode="position", **{field: float("nan")}
        )
