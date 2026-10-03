# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Control-thread boundaries shared by asynchronous inference backends."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets")

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


@pytest.mark.parametrize("return_home", [True, False])
def test_terminal_fault_teardown_honors_configured_return(return_home, monkeypatch):
    engine = GateEngine()
    engine.failed = True
    ctx, robot = make_dispatch_context(engine)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", lambda _: None)
    strategy._teardown_hardware(ctx.hardware, return_to_initial_position=return_home)
    assert robot.sent[0] == {"joint.pos": 7.0}
    if return_home:
        positions = [action["joint.pos"] for action in robot.sent]
        assert positions[-1] == 0.0
        assert positions == sorted(positions, reverse=True)
        assert robot.reads == 2
    else:
        assert robot.sent == [{"joint.pos": 7.0}]
        assert robot.reads == 1
    assert not robot.is_connected


@pytest.mark.parametrize("operation", ["get_observation", "send_action", "hold"])
def test_hardware_io_failure_prevents_teardown_motion(operation, monkeypatch, caplog):
    engine = GateEngine()
    engine.failed = True
    engine.stop = Mock()
    ctx, robot = make_dispatch_context(engine)
    wrapper = ctx.hardware.robot_wrapper
    failing_method = "get_observation" if operation == "get_observation" else "send_action"
    method = Mock(side_effect=OSError("device unavailable"))
    monkeypatch.setattr(robot, failing_method, method)
    with pytest.raises(OSError, match="device unavailable"):
        if operation == "send_action":
            wrapper.send_action({"joint.pos": 4.0})
        else:
            getattr(wrapper, operation)()
    assert wrapper.hardware_failure is not None
    assert "device unavailable" in wrapper.hardware_failure
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy.hold_control_state(ctx.hardware)
    strategy._teardown_hardware(ctx.hardware, return_to_initial_position=True)
    assert method.call_count == 1, "A failed hardware interface must not be retried for hold or homing"
    assert robot.sent == []
    assert not robot.is_connected
    engine.stop.assert_called_once()
    assert "robot I/O failed" in caplog.text


def test_hardware_failure_is_latched_and_does_not_treat_interrupt_as_io_failure(monkeypatch):
    wrapper = ThreadSafeRobot(PositionRobot())
    original_read = wrapper.inner.get_observation
    monkeypatch.setattr(wrapper.inner, "get_observation", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        wrapper.get_observation()
    assert wrapper.hardware_failure is None
    monkeypatch.setattr(wrapper.inner, "get_observation", Mock(side_effect=OSError("first read failed")))
    with pytest.raises(OSError):
        wrapper.get_observation()
    failure = wrapper.hardware_failure
    monkeypatch.setattr(wrapper.inner, "get_observation", original_read)
    wrapper.get_observation()
    assert wrapper.hardware_failure == failure
    monkeypatch.setattr(wrapper.inner, "send_action", Mock(side_effect=OSError("later write failed")))
    with pytest.raises(OSError):
        wrapper.send_action({"joint.pos": 1.0})
    assert wrapper.hardware_failure == failure


def test_teardown_hold_failure_still_stops_engine_and_disconnects(monkeypatch, caplog):
    engine = GateEngine()
    engine.failed = True
    engine.stop = Mock()
    ctx, robot = make_dispatch_context(engine)
    failed_write = Mock(side_effect=OSError("hold write failed"))
    monkeypatch.setattr(robot, "send_action", failed_write)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._teardown_hardware(ctx.hardware)
    engine.stop.assert_called_once()
    failed_write.assert_called_once()
    assert not robot.is_connected
    assert "hold write failed" in caplog.text


def test_engine_stop_failure_still_runs_local_shutdown(monkeypatch):
    engine = GateEngine()
    engine.failed = True
    engine.stop = Mock(side_effect=RuntimeError("remote close failed"))
    ctx, robot = make_dispatch_context(engine)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="remote close failed"):
        strategy._teardown_hardware(ctx.hardware)
    assert robot.sent[0] == {"joint.pos": 7.0}
    assert robot.sent[-1] == {"joint.pos": 0.0}
    assert not robot.is_connected


@pytest.mark.parametrize("operation", ["read", "write"])
def test_return_move_io_failure_aborts_and_disconnects(operation, monkeypatch, caplog):
    engine = GateEngine()
    ctx, robot = make_dispatch_context(engine)
    # A normal shutdown enters the return move directly, without a fault hold.
    method = "get_observation" if operation == "read" else "send_action"
    failed_io = Mock(side_effect=OSError("return move failed"))
    monkeypatch.setattr(robot, method, failed_io)
    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", lambda _: None)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._teardown_hardware(ctx.hardware)
    failed_io.assert_called_once()
    assert ctx.hardware.robot_wrapper.hardware_failure is not None
    assert not robot.is_connected
    assert "return move failed" in caplog.text


def test_robot_disconnect_failure_does_not_skip_teleoperator_cleanup(monkeypatch):
    ctx, robot = make_dispatch_context(GateEngine())
    teleop = SimpleNamespace(is_connected=True, disconnect=Mock())
    ctx.hardware.teleop = teleop
    monkeypatch.setattr(robot, "disconnect", Mock(side_effect=OSError("disconnect failed")))
    strategy = BaseStrategy(BaseStrategyConfig())
    with pytest.raises(OSError, match="disconnect failed"):
        strategy._teardown_hardware(ctx.hardware, return_to_initial_position=False)
    teleop.disconnect.assert_called_once()


@pytest.mark.parametrize("disable_torque", [True, False])
def test_disconnect_logs_torque_configuration_without_claiming_pose_retention(disable_torque, caplog):
    ctx, robot = make_dispatch_context(GateEngine())
    robot.config = SimpleNamespace(disable_torque_on_disconnect=disable_torque)
    strategy = BaseStrategy(BaseStrategyConfig())
    with caplog.at_level("INFO"):
        strategy._teardown_hardware(ctx.hardware, return_to_initial_position=False)
    assert f"disable_torque_on_disconnect={disable_torque}" in caplog.text
    assert "config" in caplog.text.lower()
    assert "leaving robot in final pose" not in caplog.text


def test_teardown_without_initial_position_reports_missing_capture(caplog):
    ctx, robot = make_dispatch_context(GateEngine())
    ctx.hardware.initial_position = None
    strategy = BaseStrategy(BaseStrategyConfig())
    with caplog.at_level("INFO"):
        strategy._teardown_hardware(ctx.hardware, return_to_initial_position=True)
    assert "captur" in caplog.text.lower()
    assert "disabled by config" not in caplog.text
    assert robot.sent == []
    assert not robot.is_connected


@pytest.mark.parametrize("bad_position", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("source", ["observed", "initial"])
def test_return_move_rejects_nonfinite_positions_before_sending(bad_position, source, caplog):
    ctx, robot = make_dispatch_context(GateEngine())
    if source == "observed":
        robot.position = bad_position
    else:
        ctx.hardware.initial_position["joint.pos"] = bad_position
    assert not BaseStrategy.return_to_initial_position(ctx.hardware)
    assert robot.sent == []
    assert "finite" in caplog.text


def test_return_move_requires_all_target_joints(monkeypatch):
    ctx, robot = make_dispatch_context(GateEngine())
    monkeypatch.setattr(robot, "get_observation", lambda: {})
    assert not BaseStrategy.return_to_initial_position(ctx.hardware)
    assert robot.sent == []


def test_terminal_fault_shutdown_orders_hold_stop_home_disconnect(monkeypatch):
    engine = GateEngine()
    engine.failed = True
    ctx, robot = make_dispatch_context(engine)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._interpolator = ActionInterpolator()
    strategy._interpolator.add(torch.tensor([123.0]))
    strategy._cached_obs_processed = {"joint.pos": 123.0}
    events = []
    monkeypatch.setattr(robot, "send_action", lambda action: events.append(("action", action.copy())))
    monkeypatch.setattr(engine, "stop", lambda: events.append(("engine_stop", None)))
    monkeypatch.setattr(robot, "disconnect", lambda: events.append(("disconnect", None)))
    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", lambda _: None)
    strategy._teardown_hardware(ctx.hardware)
    assert events[0] == ("action", {"joint.pos": 7.0})
    assert events[1] == ("engine_stop", None)
    assert events[-2] == ("action", {"joint.pos": 0.0})
    assert events[-1] == ("disconnect", None)
    assert strategy._interpolator.get() is None
    assert strategy._cached_obs_processed is None


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


def test_omx_hold_uses_position_driver_without_an_extra_sensor_read():
    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    measured = {name: float(index) for index, name in enumerate(names)}
    writes = []
    reads = []
    robot = OmxFollower.__new__(OmxFollower)
    robot.id = "test_omx"
    robot.config = OmxFollowerConfig(port="unused")
    robot.cameras = {}
    robot.bus = SimpleNamespace(
        motors=dict.fromkeys(names),
        is_connected=True,
        sync_read=lambda register: reads.append(register) or measured.copy(),
        sync_write=lambda register, values: writes.append((register, values.copy())),
    )
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    wrapper.get_observation()
    wrapper.hold()
    wrapper.hold()
    assert reads == ["Present_Position"]
    assert writes == [("Goal_Position", measured), ("Goal_Position", measured)]
    assert not robot.config.use_degrees


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
