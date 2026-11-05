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

from lerobot.inference import InferenceEngine, RemoteInferenceConfig
from lerobot.rollout.configs import BaseStrategyConfig, RolloutConfig
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator

pytestmark = pytest.mark.dataset


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


def skip_return_waits(monkeypatch):
    """Skip the return move's sleeps, and end its settle wait at the first tick without progress."""
    monkeypatch.setattr("lerobot.rollout.strategies.core.precise_sleep", lambda _: None)
    monkeypatch.setattr("lerobot.rollout.strategies.core._RETURN_SETTLE_S", 0.0)


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
    applied_before_hold = robot.sent[-1].copy()

    engine.allowed = False
    assert send_next_action(obs, obs, ctx, interpolator) is None
    assert engine.pulls == 2
    assert interpolator.needs_new_action()
    assert robot.sent[-1] == applied_before_hold
    assert applied_before_hold != {"joint.pos": 7.0}
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
    skip_return_waits(monkeypatch)
    strategy._teardown_hardware(ctx.hardware, return_to_initial_position=return_home)
    assert robot.sent[0] == {"joint.pos": 7.0}
    if return_home:
        positions = [action["joint.pos"] for action in robot.sent]
        assert positions[-1] == 0.0
        assert positions == sorted(positions, reverse=True)
        assert robot.reads > 2  # the return keeps reading the robot until it settles
    else:
        assert robot.sent == [{"joint.pos": 7.0}]
        assert robot.reads == 1
    assert not robot.is_connected


class SpeedLimitedRobot(PositionRobot):
    """Moves toward each command by at most `speed`, like a robot that limits its own joint speed."""

    def __init__(self, speed):
        super().__init__()
        self.speed = speed

    def send_action(self, action):
        error = action["joint.pos"] - self.position
        self.position += max(-self.speed, min(self.speed, error))
        return super().send_action(action)


def return_home(robot, monkeypatch, **kwargs):
    monkeypatch.setattr("lerobot.rollout.strategies.core._RETURN_SETTLE_S", 0.05)
    hw = SimpleNamespace(robot_wrapper=ThreadSafeRobot(robot), initial_position={"joint.pos": 0.0})
    return BaseStrategy.return_to_initial_position(hw, duration_s=0.02, fps=500, **kwargs)


def test_return_waits_for_a_speed_limited_robot_to_arrive(monkeypatch):
    robot = SpeedLimitedRobot(speed=0.1)
    assert return_home(robot, monkeypatch)
    assert robot.position == 0.0  # the 0.02 s interpolation alone leaves it at 6.0


def test_return_fails_while_the_robot_is_still_moving_at_the_timeout(monkeypatch, caplog):
    robot = SpeedLimitedRobot(speed=0.01)
    assert not return_home(robot, monkeypatch, timeout_s=0.05)
    assert "still moving" in caplog.text


def test_return_warns_when_the_robot_stops_short(monkeypatch, caplog):
    assert return_home(PositionRobot(), monkeypatch)  # this robot never moves
    assert "stopped 7 from its initial position, 0% of the way back" in caplog.text


def test_return_ends_as_soon_as_a_stop_is_requested(monkeypatch):
    robot = SpeedLimitedRobot(speed=0.1)
    assert not return_home(robot, monkeypatch, stop_requested=lambda: len(robot.sent) >= 3)
    assert len(robot.sent) == 3


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("invalid_first", [False, True])
def test_return_rejects_nonfinite_settle_observation(
    monkeypatch, invalid: float, invalid_first: bool
) -> None:
    robot = PositionRobot()
    keys = ["joint.pos", "other.pos"] if invalid_first else ["other.pos", "joint.pos"]
    readings = iter([dict.fromkeys(keys, 7.0)])
    monkeypatch.setattr(
        robot, "get_observation", lambda: next(readings, {"joint.pos": invalid, "other.pos": 0.0})
    )
    skip_return_waits(monkeypatch)
    hw = SimpleNamespace(robot_wrapper=ThreadSafeRobot(robot), initial_position=dict.fromkeys(keys, 0.0))
    assert not BaseStrategy.return_to_initial_position(hw, duration_s=0.02, fps=50)


@pytest.mark.parametrize("operation", ["send_action", "hold"])
def test_hardware_io_failure_prevents_teardown_motion(operation, monkeypatch, caplog):
    engine = GateEngine()
    engine.failed = True
    engine.stop = Mock()
    ctx, robot = make_dispatch_context(engine)
    wrapper = ctx.hardware.robot_wrapper
    method = Mock(side_effect=OSError("device unavailable"))
    monkeypatch.setattr(robot, "send_action", method)
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


def test_command_failure_is_latched_and_does_not_treat_interrupt_as_io_failure(monkeypatch):
    wrapper = ThreadSafeRobot(PositionRobot())
    original_send = wrapper.inner.send_action
    monkeypatch.setattr(wrapper.inner, "send_action", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        wrapper.send_action({"joint.pos": 1.0})
    assert wrapper.hardware_failure is None
    monkeypatch.setattr(wrapper.inner, "send_action", Mock(side_effect=OSError("first write failed")))
    with pytest.raises(OSError):
        wrapper.send_action({"joint.pos": 1.0})
    failure = wrapper.hardware_failure
    monkeypatch.setattr(wrapper.inner, "send_action", original_send)
    wrapper.send_action({"joint.pos": 1.0})
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
    skip_return_waits(monkeypatch)
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
    skip_return_waits(monkeypatch)
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._teardown_hardware(ctx.hardware)
    failed_io.assert_called_once()
    assert robot.sent == []
    assert (ctx.hardware.robot_wrapper.hardware_failure is not None) == (operation == "write")
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


@pytest.mark.parametrize(("bad_position", "source"), [(float("nan"), "observed"), (float("inf"), "initial")])
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
    monkeypatch.setattr(
        robot, "send_action", lambda action: events.append(("action", action.copy())) or action
    )
    monkeypatch.setattr(engine, "stop", lambda: events.append(("engine_stop", None)))
    monkeypatch.setattr(robot, "disconnect", lambda: events.append(("disconnect", None)))
    skip_return_waits(monkeypatch)
    strategy._teardown_hardware(ctx.hardware)
    assert events[0] == ("action", {"joint.pos": 7.0})
    assert events[1] == ("engine_stop", None)
    assert events[-2] == ("action", {"joint.pos": 0.0})
    assert events[-1] == ("disconnect", None)
    assert strategy._interpolator.get() is None
    assert strategy._cached_obs_processed is None


def test_hold_before_first_command_retains_initial_measured_pose_and_rejects_velocity_modes():
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


def test_hold_retains_actual_clipped_target_and_gripper_instead_of_new_measurements(monkeypatch):
    robot = PositionRobot()
    robot.action_features = {"joint.pos": float, "gripper.pos": float}
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    sent = []

    def clipped_send(action):
        sent.append(action.copy())
        return {"joint.pos": min(action["joint.pos"], 8.0), "gripper.pos": action["gripper.pos"]}

    monkeypatch.setattr(robot, "send_action", clipped_send)
    actual = wrapper.send_action({"joint.pos": 20.0, "gripper.pos": 4.0})
    actual["gripper.pos"] = 99.0  # The retained command must be a private snapshot.
    monkeypatch.setattr(robot, "get_observation", lambda: {"joint.pos": 3.0, "gripper.pos": 50.0})
    wrapper.get_observation()
    wrapper.hold()
    wrapper.hold()
    assert sent == [
        {"joint.pos": 20.0, "gripper.pos": 4.0},
        {"joint.pos": 8.0, "gripper.pos": 4.0},
        {"joint.pos": 8.0, "gripper.pos": 4.0},
    ]


def test_repeated_hold_retains_each_driver_returned_target(monkeypatch):
    robot = PositionRobot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    wrapper.get_observation()
    sent = []

    def clipped_send(action):
        sent.append(action.copy())
        return {"joint.pos": action["joint.pos"] - 1.0}

    monkeypatch.setattr(robot, "send_action", clipped_send)
    wrapper.hold()
    wrapper.hold()
    assert sent == [{"joint.pos": 7.0}, {"joint.pos": 6.0}]


@pytest.mark.parametrize("observation", [None, {}, {"joint.pos": float("nan")}])
def test_first_hold_requires_complete_finite_measured_positions(observation, monkeypatch):
    robot = PositionRobot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    if observation is not None:
        monkeypatch.setattr(robot, "get_observation", lambda: observation)
        wrapper.get_observation()
    with pytest.raises(RuntimeError, match="Position hold requires"):
        wrapper.hold()
    assert robot.sent == []


@pytest.mark.parametrize(
    ("bad_return", "operation"),
    [(None, "send_action"), ({}, "hold"), ({"joint.pos": float("nan")}, "send_action")],
)
def test_hold_never_substitutes_requested_action_for_invalid_driver_return(
    bad_return, operation, monkeypatch
):
    robot = PositionRobot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    wrapper.get_observation()
    monkeypatch.setattr(robot, "send_action", lambda action: bad_return)
    with pytest.raises(RuntimeError, match="Position hold requires"):
        if operation == "hold":
            wrapper.hold()
        else:
            wrapper.send_action({"joint.pos": 2.0})
    assert wrapper.hardware_failure is not None


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
    remote = RemoteInferenceConfig(deployment="test", semantics="so101-degrees-v1")
    cfg = RolloutConfig(robot=SimpleNamespace(), inference=remote)
    assert cfg.policy is None
    assert cfg.device is None
    with pytest.raises(ValueError, match="policy/device"):
        RolloutConfig(robot=SimpleNamespace(), inference=remote, device="cpu")


@pytest.mark.parametrize("field", ["refill_seconds", "action_timeout_s", "max_observation_age_s"])
def test_remote_budgets_reject_nonfinite_values(field):
    with pytest.raises(ValueError, match="finite and positive"):
        RemoteInferenceConfig(deployment="test", semantics="profile", **{field: float("nan")})
