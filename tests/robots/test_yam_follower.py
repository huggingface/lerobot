#!/usr/bin/env python

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

import math
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.yam_follower import (
    YamFollower,
    YamFollowerConfig,
    mit_arm,
    yam_follower as robot_module,
)
from lerobot.robots.yam_follower.config_yam_follower import (
    JOINT_LIMITS_RAD,
    MOTOR_NAMES,
    YAM_FEATURE_NAMES,
    motor_feature_names,
)


def make_robot(tmp_path, **overrides):
    config = {
        "id": "test",
        "calibration_dir": tmp_path,
        "port": "can0",
        "gripper_closed_deg": math.degrees(0.1),
        "gripper_open_deg": math.degrees(6.1),
        "gravity_compensation": False,
        # Equal to the former radian defaults, so the control numbers below stay unchanged.
        "max_joint_speed_deg_s": math.degrees(0.3),
        "max_tracking_error_deg": math.degrees(0.15),
        **overrides,
    }
    return YamFollower(YamFollowerConfig(**config))


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    return make_robot(tmp_path)


def params(robot):
    return robot_module.yam_arm_params(robot.config)


def raw_states(gripper=0.1):
    position = np.array([0, 0, 0, 0, 0, 0, gripper], dtype=float)
    return mit_arm.MotorStates(position=position, velocity=np.zeros(7), torque=np.zeros(7))


def joint_state(position=None):
    position = np.zeros(7) if position is None else np.asarray(position, dtype=float)
    return mit_arm.JointState(position=position, velocity=np.zeros(7), torque=np.zeros(7))


def mock_bus(states=None):
    bus = MagicMock(spec=robot_module._YamBus)
    bus.enabled = False
    bus.read_states.return_value = raw_states() if states is None else states
    bus.can_timeouts.return_value = dict.fromkeys(MOTOR_NAMES, 8000)
    return bus


def attach_bus(robot, bus):
    robot.bus = robot.servo.bus = bus
    return bus


def mock_hardware(robot, monkeypatch):
    attach_bus(robot, mock_bus())
    monkeypatch.setattr(robot.servo, "start", MagicMock())


def ready(robot):
    robot._connected = True
    robot.config.read_only = False
    robot.servo.active = True
    robot.servo.updated_at = time.monotonic()
    return dict.fromkeys(robot.action_features, 0.0)


@pytest.mark.parametrize("opened", [6.1, -5.9])
def test_units_order_and_gripper_polarity(robot, opened):
    robot.config.gripper_open_deg = math.degrees(opened)
    raw = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, (0.1 + opened) / 2]
    robot.config.joint_signs[0] = -1
    robot.config.joint_offsets_deg[0] = math.degrees(0.2)
    decoded = mit_arm.motor_to_joint(np.asarray(raw), params(robot))
    np.testing.assert_allclose(decoded, [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.5])
    np.testing.assert_allclose(mit_arm.joint_to_motor(decoded, params(robot)), raw)


def test_public_units_are_degrees_and_percent_or_internal_units():
    internal = np.array([0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.5])
    public = robot_module.to_public(internal, use_degrees=True)
    np.testing.assert_allclose(public, [*np.rad2deg(internal[:6]), 50.0])
    np.testing.assert_allclose(robot_module.from_public(public, use_degrees=True), internal)
    np.testing.assert_allclose(robot_module.to_public(internal, use_degrees=False), internal)
    np.testing.assert_allclose(robot_module.from_public(internal, use_degrees=False), internal)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_actions_clip_finite_overshoot_and_reject_non_finite(use_degrees):
    action = dict.fromkeys(YAM_FEATURE_NAMES, 0.0)
    action["gripper.pos"] = 100.4 if use_degrees else 1.004
    action["joint_2.pos"] = -1.0  # below the 0 lower limit in either unit
    target = robot_module.action_to_target(action, use_degrees)
    assert target[6] == 1.0
    assert target[1] == 0.0
    action["joint_1.pos"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        robot_module.action_to_target(action, use_degrees)


def test_measured_pose_has_wider_margin_than_action(robot):
    states = raw_states(gripper=6.4)
    states.position[0] = JOINT_LIMITS_RAD[0][0] - 0.08
    bus = attach_bus(robot, mock_bus(states))
    measured = mit_arm.read_joint_state(bus, params(robot)).position
    assert measured[0] == pytest.approx(states.position[0])
    assert measured[6] == 1.0
    # Actions are clipped to the model limits instead of being rejected.
    action = dict(zip(YAM_FEATURE_NAMES, robot_module.to_public(measured, True), strict=True))
    assert robot_module.action_to_target(action, use_degrees=True)[0] == JOINT_LIMITS_RAD[0][0]
    states.position[0] = JOINT_LIMITS_RAD[0][0] - 0.2
    with pytest.raises(ValueError, match="outside"):
        mit_arm.read_joint_state(bus, params(robot))


def test_joint_state_maps_velocity_and_torque_to_joint_frame(robot):
    robot.config.joint_signs[0] = -1
    states = raw_states(gripper=3.1)
    states.velocity[:] = [0.5, 0, 0, 0, 0, 0, 3.0]
    states.torque[:] = [2.0, 0, 0, 0, 0, 0, 0.2]
    state = mit_arm.read_joint_state(mock_bus(states), params(robot))
    assert state.velocity[0] == pytest.approx(-0.5)
    assert state.torque[0] == pytest.approx(-2.0)
    assert state.velocity[6] == pytest.approx(0.5)  # 3 rad/s over a 6 rad stroke
    assert state.torque[6] == pytest.approx(0.2)


def test_motorbridge_mapping_and_mit_radians(monkeypatch):
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    factory = MagicMock(return_value=controller)
    monkeypatch.setattr(robot_module, "Controller", factory, raising=False)
    monkeypatch.setattr(robot_module, "can", SimpleNamespace(Bus=MagicMock()), raising=False)
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.open()
    factory.assert_called_once_with(channel="can0")
    assert [call.args for call in controller.add_damiao_motor.call_args_list] == [
        (i + 1, i + 17, "4340" if i < 3 else "4310") for i in range(7)
    ]
    motors = dict(bus.motors)
    bus.enable(np.array([0.2, 0.5, 0.3, 0, 0, 0, 6.1]))
    motors["joint_1"].send_mit.assert_called_once_with(0.2, 0, 0, 0, 0)
    motors["gripper"].send_mit.assert_called_once_with(6.1, 0, 0, 0, 0)
    controller.enable_all.assert_called_once()
    bus.close()
    controller.disable_all.assert_called_once()
    controller.close.assert_called_once()


def test_failed_torque_shutdown_blocks_reconnection(monkeypatch):
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    monkeypatch.setattr(robot_module, "Controller", MagicMock(return_value=controller), raising=False)
    monkeypatch.setattr(robot_module, "can", SimpleNamespace(Bus=MagicMock()), raising=False)
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.open()
    bus.enabled = True
    controller.disable_all.side_effect = RuntimeError("CAN write failed")
    bus.close()
    with pytest.raises(RuntimeError, match="torque-disable command failed"):
        bus.open()


def frame(i, age=0, status=1):
    return SimpleNamespace(
        is_error_frame=False,
        is_remote_frame=False,
        is_extended_id=False,
        dlc=8,
        arbitration_id=i + 16,
        data=bytes([status << 4 | i]) + bytes(7),
        timestamp=time.time() - age,
    )


def feedback_bus(frames):
    """An opened bus whose python-can monitor replays ``frames``."""
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.controller = MagicMock()
    queued = deque(frames)
    bus.monitor = SimpleNamespace(recv=lambda timeout: queued.popleft() if queued else None)
    bus.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in bus.motors.values():
        motor.get_state.return_value = SimpleNamespace(pos=0.1, vel=0.2, torq=0.3, status_code=1)
    return bus


def test_cached_motorbridge_states_do_not_hide_missing_motor():
    bus = feedback_bus([frame(i) for i in range(1, 7)])
    with pytest.raises(ConnectionError, match="stale"):
        bus.read_states(wait=False)


def test_queued_old_packets_do_not_count_as_fresh():
    bus = feedback_bus([frame(i, age=1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="stale"):
        bus.read_states(wait=False)


def test_fault_in_any_feedback_packet_fails():
    bus = feedback_bus([frame(i, status=13 if i == 3 else 1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="fault 0xd"):
        bus.read_states(wait=False)


def test_fresh_feedback_accepts_stationary_motors():
    bus = feedback_bus([frame(i) for i in range(1, 8)])
    states = bus.read_states(wait=False)
    np.testing.assert_allclose(states.position, np.full(7, 0.1))
    np.testing.assert_allclose(states.velocity, np.full(7, 0.2))
    np.testing.assert_allclose(states.torque, np.full(7, 0.3))


def test_disabled_motor_after_enable_grace_is_a_fault():
    bus = feedback_bus([frame(i, status=0 if i == 3 else 1) for i in range(1, 8)])
    bus.enabled = True
    bus._enabled_at = time.monotonic() - 1
    with pytest.raises(ConnectionError, match="motor 3 unexpectedly disabled"):
        bus.read_states(wait=False)


def test_disabled_motor_is_allowed_during_enable_transition():
    bus = feedback_bus([frame(i, status=0) for i in range(1, 8)])
    bus.enabled = True
    bus._enabled_at = time.monotonic()
    np.testing.assert_allclose(bus.read_states(wait=False).position, np.full(7, 0.1))


def test_old_disabled_frame_does_not_override_new_enabled_frame():
    bus = feedback_bus([frame(3, status=0), *[frame(i) for i in range(1, 8)]])
    bus.enabled = True
    bus._enabled_at = time.monotonic() - 1
    np.testing.assert_allclose(bus.read_states(wait=False).position, np.full(7, 0.1))


def test_disabled_motor_is_allowed_before_enable():
    bus = feedback_bus([frame(i, status=0) for i in range(1, 8)])
    np.testing.assert_allclose(bus.read_states(wait=False).position, np.full(7, 0.1))


def test_slew_gripper_torque_and_gravity_feedforward(robot):
    robot.config.max_gripper_speed_s = 2
    position = np.array([0, 0.5, 0.5, 0, 0, 0, 0.5])
    command, packet = mit_arm.control_step(
        params(robot), position, position + 0.1, position, gravity=np.ones(6), dt=0.01
    )
    assert packet["joint_1"] == pytest.approx((0.003, 0, 80, 5, 1))
    assert packet["joint_2"][-1] == pytest.approx(1.1)
    assert command[6] == pytest.approx(0.52)  # gripper slews at 2 strokes/s
    assert "gripper" not in packet  # commanded by its force limiter


@pytest.mark.parametrize("polarity", [-1, 1])
def test_gripper_uses_i2rt_gains_and_force_limit(robot, polarity):
    robot.config.gripper_open_deg = robot.config.gripper_closed_deg + polarity * math.degrees(6.0)
    grip = params(robot).gripper
    assert (grip.kp, grip.kd, grip.force_limit_n, grip.finger_stroke_m) == (20.0, 0.5, 50.0, 0.096)
    limiter = mit_arm.GripperForceLimiter(grip)
    # Free motion goes straight to the target, without a torque cap.
    command = limiter.command(measured=0.5, velocity=1.0, torque=0.1, commanded=1.0, now=0.0)
    assert command == pytest.approx((grip.to_raw(1.0), 0, 20, 0.5, 0))
    # Blocked while closing; the limit is 50 N * 0.096 m / 6 rad + 0.3 Nm = 1.1 Nm at the motor.
    at_limit = mit_arm.GripperForceLimiter(grip)
    command = at_limit.command(measured=0.5, velocity=0.0, torque=1.1, commanded=0.0, now=0.0)
    assert at_limit.blocked
    assert command[0] == pytest.approx(grip.to_raw(0.5))  # pressing with the limit: hold
    harder = mit_arm.GripperForceLimiter(grip)
    command = harder.command(measured=0.5, velocity=0.0, torque=2.1, commanded=0.0, now=0.0)
    # 1 Nm too much: back off 1 Nm / kp 20 = 0.05 rad toward open.
    assert command[0] == pytest.approx(grip.to_raw(0.5) + polarity * 0.05)


def test_control_step_clamps_tracking_limits_and_gravity(robot):
    # Characterizes the full control step so refactors must reproduce it exactly.
    robot.config.joint_signs[1] = -1
    position = np.array([0.5, 0.0, 1.0, 0, 0, 0, 0.5])
    previous = np.array([0.9, 0.0, 1.0, 0, 0, 0, 0.5])
    target = np.array([0.9, -0.2, 1.5, 0, 0, 0, 1.0])
    gravity = np.array([20.0, 2, 1, 1, 1, 1])
    command, packet = mit_arm.control_step(params(robot), position, target, previous, gravity, dt=0.05)
    # joint_1 tracking band, joint_2 lower limit, joint_3 slew; the gripper slews freely.
    np.testing.assert_allclose(command, [0.65, 0.0, 1.015, 0, 0, 0, 1.0])
    np.testing.assert_allclose(previous, [0.9, 0.0, 1.0, 0, 0, 0, 0.5])  # inputs are not mutated
    assert packet["joint_1"] == pytest.approx((0.65, 0, 80, 5, 10.0))  # gravity clipped to 10 Nm
    assert packet["joint_2"] == pytest.approx((0.0, 0, 80, 5, -2.2))  # factor 1.1, sign -1
    assert packet["joint_3"] == pytest.approx((1.015, 0, 80, 5, 1.1))
    assert packet["joint_4"] == pytest.approx((0.0, 0, 10, 1.5, 1.2))


def test_gravity_matches_reference_torques(robot):
    pytest.importorskip("placo")
    robot.config.read_only = False
    robot.config.gravity_compensation = True
    robot._load_control_model()
    assert robot.gravity_model is not None
    pose = np.array([0.2, 1.0, 1.1, -0.5, 0.3, -0.2, 0.5])
    expected = [0.0, -1.3779441132, 5.9408414183, 1.0582110186, -0.0023627509, -0.0002296899]
    np.testing.assert_allclose(robot._gravity_torque(pose), expected, atol=1e-6)


def test_single_arm_features_are_not_prefixed(robot):
    assert tuple(robot.action_features) == YAM_FEATURE_NAMES
    assert robot.observation_features == robot.action_features
    assert YAM_FEATURE_NAMES[0] == "joint_1.pos"


def test_velocity_and_torque_features_are_grouped_per_motor(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    robot = make_robot(tmp_path, use_velocity_and_torque=True)
    assert tuple(robot.observation_features) == motor_feature_names(use_velocity_and_torque=True)
    assert tuple(robot.observation_features)[:3] == ("joint_1.pos", "joint_1.vel", "joint_1.torque")
    assert tuple(robot.action_features) == YAM_FEATURE_NAMES


@pytest.mark.parametrize("use_degrees", [True, False])
def test_observation_uses_public_units(tmp_path, monkeypatch, use_degrees):
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    robot = make_robot(tmp_path, use_degrees=use_degrees, use_velocity_and_torque=True)
    ready(robot)
    robot.servo.state = mit_arm.JointState(
        position=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]),
        velocity=np.array([1.0, 0, 0, 0, 0, 0, 0.25]),
        torque=np.array([2.0, 0, 0, 0, 0, 0, 0.3]),
    )
    observation = robot.get_observation()
    scale, gripper_scale = (math.degrees(1), 100.0) if use_degrees else (1.0, 1.0)
    assert observation["joint_1.pos"] == pytest.approx(0.1 * scale)
    assert observation["joint_1.vel"] == pytest.approx(1.0 * scale)
    assert observation["joint_1.torque"] == pytest.approx(2.0)
    assert observation["gripper.pos"] == pytest.approx(0.5 * gripper_scale)
    assert observation["gripper.vel"] == pytest.approx(0.25 * gripper_scale)


def test_single_arm_action_updates_target(robot):
    action = ready(robot)
    action["joint_1.pos"] = math.degrees(0.2)
    action["gripper.pos"] = 50.0
    assert robot.send_action(action) == pytest.approx(action)
    np.testing.assert_allclose(robot.servo.target, [0.2, 0, 0, 0, 0, 0, 0.5])


def test_single_arm_rejects_partial_action(robot):
    ready(robot)
    with pytest.raises(ValueError, match="all seven"):
        robot.send_action({"joint_1.pos": 0.2})


def test_calibration_connection_never_enables_motor(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = False
    robot.connect(calibrate=False)
    robot.bus.set_mit_mode.assert_not_called()
    robot.bus.enable.assert_not_called()
    robot.servo.start.assert_not_called()
    robot.disconnect()


@pytest.mark.parametrize("read_only", [False, True])
def test_connect_enables_torque_only_after_mit_mode(robot, monkeypatch, read_only):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = read_only
    robot.connect()
    calls = [name for name, *_ in robot.bus.mock_calls if name in ("open", "set_mit_mode", "enable")]
    assert calls == (["open"] if read_only else ["open", "set_mit_mode", "enable"])
    robot.servo.start.assert_called_once()
    robot.disconnect()


def test_start_pose_check_uses_degrees(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = False
    states = raw_states()
    states.position[0] = math.radians(robot.config.initial_tolerance_deg + 1)
    robot.bus.read_states.return_value = states
    with pytest.raises(ValueError, match="initial pose"):
        robot.connect()
    robot.bus.enable.assert_not_called()


def test_connect_requires_the_motor_can_timeout(robot, monkeypatch, caplog):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = False
    robot.bus.can_timeouts.return_value = {**dict.fromkeys(MOTOR_NAMES, 8000), "joint_2": 0}
    with pytest.raises(ValueError, match="CAN timeout of joint_2 is off"):
        robot.connect()
    robot.bus.enable.assert_not_called()
    robot.config.require_motor_can_timeout = False
    robot.connect()
    assert "CAN timeout of joint_2 is off" in caplog.text
    robot.bus.enable.assert_called_once()
    robot.disconnect()


def test_read_only_connect_skips_the_can_timeout_check(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect()
    robot.bus.can_timeouts.assert_not_called()
    robot.disconnect()


def test_bus_reads_the_can_timeout_register():
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in bus.motors.values():
        motor.get_register_u32.return_value = 8000
    assert bus.can_timeouts() == dict.fromkeys(MOTOR_NAMES, 8000)
    bus.motors["gripper"].get_register_u32.assert_called_once_with(9)


def test_fault_damping_comes_from_the_config(robot):
    robot.config.fault_damping_kd = [4.0, 4.0, 4.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(params(robot).fault_damping_kd, [4, 4, 4, 1, 1, 1, robot.config.gripper_kd])


def test_float_mode_requires_gravity_compensation(tmp_path):
    with pytest.raises(ValueError, match="needs gravity_compensation"):
        YamFollowerConfig(port="can0", idle_mode="float", gravity_compensation=False)
    with pytest.raises(ValueError, match="idle_mode"):
        YamFollowerConfig(port="can0", idle_mode="limp")


@pytest.mark.parametrize("friction_compensation", [False, True])
def test_float_parameters_come_from_the_config(robot, friction_compensation):
    robot.config.friction_compensation = friction_compensation
    arm = params(robot)
    np.testing.assert_allclose(arm.float_kd, [0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
    expected = [0.3, 0.3, 0.3, 0.06, 0.06, 0.06] if friction_compensation else [0.0] * 6
    np.testing.assert_allclose(arm.coulomb_friction, expected)


def test_configure_refuses_while_servo_holds_the_arm(robot):
    robot.servo.active = True
    with pytest.raises(RuntimeError, match="before the servo starts"):
        robot.configure()


def test_start_refuses_second_servo_before_enabling_again(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = False
    robot.connect()
    robot.servo.active = True
    with pytest.raises(RuntimeError, match="already running"):
        robot.start()
    robot.bus.enable.assert_called_once()
    robot.disconnect()


def test_servo_start_refuses_second_thread(robot):
    attach_bus(robot, mock_bus())
    robot.servo.seed(joint_state())
    robot.servo.start()
    original_thread = robot.servo._thread
    with pytest.raises(RuntimeError, match="already running"):
        robot.servo.start()
    assert robot.servo._thread is original_thread
    robot.servo.stop()


def test_foreground_watchdog_allows_one_late_feedback_window(robot):
    robot.servo.updated_at = time.monotonic() - 1.5 * robot.config.feedback_timeout_s
    robot.servo.check_healthy()
    robot.servo.updated_at = time.monotonic() - 3 * robot.config.feedback_timeout_s
    with pytest.raises(ConnectionError, match="stale"):
        robot.servo.check_healthy()


def test_disconnect_can_retry_after_servo_join_timeout(robot):
    bus = attach_bus(robot, mock_bus())
    robot._connected = True
    robot.servo.stop = MagicMock(side_effect=RuntimeError("YAM servo did not stop"))
    with pytest.raises(RuntimeError, match="did not stop"):
        robot.disconnect()
    bus.disable.assert_called_once()
    bus.close.assert_not_called()
    assert robot.is_connected
    robot.servo.stop.side_effect = None
    robot.disconnect()
    bus.close.assert_called_once()
    assert not robot.is_connected


def test_gravity_model_loads_before_hardware(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = False
    robot.config.gravity_compensation = True
    events = []

    def load_model():
        if robot.gravity_model is None:
            events.append("model")
            robot.gravity_model = object()

    robot._load_control_model = load_model
    robot.bus.open.side_effect = lambda: events.append("connect")
    robot.connect()
    robot.disconnect()
    assert events[:2] == ["model", "connect"]


def measure_gripper(robot, monkeypatch, *, closed=0.1, opened=6.1, answers=()):
    """Run calibrate() with the gripper reading ``closed``/``opened`` raw radians at the prompts."""
    pending = list(answers)
    prompts = []

    def prompt(text):
        prompts.append(text)
        if pending:
            return pending.pop(0)
        robot.bus.read_states.return_value = raw_states(opened if "open" in text else closed)
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(robot_module.time, "sleep", lambda _: None)
    robot.calibrate()
    return prompts


def test_single_arm_calibration_preserves_factory_zeros(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    prompts = measure_gripper(robot, monkeypatch)
    calibration = robot.calibration["gripper"]
    assert (calibration.homing_offset, calibration.drive_mode) == (0, 0)
    # Raw motor angles of the stops, in whole degrees.
    assert (calibration.range_min, calibration.range_max) == (6, 350)
    assert robot.config.gripper_open_deg == 350.0
    assert robot.params is not None
    assert robot.params.gripper.open == pytest.approx(6.1, abs=0.01)
    assert all("[test]" in text for text in prompts)
    assert robot.calibration_fpath.is_file()
    robot.bus.enable.assert_not_called()
    robot.disconnect()


def saved_calibration_robot(tmp_path, monkeypatch, *, closed=0.1, opened=6.1):
    """Calibrate once, then build a new robot that only knows the saved file."""
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    first = make_robot(tmp_path)
    mock_hardware(first, monkeypatch)
    first.connect(calibrate=False)
    measure_gripper(first, monkeypatch, closed=closed, opened=opened)
    first.disconnect()
    return make_robot(tmp_path, gripper_closed_deg=None, gripper_open_deg=None)


@pytest.mark.parametrize("closed,opened", [(0.1, 6.1), (6.1, 0.1)])
def test_saved_calibration_loads_either_polarity(tmp_path, monkeypatch, closed, opened):
    robot = saved_calibration_robot(tmp_path, monkeypatch, closed=closed, opened=opened)
    assert robot.is_calibrated
    assert robot.config.gripper_closed_deg == round(math.degrees(closed))
    assert robot.config.gripper_open_deg == round(math.degrees(opened))


def test_existing_calibration_is_kept_unless_recalibrating(tmp_path, monkeypatch):
    robot = saved_calibration_robot(tmp_path, monkeypatch)
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    robot.bus.read_states.reset_mock()
    prompts = measure_gripper(robot, monkeypatch, answers=[""])
    assert len(prompts) == 1 and "ENTER" in prompts[0]
    robot.bus.read_states.assert_not_called()

    prompts = measure_gripper(robot, monkeypatch, opened=5.1, answers=["c"])
    assert len(prompts) == 3
    assert robot.config.gripper_open_deg == round(math.degrees(5.1))
    robot.disconnect()


def test_count_based_calibration_file_asks_to_recalibrate(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    robot = make_robot(tmp_path)
    robot.calibration = {
        "gripper": robot_module.MotorCalibration(
            id=7, drive_mode=0, homing_offset=0, range_min=33000, range_max=48000
        )
    }
    robot._save_calibration()
    with pytest.raises(ValueError, match="run lerobot-calibrate again"):
        make_robot(tmp_path, gripper_closed_deg=None, gripper_open_deg=None)


def test_servo_error_disables_single_arm(robot):
    bus = attach_bus(robot, mock_bus())
    bus.read_states.side_effect = ConnectionError("lost")
    robot.servo._run()
    assert isinstance(robot.servo.failure, ConnectionError)
    assert robot.servo.stop_event.is_set()
    bus.disable.assert_called_once()


def test_servo_waits_for_asynchronous_feedback(robot):
    attach_bus(robot, feedback_bus([]))
    replies = deque([None, *[frame(i) for i in range(1, 8)], None])

    def receive(timeout):
        value = replies.popleft() if replies else None
        if not replies:
            robot.servo.stop_event.set()
        return value

    robot.bus.monitor.recv = receive
    robot.servo._run()
    assert robot.servo.failure is None
    assert robot.servo.updated_at > 0


def test_gc_freeze_lasts_exactly_as_long_as_the_servo(robot, monkeypatch):
    guard = MagicMock()
    monkeypatch.setattr(mit_arm, "_ControlGC", guard)
    attach_bus(robot, mock_bus())
    robot.servo.seed(joint_state())
    robot.servo.start()
    guard.acquire.assert_called_once()
    guard.release.assert_not_called()
    robot.servo.stop()
    guard.release.assert_called_once()


def test_gc_freeze_can_be_disabled(robot, monkeypatch):
    guard = MagicMock()
    monkeypatch.setattr(mit_arm, "_ControlGC", guard)
    attach_bus(robot, mock_bus())
    robot.config.freeze_gc = False
    robot.servo.seed(joint_state())
    robot.servo.start()
    robot.servo.stop()
    guard.acquire.assert_not_called()
    guard.release.assert_not_called()


def test_connect_failure_before_servo_never_freezes_gc(robot, monkeypatch):
    guard = MagicMock()
    monkeypatch.setattr(mit_arm, "_ControlGC", guard)
    attach_bus(robot, mock_bus()).open.side_effect = ConnectionError("no adapter")
    with pytest.raises(ConnectionError, match="no adapter"):
        robot.connect()
    guard.acquire.assert_not_called()
    guard.release.assert_not_called()
