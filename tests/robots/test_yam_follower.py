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
    YamFollowerRobotConfig,
    mit_control,
    yam_follower as robot_module,
)
from lerobot.robots.yam_follower.config_yam_follower import (
    JOINT_LIMITS_RAD,
    MOTOR_NAMES,
    YAM_FEATURE_NAMES,
)


def gripper_calibration(closed=0.1, opened=6.1):
    """Gripper calibration with limits at exactly `closed` and `opened` radians; `calibrate()` would round them."""
    low, high = sorted(math.degrees(value) for value in (closed, opened))
    return robot_module.MotorCalibration(
        id=7, drive_mode=int(opened < closed), homing_offset=0, range_min=low, range_max=high
    )


def make_robot(tmp_path, *, gripper=(0.1, 6.1), **overrides):
    config = {
        "id": "test",
        "calibration_dir": tmp_path,
        "port": "can0",
        "gravity_compensation": False,
        # The control numbers below use 0.3 rad/s and 0.15 rad.
        "max_joint_speed_deg_s": math.degrees(0.3),
        "max_tracking_error_deg": math.degrees(0.15),
        **overrides,
    }
    robot = YamFollower(YamFollowerRobotConfig(**config))
    if gripper is not None:
        robot.calibration = {"gripper": gripper_calibration(*gripper)}
        robot._refresh_params()
    return robot


@pytest.fixture(autouse=True)
def without_hardware_packages(monkeypatch):
    """Build robots without the yam extra; their hardware is mocked."""
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)


@pytest.fixture
def robot(tmp_path):
    return make_robot(tmp_path)


def params(robot):
    return robot_module._yam_arm_params(robot.config, robot.calibration["gripper"])


def raw_states(gripper=0.1):
    position = np.array([0, 0, 0, 0, 0, 0, gripper], dtype=float)
    return mit_control.MotorStates(position=position, velocity=np.zeros(7), torque=np.zeros(7))


def assert_mit_command(command, *, position, velocity, kp, kd, torque):
    assert command.position == pytest.approx(position)
    assert command.velocity == pytest.approx(velocity)
    assert command.kp == pytest.approx(kp)
    assert command.kd == pytest.approx(kd)
    assert command.torque == pytest.approx(torque)


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
    robot.servo.active = True
    robot.servo.updated_at = time.monotonic()
    return dict.fromkeys(robot.action_features, 0.0)


@pytest.mark.parametrize("opened", [6.1, -5.9])
def test_units_order_and_gripper_polarity(robot, opened):
    robot.calibration["gripper"] = gripper_calibration(0.1, opened)
    raw = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, (0.1 + opened) / 2]
    decoded = mit_control._motor_to_joint(np.asarray(raw), params(robot))
    np.testing.assert_allclose(decoded, [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.5])
    np.testing.assert_allclose(mit_control.joint_to_motor(decoded, params(robot)), raw)


@pytest.mark.parametrize("use_degrees", [True, False])
def test_actions_are_clipped_to_the_limits(robot, use_degrees):
    robot.config.use_degrees = use_degrees
    action = ready(robot)
    action["gripper.pos"] = 100.4 if use_degrees else 1.004
    action["joint_2.pos"] = -1.0  # below the 0 lower limit in either unit
    robot.send_action(action)
    assert robot.servo.target[6] == 1.0
    assert robot.servo.target[1] == 0.0


def test_measured_position_has_wider_tolerance_than_action(robot):
    states = raw_states(gripper=6.4)
    states.position[0] = JOINT_LIMITS_RAD[0][0] - 0.08
    bus = attach_bus(robot, mock_bus(states))
    measured = mit_control.read_joint_state(bus, params(robot)).position
    assert measured[0] == pytest.approx(states.position[0])
    assert measured[6] == 1.0
    # Actions are clipped to the model limits instead of being rejected.
    ready(robot)
    robot.send_action(dict(zip(YAM_FEATURE_NAMES, robot_module._to_public(measured, True), strict=True)))
    assert robot.servo.target[0] == JOINT_LIMITS_RAD[0][0]
    states.position[0] = JOINT_LIMITS_RAD[0][0] - 0.2
    with pytest.raises(ValueError, match="outside"):
        mit_control.read_joint_state(bus, params(robot))


def test_motorbridge_mapping_and_mit_radians(monkeypatch):
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    factory = MagicMock(return_value=controller)
    monitor = MagicMock()
    monkeypatch.setattr(robot_module, "Controller", factory, raising=False)
    monkeypatch.setattr(robot_module, "can", SimpleNamespace(Bus=monitor), raising=False)
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.open()
    factory.assert_called_once_with(channel="can0")
    # The monitor only receives the seven feedback IDs, without error frames.
    assert [f["can_id"] for f in monitor.call_args.kwargs["can_filters"]] == list(range(17, 24))
    assert monitor.call_args.kwargs["ignore_rx_error_frames"] is True
    assert [call.args for call in controller.add_damiao_motor.call_args_list] == [
        (i + 1, i + 17, "4340" if i < 3 else "4310") for i in range(7)
    ]
    motors = dict(bus.motors)
    bus.enable(np.array([0.2, 0.5, 0.3, 0, 0, 0, 6.1]))
    motors["joint_1"].send_mit.assert_called_once_with(0.2, 0, 0, 0, 0)
    motors["gripper"].send_mit.assert_called_once_with(6.1, 0, 0, 0, 0)
    bus.send_mit(
        "joint_1",
        mit_control.MitCommand(position=0.1, velocity=0.2, kp=3.0, kd=4.0, torque=5.0),
    )
    assert motors["joint_1"].send_mit.call_args_list[-1].args == (0.1, 0.2, 3.0, 4.0, 5.0)
    controller.enable_all.assert_called_once()
    bus.close()
    controller.disable_all.assert_called_once()
    controller.close.assert_called_once()


def test_failed_torque_disable_still_closes_the_bus(monkeypatch):
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    monitor = MagicMock()
    monkeypatch.setattr(robot_module, "Controller", MagicMock(return_value=controller), raising=False)
    monkeypatch.setattr(
        robot_module, "can", SimpleNamespace(Bus=MagicMock(return_value=monitor)), raising=False
    )
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.open()
    bus.enabled = True
    controller.disable_all.side_effect = RuntimeError("CAN write failed")
    bus.close()
    controller.close.assert_called_once()
    monitor.shutdown.assert_called_once()


def frame(motor_id, age=0, status=1):
    return SimpleNamespace(
        arbitration_id=motor_id + 0x10,
        data=bytes([status << 4]) + bytes(7),
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
        motor.get_state.return_value = SimpleNamespace(pos=0.1, vel=0.2, torq=0.3)
    return bus


@pytest.mark.parametrize(
    ("motor_ids", "age"), [(range(1, 7), 0), (range(1, 8), 1)], ids=["a motor never answered", "old frames"]
)
def test_missing_or_old_feedback_is_stale(motor_ids, age):
    # MotorBridge's cached states can't show this; the monitor's frame times do.
    bus = feedback_bus([frame(i, age=age) for i in motor_ids])
    with pytest.raises(ConnectionError, match="stale"):
        bus.read_states()


def test_fault_in_any_feedback_packet_fails():
    bus = feedback_bus([frame(i, status=13 if i == 3 else 1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="fault 0xd"):
        bus.read_states()


def test_fresh_feedback_accepts_stationary_motors():
    bus = feedback_bus([frame(i) for i in range(1, 8)])
    states = bus.read_states()
    np.testing.assert_allclose(states.position, np.full(7, 0.1))
    np.testing.assert_allclose(states.velocity, np.full(7, 0.2))
    np.testing.assert_allclose(states.torque, np.full(7, 0.3))


def test_disabled_motor_after_enable_grace_is_a_fault():
    bus = feedback_bus([frame(i, status=0 if i == 3 else 1) for i in range(1, 8)])
    bus.enabled = True
    bus._enabled_at = time.monotonic() - 1
    with pytest.raises(ConnectionError, match="motor 3 unexpectedly disabled"):
        bus.read_states()


@pytest.mark.parametrize(
    ("enabled", "enabled_ago_s"), [(False, 1.0), (True, 0.0)], ids=["torque disabled", "torque just enabled"]
)
def test_disabled_motors_are_expected_while_torque_is_off_or_turning_on(enabled, enabled_ago_s):
    bus = feedback_bus([frame(i, status=0) for i in range(1, 8)])
    bus.enabled = enabled
    bus._enabled_at = time.monotonic() - enabled_ago_s
    np.testing.assert_allclose(bus.read_states().position, np.full(7, 0.1))


def test_old_disabled_frame_does_not_override_new_enabled_frame():
    bus = feedback_bus([frame(3, status=0), *[frame(i) for i in range(1, 8)]])
    bus.enabled = True
    bus._enabled_at = time.monotonic() - 1
    np.testing.assert_allclose(bus.read_states().position, np.full(7, 0.1))


def test_yam_parameters_reach_the_motor_commands(robot):
    robot.config.max_gripper_speed = 2
    position = np.array([0.5, 0.5, 1.0, 0, 0, 0, 0.5])
    target = np.r_[position[:6], 1.0]
    gravity = np.array([30.0, 15, 1, 1, 1, 1])
    command, packet = mit_control._control_step(params(robot), position, target, position, gravity, dt=0.01)
    # I2RT's gains, the per-joint gravity factors, and the 25 Nm cap on gravity torque.
    assert [(c.kp, c.kd) for c in packet] == [(80, 5)] * 3 + [(10, 1.5)] * 3
    np.testing.assert_allclose([c.torque for c in packet], [25.0, 16.5, 1.1, 1.2, 1.0, 1.0])
    assert command[6] == pytest.approx(0.52)  # the gripper slews at 2 strokes/s


@pytest.mark.parametrize("polarity", [-1, 1])
def test_gripper_uses_i2rt_gains_and_force_limit(robot, polarity):
    robot.calibration["gripper"] = gripper_calibration(0.1, 0.1 + polarity * 6.0)
    grip = params(robot).gripper
    assert (grip.kp, grip.kd, grip.force_limit_n, grip.finger_stroke_m) == (20.0, 0.5, 50.0, 0.096)
    limiter = mit_control._GripperForceLimiter(grip)
    # Free motion goes straight to the target, without a torque cap.
    command = limiter.command(measured=0.5, velocity=1.0, torque=0.1, commanded=1.0, now=0.0)
    assert_mit_command(command, position=grip.to_raw(1.0), velocity=0, kp=20, kd=0.5, torque=0)
    # Blocked while closing; the limit is 50 N * 0.096 m / 6 rad + 0.3 Nm = 1.1 Nm at the motor.
    at_limit = mit_control._GripperForceLimiter(grip)
    command = at_limit.command(measured=0.5, velocity=0.0, torque=1.1, commanded=0.0, now=0.0)
    assert at_limit.blocked
    assert command.position == pytest.approx(grip.to_raw(0.5))  # pressing with the limit: hold
    harder = mit_control._GripperForceLimiter(grip)
    command = harder.command(measured=0.5, velocity=0.0, torque=2.1, commanded=0.0, now=0.0)
    # 1 Nm too much: back off 1 Nm / kp 20 = 0.05 rad toward open.
    assert command.position == pytest.approx(grip.to_raw(0.5) + polarity * 0.05)


def test_gravity_model_is_loaded_from_bucket_cache(robot, monkeypatch, tmp_path):
    model_path = tmp_path / "linear_4310.urdf"
    ensure_model = MagicMock(return_value=model_path)
    gravity_model = MagicMock()
    monkeypatch.setattr(robot_module, "_GRAVITY_MODEL_LOCAL_PATH", model_path)
    monkeypatch.setattr(robot_module, "ensure_bucket_file", ensure_model)
    monkeypatch.setattr(robot_module, "GravityCompensation", gravity_model)
    robot.config.gravity_compensation = True
    mock_hardware(robot, monkeypatch)
    robot.connect()
    robot.disconnect()
    ensure_model.assert_called_once_with(
        "lerobot/robot-urdfs",
        "yam/v1/linear_4310.urdf",
        model_path,
        token=False,
    )
    gravity_model.assert_called_once_with(
        model_path, tuple(f"joint{i}" for i in range(1, 7)), base_frame="base"
    )


def test_features_are_positions_with_optional_velocity_and_torque_per_motor(robot, tmp_path):
    assert tuple(robot.action_features) == YAM_FEATURE_NAMES
    assert robot.observation_features == robot.action_features
    names = tuple(make_robot(tmp_path, use_velocity_and_torque=True).observation_features)
    assert len(names) == 21
    assert names[:3] == ("joint_1.pos", "joint_1.vel", "joint_1.torque")
    assert names[-3:] == ("gripper.pos", "gripper.vel", "gripper.torque")


@pytest.mark.parametrize("use_degrees", [True, False])
def test_observation_uses_public_units(tmp_path, monkeypatch, use_degrees):
    robot = make_robot(tmp_path, use_degrees=use_degrees, use_velocity_and_torque=True)
    ready(robot)
    robot.servo.state = mit_control.JointState(
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
    with pytest.raises(KeyError, match="joint_2.pos"):
        robot.send_action({"joint_1.pos": 0.2})


def test_calibration_connection_never_enables_motor(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    robot.bus.set_mit_mode.assert_not_called()
    robot.bus.enable.assert_not_called()
    robot.servo.start.assert_not_called()
    robot.disconnect()


def test_open_without_calibration_touches_no_hardware(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gripper=None)
    mock_hardware(robot, monkeypatch)
    with pytest.raises(ValueError, match="lerobot-calibrate"):
        robot.open()
    robot.bus.open.assert_not_called()


def test_connect_calibrates_missing_gripper_limits(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gripper=None)
    mock_hardware(robot, monkeypatch)

    def prompt(text):
        robot.bus.read_states.return_value.position[6] = 6.1 if "open" in text else 0.1
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(robot_module.time, "sleep", lambda _: None)
    robot.connect()

    assert robot.is_calibrated
    assert robot.bus.open.call_count == 2
    robot.bus.enable.assert_called_once()
    robot.disconnect()


def test_connect_enables_torque_only_after_mit_mode(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect()
    calls = [name for name, *_ in robot.bus.mock_calls if name in ("open", "set_mit_mode", "enable")]
    assert calls == ["open", "set_mit_mode", "enable"]
    robot.servo.start.assert_called_once()
    robot.disconnect()


def test_initial_position_check_uses_degrees(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    states = raw_states()
    states.position[0] = math.radians(robot.config.initial_tolerance_deg + 1)
    robot.bus.read_states.return_value = states
    with pytest.raises(ValueError, match="initial_position_deg"):
        robot.connect()
    robot.bus.enable.assert_not_called()


@pytest.mark.parametrize(("timeout", "detail"), [(0, "joint_2=off"), (8001, "joint_2=400.05 ms")])
def test_connect_requires_a_prompt_motor_can_timeout(robot, monkeypatch, timeout, detail):
    mock_hardware(robot, monkeypatch)
    robot.bus.can_timeouts.return_value = {**dict.fromkeys(MOTOR_NAMES, 8000), "joint_2": timeout}
    with pytest.raises(ValueError, match="unsafe motor CAN timeout") as exc_info:
        robot.connect()
    robot.bus.enable.assert_not_called()
    assert detail in str(exc_info.value)


def test_bus_reads_the_can_timeout_register():
    bus = robot_module._YamBus("can0", feedback_timeout_s=0.2)
    bus.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in bus.motors.values():
        motor.get_register_u32.return_value = 8000
    assert bus.can_timeouts() == dict.fromkeys(MOTOR_NAMES, 8000)
    bus.motors["gripper"].get_register_u32.assert_called_once_with(9)


def test_float_mode_requires_gravity_compensation():
    with pytest.raises(ValueError, match="needs gravity_compensation"):
        YamFollowerRobotConfig(port="can0", idle_mode="float", gravity_compensation=False)


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
    robot.connect()
    robot.servo.active = True
    with pytest.raises(RuntimeError, match="already running"):
        robot.start()
    robot.bus.enable.assert_called_once()
    robot.disconnect()


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
    robot.config.gravity_compensation = True
    events = []

    def load_model():
        if robot.gravity_model is None:
            events.append("model")
            robot.gravity_model = object()

    robot._load_gravity_model = load_model
    robot.bus.open.side_effect = lambda: events.append("connect")
    robot.connect()
    robot.disconnect()
    assert events[:2] == ["model", "connect"]


def measure_gripper(robot, monkeypatch, *, closed=0.1, opened=6.1, answers=()):
    """Run calibrate() with the gripper motor at `closed` and `opened` radians at the prompts."""
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


def test_single_arm_calibration_preserves_factory_zeros(tmp_path, monkeypatch):
    robot = make_robot(tmp_path, gripper=None)
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    prompts = measure_gripper(robot, monkeypatch)
    calibration = robot.calibration["gripper"]
    assert (calibration.homing_offset, calibration.drive_mode) == (0, 0)
    # Motor angles of the gripper limits, in whole degrees.
    assert (calibration.range_min, calibration.range_max) == (6, 350)
    assert robot.servo.params is not None
    assert robot.servo.params.gripper.open == pytest.approx(6.1, abs=0.01)
    assert all("[test]" in text for text in prompts)
    assert robot.calibration_fpath.is_file()
    robot.bus.enable.assert_not_called()
    robot.disconnect()


def saved_calibration_robot(tmp_path, monkeypatch, *, closed=0.1, opened=6.1):
    """Calibrate once, then build a new robot that only knows the saved file."""
    first = make_robot(tmp_path, gripper=None)
    mock_hardware(first, monkeypatch)
    first.connect(calibrate=False)
    measure_gripper(first, monkeypatch, closed=closed, opened=opened)
    first.disconnect()
    return make_robot(tmp_path, gripper=None)


@pytest.mark.parametrize("closed,opened", [(0.1, 6.1), (6.1, 0.1)])
def test_saved_calibration_loads_either_polarity(tmp_path, monkeypatch, closed, opened):
    robot = saved_calibration_robot(tmp_path, monkeypatch, closed=closed, opened=opened)
    assert robot.is_calibrated
    assert robot.servo.params.gripper.closed == pytest.approx(math.radians(round(math.degrees(closed))))
    assert robot.servo.params.gripper.open == pytest.approx(math.radians(round(math.degrees(opened))))


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
    assert robot.servo.params.gripper.open == pytest.approx(math.radians(round(math.degrees(5.1))))
    robot.disconnect()


@pytest.mark.parametrize("opened", [0.2, 11.0])  # strokes outside 0.5 to 10 rad
def test_calibration_refuses_an_implausible_stroke(robot, monkeypatch, opened):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    with pytest.raises(ValueError, match="Gripper stroke"):
        measure_gripper(robot, monkeypatch, opened=opened, answers=["c"])
    assert not robot.calibration_fpath.is_file()
    robot.disconnect()


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
