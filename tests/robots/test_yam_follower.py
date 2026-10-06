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

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.yam_follower import (
    YamFollower,
    YamFollowerConfig,
    yam_follower as robot_module,
)
from lerobot.robots.yam_follower.config_yam_follower import JOINT_LIMITS, MOTOR_NAMES, YAM_FEATURE_NAMES


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    return YamFollower(
        YamFollowerConfig(
            id="test",
            calibration_dir=tmp_path,
            port="can0",
            gripper_closed_rad=0.1,
            gripper_open_rad=6.1,
            gravity_compensation=False,
        )
    )


def states(gripper=0.1):
    return {
        name: SimpleNamespace(pos=gripper if name == "gripper" else 0, status_code=0) for name in MOTOR_NAMES
    }


def mock_hardware(robot, monkeypatch):
    monkeypatch.setattr(robot, "_open_hardware", MagicMock())
    monkeypatch.setattr(robot, "_read_feedback", MagicMock(return_value=states()))
    monkeypatch.setattr(robot, "_close_hardware", MagicMock())
    monkeypatch.setattr(robot, "_configure_control", MagicMock())
    monkeypatch.setattr(robot, "_enable_motors", MagicMock())
    monkeypatch.setattr(robot, "_start_servo", MagicMock())


def ready(robot):
    robot._connected = True
    robot.config.read_only = False
    robot.updated_at = time.monotonic()
    return dict.fromkeys(robot.action_features, 0.0)


@pytest.mark.parametrize("opened", [6.1, -5.9])
def test_units_order_and_gripper_polarity(robot, opened):
    robot.config.gripper_open_rad = opened
    raw = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, (0.1 + opened) / 2]
    robot.config.joint_signs[0] = -1
    robot.config.joint_offsets_rad[0] = 0.2
    feedback = {name: SimpleNamespace(pos=value) for name, value in zip(MOTOR_NAMES, raw, strict=True)}
    decoded = robot_module.motor_to_joint(
        np.asarray([feedback[name].pos for name in MOTOR_NAMES]), robot.config
    )
    np.testing.assert_allclose(decoded, [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.5])
    np.testing.assert_allclose(robot_module.joint_to_motor(decoded, robot.config), raw)


def test_position_validation_and_clipping_are_pure():
    values = np.array([JOINT_LIMITS[0][0] - 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    robot_module.validate_positions(values, joint_tolerance_rad=0.03)
    clipped = robot_module.clip_to_limits(values)
    assert values[0] == JOINT_LIMITS[0][0] - 0.02
    assert clipped[0] == JOINT_LIMITS[0][0]
    assert clipped[6] == values[6]
    with pytest.raises(ValueError):
        robot_module.validate_positions(values, joint_tolerance_rad=0.0)


def test_motorbridge_mapping_and_mit_radians(robot, monkeypatch):
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    factory = MagicMock(return_value=controller)
    monkeypatch.setattr(robot_module, "Controller", factory, raising=False)
    monkeypatch.setattr(robot_module, "can", SimpleNamespace(Bus=MagicMock()), raising=False)
    robot._open_hardware()
    factory.assert_called_once_with(channel="can0")
    assert [call.args for call in controller.add_damiao_motor.call_args_list] == [
        (i + 1, i + 17, "4340" if i < 3 else "4310") for i in range(7)
    ]
    robot.config.read_only = False
    robot.position = np.array([0.2, 0.5, 0.3, 0, 0, 0, 1.0])
    robot._enable_motors()
    robot.motors["joint_0"].send_mit.assert_called_once_with(0.2, 0, 0, 0, 0)
    robot.motors["gripper"].send_mit.assert_called_once_with(6.1, 0, 0, 0, 0)
    robot._close_hardware()
    controller.close.assert_called_once()


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


def monitor(robot, frames):
    robot.bus = MagicMock()
    queued = deque(frames)
    robot.monitor = SimpleNamespace(recv=lambda timeout: queued.popleft() if queued else None)
    robot.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in robot.motors.values():
        motor.get_state.return_value = SimpleNamespace(pos=0.1, status_code=1)


def test_cached_motorbridge_states_do_not_hide_missing_motor(robot):
    monitor(robot, [frame(i) for i in range(1, 7)])
    with pytest.raises(ConnectionError, match="stale"):
        robot._read_feedback()


def test_queued_old_packets_do_not_count_as_fresh(robot):
    monitor(robot, [frame(i, age=1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="stale"):
        robot._read_feedback()


def test_fault_in_any_feedback_packet_fails(robot):
    monitor(robot, [frame(i, status=13 if i == 3 else 1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="fault 0xd"):
        robot._read_feedback()


def test_fresh_feedback_accepts_stationary_motors(robot):
    monitor(robot, [frame(i) for i in range(1, 8)])
    assert len(robot._read_feedback()) == 7


def test_slew_gripper_torque_and_gravity_feedforward(robot, monkeypatch):
    robot.config.max_gripper_speed_s = 2
    robot.position = np.array([0, 0.5, 0.5, 0, 0, 0, 0.5])
    robot.command = robot.position.copy()
    robot.target = robot.position + 0.1
    robot.gravity_model = object()
    monkeypatch.setattr(robot, "_gravity_torque", lambda position: np.ones(6))
    packet = robot._command_packet(robot.position, 0.01)
    assert packet["joint_0"] == pytest.approx((0.003, 0, 80, 5, 1))
    assert packet["joint_1"][-1] == pytest.approx(1.1)
    assert packet["gripper"] == pytest.approx((3.2, 0, 5, 0.005, 0))


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("polarity", [-1, 1])
def test_default_gripper_slew_preserves_torque_bound(robot, direction, polarity):
    robot.config.gripper_open_rad = robot.config.gripper_closed_rad + polarity * 6.0
    robot.position = np.array([0, 0.5, 0.5, 0, 0, 0, 0.5])
    robot.command = robot.position.copy()
    robot.target = robot.position.copy()
    robot.target[6] = 1 if direction > 0 else 0
    packet = robot._command_packet(robot.position, 0.01)["gripper"]
    assert robot.command[6] == pytest.approx(0.5 + direction * 0.12)
    raw_measured = 0.1 + polarity * 3.0
    assert packet == pytest.approx((raw_measured + direction * polarity * 0.1, 0, 5, 0.005, 0))
    assert (packet[0] - raw_measured) * packet[2] == pytest.approx(direction * polarity * 0.5)


def test_command_packet_clamps_tracking_limits_and_gravity(robot, monkeypatch):
    # Characterizes the full control step so refactors must reproduce it exactly.
    robot.config.joint_signs[1] = -1
    robot.position = np.array([0.5, 0.0, 1.0, 0, 0, 0, 0.5])
    robot.command = np.array([0.9, 0.0, 1.0, 0, 0, 0, 0.5])
    robot.target = np.array([0.9, -0.2, 1.5, 0, 0, 0, 1.0])
    monkeypatch.setattr(robot, "_gravity_torque", lambda position: np.array([20.0, 2, 1, 1, 1, 1]))
    packet = robot._command_packet(robot.position, 0.05)
    # joint_0 tracking band, joint_1 lower limit, joint_2 slew; the gripper slews freely.
    np.testing.assert_allclose(robot.command, [0.65, 0.0, 1.015, 0, 0, 0, 1.0])
    assert packet["joint_0"] == pytest.approx((0.65, 0, 80, 5, 10.0))  # gravity clipped to 10 Nm
    assert packet["joint_1"] == pytest.approx((0.0, 0, 80, 5, -2.2))  # factor 1.1, sign -1
    assert packet["joint_2"] == pytest.approx((1.015, 0, 80, 5, 1.1))
    assert packet["joint_3"] == pytest.approx((0.0, 0, 10, 1.5, 1.2))
    assert packet["gripper"] == pytest.approx((3.2, 0, 5, 0.005, 0))  # 0.1 rad torque band


def test_gravity_matches_reference_torques(robot):
    pytest.importorskip("placo")
    robot.config.read_only = False
    robot.config.gravity_compensation = True
    robot._load_control_model()
    assert robot.gravity_model is not None
    pose = np.array([0.2, 1.0, 1.1, -0.5, 0.3, -0.2, 0.5])
    expected = [0.0, -1.3779441132, 5.9408414183, 1.0582110186, -0.0023627509, -0.0002296899]
    np.testing.assert_allclose(robot._gravity_torque(pose), expected, atol=1e-9)


def test_single_arm_features_are_not_prefixed(robot):
    assert tuple(robot.action_features) == YAM_FEATURE_NAMES
    assert robot.observation_features == robot.action_features


def test_single_arm_action_updates_target(robot):
    action = ready(robot)
    action["joint_0.pos"] = 0.2
    action["gripper.pos"] = 0.5
    assert robot.send_action(action) == action
    np.testing.assert_allclose(robot.target, [0.2, 0, 0, 0, 0, 0, 0.5])


def test_single_arm_rejects_partial_action(robot):
    ready(robot)
    with pytest.raises(ValueError, match="all seven"):
        robot.send_action({"joint_0.pos": 0.2})


def test_calibration_connection_never_enables_motor(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.config.read_only = False
    robot.connect(calibrate=False)
    robot._configure_control.assert_not_called()
    robot._enable_motors.assert_not_called()
    robot.disconnect()


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
    robot._open_hardware.side_effect = lambda: events.append("connect")
    robot.connect()
    robot.disconnect()
    assert events[:2] == ["model", "connect"]


def test_single_arm_calibration_preserves_factory_zeros(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)

    def prompt(text):
        robot._read_feedback.return_value["gripper"].pos = 6.1 if "open" in text else 0.1

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(robot_module.time, "sleep", lambda _: None)
    robot.calibrate()
    assert robot.calibration["gripper"].homing_offset == 0
    assert robot.calibration["gripper"].drive_mode == 0
    assert robot.config.gripper_open_rad == pytest.approx(6.1, abs=0.0002)
    assert robot.calibration_fpath.is_file()
    robot._enable_motors.assert_not_called()
    robot.disconnect()


def test_servo_error_disables_single_arm(robot):
    robot.enabled = True
    robot.bus = MagicMock()
    robot._read_feedback = MagicMock(side_effect=ConnectionError("lost"))
    robot._run()
    assert isinstance(robot._failure, ConnectionError)
    assert robot._stop.is_set()
    robot.bus.disable_all.assert_called_once()


def test_servo_waits_for_asynchronous_feedback(robot):
    monitor(robot, [])
    replies = deque([None, *[frame(i) for i in range(1, 8)], None])

    def receive(timeout):
        value = replies.popleft() if replies else None
        if not replies:
            robot._stop.set()
        return value

    robot.monitor.recv = receive
    robot._run()
    assert robot._failure is None
    assert robot.updated_at > 0


@pytest.mark.parametrize("enabled,preexisting", [(True, False), (True, True), (False, False)])
def test_gc_freeze_is_shared_and_preserves_callers_state(monkeypatch, enabled, preexisting):
    gc_mock = MagicMock()
    gc_mock.isenabled.return_value = enabled
    gc_mock.get_freeze_count.return_value = int(preexisting)
    monkeypatch.setattr(robot_module, "gc", gc_mock)
    guard = robot_module._ControlGC
    assert guard._users == 0
    guard.acquire()
    guard.acquire()
    guard.release()
    gc_mock.unfreeze.assert_not_called()
    guard.release()
    assert gc_mock.collect.call_count == int(enabled)
    assert gc_mock.freeze.call_count == int(enabled)
    assert gc_mock.unfreeze.call_count == int(enabled and not preexisting)
    assert guard._users == 0


def test_connect_failure_releases_gc_freeze(robot, monkeypatch):
    robot._open_hardware = MagicMock(side_effect=ConnectionError("no adapter"))
    guard = MagicMock()
    monkeypatch.setattr(robot_module, "_ControlGC", guard)
    with pytest.raises(ConnectionError, match="no adapter"):
        robot.connect()
    guard.acquire.assert_called_once()
    guard.release.assert_called_once()
    assert not robot._gc_acquired
