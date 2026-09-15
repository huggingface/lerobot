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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors.robstride.tables import PrivateControlMode
from lerobot.robots.rebot_b601_rs_follower import (
    RebotB601RSFollower,
    RebotB601RSFollowerRobotConfig,
)

_MODULE = "lerobot.robots.rebot_b601_rs_follower.rebot_b601_rs_follower"

_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="RobstridePrivateMotorsBusMock")
    bus.is_connected = False

    def _connect(handshake: bool = True) -> None:
        bus.is_connected = True

    def _disconnect(disable_torque: bool = True) -> None:
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    # Present positions in degrees; each motor's value encodes its 1-indexed order.
    bus.sync_read.return_value = {name: float(idx) for idx, name in enumerate(_MOTOR_NAMES, 1)}
    return bus


def _make_robot(**config_kwargs) -> tuple[RebotB601RSFollower, MagicMock]:
    bus_mock = _make_bus_mock()
    with patch(f"{_MODULE}.RobstridePrivateMotorsBus", return_value=bus_mock):
        robot = RebotB601RSFollower(RebotB601RSFollowerRobotConfig(**config_kwargs))
    return robot, bus_mock


@pytest.fixture
def follower():
    robot, _ = _make_robot()
    robot.connect(calibrate=False)
    yield robot
    if robot.is_connected:
        robot.disconnect()


def test_features_match_joints():
    robot, _ = _make_robot()
    expected = {f"{m}.pos" for m in _MOTOR_NAMES}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected
    assert "gripper.pos" in expected


def test_motors_built_from_config():
    bus_mock = _make_bus_mock()
    with patch(f"{_MODULE}.RobstridePrivateMotorsBus", return_value=bus_mock) as bus_cls:
        RebotB601RSFollower(RebotB601RSFollowerRobotConfig())
    motors = bus_cls.call_args.kwargs["motors"]
    assert [motors[name].id for name in _MOTOR_NAMES] == [1, 2, 3, 4, 5, 6, 7]
    # B601-RS fitting: rs06 on the shoulder/elbow joints, rs00 on the wrists and gripper.
    assert motors["shoulder_pan"].motor_type_str == "rs06"
    assert motors["elbow_flex"].motor_type_str == "rs06"
    assert motors["wrist_flex"].motor_type_str == "rs00"
    assert motors["gripper"].motor_type_str == "rs00"


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected
    follower.bus.disconnect.assert_called_once_with(True)


def test_connect_configures_position_mode(follower):
    follower.bus.configure_motors.assert_called_once_with(PrivateControlMode.POSITION)
    assert follower.bus.set_position_speed_limit.call_count == len(_MOTOR_NAMES)
    follower.bus.set_position_speed_limit.assert_any_call("shoulder_pan", 150.0)
    follower.bus.set_position_speed_limit.assert_any_call("gripper", 900.0)
    follower.bus.enable_torque.assert_called_once()


def test_get_observation_returns_positions(follower):
    obs = follower.get_observation()
    assert set(obs) == {f"{m}.pos" for m in _MOTOR_NAMES}
    for idx, motor in enumerate(_MOTOR_NAMES, 1):
        assert obs[f"{motor}.pos"] == pytest.approx(float(idx))


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-150, 150); request beyond the upper bound.
    returned = follower.send_action({"shoulder_pan.pos": 999.0})
    assert returned["shoulder_pan.pos"] == 150.0
    # Default control_mode is "position", so goals go through sync_write("Goal_Position", ...).
    goals = follower.bus.sync_write.call_args[0][1]
    assert goals["shoulder_pan"] == 150.0


def test_send_action_fills_wrist_yaw(follower):
    returned = follower.send_action({"shoulder_pan.pos": 10.0})
    assert returned["wrist_yaw.pos"] == 0.0
    goals = follower.bus.sync_write.call_args[0][1]
    assert goals["wrist_yaw"] == 0.0


def test_mit_mode_routes_to_mit_batch():
    robot, bus = _make_robot(control_mode="mit")
    robot.connect(calibrate=False)
    bus.configure_motors.assert_called_once_with(PrivateControlMode.MIT)
    bus.set_position_speed_limit.assert_not_called()
    robot.send_action({"shoulder_pan.pos": 10.0, "gripper.pos": -10.0})
    commands = bus._mit_control_batch.call_args[0][0]
    # Arm joints use mit_kp/mit_kd (motor order); the gripper uses its own gains.
    assert commands["shoulder_pan"] == (50.0, 3.0, 10.0, 0.0, 0.0)
    assert commands["gripper"] == (50.0, 4.0, -10.0, 0.0, 0.0)
    bus.sync_write.assert_not_called()
    robot.disconnect()


def test_max_relative_target_caps_step():
    robot, bus = _make_robot(max_relative_target=5.0)
    robot.connect(calibrate=False)
    # The bus mock reports shoulder_pan at 1.0 deg, so a far goal is capped to 1.0 + 5.0.
    returned = robot.send_action({"shoulder_pan.pos": 100.0})
    assert returned["shoulder_pan.pos"] == pytest.approx(6.0)
    robot.disconnect()


def test_disconnect_keeps_torque_when_configured():
    robot, bus = _make_robot(disable_torque_on_disconnect=False)
    robot.connect(calibrate=False)
    robot.disconnect()
    bus.disconnect.assert_called_once_with(False)


def test_configure_refuses_multi_turn_wrapped_reading():
    # A gripper that woke 2*pi-wrapped reads physical + 360 deg; configure() must refuse
    # to enable torque instead of letting a later action slam it into its stop.
    robot, bus = _make_robot()
    positions = {name: float(idx) for idx, name in enumerate(_MOTOR_NAMES, 1)}
    positions["gripper"] = -20.0 + 360.0  # closed-ish gripper, one full wrap up
    bus.sync_read.return_value = positions
    with pytest.raises(RuntimeError, match="multi-turn"):
        robot.connect(calibrate=False)
    bus.enable_torque.assert_not_called()


def test_calibrate_zeroes_and_persists(monkeypatch):
    robot, bus = _make_robot()
    robot.connect(calibrate=False)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "")
    monkeypatch.setattr(robot, "_save_calibration", lambda: None)
    robot.calibrate()
    bus.set_zero_position.assert_called_once()
    bus.save_parameters.assert_called_once()
    bus.write_calibration.assert_called()
    robot.disconnect()
