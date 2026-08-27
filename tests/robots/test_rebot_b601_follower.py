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
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)
from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderConfig

_MODULE = "lerobot.robots.rebot_b601_follower.rebot_b601_follower"
_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


def _per_joint(value: float | list[float] | Mapping[str, float]) -> dict[str, float]:
    """Compare legacy scalar/list inputs and normalized mapping values uniformly."""
    if isinstance(value, Mapping):
        return {joint: float(value[joint]) for joint in _JOINTS}
    if isinstance(value, list):
        return dict(zip(_JOINTS, value, strict=True))
    return dict.fromkeys(_JOINTS, float(value))


def _make_motor_mock(position_rad: float = 0.0) -> MagicMock:
    motor = MagicMock(name="MotorMock")
    state = MagicMock()
    state.pos = position_rad
    motor.get_state.return_value = state
    return motor


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MotorBridgeControllerMock")
    # add_damiao_motor returns a fresh motor mock; position encodes the call order.
    bus._motor_count = 0

    def _add_motor(_send_id, _recv_id, _model):
        bus._motor_count += 1
        return _make_motor_mock(position_rad=math.radians(bus._motor_count))

    bus.add_damiao_motor.side_effect = _add_motor
    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(port="/dev/null")
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null"))
    expected = {f"{m}.pos" for m in robot.motor_names}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected
    assert "gripper.pos" in expected


def test_shipped_dm_defaults_are_preserved():
    config = RebotB601FollowerRobotConfig(port="/dev/null")

    assert config.can_adapter == "damiao"
    assert config.control_mode == "mit"
    assert config.gripper_control_mode == "force_pos"
    assert config.gripper_torque_ratio == 0.07
    assert config.joint_limits == {
        "shoulder_pan": (-150.0, 150.0),
        "shoulder_lift": (-200.0, 1.0),
        "elbow_flex": (-200.0, 1.0),
        "wrist_flex": (-80.0, 90.0),
        "wrist_yaw": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
        "gripper": (-270.0, 0.0),
    }
    assert _per_joint(config.mit_kp) == dict(
        zip(_JOINTS, [45.0, 45.0, 45.0, 8.0, 9.0, 8.0, 8.0], strict=True)
    )
    assert _per_joint(config.mit_kd) == dict(
        zip(_JOINTS, [12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0], strict=True)
    )

    leader = RebotArm102LeaderConfig(port="/dev/null")
    assert leader.joint_ranges == {
        "shoulder_pan": [-150, 150],
        "shoulder_lift": [-200, 1],
        "elbow_flex": [-200, 1],
        "wrist_flex": [-80, 90],
        "wrist_yaw": [-90, 90],
        "wrist_roll": [-90, 90],
        "gripper": [-270, 0],
    }


def test_legacy_dm_config_accepts_gain_lists_without_motor_family():
    kp = [40.0, 41.0, 42.0, 7.0, 8.0, 9.0, 6.0]
    kd = [10.0, 11.0, 12.0, 0.7, 0.8, 0.9, 0.2]
    velocity = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 800.0]

    config = RebotB601FollowerRobotConfig(
        port="/dev/null",
        mit_kp=kp,
        mit_kd=kd,
        pos_vel_velocity=velocity,
        gripper_mit_kp=6.0,
        gripper_mit_kd=0.2,
    )

    assert config.can_adapter == "damiao"
    assert _per_joint(config.mit_kp) == dict(zip(_JOINTS, kp, strict=True))
    assert _per_joint(config.mit_kd) == dict(zip(_JOINTS, kd, strict=True))
    assert _per_joint(config.pos_vel_velocity) == dict(zip(_JOINTS, velocity, strict=True))
    assert config.gripper_mit_kp == 6.0
    assert config.gripper_mit_kd == 0.2


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected


def test_get_observation_converts_to_degrees(follower):
    obs = follower.get_observation()
    assert set(obs) == {f"{m}.pos" for m in follower.motor_names}
    # The bus mock seeds each motor's position with its 1-indexed creation order (radians).
    for idx, motor in enumerate(follower.motor_names, 1):
        assert obs[f"{motor}.pos"] == pytest.approx(math.degrees(math.radians(idx)))


def test_dm_public_positions_equal_motor_positions(follower):
    obs = follower.get_observation()

    for motor_name, motor in follower.motors.items():
        assert obs[f"{motor_name}.pos"] == pytest.approx(math.degrees(motor.get_state().pos))


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-150, 150); request beyond the upper bound.
    returned = follower.send_action({"shoulder_pan.pos": 999.0})
    assert returned["shoulder_pan.pos"] == 150.0
    # Default control_mode is "mit", so arm joints are driven via send_mit.
    follower.motors["shoulder_pan"].send_mit.assert_called_once()


def test_send_action_routes_gripper_to_force_pos(follower):
    follower.send_action({"gripper.pos": -10.0})
    follower.motors["gripper"].send_force_pos.assert_called_once()
    follower.motors["gripper"].send_pos_vel.assert_not_called()


def test_gripper_mit_mode_routes_to_send_mit():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(port="/dev/null", gripper_control_mode="mit")
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        robot.send_action({"gripper.pos": -10.0})
        robot.motors["gripper"].send_mit.assert_called_once()
        robot.motors["gripper"].send_force_pos.assert_not_called()


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601FollowerConfig(
            left_arm_config=RebotB601FollowerConfig(port="/dev/null0"),
            right_arm_config=RebotB601FollowerConfig(port="/dev/null1"),
        )
        robot = BiRebotB601Follower(cfg)
    assert any(k.startswith("left_") for k in robot.action_features)
    assert any(k.startswith("right_") for k in robot.action_features)
    assert "left_gripper.pos" in robot.action_features
    assert "right_gripper.pos" in robot.action_features
