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
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_rs_follower import BiRebotB601RSFollower, BiRebotB601RSFollowerConfig
from lerobot.robots.rebot_b601_rs_follower import (
    RebotB601RSFollower,
    RebotB601RSFollowerConfig,
    RebotB601RSFollowerRobotConfig,
)

# RebotB601RSFollower is a standalone class (a self-contained copy of the
# Damiao variant), so the motorbridge symbols it looks up live in its own
# module — patch there.
_MODULE = "lerobot.robots.rebot_b601_rs_follower.rebot_b601_rs_follower"


def _make_motor_mock(position_rad: float = 0.0) -> MagicMock:
    motor = MagicMock(name="RobStrideMotorMock")
    state = MagicMock()
    state.pos = position_rad
    state.vel = 0.0
    motor.get_state.return_value = state
    return motor


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MotorBridgeControllerMock")
    # add_robstride_motor returns a fresh motor mock; position encodes call order.
    bus._motor_count = 0

    def _add_motor(_send_id, _recv_id, _model):
        bus._motor_count += 1
        return _make_motor_mock(position_rad=math.radians(bus._motor_count))

    bus.add_robstride_motor.side_effect = _add_motor
    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        # RS default can_adapter is "socketcan": connect() calls the controller
        # constructor directly: MotorBridgeController(channel=port).
        controller_cls.return_value = bus_mock
        cfg = RebotB601RSFollowerRobotConfig(port="can0")
        robot = RebotB601RSFollower(cfg)
        robot.connect(calibrate=False)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RebotB601RSFollower(RebotB601RSFollowerRobotConfig(port="can0"))
    expected = {f"{m}.pos" for m in robot.motor_names}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected
    assert "gripper.pos" in expected


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected


def test_registers_robstride_motors_not_damiao(follower):
    # The RS variant must register motors via add_robstride_motor, never
    # add_damiao_motor.
    assert follower.bus.add_robstride_motor.call_count == len(follower.motor_names)
    follower.bus.add_damiao_motor.assert_not_called()


def test_uses_shared_feedback_can_id(follower):
    # All RobStride motors share the 0xFD feedback id.
    for (_send_id, recv_id), motor_name in zip(
        follower.config.motor_can_ids.values(), follower.motor_names
    ):
        assert recv_id == 0xFD, f"{motor_name} has non-shared recv_id {recv_id:#x}"


def test_get_observation_converts_to_degrees(follower):
    obs = follower.get_observation()
    assert set(obs) == {f"{m}.pos" for m in follower.motor_names}
    # The bus mock seeds each motor's position with its 1-indexed creation order (radians).
    for idx, motor in enumerate(follower.motor_names, 1):
        assert obs[f"{motor}.pos"] == pytest.approx(math.degrees(math.radians(idx)))


def test_send_action_applies_joint_directions(follower):
    # RobStride motors are mounted opposite to the Damiao variant, so the
    # leader's Damiao-convention action is sign-flipped before reaching the motor.
    # shoulder_pan direction is -1; +30 deg -> motor sees -30 deg.
    follower.send_action({"shoulder_pan.pos": 30.0})
    pos_rad, _vel, _kp, _kd, _tau = follower.motors["shoulder_pan"].send_mit.call_args.args
    assert pos_rad == pytest.approx(math.radians(-30.0))


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-145, 145) with direction -1; -999 -> +999 -> clip to 145.
    # Asserting +145 (not -145) also proves the direction flip runs before clipping.
    returned = follower.send_action({"shoulder_pan.pos": -999.0})
    assert returned["shoulder_pan.pos"] == 145.0
    # control_mode is "mit", so arm joints are driven via send_mit.
    follower.motors["shoulder_pan"].send_mit.assert_called_once()


def test_send_action_routes_gripper_to_mit(follower):
    # RS gripper_control_mode is "mit": the gripper (rs-00) must be driven via
    # send_mit, never send_force_pos (a Damiao-only mode).
    follower.send_action({"gripper.pos": -10.0})
    follower.motors["gripper"].send_mit.assert_called_once()
    follower.motors["gripper"].send_force_pos.assert_not_called()
    follower.motors["gripper"].send_pos_vel.assert_not_called()


def test_gripper_uses_force_limited_impedance(follower):
    # The RS gripper is driven in force mode: send_mit(0, 0, 0, kd_damping, tau_ff)
    # where tau_ff is an impedance torque clamped to a safe limit — NOT a plain
    # position-MIT command (which would push to the target regardless of force).
    follower.send_action({"gripper.pos": -100.0})
    args = follower.motors["gripper"].send_mit.call_args.args
    pos_des, vel_des, kp, kd, tau_ff = args
    assert pos_des == 0.0  # no position setpoint
    assert kp == 0.0  # pure feedforward torque (no position stiffness at the motor)
    # On the first call the estimated state velocity is 0, so the gentler hold
    # limit applies.
    assert abs(tau_ff) <= follower.config.gripper_mit_hold_torque_limit


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601RSFollowerConfig(
            left_arm_config=RebotB601RSFollowerConfig(port="can0"),
            right_arm_config=RebotB601RSFollowerConfig(port="can1"),
        )
        robot = BiRebotB601RSFollower(cfg)
    assert any(k.startswith("left_") for k in robot.action_features)
    assert any(k.startswith("right_") for k in robot.action_features)
    assert "left_gripper.pos" in robot.action_features
    assert "right_gripper.pos" in robot.action_features


def test_bimanual_forwards_rs_specific_config_fields():
    # The bimanual wrapper must forward the RS-specific config fields
    # (joint_directions + gripper MIT torque limits) to each single arm — they
    # are not present on the Damiao variant, so they must be passed explicitly.
    custom_dirs = {
        "shoulder_pan": -1.0, "shoulder_lift": -1.0, "elbow_flex": -1.0,
        "wrist_flex": -1.0, "wrist_yaw": -1.0, "wrist_roll": -1.0, "gripper": -2.0,
    }
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601RSFollowerConfig(
            left_arm_config=RebotB601RSFollowerConfig(
                port="can0",
                joint_directions=custom_dirs,
                gripper_mit_torque_limit=5.0,
                gripper_mit_hold_torque_limit=2.0,
            ),
            right_arm_config=RebotB601RSFollowerConfig(port="can1"),
        )
        robot = BiRebotB601RSFollower(cfg)
    # Left arm received the custom values.
    assert robot.left_arm.config.joint_directions["gripper"] == -2.0
    assert robot.left_arm.config.gripper_mit_torque_limit == 5.0
    assert robot.left_arm.config.gripper_mit_hold_torque_limit == 2.0
    # Right arm kept the defaults.
    assert robot.right_arm.config.joint_directions["gripper"] == -1.0
    assert robot.right_arm.config.gripper_mit_torque_limit == 3.5
    assert robot.right_arm.config.gripper_mit_hold_torque_limit == 1.0
