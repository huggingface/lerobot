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
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    MotorFamily,
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)
from lerobot.robots.rebot_b601_follower.motor_family import RS_PROFILE
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


FAMILIES = [MotorFamily.DM, MotorFamily.RS]


def _make_motor_mock(position_rad: float | None = 0.0) -> MagicMock:
    motor = MagicMock(name="MotorMock")
    if position_rad is None:
        motor.get_state.return_value = None
        return motor
    state = MagicMock()
    state.pos = position_rad
    state.vel = 0.0
    motor.get_state.return_value = state
    return motor


def _make_bus_mock(positions_deg: list[float | None] | None = None) -> MagicMock:
    """Bus whose motors report a known position, in joint declaration order."""
    bus = MagicMock(name="MotorBridgeControllerMock")
    bus._motor_count = 0

    def _add_motor(send_id, recv_id, model):
        index = bus._motor_count
        bus._motor_count += 1
        # Default seeds each motor with its 1-indexed creation order, in degrees.
        position = index + 1 if positions_deg is None else positions_deg[index]
        motor = _make_motor_mock(position_rad=None if position is None else math.radians(position))
        motor.model = model
        return motor

    bus.add_damiao_motor.side_effect = _add_motor
    bus.add_robstride_motor.side_effect = _add_motor
    return bus


@contextmanager
def _connected(motor_family, *, positions_deg=None, **config_kwargs):
    bus_mock = _make_bus_mock(positions_deg)
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        # The Damiao serial bridge and native CAN path build the controller
        # differently; both resolve to the same mock here.
        controller_cls.from_dm_serial.return_value = bus_mock
        controller_cls.return_value = bus_mock
        config = RebotB601FollowerRobotConfig(motor_family=motor_family, port="/dev/null", **config_kwargs)
        robot = RebotB601Follower(config)
        robot.connect(calibrate=False)
        try:
            yield robot
        finally:
            if robot.is_connected:
                robot.disconnect()


def _build(motor_family, **config_kwargs) -> RebotB601Follower:
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        return RebotB601Follower(
            RebotB601FollowerRobotConfig(motor_family=motor_family, port="/dev/null", **config_kwargs)
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_features_match_joints(family):
    robot = _build(family)
    expected = {f"{motor}.pos" for motor in robot.motor_names}
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
    assert config.mit_kp == dict(zip(_JOINTS, [45.0, 45.0, 45.0, 8.0, 9.0, 8.0, 8.0], strict=True))
    assert config.mit_kd == dict(zip(_JOINTS, [12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 0.3], strict=True))

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


def test_scalar_joint_tuning_applies_to_every_joint():
    config = RebotB601FollowerRobotConfig(
        port="/dev/null",
        mit_kp=6.0,
        mit_kd=0.2,
        pos_vel_velocity=100.0,
    )

    assert config.can_adapter == "damiao"
    assert config.mit_kp == dict.fromkeys(_JOINTS, 6.0)
    assert config.mit_kd == dict.fromkeys(_JOINTS, 0.2)
    assert config.pos_vel_velocity == dict.fromkeys(_JOINTS, 100.0)


def test_named_joint_tuning_is_preserved():
    kp = {joint: float(index) for index, joint in enumerate(_JOINTS, start=1)}
    kd = {joint: index / 10 for index, joint in enumerate(_JOINTS, start=1)}
    velocity = {joint: float(index * 100) for index, joint in enumerate(_JOINTS, start=1)}

    config = RebotB601FollowerRobotConfig(
        port="/dev/null",
        mit_kp=kp,
        mit_kd=kd,
        pos_vel_velocity=velocity,
    )

    assert config.mit_kp == kp
    assert config.mit_kd == kd
    assert config.pos_vel_velocity == velocity


@pytest.mark.parametrize(
    ("family", "adapter", "uses_serial_bridge"),
    [
        (MotorFamily.DM, None, True),
        (MotorFamily.DM, "socketcan", False),
        (MotorFamily.RS, None, False),
    ],
)
def test_connect_uses_the_configured_transport(family, adapter, uses_serial_bridge):
    bus = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus
        controller_cls.return_value = bus
        kwargs = {} if adapter is None else {"can_adapter": adapter}
        robot = RebotB601Follower(
            RebotB601FollowerRobotConfig(motor_family=family, port="/dev/null", **kwargs)
        )
        robot.connect(calibrate=False)

        if uses_serial_bridge:
            controller_cls.from_dm_serial.assert_called_once_with(serial_port="/dev/null", baud=921600)
            controller_cls.assert_not_called()
        else:
            controller_cls.assert_called_once_with(channel="/dev/null")
            controller_cls.from_dm_serial.assert_not_called()

        robot.disconnect()
        assert not robot.is_connected


@pytest.mark.parametrize(
    ("family", "expected_position"),
    [(MotorFamily.DM, 10.0), (MotorFamily.RS, -10.0)],
)
def test_get_observation_uses_the_public_coordinate_frame(family, expected_position):
    with _connected(family, positions_deg=[10.0] * len(_JOINTS)) as robot:
        obs = robot.get_observation()
        assert set(obs) == {f"{motor}.pos" for motor in robot.motor_names}
        assert all(position == pytest.approx(expected_position) for position in obs.values())


@pytest.mark.parametrize(
    ("family", "expected_factory", "unused_factory", "expected_model"),
    [
        (MotorFamily.DM, "add_damiao_motor", "add_robstride_motor", "4340P"),
        (MotorFamily.RS, "add_robstride_motor", "add_damiao_motor", "rs-06"),
    ],
)
def test_registers_motors_with_the_family_factory(family, expected_factory, unused_factory, expected_model):
    with _connected(family) as robot:
        assert getattr(robot.bus, expected_factory).call_count == len(robot.motor_names)
        getattr(robot.bus, unused_factory).assert_not_called()
        # The three proximal joints carry the larger motor model.
        assert robot.motors["shoulder_pan"].model == expected_model


@pytest.mark.parametrize(
    ("family", "expected_public_position", "expected_motor_position"),
    [(MotorFamily.DM, 150.0, 150.0), (MotorFamily.RS, 145.0, -145.0)],
)
def test_send_action_clips_to_the_family_joint_limits(
    family, expected_public_position, expected_motor_position
):
    with _connected(family) as robot:
        returned = robot.send_action({"shoulder_pan.pos": 999.0})
        # The return value stays in the public robot frame for both families.
        assert returned["shoulder_pan.pos"] == expected_public_position
        robot.motors["shoulder_pan"].send_mit.assert_called_once()
        motor_position = math.degrees(robot.motors["shoulder_pan"].send_mit.call_args.args[0])
        assert motor_position == pytest.approx(expected_motor_position)


def test_rs_observation_can_be_sent_back_to_hold_position():
    with _connected(MotorFamily.RS, positions_deg=[10.0] * 7) as robot:
        observed = robot.get_observation()["shoulder_pan.pos"]
        robot.send_action({"shoulder_pan.pos": observed})
        motor_position = robot.motors["shoulder_pan"].send_mit.call_args.args[0]
        assert observed == -10.0
        assert math.degrees(motor_position) == pytest.approx(10.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_partial_action_does_not_command_unspecified_joints(family):
    positions = [0.0, 0.0, 0.0, 0.0, -2.393, 0.0, 0.0]
    with _connected(family, positions_deg=positions, max_relative_target=2.0) as robot:
        returned = robot.send_action({"shoulder_pan.pos": 1.0})

        assert returned == {"shoulder_pan.pos": 1.0}
        robot.motors["shoulder_pan"].send_mit.assert_called_once()
        for motor_name in set(robot.motor_names) - {"shoulder_pan"}:
            robot.motors[motor_name].send_mit.assert_not_called()
            robot.motors[motor_name].send_pos_vel.assert_not_called()
            robot.motors[motor_name].send_force_pos.assert_not_called()


@pytest.mark.parametrize(
    ("config_kwargs", "action", "motor_name", "method"),
    [
        ({}, {"gripper.pos": -10.0}, "gripper", "send_force_pos"),
        ({"gripper_control_mode": "mit"}, {"gripper.pos": -10.0}, "gripper", "send_mit"),
        ({"control_mode": "pos_vel"}, {"shoulder_pan.pos": 10.0}, "shoulder_pan", "send_pos_vel"),
    ],
)
def test_dm_routes_actions_by_control_mode(config_kwargs, action, motor_name, method):
    with _connected(MotorFamily.DM, **config_kwargs) as robot:
        robot.send_action(action)
        getattr(robot.motors[motor_name], method).assert_called_once()


def test_rs_gripper_uses_force_limited_impedance():
    # The impedance strategy drives the gripper purely by a feedforward torque:
    # position setpoint and kp are zero so grip force stays bounded.
    with _connected(MotorFamily.RS) as robot:
        returned = robot.send_action({"gripper.pos": -999.0})
        position, velocity, kp, kd, tau = robot.motors["gripper"].send_mit.call_args.args
        assert (position, velocity, kp) == (0.0, 0.0, 0.0)
        assert kd > 0.0
        assert abs(tau) <= robot.config.gripper_hold_torque_limit
        assert returned["gripper.pos"] == -270.0


def test_rs_gripper_disables_torque_after_feedback_failure():
    with _connected(MotorFamily.RS) as robot:
        gripper = robot.motors["gripper"]
        robot.bus.poll_feedback_once.side_effect = RuntimeError("temporary CAN error")
        gripper.send_mit.side_effect = RuntimeError("CAN transmit failed")
        with pytest.raises(RuntimeError, match="temporary CAN error"):
            robot.send_action({"gripper.pos": -100.0})
        assert gripper.send_mit.call_args.args[4] == 0.0
        robot.bus.disable_all.assert_called()
        assert robot.bus is not None


@pytest.mark.parametrize("family", FAMILIES)
def test_mode_switching_happens_with_torque_disabled(family):
    with _connected(family) as robot:
        calls = [name for name, _, _ in robot.bus.mock_calls if name in ("disable_all", "enable_all")]
        assert calls[-1] == "enable_all"
        assert all(call == "disable_all" for call in calls[:-1])


def test_observation_propagates_motorbridge_feedback_errors():
    with _connected(MotorFamily.DM) as robot:
        bus = robot.bus
        bus.poll_feedback_once.side_effect = RuntimeError("CAN poll failed")

        with pytest.raises(RuntimeError, match="CAN poll failed"):
            robot.get_observation()

        assert robot.bus is bus


def test_partial_action_subsets_per_joint_relative_limits():
    limits = dict.fromkeys(_JOINTS, 1.0)
    with _connected(MotorFamily.DM, max_relative_target=limits) as robot:
        returned = robot.send_action({"shoulder_pan.pos": 100.0, "wrist_yaw.pos": 5.0})
        assert returned["shoulder_pan.pos"] == 2.0
        assert returned["wrist_yaw.pos"] == 5.0


def test_explicit_config_values_are_passed_through():
    config = RebotB601FollowerRobotConfig(
        motor_family=MotorFamily.RS,
        port="/dev/ttyACM0",
        can_adapter="damiao",
        gripper_control_mode="mit",
        gripper_torque_limit=1.0,
        gripper_hold_torque_limit=2.0,
    )
    assert config.can_adapter == "damiao"
    assert config.gripper_control_mode == "mit"
    assert config.gripper_torque_limit == 1.0
    assert config.gripper_hold_torque_limit == 2.0


def test_rs_defaults_are_selected():
    config = RebotB601FollowerRobotConfig(motor_family=MotorFamily.RS, port="can0")
    assert config.can_adapter == "socketcan"
    assert config.gripper_control_mode == "mit_impedance"
    assert config.motor_can_ids == RS_PROFILE.motor_can_ids
    assert config.mit_kp == RS_PROFILE.mit_kp
    assert config.joint_limits == RS_PROFILE.joint_limits


def test_bimanual_accepts_per_arm_motor_families():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = BiRebotB601Follower(
            BiRebotB601FollowerConfig(
                id="pair",
                left_arm_config=RebotB601FollowerConfig(motor_family=MotorFamily.DM, port="/dev/null0"),
                right_arm_config=RebotB601FollowerConfig(
                    motor_family=MotorFamily.RS,
                    port="can0",
                    max_relative_target=5.0,
                    gripper_torque_limit=2.0,
                ),
            )
        )
    assert "left_gripper.pos" in robot.action_features
    assert "right_gripper.pos" in robot.action_features
    assert robot.left_arm.config.motor_family is MotorFamily.DM
    assert robot.right_arm.config.motor_family is MotorFamily.RS
    assert robot.right_arm.config.max_relative_target == 5.0
    assert robot.right_arm.config.gripper_torque_limit == 2.0
    assert robot.right_arm.config.id == "pair_right"
