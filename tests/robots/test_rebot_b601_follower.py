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
from unittest.mock import MagicMock, call, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    ArmControlMode,
    GripperControlMode,
    MotorFamily,
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)
from lerobot.robots.rebot_b601_follower.motor_family import (
    DM_PROFILE,
    JOINT_NAMES,
    MOTOR_PROFILES,
    RS_PROFILE,
)
from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderConfig

_MODULE = "lerobot.robots.rebot_b601_follower.rebot_b601_follower"
_EXPECTED_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


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


def test_motor_family_structure_is_complete():
    assert JOINT_NAMES == _EXPECTED_JOINT_NAMES
    assert set(MOTOR_PROFILES) == set(MotorFamily)


@pytest.mark.parametrize("family", MotorFamily)
def test_features_match_joints(family):
    robot = _build(family)
    expected = {f"{motor}.pos" for motor in robot.motor_names}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected
    assert "gripper.pos" in expected


def test_shipped_dm_defaults_are_preserved():
    config = RebotB601FollowerRobotConfig(port="/dev/null")

    assert config.can_adapter == "damiao"
    assert config.control_mode is ArmControlMode.MIT
    assert config.gripper_control_mode is GripperControlMode.FORCE_POS
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
    assert config.mit_kp == dict(zip(JOINT_NAMES, [45.0, 45.0, 45.0, 8.0, 9.0, 8.0, 8.0], strict=True))
    assert config.mit_kd == dict(zip(JOINT_NAMES, [12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 0.3], strict=True))


def test_dm_joint_limits_match_leader_ranges():
    follower = RebotB601FollowerRobotConfig(port="/dev/null")
    leader = RebotArm102LeaderConfig(port="/dev/null")

    assert follower.joint_limits == {joint: tuple(limits) for joint, limits in leader.joint_ranges.items()}


def test_scalar_joint_tuning_applies_to_every_joint():
    config = RebotB601FollowerRobotConfig(
        port="/dev/null",
        mit_kp=6.0,
        mit_kd=0.2,
        pos_vel_velocity=100.0,
    )

    assert config.can_adapter == "damiao"
    assert config.mit_kp == dict.fromkeys(JOINT_NAMES, 6.0)
    assert config.mit_kd == dict.fromkeys(JOINT_NAMES, 0.2)
    assert config.pos_vel_velocity == dict.fromkeys(JOINT_NAMES, 100.0)


def test_named_joint_tuning_is_preserved():
    kp = {joint: float(index) for index, joint in enumerate(JOINT_NAMES, start=1)}
    kd = {joint: index / 10 for index, joint in enumerate(JOINT_NAMES, start=1)}
    velocity = {joint: float(index * 100) for index, joint in enumerate(JOINT_NAMES, start=1)}

    config = RebotB601FollowerRobotConfig(
        port="/dev/null",
        mit_kp=kp,
        mit_kd=kd,
        pos_vel_velocity=velocity,
    )

    assert config.mit_kp == kp
    assert config.mit_kd == kd
    assert config.pos_vel_velocity == velocity


def test_named_joint_tuning_overrides_profile_defaults():
    config = RebotB601FollowerRobotConfig(
        port="/dev/null",
        mit_kp={"gripper": 5.0},
    )

    assert config.mit_kp == {**DM_PROFILE.mit_kp, "gripper": 5.0}


@pytest.mark.parametrize(("family", "profile"), [(MotorFamily.DM, DM_PROFILE), (MotorFamily.RS, RS_PROFILE)])
def test_named_joint_limits_override_profile_defaults(family, profile):
    gripper_limits = (-180.0, 0.0)
    config = RebotB601FollowerRobotConfig(
        motor_family=family,
        port="/dev/null",
        joint_limits={"gripper": gripper_limits},
    )

    assert config.joint_limits == {**profile.joint_limits, "gripper": gripper_limits}


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
    assert config.joint_limits == {
        "shoulder_pan": (-145.0, 145.0),
        "shoulder_lift": (-170.0, 0.0),
        "elbow_flex": (-200.0, 0.0),
        "wrist_flex": (-90.0, 80.0),
        "wrist_yaw": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
        "gripper": (-270.0, 0.0),
    }


@pytest.mark.parametrize(
    ("config_kwargs", "mode_type"),
    [
        ({"control_mode": "force_pos"}, "ArmControlMode"),
        ({"gripper_control_mode": "pos_vel"}, "GripperControlMode"),
    ],
)
def test_control_modes_reject_values_for_the_wrong_motor_group(config_kwargs, mode_type):
    with pytest.raises(ValueError, match=mode_type):
        RebotB601FollowerRobotConfig(port="/dev/null", **config_kwargs)


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


def test_connect_rejects_unknown_can_adapter():
    robot = _build(MotorFamily.DM, can_adapter="unknown")

    with pytest.raises(ValueError, match="Unsupported can_adapter 'unknown'"):
        robot.connect(calibrate=False)


@pytest.mark.parametrize(
    ("family", "config_kwargs", "message"),
    [
        (MotorFamily.RS, {"control_mode": "pos_vel"}, "POS_VEL requires pos_vel_velocity"),
        (
            MotorFamily.DM,
            {"gripper_control_mode": "mit_impedance"},
            "MIT impedance requires moving and holding gripper torque limits",
        ),
        (
            MotorFamily.RS,
            {"gripper_control_mode": "force_pos"},
            "FORCE_POS requires gripper velocity and torque ratio settings",
        ),
    ],
)
def test_missing_mode_tuning_fails_before_opening_hardware(family, config_kwargs, message):
    bus = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus
        controller_cls.return_value = bus
        robot = RebotB601Follower(
            RebotB601FollowerRobotConfig(
                motor_family=family,
                port="/dev/null",
                **config_kwargs,
            )
        )

        with pytest.raises(ValueError, match=message):
            robot.connect(calibrate=False)

    controller_cls.from_dm_serial.assert_not_called()
    controller_cls.assert_not_called()
    bus.enable_all.assert_not_called()


def test_configure_validates_control_settings_independently():
    robot = _build(MotorFamily.RS, control_mode="pos_vel")
    robot.bus = MagicMock()

    with pytest.raises(ValueError, match="POS_VEL requires pos_vel_velocity"):
        robot.configure()

    robot.bus.disable_all.assert_not_called()


@pytest.mark.parametrize(
    ("family", "expected_position"),
    [(MotorFamily.DM, 10.0), (MotorFamily.RS, -10.0)],
)
def test_get_observation_uses_the_public_coordinate_frame(family, expected_position):
    with _connected(family, positions_deg=[10.0] * len(JOINT_NAMES)) as robot:
        obs = robot.get_observation()
        assert set(obs) == {f"{motor}.pos" for motor in robot.motor_names}
        assert all(position == pytest.approx(expected_position) for position in obs.values())


@pytest.mark.parametrize(
    ("family", "expected_factory", "unused_factory", "profile"),
    [
        (MotorFamily.DM, "add_damiao_motor", "add_robstride_motor", DM_PROFILE),
        (MotorFamily.RS, "add_robstride_motor", "add_damiao_motor", RS_PROFILE),
    ],
)
def test_registers_motors_with_the_family_profile(family, expected_factory, unused_factory, profile):
    with _connected(family) as robot:
        expected_calls = [
            call(send_id, recv_id, profile.motor_models[motor_name])
            for motor_name, (send_id, recv_id) in profile.motor_can_ids.items()
        ]
        assert getattr(robot.bus, expected_factory).call_args_list == expected_calls
        getattr(robot.bus, unused_factory).assert_not_called()


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


@pytest.mark.parametrize(
    ("target", "expected_public_position", "expected_motor_position"),
    [(999.0, 80.0, -80.0), (-999.0, -90.0, 90.0)],
)
def test_rs_asymmetric_joint_limits_use_public_coordinates(
    target, expected_public_position, expected_motor_position
):
    with _connected(MotorFamily.RS) as robot:
        returned = robot.send_action({"wrist_flex.pos": target})

        assert returned["wrist_flex.pos"] == expected_public_position
        motor_position = math.degrees(robot.motors["wrist_flex"].send_mit.call_args.args[0])
        assert motor_position == pytest.approx(expected_motor_position)


def test_rs_calibration_converts_public_joint_limits_to_motor_coordinates():
    with (
        _connected(MotorFamily.RS) as robot,
        patch("builtins.input", return_value=""),
        patch(f"{_MODULE}.time.sleep"),
        patch.object(robot, "_save_calibration"),
    ):
        robot.calibrate()

        assert robot.calibration["wrist_flex"].range_min == -80
        assert robot.calibration["wrist_flex"].range_max == 90
        assert robot.calibration["gripper"].range_min == 0
        assert robot.calibration["gripper"].range_max == 270


def test_rs_observation_can_be_sent_back_to_hold_position():
    with _connected(MotorFamily.RS, positions_deg=[10.0] * 7) as robot:
        observed = robot.get_observation()["shoulder_pan.pos"]
        robot.send_action({"shoulder_pan.pos": observed})
        motor_position = robot.motors["shoulder_pan"].send_mit.call_args.args[0]
        assert observed == -10.0
        assert math.degrees(motor_position) == pytest.approx(10.0)


@pytest.mark.parametrize("family", MotorFamily)
def test_partial_action_does_not_command_unspecified_joints(family):
    with _connected(
        family,
        positions_deg=[0.0] * len(JOINT_NAMES),
        max_relative_target=2.0,
    ) as robot:
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
    with _connected(MotorFamily.RS) as robot:
        returned = robot.send_action({"gripper.pos": -999.0})
        gripper = robot.motors["gripper"]
        position, velocity, kp, kd, tau = gripper.send_mit.call_args.args
        assert (position, velocity, kp) == (0.0, 0.0, 0.0)
        assert kd > 0.0
        assert abs(tau) <= robot.config.gripper_hold_torque_limit
        assert returned["gripper.pos"] == -270.0
        gripper.send_force_pos.assert_not_called()


def test_rs_gripper_reuses_relative_limit_feedback():
    with _connected(MotorFamily.RS, max_relative_target=5.0) as robot:
        robot.send_action({"gripper.pos": -10.0})

        robot.bus.poll_feedback_once.assert_called_once()


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


def test_rs_observation_disables_torque_after_feedback_failure(caplog):
    with _connected(MotorFamily.RS) as robot:
        robot.bus.poll_feedback_once.side_effect = RuntimeError("temporary CAN error")

        with pytest.raises(RuntimeError, match="temporary CAN error"):
            robot.get_observation()

        assert robot.motors["gripper"].send_mit.call_args.args[4] == 0.0
        robot.bus.disable_all.assert_called()
        assert "Motor feedback failed; disabling all motor torque." in caplog.messages


@pytest.mark.parametrize("family", MotorFamily)
def test_mode_switching_happens_with_torque_disabled(family):
    with _connected(family) as robot:
        events = []
        robot.bus.disable_all.side_effect = lambda: events.append("disable")
        robot.bus.enable_all.side_effect = lambda: events.append("enable")
        for motor_name, motor in robot.motors.items():
            motor.ensure_mode.side_effect = lambda _mode, name=motor_name: events.append(name)

        robot.configure()

        assert events == ["disable", *robot.motor_names, "enable"]


def test_configure_delegates_mode_errors_to_motorbridge():
    with _connected(MotorFamily.DM) as robot:
        motor = robot.motors["shoulder_pan"]
        motor.ensure_mode.reset_mock()
        motor.ensure_mode.side_effect = RuntimeError("mode switch failed")
        robot.bus.enable_all.reset_mock()

        with pytest.raises(RuntimeError, match="mode switch failed"):
            robot.configure()

        motor.ensure_mode.assert_called_once()
        robot.bus.enable_all.assert_not_called()


def test_observation_propagates_motorbridge_feedback_errors():
    with _connected(MotorFamily.DM) as robot:
        bus = robot.bus
        bus.poll_feedback_once.side_effect = RuntimeError("CAN poll failed")

        with pytest.raises(RuntimeError, match="CAN poll failed"):
            robot.get_observation()

        assert robot.bus is bus


def test_observation_reports_unavailable_motor_feedback():
    positions = [0.0] * len(JOINT_NAMES)
    positions[2] = None
    with (
        _connected(MotorFamily.DM, positions_deg=positions) as robot,
        pytest.raises(RuntimeError, match="No motor feedback available for: elbow_flex"),
    ):
        robot.get_observation()


def test_partial_action_subsets_per_joint_relative_limits():
    limits = dict.fromkeys(JOINT_NAMES, 1.0)
    with _connected(MotorFamily.DM, max_relative_target=limits) as robot:
        returned = robot.send_action({"shoulder_pan.pos": 100.0, "wrist_yaw.pos": 5.0})
        assert returned["shoulder_pan.pos"] == 2.0
        assert returned["wrist_yaw.pos"] == 5.0


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
