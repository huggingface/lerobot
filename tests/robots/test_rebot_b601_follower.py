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

import threading
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError

_MODULE = "lerobot.robots.rebot_b601_follower.rebot_b601_follower"


class _FakeGravity:
    urdf_path = "/tmp/rebot.urdf"

    def torque(self, positions):
        return {name: 0.5 for name in positions if name != "gripper"}


def _make_motor_mock(position_rad: float = 0.0) -> MagicMock:
    motor = MagicMock(name="MotorMock")
    state = MagicMock()
    state.pos = position_rad
    motor.get_state.return_value = state
    return motor


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MotorBridgeControllerMock")

    def _add_motor(_send_id, _recv_id, _model):
        # Keep the startup pose within the B601 soft joint limits so teardown
        # can exercise safe_home instead of failing validation.
        return _make_motor_mock()

    bus.add_damiao_motor.side_effect = _add_motor
    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
        patch(f"{_MODULE}.B601GravityFeedforward", _FakeGravity),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(
            port="/dev/null",
            home_duration_s=0.0,
            home_hz=1000.0,
            gripper_open_duration_s=0.0,
            gripper_close_duration_s=0.0,
        )
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


def test_partial_home_action_is_rejected_before_connect():
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        pytest.raises(ValueError, match="must specify every arm joint"),
    ):
        RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null", home_action={"shoulder_pan": 0.0}))


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected


def test_motor_lifecycle_uses_bus_when_camera_is_disconnected(follower):
    follower.cameras = {"failed_camera": MagicMock(is_connected=False)}
    assert not follower.is_connected

    with pytest.raises(DeviceAlreadyConnectedError, match="motor bus is already connected"):
        follower.connect(calibrate=False)

    follower.disable_torque()
    follower.bus.disable_all.assert_called_once()
    follower.cameras = {}


def test_single_arm_connect_failure_emergency_disables_and_closes_resources():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
        patch(f"{_MODULE}.B601GravityFeedforward", _FakeGravity),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null"))
        with (
            patch.object(robot, "configure", side_effect=RuntimeError("configure failed")),
            pytest.raises(RuntimeError, match="configure failed"),
        ):
            robot.connect(calibrate=False)

    bus_mock.disable_all.assert_called_once()
    bus_mock.close.assert_called_once()
    assert robot.bus is None


def test_connect_captures_startup_arm_pose_without_gripper(follower):
    assert follower._startup_home_action == {
        motor_name: pytest.approx(0.0) for motor_name in follower.motor_names[:-1]
    }
    assert "gripper" not in follower._startup_home_action


def test_safe_home_rejects_target_outside_joint_limits(follower):
    follower.config.home_action = {
        motor_name: (999.0 if motor_name == "shoulder_pan" else 0.0)
        for motor_name in follower.motor_names[:-1]
    }
    follower._startup_home_action = {}

    assert not follower.safe_home()
    assert not follower._safe_home_succeeded
    follower.config.home_on_disconnect = False


def test_get_observation_converts_to_degrees(follower):
    obs = follower.get_observation()
    assert set(obs) == {f"{m}.pos" for m in follower.motor_names}
    assert all(position == pytest.approx(0.0) for position in obs.values())


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-150, 150); request beyond the upper bound.
    returned = follower.send_action({"shoulder_pan.pos": 999.0})
    assert returned["shoulder_pan.pos"] == 150.0
    # Default control_mode is "mit", so arm joints are driven via send_mit.
    follower.motors["shoulder_pan"].send_mit.assert_called_once()


def test_mit_actions_always_use_gravity_feedforward(follower):
    class FakeGravity:
        urdf_path = "/tmp/rebot.urdf"

        def torque(self, _positions):
            return dict.fromkeys(follower.motor_names[:-1], 1.25)

    follower._gravity_feedforward = FakeGravity()
    follower.send_action({"shoulder_pan.pos": 1.0})

    assert follower.motors["shoulder_pan"].send_mit.call_args.args[-1] == pytest.approx(1.25)


def test_mit_action_rejects_missing_arm_feedback(follower):
    motor = follower.motors["shoulder_pan"]
    original_state = motor.get_state.return_value
    motor.get_state.return_value = None

    try:
        with pytest.raises(RuntimeError, match="invalid feedback"):
            follower.send_action({"shoulder_pan.pos": 1.0})
    finally:
        motor.get_state.return_value = original_state


def test_safe_home_uses_gravity_feedforward_for_each_mit_command(follower):
    class FakeGravity:
        urdf_path = "/tmp/rebot.urdf"

        def torque(self, _positions):
            return dict.fromkeys(follower.motor_names[:-1], 2.5)

    follower._gravity_feedforward = FakeGravity()
    follower.config.home_action = dict.fromkeys(follower.motor_names[:-1], 0.0)
    follower._startup_home_action = {}
    follower.config.open_gripper_before_home = False
    follower.config.close_gripper_after_home = False

    assert follower.safe_home()
    for motor_name in follower.motor_names[:-1]:
        assert all(
            call.args[-1] == pytest.approx(2.5)
            for call in follower.motors[motor_name].send_mit.call_args_list
        )


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
        patch(f"{_MODULE}.B601GravityFeedforward", _FakeGravity),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(
            port="/dev/null", gripper_control_mode="mit", home_on_disconnect=False
        )
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        robot.send_action({"gripper.pos": -10.0})
        robot.motors["gripper"].send_mit.assert_called_once()
        assert robot.motors["gripper"].send_mit.call_args.args[-1] == pytest.approx(0.0)
        robot.motors["gripper"].send_force_pos.assert_not_called()


def test_safe_home_uses_explicit_physical_joint_targets(follower):
    home = {
        "shoulder_pan": 1.0,
        "shoulder_lift": -1.0,
        "elbow_flex": -2.0,
        "wrist_flex": 3.0,
        "wrist_yaw": 4.0,
        "wrist_roll": 5.0,
    }
    follower.config.home_action = home
    follower._startup_home_action = {}

    assert follower.safe_home()
    assert follower._safe_home_succeeded
    follower.motors["gripper"].send_force_pos.assert_called()
    for motor_name in follower.motor_names[:-1]:
        follower.motors[motor_name].send_mit.assert_called()


def test_safe_home_holds_arm_before_and_during_gripper_motion(follower):
    home = {
        "shoulder_pan": 1.0,
        "shoulder_lift": -1.0,
        "elbow_flex": -2.0,
        "wrist_flex": 3.0,
        "wrist_yaw": 4.0,
        "wrist_roll": 5.0,
    }
    follower.config.home_action = home
    follower._startup_home_action = {}
    follower.config.open_gripper_before_home = True
    follower.config.gripper_open_duration_s = 0.0
    follower.config.gripper_close_duration_s = 0.0

    with patch.object(follower, "_send_goal_positions", wraps=follower._send_goal_positions) as send_goals:
        assert follower.safe_home()

    sent_goals = [call.args[0] for call in send_goals.call_args_list]
    startup_hold = sent_goals[0]
    assert startup_hold == {motor_name: pytest.approx(0.0) for motor_name in follower.motor_names[:-1]}
    assert any(
        all(goal[motor_name] == pytest.approx(0.0) for motor_name in follower.motor_names[:-1])
        and goal.get("gripper") == follower.config.gripper_open_position_deg
        for goal in sent_goals
    )
    assert any(
        all(goal[motor_name] == home[motor_name] for motor_name in follower.motor_names[:-1])
        and goal.get("gripper") == follower.config.gripper_closed_position_deg
        for goal in sent_goals
    )


def test_safe_home_starts_from_latest_measured_pose_not_last_policy_target(follower):
    last_target = {
        "shoulder_pan": 4.0,
        "shoulder_lift": -56.0,
        "elbow_flex": -45.0,
        "wrist_flex": 3.0,
        "wrist_yaw": -7.0,
        "wrist_roll": -3.0,
    }
    follower.send_action({f"{name}.pos": value for name, value in last_target.items()})
    follower.config.home_action = dict.fromkeys(last_target, 0.0)
    follower._startup_home_action = {}
    follower.config.open_gripper_before_home = False
    follower.config.close_gripper_after_home = False
    follower.config.home_velocity_deg_s = 1_000_000.0

    with patch.object(follower, "_send_goal_positions", wraps=follower._send_goal_positions) as send_goals:
        assert follower.safe_home()

    first_safe_home_target = send_goals.call_args_list[0].args[0]
    assert first_safe_home_target == {
        motor_name: pytest.approx(0.0) for motor_name in follower.motor_names[:-1]
    }


def test_safe_home_uses_seeed_two_stage_joint_order(follower):
    home = {
        "shoulder_pan": 1.0,
        "shoulder_lift": -1.0,
        "elbow_flex": -2.0,
        "wrist_flex": 3.0,
        "wrist_yaw": 4.0,
        "wrist_roll": 5.0,
    }
    follower.config.home_action = home
    follower._startup_home_action = {}
    follower.config.home_duration_s = 0.0
    follower.config.home_hz = 1000.0
    follower.config.home_velocity_deg_s = 1_000_000.0
    follower.config.open_gripper_before_home = False
    follower.config.close_gripper_after_home = False

    with patch.object(follower, "_send_goal_positions", wraps=follower._send_goal_positions) as send_goals:
        assert follower.safe_home()

    sent_goals = [call.args[0] for call in send_goals.call_args_list]
    first_stage = sent_goals[1]
    assert first_stage["shoulder_pan"] == home["shoulder_pan"]
    assert first_stage["wrist_flex"] == home["wrist_flex"]
    assert first_stage["wrist_yaw"] == home["wrist_yaw"]
    assert first_stage["wrist_roll"] == home["wrist_roll"]
    assert first_stage["shoulder_lift"] == pytest.approx(0.0)
    assert first_stage["elbow_flex"] == pytest.approx(0.0)
    assert "gripper" not in first_stage

    second_stage = sent_goals[2]
    assert all(second_stage[motor_name] == home[motor_name] for motor_name in follower.motor_names[:-1])
    assert "gripper" not in second_stage


def test_normal_action_invalidates_completed_safe_home(follower):
    follower._safe_home_succeeded = True

    follower.send_action({"shoulder_pan.pos": 0.0})

    assert not follower._safe_home_succeeded


def test_disconnect_keeps_bus_and_torque_when_safe_home_fails(follower):
    bus = follower.bus
    follower._startup_home_action = {}
    follower.config.home_action = {}

    with pytest.raises(RuntimeError, match="Refusing to disconnect"):
        follower.disconnect()

    assert follower.bus is bus
    bus.close.assert_not_called()
    bus.disable_all.assert_not_called()
    for motor in follower.motors.values():
        motor.disable.assert_not_called()
        motor.close.assert_not_called()
    follower.config.home_on_disconnect = False


def test_emergency_disable_skips_safe_home_on_disconnect(follower):
    bus = follower.bus
    follower._startup_home_action = {}

    follower.emergency_disable()
    follower.disconnect()

    bus.disable_all.assert_called_once()
    bus.close.assert_called_once()
    assert follower.bus is None


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


def test_bimanual_safe_home_runs_both_arms_concurrently():
    barrier = threading.Barrier(2, timeout=1.0)

    class FakeArm:
        def safe_home(self):
            barrier.wait()
            return True

    robot = object.__new__(BiRebotB601Follower)
    robot.left_arm = FakeArm()
    robot.right_arm = FakeArm()

    assert robot.safe_home()


def test_bimanual_disconnect_keeps_both_buses_when_one_home_fails():
    class FakeArm:
        def __init__(self, home_result):
            self.config = MagicMock(home_on_disconnect=True)
            self._emergency_disable_requested = False
            self.home_result = home_result
            self.disconnect = MagicMock()

        def safe_home(self):
            return self.home_result

    robot = object.__new__(BiRebotB601Follower)
    robot.left_arm = FakeArm(True)
    robot.right_arm = FakeArm(False)

    with pytest.raises(RuntimeError, match="Refusing to disconnect"):
        robot.disconnect()

    robot.left_arm.disconnect.assert_not_called()
    robot.right_arm.disconnect.assert_not_called()


def test_bimanual_connect_failure_cleans_up_left_arm():
    class FakeArm:
        def __init__(self, connect_error=None):
            self.bus = None
            self.connect_error = connect_error
            self._startup_home_action = {}
            self.config = MagicMock(home_action={})
            self.emergency_disable = MagicMock()
            self.disconnect = MagicMock(side_effect=self._disconnect)

        def connect(self, _calibrate):
            self.bus = MagicMock()
            if self.connect_error is not None:
                raise self.connect_error
            self._startup_home_action = {"shoulder_pan": 0.0}

        def _disconnect(self):
            self.bus = None

    robot = object.__new__(BiRebotB601Follower)
    robot.left_arm = FakeArm()
    robot.right_arm = FakeArm(RuntimeError("right arm failed"))

    with pytest.raises(RuntimeError, match="right arm failed"):
        robot.connect(calibrate=False)

    robot.left_arm.emergency_disable.assert_not_called()
    robot.left_arm.disconnect.assert_called_once()
    robot.right_arm.emergency_disable.assert_called_once()
    robot.right_arm.disconnect.assert_called_once()
