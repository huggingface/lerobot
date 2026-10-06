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

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower  # noqa: E402
from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower  # noqa: E402
from lerobot.robots.bi_so_follower import BiSOFollower  # noqa: E402
from lerobot.robots.hope_jr import HopeJrArm, HopeJrArmConfig, HopeJrHand, HopeJrHandConfig  # noqa: E402
from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig  # noqa: E402
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig  # noqa: E402
from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig  # noqa: E402
from lerobot.robots.rebot_b601_follower import RebotB601Follower  # noqa: E402
from lerobot.robots.so_follower import SOFollower  # noqa: E402
from lerobot.rollout.robot_wrapper import ThreadSafeRobot  # noqa: E402


@pytest.mark.parametrize(
    ("robot_cls", "config_cls", "bus_name", "clips"),
    [
        (KochFollower, KochFollowerConfig, "DynamixelMotorsBus", True),
        (HopeJrArm, HopeJrArmConfig, "FeetechMotorsBus", True),
        (HopeJrHand, HopeJrHandConfig, "FeetechMotorsBus", False),
    ],
)
def test_servo_hold_reuses_complete_measured_then_applied_targets(
    tmp_path, robot_cls, config_cls, bus_name, clips
):
    bus = MagicMock(is_connected=True)

    def make_bus(**kwargs):
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
        bus.read.return_value = 0.0
        return bus

    kwargs = {"max_relative_target": 2.0} if clips else {"side": "right"}
    with patch(f"{robot_cls.__module__}.{bus_name}", side_effect=make_bus):
        robot = robot_cls(config_cls(port="/dev/null", calibration_dir=tmp_path, **kwargs))
    wrapped = ThreadSafeRobot(robot)
    wrapped.configure_position_hold()
    measured = wrapped.get_observation()
    wrapped.hold()
    bus.sync_write.assert_called_with("Goal_Position", dict.fromkeys(bus.motors, 0.0))

    applied = wrapped.send_action(dict.fromkeys(robot.action_features, 10.0))
    expected = dict.fromkeys(robot.action_features, 2.0 if clips else 10.0)
    assert set(measured) == set(robot.action_features)
    assert applied == expected
    bus.sync_read.reset_mock()
    bus.read.reset_mock()
    wrapped.hold()
    wrapped.hold()
    bus.sync_write.assert_called_with(
        "Goal_Position", {key.removesuffix(".pos"): value for key, value in expected.items()}
    )
    # Holding may read motors for clipping, but never needs another observation/camera sample.
    bus.read.assert_not_called()
    assert wrapped.hardware_failure is None


def test_openarm_hold_preserves_clipped_absolute_targets_and_zero_feedforward(tmp_path):
    bus = MagicMock(is_connected=True)

    def make_bus(**kwargs):
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
        bus.sync_read_all_states.return_value = {
            name: {"position": 0.0, "velocity": 0.0, "torque": 0.0} for name in bus.motors
        }
        return bus

    with patch(f"{OpenArmFollower.__module__}.DamiaoMotorsBus", side_effect=make_bus):
        robot = OpenArmFollower(
            OpenArmFollowerConfig(port="can0", calibration_dir=tmp_path, max_relative_target=2.0)
        )
    wrapped = ThreadSafeRobot(robot)
    wrapped.configure_position_hold()
    measured = wrapped.get_observation()
    wrapped.hold()
    assert {key: command[2] for key, command in bus._mit_control_batch.call_args.args[0].items()} == {
        key.removesuffix(".pos"): value for key, value in measured.items()
    }
    applied = wrapped.send_action(dict.fromkeys(robot.action_features, 100.0))
    assert applied == {**dict.fromkeys(robot.action_features, 2.0), "gripper.pos": 0.0}
    commands = bus._mit_control_batch.call_args.args[0]
    assert all(command[3:] == (0.0, 0.0) for command in commands.values())
    assert all(command[0] > 0.0 for command in commands.values())
    bus.sync_read_all_states.reset_mock()
    wrapped.hold()
    wrapped.hold()
    bus._mit_control_batch.assert_called_with(commands)
    bus.sync_read_all_states.assert_not_called()


def test_openarm_mixed_action_schema_does_not_claim_position_hold(tmp_path):
    with patch(f"{OpenArmFollower.__module__}.DamiaoMotorsBus"):
        robot = OpenArmFollower(
            OpenArmFollowerConfig(port="can0", calibration_dir=tmp_path, use_velocity_and_torque=True)
        )
    assert not robot.supports_position_hold
    with pytest.raises(ValueError, match="no supported local position-hold contract"):
        ThreadSafeRobot(robot).configure_position_hold()


@pytest.fixture(
    params=[
        (BiSOFollower, SOFollower),
        (BiOpenArmFollower, OpenArmFollower),
        (BiRebotB601Follower, RebotB601Follower),
    ],
    ids=["so", "openarm", "rebot"],
)
def bimanual_robot(request, monkeypatch, tmp_path):
    wrapper_type, arm_type = request.param

    def make_robot(*, right_supports_hold=True):
        arms = []
        for supports_hold in (True, right_supports_hold):
            arm = Mock()
            arm.supports_position_hold = supports_hold
            arm.is_connected = True
            arm.cameras = {}
            arm._motors_ft = {"shoulder.pos": float, "gripper.pos": float}
            arm.send_action.side_effect = lambda action, *args: {
                key: min(10.0, max(-10.0, value)) for key, value in action.items()
            }
            arms.append(arm)
        monkeypatch.setattr(f"{wrapper_type.__module__}.{arm_type.__name__}", Mock(side_effect=arms))
        config = wrapper_type.config_class(
            calibration_dir=tmp_path,
            left_arm_config=arm_type.config_class(port="/dev/null0"),
            right_arm_config=arm_type.config_class(port="/dev/null1"),
        )
        return wrapper_type(config)

    return make_robot


def test_hold_replays_both_arms_applied_targets(bimanual_robot):
    robot = bimanual_robot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    applied = wrapper.send_action(
        {
            "left_shoulder.pos": 20.0,
            "left_gripper.pos": 2.0,
            "right_shoulder.pos": -20.0,
            "right_gripper.pos": 3.0,
        }
    )

    assert applied == {
        "left_shoulder.pos": 10.0,
        "left_gripper.pos": 2.0,
        "right_shoulder.pos": -10.0,
        "right_gripper.pos": 3.0,
    }
    wrapper.hold()
    assert robot.left_arm.send_action.call_args.args[0] == {"shoulder.pos": 10.0, "gripper.pos": 2.0}
    assert robot.right_arm.send_action.call_args.args[0] == {"shoulder.pos": -10.0, "gripper.pos": 3.0}


def test_hold_requires_both_arms_capabilities(bimanual_robot):
    robot = bimanual_robot(right_supports_hold=False)
    assert not robot.supports_position_hold
    with pytest.raises(ValueError, match="no supported local position-hold contract"):
        ThreadSafeRobot(robot).configure_position_hold()


@pytest.mark.parametrize("incomplete_reply", [False, True], ids=["write-failure", "missing-target"])
def test_partial_bimanual_application_is_a_hardware_failure(bimanual_robot, incomplete_reply):
    robot = bimanual_robot()
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    if incomplete_reply:
        robot.right_arm.send_action.side_effect = lambda action, *args: {"shoulder.pos": 1.0}
        message = "applied position target for every actuator"
    else:
        robot.right_arm.send_action.side_effect = RuntimeError("right arm write failed")
        message = "right arm write failed"

    with pytest.raises(RuntimeError, match=message):
        wrapper.send_action(dict.fromkeys(robot.action_features, 1.0))

    robot.left_arm.send_action.assert_called_once()
    assert wrapper.hardware_failure is not None
    assert message in wrapper.hardware_failure


@pytest.fixture
def make_reachy2(monkeypatch, tmp_path):
    monkeypatch.setattr("lerobot.robots.reachy2.robot_reachy2.require_package", lambda *args, **kwargs: None)

    def make_robot(**overrides):
        return Reachy2Robot(Reachy2RobotConfig(calibration_dir=tmp_path, **overrides))

    return make_robot


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"with_mobile_base": False, "use_external_commands": True},
        {"with_mobile_base": False, "max_relative_target": 5.0},
    ],
    ids=["mobile-base", "external-commands", "clipped-return"],
)
def test_reachy2_hold_rejects_unsupported_command_contracts(make_reachy2, config):
    robot = make_reachy2(**config)
    assert not robot.supports_position_hold
    with pytest.raises(ValueError, match="no supported local position-hold contract"):
        ThreadSafeRobot(robot).configure_position_hold()


def test_reachy2_position_only_hold_replays_submitted_joint_targets(make_reachy2):
    robot = make_reachy2(with_mobile_base=False)
    sdk = Mock()
    sdk.is_connected.return_value = True
    sdk.joints = {name: SimpleNamespace(present_position=0.0) for name in robot.joints_dict.values()}
    robot.reachy = sdk
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_position_hold()
    wrapper.get_observation()
    wrapper.hold()
    assert all(joint.goal_position == 0.0 for joint in sdk.joints.values())
    sdk.send_goal_positions.assert_called_once()
    sdk.send_goal_positions.reset_mock()
    target = {name: float(index) for index, name in enumerate(robot.action_features)}

    assert wrapper.send_action(target) == target
    assert {name: sdk.joints[joint].goal_position for name, joint in robot.joints_dict.items()} == target
    for joint in sdk.joints.values():
        joint.goal_position = -100.0
    wrapper.hold()

    assert {name: sdk.joints[joint].goal_position for name, joint in robot.joints_dict.items()} == target
    assert sdk.send_goal_positions.call_count == 2
    sdk.mobile_base.set_goal_speed.assert_not_called()
    robot.reachy = None
