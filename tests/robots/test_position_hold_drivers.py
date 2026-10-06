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

from lerobot.robots.hope_jr import HopeJrArm, HopeJrArmConfig, HopeJrHand, HopeJrHandConfig
from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

pytest.importorskip("datasets")

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
