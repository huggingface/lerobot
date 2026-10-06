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
from unittest.mock import Mock

import pytest

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig  # noqa: E402
from lerobot.rollout.robot_wrapper import ThreadSafeRobot  # noqa: E402


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
