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

from unittest.mock import Mock

import pytest

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower  # noqa: E402
from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower  # noqa: E402
from lerobot.robots.bi_so_follower import BiSOFollower  # noqa: E402
from lerobot.robots.openarm_follower import OpenArmFollower  # noqa: E402
from lerobot.robots.rebot_b601_follower import RebotB601Follower  # noqa: E402
from lerobot.robots.so_follower import SOFollower  # noqa: E402
from lerobot.rollout.robot_wrapper import ThreadSafeRobot  # noqa: E402


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
