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

from lerobot.robots.roarm_m3_follower import (
    JOINT_NAMES,
    RoarmM3Follower,
    RoarmM3FollowerConfig,
)
from lerobot.robots.roarm_m3_follower.async_worker import AsyncArmWorker

_MODULE = "lerobot.robots.roarm_m3_follower.roarm_m3_follower"


def _make_arm_mock(positions) -> MagicMock:
    arm = MagicMock(name="RoarmMock")
    arm.joints_angle_get.return_value = list(positions)
    return arm


@pytest.fixture
def follower():
    arm = _make_arm_mock([3.0, -98.0, 182.0, -2.0, 4.0, 100.0])
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.roarm", return_value=arm),
    ):
        robot = RoarmM3Follower(RoarmM3FollowerConfig(id="test", port="/dev/null"))
        robot.connect(calibrate=False)
        yield robot, arm
        if robot.is_connected:
            robot.disconnect()


def test_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RoarmM3Follower(RoarmM3FollowerConfig(id="test", port="/dev/null"))
    expected = {f"{n}.pos" for n in JOINT_NAMES}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected  # no cameras configured
    assert "gripper.pos" in expected


def test_connect_disconnect(follower):
    robot, _ = follower
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected


def test_get_observation_returns_degrees(follower):
    robot, _ = follower
    obs = robot.get_observation()
    assert set(obs) == {f"{n}.pos" for n in JOINT_NAMES}
    assert obs["shoulder.pos"] == pytest.approx(-98.0)
    assert obs["gripper.pos"] == pytest.approx(100.0)


def test_send_action_floor_roundtrip(follower):
    """Known degrees in -> known degrees out: floor quantization (73.6 -> 73)."""
    robot, _ = follower
    target = dict(zip([f"{n}.pos" for n in JOINT_NAMES], [73.6, -98.2, 182.7, -2.1, 4.9, 100.4], strict=True))
    sent = robot.send_action(target)
    assert sent["base.pos"] == 73.0
    assert sent["shoulder.pos"] == -98.0
    assert sent["elbow.pos"] == 182.0
    assert sent["gripper.pos"] == 100.0


def test_send_action_clamps_limits(follower):
    robot, _ = follower
    sent = robot.send_action({"base.pos": 999.0, "wrist_roll.pos": 130.0})
    assert sent["base.pos"] == 190.0  # ANGLES_MAX base
    assert sent["wrist_roll.pos"] == 90.0  # +-90 cable-protection limit


def test_force_control_gripper_double_write():
    """force_control_gripper=True adds the gripper-only T:121 write; off sends one write."""
    arm_on = _make_arm_mock([0, 0, 0, 0, 0, 100.0])
    worker_on = AsyncArmWorker(arm_on, "test", force_control_gripper=True)
    worker_on._execute_write([3, -98, 182, -2, 4, 73], 500, 50)
    arm_on.joints_angle_ctrl.assert_called_once()
    arm_on.joint_angle_ctrl.assert_called_once()  # the second, gripper-only write

    arm_off = _make_arm_mock([0, 0, 0, 0, 0, 100.0])
    worker_off = AsyncArmWorker(arm_off, "test", force_control_gripper=False)
    worker_off._execute_write([3, -98, 182, -2, 4, 73], 500, 50)
    arm_off.joints_angle_ctrl.assert_called_once()
    arm_off.joint_angle_ctrl.assert_not_called()


def test_disable_torque(follower):
    robot, arm = follower
    robot.disable_torque()
    arm.torque_set.assert_any_call(cmd=0)
