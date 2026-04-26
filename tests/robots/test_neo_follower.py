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

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.robots.nero_follower.config_nero_follower import NEOFollowerRobotConfig


@pytest.fixture
def neo_fixture(monkeypatch):
    fake_effector = MagicMock(name="AgxEffectorMock")
    fake_arm = MagicMock(name="AgxArmMock")
    fake_arm.is_connected.return_value = False
    fake_arm.enable.return_value = True
    fake_arm.get_joint_angles.return_value = SimpleNamespace(msg=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    fake_arm.init_effector.return_value = fake_effector
    fake_arm.OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="AGX_GRIPPER"))

    def _connect():
        fake_arm.is_connected.return_value = True

    def _disconnect():
        fake_arm.is_connected.return_value = False

    fake_arm.connect.side_effect = _connect
    fake_arm.disconnect.side_effect = _disconnect

    fake_pyagxarm = SimpleNamespace(
        create_agx_arm_config=MagicMock(return_value={"robot": "nero"}),
        AgxArmFactory=SimpleNamespace(create_arm=MagicMock(return_value=fake_arm)),
        ArmModel=SimpleNamespace(NERO="NERO"),
        NeroFW=SimpleNamespace(DEFAULT="default"),
    )
    monkeypatch.setitem(sys.modules, "pyAgxArm", fake_pyagxarm)

    module = importlib.import_module("lerobot.robots.nero_follower.nero_follower")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "make_cameras_from_configs", lambda _cfg: {})

    robot = module.NEOFollower(NEOFollowerRobotConfig(cameras={}))
    robot.connect()

    yield robot, fake_arm, fake_effector, module

    if robot.is_connected:
        robot.disconnect()


def test_connect_primes_joint_target_cache(neo_fixture):
    robot, fake_arm, _, module = neo_fixture

    assert robot._last_joint_targets == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    fake_arm.set_speed_percent.assert_called_with(robot.config.speed_percent)
    fake_arm.init_effector.assert_called_once_with(fake_arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    assert robot.is_connected
    assert module.NERO_JOINTS == [f"joint{i}" for i in range(1, 8)]


def test_send_action_full_joint_and_gripper(neo_fixture):
    robot, fake_arm, fake_effector, module = neo_fixture
    fake_arm.move_j.reset_mock()
    fake_effector.move_gripper_deg.reset_mock()

    action = {f"{joint}.pos": float(i) for i, joint in enumerate(module.NERO_JOINTS, start=1)}
    action["gripper.pos"] = 32.0

    returned = robot.send_action(action)

    fake_arm.move_j.assert_called_once_with([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    fake_effector.move_gripper_deg.assert_called_once_with(32.0)
    assert returned == action


def test_send_action_partial_joint_keeps_other_joints(neo_fixture):
    robot, fake_arm, _, _ = neo_fixture
    fake_arm.move_j.reset_mock()

    returned = robot.send_action({"joint1.pos": 1.5})

    fake_arm.move_j.assert_called_once_with([1.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert returned == {"joint1.pos": 1.5}


def test_send_action_ignores_raw_keyboard_chars(neo_fixture):
    robot, fake_arm, fake_effector, _ = neo_fixture
    fake_arm.move_j.reset_mock()
    fake_effector.move_gripper_deg.reset_mock()

    returned = robot.send_action({"w": None, "a": None})

    fake_arm.move_j.assert_not_called()
    fake_effector.move_gripper_deg.assert_not_called()
    assert returned == {}
