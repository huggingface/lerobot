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

"""A mobile robot is driven from a base teleoperator, with or without a teleoperator for its arms."""

from unittest.mock import PropertyMock, patch

import pytest

from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.mobile_base import MobileBaseTeleop

ARM = ("arm_shoulder.pos", "arm_gripper.pos")
BASE = ("x.vel", "theta.vel")


class FakeMobileRobot:
    name = "fake_mobile"
    arm_action_prefix = "arm_"
    action_features = dict.fromkeys(ARM + BASE, float)

    def __init__(self):
        self.seen_observations = []

    def from_keyboard_to_base_action(self, pressed_keys, observation=None):
        self.seen_observations.append(observation)
        return {"x.vel": 0.2 if "w" in pressed_keys else 0.0, "theta.vel": 0.0}


class FakeArmOnlyRobot:
    name = "fake_arm"
    action_features = dict.fromkeys(ARM, float)


class FakeTeleop:
    def __init__(self, action):
        self.action = action
        self.is_connected = True
        self.observations = []
        self.disconnect_calls = 0

    @property
    def action_features(self):
        return dict.fromkeys(self.action, float)

    def get_action(self):
        return dict(self.action)

    def on_observation(self, observation):
        self.observations.append(observation)

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False
        self.disconnect_calls += 1


@pytest.fixture
def keyboard():
    # These tests inject events directly; no pynput installation or desktop backend is needed.
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"):
        keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True):
        yield keyboard


def observation(shoulder=10.0, gripper=50.0):
    return {"arm_shoulder.pos": shoulder, "arm_gripper.pos": gripper, "x.vel": 0.0, "theta.vel": 0.0}


def test_keyboard_alone_drives_the_base_and_holds_the_arms(keyboard):
    robot = FakeMobileRobot()
    teleop = MobileBaseTeleop(robot, base=keyboard)

    keyboard.event_queue.put(("w", True))
    teleop.on_observation(observation())
    action = teleop.get_action()
    assert action == {"arm_shoulder.pos": 10.0, "arm_gripper.pos": 50.0, "x.vel": 0.2, "theta.vel": 0.0}
    assert robot.seen_observations[-1] == observation()

    # The hold is latched on the first call: the arms do not follow a sagging measurement.
    teleop.on_observation(observation(shoulder=7.0))
    action = teleop.get_action()
    assert action["arm_shoulder.pos"] == 10.0


def test_arm_teleop_actions_get_the_robot_prefix(keyboard):
    arm = FakeTeleop({"shoulder.pos": 1.0, "gripper.pos": 2.0})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=keyboard, arm=arm)

    teleop.on_observation(observation())
    action = teleop.get_action()
    assert action == {"arm_shoulder.pos": 1.0, "arm_gripper.pos": 2.0, "x.vel": 0.0, "theta.vel": 0.0}


def test_a_disconnected_keyboard_stops_the_base(keyboard):
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=keyboard)
    keyboard.event_queue.put(("w", True))
    teleop.on_observation(observation())
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False):
        assert teleop.get_action()["x.vel"] == 0.0


def test_a_base_teleop_in_robot_action_space_is_used_as_is():
    base = FakeTeleop({"x.vel": 0.5, "theta.vel": -0.1})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=base)
    teleop.on_observation(observation())
    assert teleop.get_action() == {
        "arm_shoulder.pos": 10.0,
        "arm_gripper.pos": 50.0,
        "x.vel": 0.5,
        "theta.vel": -0.1,
    }


def test_a_plain_keyboard_needs_the_robot_hook(keyboard):
    with pytest.raises(TypeError, match="from_keyboard_to_base_action"):
        MobileBaseTeleop(FakeArmOnlyRobot(), base=keyboard)


def test_holding_needs_the_held_actions_in_the_observation():
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=FakeTeleop({"x.vel": 0.0, "theta.vel": 0.0}))
    teleop.on_observation({"arm_shoulder.pos": 0.0})
    with pytest.raises(ValueError, match="arm_gripper.pos"):
        teleop.get_action()


def test_from_list_accepts_keyboard_alone_or_with_one_arm(keyboard):
    robot = FakeMobileRobot()
    arm = FakeTeleop({"shoulder.pos": 0.0, "gripper.pos": 0.0})
    assert MobileBaseTeleop.from_list(robot, [keyboard]).arm is None
    assert MobileBaseTeleop.from_list(robot, [arm, keyboard]).arm is arm
    with pytest.raises(ValueError):
        MobileBaseTeleop.from_list(robot, [arm])


def test_observation_reaches_both_devices_before_actions():
    class ObservedTeleop(FakeTeleop):
        def get_action(self):
            assert self.observations
            return dict.fromkeys(self.action, self.observations[-1]["arm_shoulder.pos"])

    base = ObservedTeleop({"x.vel": 0.0, "theta.vel": 0.0})
    arm = ObservedTeleop({"shoulder.pos": 0.0, "gripper.pos": 0.0})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    for shoulder in (10.0, 7.0):
        obs = observation(shoulder=shoulder)
        teleop.on_observation(obs)
        action = teleop.get_action()
        assert action["arm_shoulder.pos"] == action["x.vel"] == shoulder
        assert base.observations[-1] is arm.observations[-1] is obs
    assert len(base.observations) == len(arm.observations) == 2


def test_mobile_action_requires_an_observation():
    teleop = MobileBaseTeleop(FakeMobileRobot(), FakeTeleop({"x.vel": 0.0}))
    with pytest.raises(RuntimeError, match="on_observation"):
        teleop.get_action()


def test_failed_base_connection_releases_the_arm():
    base, arm = FakeTeleop({}), FakeTeleop({})
    base.is_connected = arm.is_connected = False
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    with (
        patch.object(base, "connect", side_effect=RuntimeError("Base connection failed")),
        pytest.raises(RuntimeError, match="Base connection failed"),
    ):
        teleop.connect()
    assert arm.disconnect_calls == 1
    assert not arm.is_connected


def test_disconnect_releases_arm_even_when_base_cleanup_fails():
    base, arm = FakeTeleop({}), FakeTeleop({})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    with (
        patch.object(base, "disconnect", side_effect=RuntimeError("Base cleanup failed")),
        pytest.raises(RuntimeError, match="Base cleanup failed"),
    ):
        teleop.disconnect()
    assert arm.disconnect_calls == 1
    assert not arm.is_connected
