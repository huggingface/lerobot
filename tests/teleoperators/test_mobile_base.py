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

"""Composition and legacy compatibility without a robot keyboard interface."""

from unittest.mock import PropertyMock, patch

import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.processor.keyboard_base_processor import (
    KeyboardToBaseActionProcessorStep,
    make_keyboard_base_processor,
)
from lerobot.robots.lekiwi import LeKiwiClientConfig
from lerobot.teleoperators.keyboard import (
    KeyboardBaseTeleop,
    KeyboardBaseTeleopConfig,
    KeyboardTeleop,
    KeyboardTeleopConfig,
)
from lerobot.teleoperators.mobile_base import MobileBaseTeleop, make_mobile_base_teleop

ARM = ("arm_shoulder.pos", "arm_gripper.pos")
BASE = ("x.vel", "y.vel", "theta.vel")


class FakeMobileRobot:
    name = "fake_mobile"
    action_features = dict.fromkeys(ARM + BASE, float)


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
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"):
        keyboard = KeyboardBaseTeleop(KeyboardBaseTeleopConfig())
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True):
        yield keyboard


@pytest.fixture
def raw_keyboard():
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"):
        keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True):
        yield keyboard


def observation(shoulder=10.0, gripper=50.0):
    return {"arm_shoulder.pos": shoulder, "arm_gripper.pos": gripper, **dict.fromkeys(BASE, 0.0)}


def test_keyboard_alone_drives_base_and_explicitly_holds_arms(keyboard):
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=keyboard, hold_uncommanded=True)
    keyboard.event_queue.put(("w", True))
    teleop.on_observation(observation())
    assert teleop.get_action() == {
        "arm_shoulder.pos": 10.0,
        "arm_gripper.pos": 50.0,
        "x.vel": 0.1,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    teleop.on_observation(observation(shoulder=7.0))
    assert teleop.get_action()["arm_shoulder.pos"] == 10.0


def test_holding_is_opt_in(keyboard):
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=keyboard)
    teleop.on_observation({})
    assert teleop.get_action() == dict.fromkeys(BASE, 0.0)


def test_arm_prefix_is_explicit(keyboard):
    arm = FakeTeleop({"shoulder.pos": 1.0, "gripper.pos": 2.0})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=keyboard, arm=arm, arm_action_prefix="arm_")
    teleop.on_observation(observation())
    assert teleop.get_action() == {
        "arm_shoulder.pos": 1.0,
        "arm_gripper.pos": 2.0,
        **dict.fromkeys(BASE, 0.0),
    }


def test_overlapping_actions_are_rejected():
    teleop = MobileBaseTeleop(FakeMobileRobot(), FakeTeleop({"x.vel": 0.1}), FakeTeleop({"x.vel": 0.2}))
    teleop.on_observation(observation())
    with pytest.raises(ValueError, match="overlapping.*x.vel"):
        teleop.get_action()


def test_disconnected_keyboard_stops_base(keyboard):
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=keyboard)
    keyboard.event_queue.put(("w", True))
    teleop.on_observation(observation())
    assert teleop.get_action()["x.vel"] == 0.1
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False):
        assert teleop.get_action() == dict.fromkeys(BASE, 0.0)


def test_base_actions_pass_through():
    action = {"x.vel": 0.5, "theta.vel": -0.1}
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=FakeTeleop(action))
    teleop.on_observation(observation())
    assert teleop.get_action() == action


def test_raw_keyboard_requires_a_processor(raw_keyboard):
    with pytest.raises(ValueError, match="base_action_processor"):
        MobileBaseTeleop(FakeMobileRobot(), base=raw_keyboard)
    with pytest.raises(ValueError, match="keyboard_base"):
        make_mobile_base_teleop(FakeMobileRobot(), base=raw_keyboard)


def test_raw_keyboard_can_use_a_processor_on_any_robot(raw_keyboard):
    processor = make_keyboard_base_processor(KeyboardToBaseActionProcessorStep())
    teleop = MobileBaseTeleop(FakeMobileRobot(), base=raw_keyboard, base_action_processor=processor)
    raw_keyboard.event_queue.put(("w", True))
    teleop.on_observation(observation())
    assert teleop.get_action()["x.vel"] == 0.1
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False):
        assert teleop.get_action() == dict.fromkeys(BASE, 0.0)


def test_legacy_lekiwi_config_is_mapped_without_robot_methods(raw_keyboard):
    robot = FakeMobileRobot()
    robot.config = LeKiwiClientConfig(remote_ip="unused", cameras={})
    robot.config.teleop_keys["forward"] = "i"
    # FakeMobileRobot has no keyboard mapping method or arm prefix attribute.
    arm = FakeTeleop({"shoulder.pos": 1.0})
    teleop = MobileBaseTeleop.from_list(robot, [arm, raw_keyboard])
    raw_keyboard.event_queue.put(("i", True))
    raw_keyboard.event_queue.put(("r", True))
    teleop.on_observation(observation())
    assert teleop.get_action() == {"arm_shoulder.pos": 1.0, "x.vel": 0.2, "y.vel": 0.0, "theta.vel": 0.0}


def test_holding_requires_measurements():
    teleop = MobileBaseTeleop(FakeMobileRobot(), FakeTeleop(dict.fromkeys(BASE, 0.0)), hold_uncommanded=True)
    teleop.on_observation({"arm_shoulder.pos": 0.0})
    with pytest.raises(ValueError, match="arm_gripper.pos"):
        teleop.get_action()


def test_from_list_accepts_keyboard_alone_or_with_arm(keyboard):
    robot = FakeMobileRobot()
    arm = FakeTeleop({"arm_shoulder.pos": 0.0, "arm_gripper.pos": 0.0})
    assert MobileBaseTeleop.from_list(robot, [keyboard]).arm is None
    assert MobileBaseTeleop.from_list(robot, [arm, keyboard]).arm is arm
    with pytest.raises(ValueError):
        MobileBaseTeleop.from_list(robot, [arm])


def test_observation_reaches_both_devices_before_actions():
    class ObservedTeleop(FakeTeleop):
        def get_action(self):
            assert self.observations
            return dict.fromkeys(self.action, self.observations[-1]["arm_shoulder.pos"])

    base = ObservedTeleop(dict.fromkeys(BASE, 0.0))
    arm = ObservedTeleop({"shoulder.pos": 0.0})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm, arm_action_prefix="arm_")
    for shoulder in (10.0, 7.0):
        obs = observation(shoulder=shoulder)
        teleop.on_observation(obs)
        action = teleop.get_action()
        assert action["arm_shoulder.pos"] == action["x.vel"] == shoulder
        assert base.observations[-1] is arm.observations[-1] is obs
    assert len(base.observations) == len(arm.observations) == 2


def test_mobile_action_requires_observation():
    teleop = MobileBaseTeleop(FakeMobileRobot(), FakeTeleop({"x.vel": 0.0}))
    with pytest.raises(RuntimeError, match="on_observation"):
        teleop.get_action()


def test_failed_base_connection_releases_arm():
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


def test_disconnect_releases_arm_even_if_base_cleanup_fails():
    base, arm = FakeTeleop({}), FakeTeleop({})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    with (
        patch.object(base, "disconnect", side_effect=RuntimeError("Base cleanup failed")),
        pytest.raises(RuntimeError, match="Base cleanup failed"),
    ):
        teleop.disconnect()
    assert arm.disconnect_calls == 1
    assert not arm.is_connected


def test_base_processor_receives_fresh_observations(raw_keyboard):
    class RelativeMapper(KeyboardToBaseActionProcessorStep):
        def action(self, action):
            return {"lift.pos": self.transition[TransitionKey.OBSERVATION]["lift.pos"] + 1.0}

    teleop = MobileBaseTeleop(
        FakeMobileRobot(),
        raw_keyboard,
        base_action_processor=make_keyboard_base_processor(RelativeMapper()),
    )
    for height in (10.0, 20.0):
        obs = {"lift.pos": height}
        teleop.on_observation(obs)
        assert teleop.get_action() == {"lift.pos": height + 1.0}
        assert obs == {"lift.pos": height}
