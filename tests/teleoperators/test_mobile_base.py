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

import gc
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from lerobot.common.control_utils import teleop_supports_feedback
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.keyboard_base_processor import (
    KeyboardToBaseActionProcessorStep,
    make_keyboard_base_processor,
)
from lerobot.robots.lekiwi import LeKiwi, LeKiwiClient, LeKiwiClientConfig, LeKiwiConfig
from lerobot.teleoperators import Teleoperator
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
    observation_features = action_features


class FakeTeleop:
    def __init__(self, action):
        self.action = action
        self.is_connected = True
        self.observations = []
        self.disconnect_calls = 0
        self.is_calibrated = False
        self.configure_calls = 0
        self.feedback_features = {}
        self.feedback = []

    @property
    def action_features(self):
        return dict.fromkeys(self.action, float)

    def get_action(self):
        return dict(self.action)

    def on_observation(self, observation):
        self.observations.append(observation)

    def connect(self, calibrate=True):
        self.is_connected = True
        if calibrate:
            self.calibrate()

    def calibrate(self):
        self.is_calibrated = True

    def configure(self):
        self.configure_calls += 1

    def send_feedback(self, feedback):
        self.feedback.append(feedback)

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
    robot = LeKiwiClient.__new__(LeKiwiClient)
    robot.config = LeKiwiClientConfig(remote_ip="unused", cameras={})
    robot.config.teleop_keys["forward"] = "i"
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


def test_robot_provided_defaults_work_without_a_known_config(raw_keyboard):
    robot = FakeMobileRobot()
    robot.arm_action_prefix = "arm_"
    robot.legacy_keyboard_bindings = dict(KeyboardToBaseActionProcessorStep().key_bindings)
    robot.legacy_keyboard_bindings["forward"] = "i"
    arm = FakeTeleop({"shoulder.pos": 12.0})
    teleop = make_mobile_base_teleop(robot, raw_keyboard, arm)
    raw_keyboard.event_queue.put(("i", True))
    teleop.on_observation(observation())
    assert teleop.get_action() == {"arm_shoulder.pos": 12.0, "x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0}


def test_wired_lekiwi_prefixes_arm_commands_for_the_motor_bus(keyboard, raw_keyboard):
    robot = LeKiwi.__new__(LeKiwi)
    robot.config = LeKiwiConfig(cameras={})
    robot.bus = MagicMock()
    robot.cameras = {}
    arm = FakeTeleop({"shoulder_pan.pos": 12.0, "gripper.pos": 50.0})
    teleop = make_mobile_base_teleop(robot, keyboard, arm)
    teleop.on_observation({})
    robot.send_action(teleop.get_action())
    robot.bus.sync_write.assert_any_call("Goal_Position", {"arm_shoulder_pan": 12.0, "arm_gripper": 50.0})
    with pytest.raises(ValueError, match="keyboard_base"):
        make_mobile_base_teleop(robot, raw_keyboard)


def test_legacy_mapping_restarts_when_bindings_change(raw_keyboard):
    keys = dict(KeyboardToBaseActionProcessorStep().key_bindings)
    processor = raw_keyboard.get_base_action_processor(keys)
    assert processor(({"w": None, "+": None}, {}))["x.vel"] == 0.2
    keys["forward"] = "i"
    updated = raw_keyboard.get_base_action_processor(keys)
    assert updated(({"w": None}, {}))["x.vel"] == 0.0
    assert updated(({"i": None}, {}))["x.vel"] == 0.1


def test_legacy_mapping_resets_on_keyboard_reconnect(raw_keyboard):
    processor = raw_keyboard.get_base_action_processor(KeyboardToBaseActionProcessorStep().key_bindings)
    assert processor(({"w": None, "+": None}, {}))["x.vel"] == 0.2
    with (
        patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False),
        patch("lerobot.teleoperators.keyboard.teleop_keyboard.PYNPUT_AVAILABLE", False),
    ):
        raw_keyboard.connect()
    assert processor(({"w": None}, {}))["x.vel"] == 0.1


@pytest.mark.parametrize("with_arm", [False, True])
@pytest.mark.parametrize("hold", [False, True])
def test_composite_implements_teleoperator_features(keyboard, with_arm, hold):
    arm = FakeTeleop({"shoulder.pos": 1.0, "gripper.pos": 2.0}) if with_arm else None
    teleop = MobileBaseTeleop(
        FakeMobileRobot(), keyboard, arm, arm_action_prefix="arm_", hold_uncommanded=hold
    )
    assert isinstance(teleop, Teleoperator)
    teleop.on_observation(observation())
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False):
        assert teleop.action_features == dict.fromkeys(ARM + BASE if with_arm or hold else BASE, float)
    assert set(teleop.get_action()) == set(teleop.action_features)


def test_raw_keyboard_features_describe_mapped_actions(raw_keyboard):
    teleop = MobileBaseTeleop(
        FakeMobileRobot(),
        raw_keyboard,
        base_action_processor=make_keyboard_base_processor(KeyboardToBaseActionProcessorStep()),
        hold_uncommanded=True,
    )
    assert teleop.action_features == FakeMobileRobot.action_features


def test_custom_processor_feature_mapping_is_respected(raw_keyboard):
    class LiftMapper(KeyboardToBaseActionProcessorStep):
        def action(self, action):
            return {"lift.pos": 1.0}

        def transform_features(self, features):
            return {
                **features,
                PipelineFeatureType.ACTION: {"lift.pos": PolicyFeature(FeatureType.ACTION, (1,))},
            }

    teleop = MobileBaseTeleop(
        FakeMobileRobot(), raw_keyboard, base_action_processor=make_keyboard_base_processor(LiftMapper())
    )
    assert teleop.action_features == {"lift.pos": float}


def test_composite_delegates_calibration_and_configuration():
    base, arm = FakeTeleop({}), FakeTeleop({})
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    assert not teleop.is_calibrated
    teleop.connect(calibrate=False)
    assert not base.is_calibrated and not arm.is_calibrated
    teleop.calibrate()
    assert teleop.is_calibrated
    teleop.configure()
    assert base.configure_calls == arm.configure_calls == 1
    base.disconnect()
    assert teleop.is_connected
    teleop.disconnect()
    assert not teleop.is_connected


def test_composite_routes_feedback_and_torque_to_children():
    base, arm = FakeTeleop({"x.vel": 0.0}), FakeTeleop({"shoulder.pos": 2.0})
    base.feedback_features = {"base_force": float}
    arm.feedback_features = {"shoulder.pos": float}
    for device in (base, arm):
        device.enable_torque = MagicMock()
        device.disable_torque = MagicMock()
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm, arm_action_prefix="arm_")
    assert teleop.feedback_features == {"base_force": float, "arm_shoulder.pos": float}
    assert teleop_supports_feedback(teleop)
    teleop.send_feedback({"base_force": 0.5, "arm_shoulder.pos": 12.0, "unused": 99.0})
    assert base.feedback == [{"base_force": 0.5}]
    assert arm.feedback == [{"shoulder.pos": 12.0}]
    teleop.enable_torque()
    teleop.disable_torque()
    for device in (base, arm):
        device.enable_torque.assert_called_once()
        device.disable_torque.assert_called_once()


def test_non_actuated_feedback_is_not_advertised_as_position_feedback():
    base = FakeTeleop({})
    base.feedback_features = {"wireless_remote": bytes}
    teleop = MobileBaseTeleop(FakeMobileRobot(), base)
    assert not teleop_supports_feedback(teleop)
    teleop.send_feedback({"wireless_remote": b"remote"})
    assert base.feedback == [{"wireless_remote": b"remote"}]
    with pytest.raises(NotImplementedError, match="torque"):
        teleop.enable_torque()


def test_composite_features_reject_collisions_before_use():
    base, arm = FakeTeleop({"x.vel": 0.0}), FakeTeleop({"x.vel": 1.0})
    base.feedback_features = arm.feedback_features = {"x.vel": float}
    teleop = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    for attr in ("action_features", "feedback_features"):
        with pytest.raises(ValueError, match="overlapping"):
            getattr(teleop, attr)
    with pytest.raises(ValueError, match="overlapping"):
        teleop.send_feedback({"x.vel": 0.0})
    assert base.feedback == arm.feedback == []


def test_borrowed_devices_survive_temporary_wrapper_collection(keyboard):
    arm = FakeTeleop({"arm_shoulder.pos": 1.0})
    wrapper = MobileBaseTeleop.from_list(FakeMobileRobot(), [arm, keyboard])
    with patch.object(keyboard, "disconnect") as disconnect:
        del wrapper
        gc.collect()
        disconnect.assert_not_called()
    assert arm.is_connected and arm.disconnect_calls == 0


def test_owned_devices_are_closed_after_partial_disconnect():
    base, arm = FakeTeleop({}), FakeTeleop({})
    wrapper = MobileBaseTeleop(FakeMobileRobot(), base, arm)
    wrapper.connect()
    base.disconnect()
    del wrapper
    gc.collect()
    assert arm.disconnect_calls == 1


def test_processor_passthrough_preserves_hardware_feature_types():
    class DiscreteBase(FakeTeleop):
        @property
        def action_features(self):
            return {"direction": int}

    class Passthrough(KeyboardToBaseActionProcessorStep):
        def action(self, action):
            return action

        def transform_features(self, features):
            return features

    teleop = MobileBaseTeleop(
        FakeMobileRobot(),
        DiscreteBase({"direction": 1}),
        base_action_processor=make_keyboard_base_processor(Passthrough()),
    )
    assert teleop.action_features == {"direction": int}
