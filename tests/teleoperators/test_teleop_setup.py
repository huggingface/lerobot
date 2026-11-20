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
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from lerobot.common.teleop_setup import make_mobile_teleop
from lerobot.processor import make_default_processors
from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep
from lerobot.teleoperators.composite import CompositeTeleoperator
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


@pytest.fixture
def keyboard(tmp_path):
    with (
        patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"),
        patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True),
    ):
        yield KeyboardTeleop(KeyboardTeleopConfig(calibration_dir=tmp_path))


@pytest.fixture
def arm(tmp_path):
    arm = MockTeleop(
        MockTeleopConfig(n_motors=1, random_values=False, static_values=[12.0], calibration_dir=tmp_path)
    )
    arm.motors = ["shoulder"]
    arm.connect()
    yield arm
    if arm.is_connected:
        arm.disconnect()


def test_base_only_drives_the_base_and_holds_the_arm(keyboard):
    robot = SimpleNamespace(action_features={"joint.pos": float, "x.vel": float})
    teleop, pipeline = make_mobile_teleop(robot, keyboard)
    assert teleop is keyboard
    keyboard.event_queue.put(("w", True))
    assert pipeline((teleop.get_action(), {"joint.pos": 10.0})) == {
        "x.vel": 0.1,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "joint.pos": 10.0,
    }


def test_keyboard_and_prefixed_arm_are_combined_and_feedback_is_routed(keyboard, arm):
    robot = SimpleNamespace(arm_action_prefix="arm_", action_features={"arm_shoulder.pos": float})
    teleop, pipeline = make_mobile_teleop(robot, keyboard, arm)
    keyboard.event_queue.put(("w", True))
    assert pipeline((teleop.get_action(), {})) == {
        "arm_shoulder.pos": 12.0,
        "x.vel": 0.1,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    with patch.object(arm, "send_feedback") as send_feedback:
        teleop.send_feedback({"arm_shoulder.pos": 5.0})
    send_feedback.assert_called_once_with({"shoulder.pos": 5.0})


def test_overlapping_device_keys_are_rejected(keyboard, arm):
    teleop = CompositeTeleoperator([(keyboard, ""), (arm, "")])
    keyboard.event_queue.put(("shoulder.pos", True))
    with pytest.raises(ValueError, match="overlapping keys"):
        teleop.get_action()


def test_escaped_keyboard_stops_the_base_and_the_arm_is_still_cleaned_up(keyboard, arm):
    robot = SimpleNamespace(action_features={})
    teleop, pipeline = make_mobile_teleop(robot, keyboard, arm)
    keyboard.event_queue.put(("w", True))
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False):
        assert pipeline((teleop.get_action(), {}))["x.vel"] == 0.0
        assert teleop.is_connected
        teleop.disconnect()
    assert not arm.is_connected


def test_setup_copies_mapping_state_and_keeps_custom_pipeline_hooks(keyboard):
    robot = SimpleNamespace(action_features={})
    mapping = KeyboardToBaseActionProcessorStep()
    _, first = make_mobile_teleop(robot, keyboard, keyboard_mapping=mapping)
    assert first(({"w": None, "+": None}, {}))["x.vel"] == 0.2
    custom = make_default_processors()[0]
    hook = MagicMock()
    custom.after_step_hooks.append(hook)
    _, second = make_mobile_teleop(robot, keyboard, keyboard_mapping=mapping, action_processor=custom)
    assert second(({"w": None}, {}))["x.vel"] == 0.1
    assert hook.call_count == len(second.steps)
    assert len(custom.steps) == len(make_default_processors()[0].steps)
