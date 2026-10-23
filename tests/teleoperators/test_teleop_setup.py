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

"""Mobile session setup keeps devices independent of mapping and robot types."""

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from lerobot.common.teleop_setup import make_mobile_teleop
from lerobot.processor import make_default_processors
from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep
from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig
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


def test_base_only_setup_returns_original_device_and_external_pipeline(keyboard):
    robot = SimpleNamespace(action_features={"joint.pos": float, "x.vel": float})
    teleop, pipeline = make_mobile_teleop(robot, keyboard)
    assert teleop is keyboard
    keyboard.event_queue.put(("w", True))
    observation = {"joint.pos": 10.0}
    raw = teleop.get_action(observation=observation)
    assert raw == {"w": None}
    assert pipeline((raw, observation)) == {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0, "joint.pos": 10.0}


def test_wired_lekiwi_arm_prefix_reaches_motor_bus(keyboard, tmp_path):
    robot = LeKiwi.__new__(LeKiwi)
    robot.config = LeKiwiConfig(cameras={})
    robot.bus = MagicMock()
    robot.cameras = {}
    arm = MockTeleop(
        MockTeleopConfig(n_motors=1, random_values=False, static_values=[12.0], calibration_dir=tmp_path)
    )
    arm.motors = ["shoulder_pan"]
    arm.connect()
    teleop, pipeline = make_mobile_teleop(robot, keyboard, arm, hold_uncommanded=False)
    observation = {}
    robot.send_action(pipeline((teleop.get_action(observation=observation), observation)))
    robot.bus.sync_write.assert_any_call("Goal_Position", {"arm_shoulder_pan": 12.0})
    teleop.disconnect()


def test_generic_robot_combines_unprefixed_nonoverlapping_devices(tmp_path):
    base, arm = (MockTeleop(MockTeleopConfig(n_motors=1, calibration_dir=tmp_path)) for _ in range(2))
    base.motors, arm.motors = ["base"], ["arm"]
    robot = SimpleNamespace(action_features={"base.pos": float, "arm.pos": float})
    teleop, pipeline = make_mobile_teleop(robot, base, arm)
    teleop.connect()
    try:
        assert set(pipeline((teleop.get_action(observation={}), {}))) == {"base.pos", "arm.pos"}
    finally:
        teleop.disconnect()


def test_setup_retains_custom_pipeline_hooks_and_does_not_mutate_it(keyboard):
    original = make_default_processors()[0]
    hook = MagicMock()
    original.after_step_hooks.append(hook)
    original_steps = list(original.steps)
    robot = SimpleNamespace(action_features={})
    _, pipeline = make_mobile_teleop(robot, keyboard, action_processor=original, hold_uncommanded=False)
    pipeline(({}, {}))
    assert len(original.steps) == len(original_steps)
    assert hook.call_count == len(pipeline.steps)


def test_fresh_setup_resets_mapping_state_without_storing_it_on_keyboard(keyboard):
    robot = SimpleNamespace(action_features={})
    mapping = KeyboardToBaseActionProcessorStep()
    _, first = make_mobile_teleop(robot, keyboard, keyboard_mapping=mapping)
    assert first(({"w": None, "+": None}, {}))["x.vel"] == 0.2
    _, second = make_mobile_teleop(robot, keyboard, keyboard_mapping=mapping)
    assert second(({"w": None}, {}))["x.vel"] == 0.1
    assert first(({"w": None}, {}))["x.vel"] == 0.2
    assert not hasattr(keyboard, "_base_action_processor")


def test_disconnected_keyboard_stops_velocities(keyboard):
    robot = SimpleNamespace(action_features={})
    teleop, pipeline = make_mobile_teleop(robot, keyboard)
    keyboard.event_queue.put(("w", True))
    assert pipeline((teleop.get_action(observation={}), {}))["x.vel"] == 0.1
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=False):
        assert pipeline((teleop.get_action(observation={}), {})) == {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }


def test_borrowed_devices_survive_composite_collection(tmp_path):
    base, arm = (MockTeleop(MockTeleopConfig(n_motors=1, calibration_dir=tmp_path)) for _ in range(2))
    base.connect()
    arm.connect()
    composite = CompositeTeleoperator([(base, "base_"), (arm, "arm_")])
    del composite
    gc.collect()
    assert base.is_connected and arm.is_connected
    base.disconnect()
    arm.disconnect()


def test_static_action_collisions_are_rejected(tmp_path):
    base, arm = (MockTeleop(MockTeleopConfig(n_motors=1, calibration_dir=tmp_path)) for _ in range(2))
    with pytest.raises(ValueError, match="overlapping keys"):
        CompositeTeleoperator([(base, ""), (arm, "")])
