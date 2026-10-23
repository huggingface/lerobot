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

"""Exercise the proposed pipeline with actual interfaces, without connecting hardware."""

import inspect
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest

from lerobot.common.control_utils import teleop_smooth_move_to, teleop_supports_feedback
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.hold_joints_processor import HoldUncommandedJointsProcessorStep
from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.composite import CompositeTeleoperator
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


class ObservedLeader(MockTeleop):
    def __init__(self, directory: Path) -> None:
        super().__init__(MockTeleopConfig(n_motors=1, calibration_dir=directory))
        self.observations: list[RobotObservation | None] = []
        self.feedback: list[RobotAction] = []
        self.torque = False

    @property
    def action_features(self) -> dict[str, type]:
        return {"joint.pos": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {"joint.pos": float}

    def get_action(self, observation: RobotObservation | None = None) -> RobotAction:
        self.observations.append(observation)
        if observation is None:
            return {"joint.pos": 0.0}
        return {"joint.pos": observation["arm_joint.pos"] + 1.0}

    def on_observation(self, observation: RobotObservation) -> None:
        raise AssertionError("The prototype must use get_action(observation=...), not the legacy hook")

    def send_feedback(self, feedback: RobotAction) -> None:
        self.feedback.append(feedback)

    def enable_torque(self) -> None:
        self.torque = True

    def disable_torque(self) -> None:
        self.torque = False


@pytest.fixture
def keyboard(tmp_path):
    with (
        patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"),
        patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True),
    ):
        yield KeyboardTeleop(KeyboardTeleopConfig(calibration_dir=tmp_path))


def make_pipeline(*steps):
    return RobotProcessorPipeline(
        steps=list(steps),
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def test_standard_keyboard_alone_uses_processors_without_a_wrapper(keyboard):
    pipeline = make_pipeline(
        KeyboardToBaseActionProcessorStep(), HoldUncommandedJointsProcessorStep(("arm_joint.pos",))
    )
    keyboard.event_queue.put(("w", True))
    first = {"arm_joint.pos": 10.0}
    action = pipeline((keyboard.get_action(observation=first), first))
    assert action == {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0, "arm_joint.pos": 10.0}
    assert first == {"arm_joint.pos": 10.0}
    later = {"arm_joint.pos": 8.0}
    assert pipeline((keyboard.get_action(observation=later), later))["arm_joint.pos"] == 10.0
    keyboard.event_queue.put(("w", False))
    assert pipeline((keyboard.get_action(observation=later), later))["x.vel"] == 0.0


def test_keyboard_and_leader_receive_fresh_observations_without_hook(keyboard, tmp_path):
    leader = ObservedLeader(tmp_path)
    composite = CompositeTeleoperator(
        {"": keyboard, "arm_": leader}, TeleoperatorConfig(calibration_dir=tmp_path)
    )
    pipeline = make_pipeline(KeyboardToBaseActionProcessorStep())
    keyboard.event_queue.put(("w", True))
    for position in (10.0, 20.0):
        observation = {"arm_joint.pos": position}
        raw = composite.get_action(observation=observation)
        assert raw == {"w": None, "arm_joint.pos": position + 1.0}
        result = pipeline((raw, observation))
        assert result == {"arm_joint.pos": position + 1.0, "x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0}
        assert leader.observations[-1] is observation
        assert raw == {"w": None, "arm_joint.pos": position + 1.0}


def test_speed_selection_survives_control_phase_boundaries(keyboard):
    pipeline = make_pipeline(KeyboardToBaseActionProcessorStep())
    keyboard.event_queue.put(("w", True))
    speeds = []
    for pressed in (True, False, True, True):
        keyboard.event_queue.put(("+", pressed))
        speeds.append(pipeline((keyboard.get_action(observation={}), {}))["x.vel"])
    assert speeds == [0.2, 0.2, 0.3, 0.3]
    pipeline.reset()
    keyboard.event_queue.put(("+", False))
    assert pipeline((keyboard.get_action(observation={}), {}))["x.vel"] == 0.1


def test_mapping_preserves_numeric_commands_even_if_their_names_are_bound_keys():
    pipeline = make_pipeline(KeyboardToBaseActionProcessorStep())
    assert pipeline(({"w": 2.0, "arm_joint.pos": 0.0}, {})) == {
        "w": 2.0,
        "arm_joint.pos": 0.0,
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }


def test_mapping_rejects_two_devices_commanding_the_same_base():
    with pytest.raises(ValueError, match="overwrite base commands"):
        make_pipeline(KeyboardToBaseActionProcessorStep())(({"w": None, "x.vel": 3.0}, {}))


def test_holding_uses_latched_positions_until_reset_and_respects_explicit_commands():
    pipeline = make_pipeline(HoldUncommandedJointsProcessorStep(("joint.pos",)))
    assert pipeline(({}, {"joint.pos": 10.0})) == {"joint.pos": 10.0}
    assert pipeline(({}, {"joint.pos": 8.0})) == {"joint.pos": 10.0}
    assert pipeline(({"joint.pos": 20.0}, {})) == {"joint.pos": 20.0}
    assert pipeline(({}, {"joint.pos": 19.0})) == {"joint.pos": 19.0}
    pipeline.reset()
    assert pipeline(({}, {"joint.pos": 30.0})) == {"joint.pos": 30.0}


@pytest.mark.parametrize("names", [("x.vel",), ("joint.pos", "joint.pos")])
def test_holding_rejects_invalid_channels(names):
    with pytest.raises(ValueError, match="position-action names"):
        HoldUncommandedJointsProcessorStep(names)


def test_holding_requires_measurements_and_does_not_partially_latch():
    pipeline = make_pipeline(HoldUncommandedJointsProcessorStep(("a.pos", "b.pos")))
    with pytest.raises(ValueError, match="b.pos"):
        pipeline(({}, {"a.pos": 10.0}))
    assert pipeline(({}, {"a.pos": 20.0, "b.pos": 30.0})) == {"a.pos": 20.0, "b.pos": 30.0}


def test_features_and_serialization_preserve_arm_and_held_channels(tmp_path):
    pipeline = make_pipeline(
        KeyboardToBaseActionProcessorStep(), HoldUncommandedJointsProcessorStep(("gripper.pos",))
    )
    feature = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
    features = pipeline.transform_features({PipelineFeatureType.ACTION: {"arm_joint.pos": feature}})
    assert set(features[PipelineFeatureType.ACTION]) == {
        "arm_joint.pos",
        "gripper.pos",
        "x.vel",
        "y.vel",
        "theta.vel",
    }
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")
    restored = RobotProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="processor.json",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    sample = ({"w": None, "arm_joint.pos": 4.0}, {"gripper.pos": 8.0})
    assert restored(sample) == pipeline(sample)


def test_feedback_routing_and_smooth_handover_are_separate_from_observation(keyboard, tmp_path, monkeypatch):
    leader = ObservedLeader(tmp_path)
    composite = CompositeTeleoperator(
        {"": keyboard, "arm_": leader}, TeleoperatorConfig(calibration_dir=tmp_path)
    )
    assert teleop_supports_feedback(composite)
    composite.send_feedback({"arm_joint.pos": 12.0})
    assert leader.feedback == [{"joint.pos": 12.0}]
    assert leader.observations == []
    keyboard.event_queue.put(("w", True))
    monkeypatch.setattr("lerobot.common.control_utils.time.sleep", lambda _: None)
    teleop_smooth_move_to(
        composite, {"arm_joint.pos": 20.0}, duration_s=0.1, fps=10, observation={"arm_joint.pos": 9.0}
    )
    assert leader.torque
    assert leader.feedback[-2:] == [{"joint.pos": 10.0}, {"joint.pos": 20.0}]
    assert leader.observations == [{"arm_joint.pos": 9.0}]
    composite.disable_torque()
    assert not leader.torque
    with pytest.raises(ValueError, match="No composite device accepts"):
        composite.send_feedback({"x.vel": 1.0})


def test_composition_supports_more_than_two_devices(tmp_path):
    devices = {prefix: ObservedLeader(tmp_path) for prefix in ("left_", "right_", "camera_")}
    composite = CompositeTeleoperator(devices, TeleoperatorConfig(calibration_dir=tmp_path))
    assert composite.get_action() == {"left_joint.pos": 0.0, "right_joint.pos": 0.0, "camera_joint.pos": 0.0}
    assert set(composite.action_features) == set(composite.get_action())
    composite.connect()
    assert all(device.is_connected for device in devices.values())
    composite.disconnect()
    assert not composite.is_connected


def test_runtime_collisions_are_rejected(keyboard, tmp_path):
    leader = ObservedLeader(tmp_path)
    composite = CompositeTeleoperator(
        {"": keyboard, "arm_": leader}, TeleoperatorConfig(calibration_dir=tmp_path)
    )
    keyboard.event_queue.put(("arm_joint.pos", True))
    with pytest.raises(ValueError, match="overlapping keys"):
        composite.get_action()


def test_partial_connection_failure_cleans_up_every_device(tmp_path, monkeypatch):
    left, right = ObservedLeader(tmp_path), ObservedLeader(tmp_path)
    composite = CompositeTeleoperator(
        {"left_": left, "right_": right}, TeleoperatorConfig(calibration_dir=tmp_path)
    )

    def fail(calibrate=True):
        right._is_connected = True
        raise RuntimeError("Connection failed")

    monkeypatch.setattr(right, "connect", fail)
    with pytest.raises(RuntimeError, match="Connection failed"):
        composite.connect()
    assert not left.is_connected and not right.is_connected


def test_failed_disconnect_still_cleans_up_other_children(tmp_path, monkeypatch):
    left, right = ObservedLeader(tmp_path), ObservedLeader(tmp_path)
    composite = CompositeTeleoperator(
        {"left_": left, "right_": right}, TeleoperatorConfig(calibration_dir=tmp_path)
    )
    composite.connect()

    def fail():
        raise RuntimeError("Disconnect failed")

    monkeypatch.setattr(right, "disconnect", fail)
    with pytest.raises(ExceptionGroup, match="Could not disconnect"):
        composite.disconnect()
    assert not left.is_connected and right.is_connected
    right._is_connected = False


def test_builtin_devices_accept_the_prototype_observation_argument():
    # Check source signatures without importing optional hardware SDKs.
    import ast

    root = Path(inspect.getfile(CompositeTeleoperator)).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_action":
                assert "observation" in [arg.arg for arg in node.args.args], path
