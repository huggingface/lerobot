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

import pytest

from lerobot.configs import FeatureType, PipelineFeatureType
from lerobot.processor import IdentityProcessorStep, MapKeyboardToSOJointPositionsStep
from lerobot.processor.converters import create_transition
from lerobot.processor.factory import SO_FOLLOWER_MOTOR_NAMES, make_default_processors
from lerobot.types import TransitionKey


def _make_observation(**overrides):
    observation = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 10.0,
        "elbow_flex.pos": -20.0,
        "wrist_flex.pos": 30.0,
        "wrist_roll.pos": -40.0,
        "gripper.pos": 50.0,
    }
    observation.update(overrides)
    return observation


def _run_step(step, action, observation=None):
    transition = create_transition(action=action, observation=observation or _make_observation())
    return step(transition)[TransitionKey.ACTION]


class _FakeConfig:
    def __init__(self, config_type: str):
        self.type = config_type


def test_no_keys_hold_current_joint_positions():
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES)

    assert _run_step(step, {}) == _make_observation()


@pytest.mark.parametrize(
    "key, motor, delta",
    [
        ("a", "shoulder_pan", -1.0),
        ("d", "shoulder_pan", 1.0),
        ("w", "shoulder_lift", 1.0),
        ("s", "shoulder_lift", -1.0),
        ("i", "elbow_flex", 1.0),
        ("k", "elbow_flex", -1.0),
        ("j", "wrist_flex", -1.0),
        ("l", "wrist_flex", 1.0),
        ("u", "wrist_roll", -1.0),
        ("o", "wrist_roll", 1.0),
    ],
)
def test_joint_keys_move_expected_motor(key, motor, delta):
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES, joint_step_size=3.0)
    observation = _make_observation()

    result = _run_step(step, {key: None}, observation)

    assert result[f"{motor}.pos"] == pytest.approx(observation[f"{motor}.pos"] + delta * 3.0)
    unchanged = set(observation) - {f"{motor}.pos"}
    for action_key in unchanged:
        assert result[action_key] == pytest.approx(observation[action_key])


def test_opposing_keys_cancel_each_other():
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES, joint_step_size=5.0)

    assert _run_step(step, {"a": None, "d": None}) == _make_observation()


def test_multiple_keys_apply_independent_joint_deltas():
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES, joint_step_size=2.0)
    observation = _make_observation()

    result = _run_step(step, {"w": None, "j": None, "o": None}, observation)

    assert result["shoulder_lift.pos"] == pytest.approx(observation["shoulder_lift.pos"] + 2.0)
    assert result["wrist_flex.pos"] == pytest.approx(observation["wrist_flex.pos"] - 2.0)
    assert result["wrist_roll.pos"] == pytest.approx(observation["wrist_roll.pos"] + 2.0)


def test_unknown_keys_are_ignored_but_output_stays_complete():
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES)

    assert _run_step(step, {"x": None}) == _make_observation()


def test_gripper_keys_are_clamped():
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES, gripper_step_size=10.0)

    assert _run_step(step, {"r": None}, _make_observation(**{"gripper.pos": 97.0}))[
        "gripper.pos"
    ] == pytest.approx(100.0)
    assert _run_step(step, {"f": None}, _make_observation(**{"gripper.pos": 3.0}))[
        "gripper.pos"
    ] == pytest.approx(0.0)


def test_missing_observation_key_raises_clear_error():
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES)
    observation = _make_observation()
    observation.pop("wrist_roll.pos")

    with pytest.raises(ValueError, match="Missing keys: \\['wrist_roll.pos'\\]"):
        _run_step(step, {"u": None}, observation)


def test_transform_features_outputs_so_action_features(policy_feature_factory):
    step = MapKeyboardToSOJointPositionsStep(motor_names=SO_FOLLOWER_MOTOR_NAMES)
    features = {
        PipelineFeatureType.ACTION: {"w": policy_feature_factory(FeatureType.ACTION, (1,))},
        PipelineFeatureType.OBSERVATION: {
            f"{motor}.pos": policy_feature_factory(FeatureType.STATE, (1,))
            for motor in SO_FOLLOWER_MOTOR_NAMES
        },
    }

    transformed = step.transform_features(features)

    assert set(transformed[PipelineFeatureType.ACTION]) == {
        f"{motor}.pos" for motor in SO_FOLLOWER_MOTOR_NAMES
    }


@pytest.mark.parametrize("robot_type", ["so100_follower", "so101_follower"])
def test_default_processors_route_keyboard_so_follower_to_keyboard_joint_processor(robot_type):
    teleop_processor, robot_processor, observation_processor = make_default_processors(
        _FakeConfig("keyboard"), _FakeConfig(robot_type)
    )

    assert isinstance(teleop_processor.steps[0], MapKeyboardToSOJointPositionsStep)
    assert isinstance(robot_processor.steps[0], IdentityProcessorStep)
    assert isinstance(observation_processor.steps[0], IdentityProcessorStep)


def test_default_processors_keep_identity_for_no_config_or_leader_config():
    no_config_processors = make_default_processors()
    leader_processors = make_default_processors(_FakeConfig("so101_leader"), _FakeConfig("so101_follower"))

    assert isinstance(no_config_processors[0].steps[0], IdentityProcessorStep)
    assert isinstance(leader_processors[0].steps[0], IdentityProcessorStep)
