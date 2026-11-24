#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import MagicMock

import numpy as np

from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots.so_follower.robot_kinematic_processor import (
    AddIKSolutionStep,
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
    InverseKinematicsEEToJoints,
)

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_KEYS = {f"ee.{k}" for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]}


def _joints(gripper_pos: float = 7.0) -> dict[str, float]:
    joints = {f"{n}.pos": float(i) for i, n in enumerate(MOTOR_NAMES) if n != "gripper"}
    joints["gripper.pos"] = gripper_pos
    return joints


def _fk_kinematics(translation: tuple[float, float, float]) -> MagicMock:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = translation
    kinematics = MagicMock()
    kinematics.forward_kinematics.return_value = transform
    return kinematics


def test_forward_kinematics_observation_step():
    kinematics = _fk_kinematics(translation=(0.1, 0.2, 0.3))
    step = ForwardKinematicsJointsToEEObservation(kinematics=kinematics, motor_names=MOTOR_NAMES)

    transition = create_transition(observation=_joints(gripper_pos=42.0))
    result = step(transition)

    observation = result[TransitionKey.OBSERVATION]
    assert set(observation) == EE_KEYS
    assert observation["ee.x"] == 0.1
    assert observation["ee.y"] == 0.2
    assert observation["ee.z"] == 0.3
    assert observation["ee.wx"] == observation["ee.wy"] == observation["ee.wz"] == 0.0
    assert observation["ee.gripper_pos"] == 42.0
    assert result[TransitionKey.ACTION] is None

    (fk_input,) = kinematics.forward_kinematics.call_args.args
    np.testing.assert_allclose(fk_input, [0.0, 1.0, 2.0, 3.0, 4.0, 42.0])


def test_forward_kinematics_action_step():
    kinematics = _fk_kinematics(translation=(-0.5, 0.0, 0.25))
    step = ForwardKinematicsJointsToEEAction(kinematics=kinematics, motor_names=MOTOR_NAMES)

    transition = create_transition(action=_joints(gripper_pos=13.0))
    result = step(transition)

    action = result[TransitionKey.ACTION]
    assert set(action) == EE_KEYS
    assert action["ee.x"] == -0.5
    assert action["ee.y"] == 0.0
    assert action["ee.z"] == 0.25
    assert action["ee.gripper_pos"] == 13.0
    assert result[TransitionKey.OBSERVATION] is None


def test_inverse_kinematics_then_add_ik_solution():
    q_target = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    kinematics = MagicMock()
    kinematics.inverse_kinematics.return_value = q_target

    ik_step = InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=MOTOR_NAMES)
    add_step = AddIKSolutionStep(ik_step=ik_step)

    ee_action = {
        "ee.x": 0.1,
        "ee.y": 0.2,
        "ee.z": 0.3,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 55.0,
    }
    transition = create_transition(action=ee_action, observation=_joints())
    result = add_step(ik_step(transition))

    action = result[TransitionKey.ACTION]
    for i, name in enumerate(MOTOR_NAMES):
        expected = 55.0 if name == "gripper" else q_target[i]
        assert action[f"{name}.pos"] == expected
    assert not EE_KEYS & set(action)

    ik_solution = result[TransitionKey.COMPLEMENTARY_DATA]["IK_solution"]
    np.testing.assert_allclose(ik_solution, q_target)
    assert ik_solution is not ik_step.q_curr


def test_add_ik_solution_without_solution_leaves_data_unchanged():
    ik_step = InverseKinematicsEEToJoints(kinematics=None, motor_names=MOTOR_NAMES)
    add_step = AddIKSolutionStep(ik_step=ik_step)

    transition = create_transition(complementary_data={"foo": "bar"})
    result = add_step(transition)

    assert result[TransitionKey.COMPLEMENTARY_DATA] == {"foo": "bar"}


def test_reset_clears_ik_solution_state():
    q_target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    kinematics = MagicMock()
    kinematics.inverse_kinematics.return_value = q_target
    ik_step = InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=MOTOR_NAMES)

    ee_action = dict.fromkeys(EE_KEYS, 0.0)
    ik_step(create_transition(action=ee_action, observation=_joints()))
    assert ik_step.q_curr is not None

    ik_step.reset()
    assert ik_step.q_curr is None
