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

import pytest

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
    InverseKinematicsRLStep,
)

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_KEYS = {f"ee.{k}" for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]}


def _joint_bucket(feature_type: FeatureType) -> dict[str, PolicyFeature]:
    return {f"{n}.pos": PolicyFeature(type=feature_type, shape=(1,)) for n in MOTOR_NAMES}


@pytest.mark.parametrize(
    ("step_cls", "bucket", "feature_type"),
    [
        (ForwardKinematicsJointsToEEAction, PipelineFeatureType.ACTION, FeatureType.ACTION),
        (ForwardKinematicsJointsToEEObservation, PipelineFeatureType.OBSERVATION, FeatureType.STATE),
    ],
)
def test_fk_feature_schema(step_cls, bucket, feature_type):
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    features[bucket] = _joint_bucket(feature_type)
    out = step_cls(kinematics=None, motor_names=MOTOR_NAMES).transform_features(features)[bucket]
    assert set(out) == EE_KEYS
    assert {feature.type for feature in out.values()} == {feature_type}


EE_ACTION = dict.fromkeys(EE_KEYS, 0.0)


@pytest.mark.parametrize(
    ("step", "action"),
    [
        (
            EEReferenceAndDelta(kinematics=None, end_effector_step_sizes={}, motor_names=MOTOR_NAMES),
            dict(EE_ACTION),
        ),
        (InverseKinematicsEEToJoints(kinematics=None, motor_names=MOTOR_NAMES), dict(EE_ACTION)),
        (GripperVelocityToJoint(), {**EE_ACTION, "ee.gripper_vel": 0.0}),
        (InverseKinematicsRLStep(kinematics=None, motor_names=MOTOR_NAMES), dict(EE_ACTION)),
    ],
    ids=[
        "ee_reference_and_delta",
        "inverse_kinematics_ee_to_joints",
        "gripper_velocity_to_joint",
        "inverse_kinematics_rl_step",
    ],
)
def test_missing_observation_raises_value_error(step, action):
    """A transition without an observation must surface the documented ValueError.

    `RobotProcessorPipeline.process_action` builds its transition with
    `create_transition(action=...)`, which sets `TransitionKey.OBSERVATION` to None.
    These steps used to call `.copy()` on that before the None check, so the guard
    below them was unreachable and an AttributeError escaped instead.
    """
    transition = create_transition(action=action)

    with pytest.raises(ValueError, match="Joints observation"):
        step(transition)


class _RecordingKinematics:
    def __init__(self):
        self.q_init = None

    def inverse_kinematics(self, q_init, t_des, **kwargs):
        self.q_init = q_init
        return q_init


def test_joints_are_read_by_motor_name_not_observation_order():
    """Joint vectors must follow `motor_names`, whatever the order of the observation dict."""
    joints = {name: float(i) for i, name in enumerate(MOTOR_NAMES)}
    observation = {f"{name}.pos": joints[name] for name in reversed(MOTOR_NAMES)}

    kinematics = _RecordingKinematics()
    step = InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=MOTOR_NAMES)
    step(create_transition(observation=observation, action=dict(EE_ACTION)))
    assert list(kinematics.q_init) == [joints[name] for name in MOTOR_NAMES]

    step = GripperVelocityToJoint(speed_factor=1.0)
    out = step(create_transition(observation=observation, action={**EE_ACTION, "ee.gripper_vel": 0.0}))
    assert out[TransitionKey.ACTION]["ee.gripper_pos"] == joints["gripper"]
