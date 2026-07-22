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

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
)

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_KEYS = {f"ee.{k}" for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]}


def _joint_bucket(feature_type: FeatureType) -> dict[str, PolicyFeature]:
    return {f"{n}.pos": PolicyFeature(type=feature_type, shape=(1,)) for n in MOTOR_NAMES}


def test_fk_action_step_types_ee_features_as_action():
    """The action FK step must emit its EE features in the ACTION bucket typed ACTION.

    Regression test: these were mistakenly typed FeatureType.STATE (copied from the
    observation variant), mis-classifying the converted end-effector actions as state.
    """
    step = ForwardKinematicsJointsToEEAction(kinematics=None, motor_names=MOTOR_NAMES)
    features = {
        PipelineFeatureType.ACTION: _joint_bucket(FeatureType.ACTION),
        PipelineFeatureType.OBSERVATION: {},
    }

    out = step.transform_features(features)[PipelineFeatureType.ACTION]

    # Joint positions consumed, EE pose produced.
    assert set(out) == EE_KEYS
    # Every produced action feature is typed ACTION, not STATE.
    assert all(feat.type == FeatureType.ACTION for feat in out.values())


def test_fk_observation_step_types_ee_features_as_state():
    """The observation FK step keeps its EE features in the OBSERVATION bucket typed STATE."""
    step = ForwardKinematicsJointsToEEObservation(kinematics=None, motor_names=MOTOR_NAMES)
    features = {
        PipelineFeatureType.ACTION: {},
        PipelineFeatureType.OBSERVATION: _joint_bucket(FeatureType.STATE),
    }

    out = step.transform_features(features)[PipelineFeatureType.OBSERVATION]

    assert set(out) == EE_KEYS
    assert all(feat.type == FeatureType.STATE for feat in out.values())
