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
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
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
