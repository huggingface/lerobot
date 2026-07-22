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

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)


def _empty_features() -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
    return {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}


def test_map_delta_to_robot_features_match_runtime_output_keys():
    """The declared action features must be exactly the keys emitted by action().

    Regression test: transform_features previously popped a non-existent
    "delta_gripper" key (leaving the real "gripper" feature dangling) and never
    declared the "gripper_vel" output.
    """
    # Build the upstream feature schema the way MapTensorToDeltaActionDictStep does.
    features = _empty_features()
    features = MapTensorToDeltaActionDictStep(use_gripper=True).transform_features(features)
    assert set(features[PipelineFeatureType.ACTION]) == {"delta_x", "delta_y", "delta_z", "gripper"}

    step = MapDeltaActionToRobotActionStep()
    features = step.transform_features(features)
    declared_keys = set(features[PipelineFeatureType.ACTION])

    # Compute the keys the runtime path actually produces.
    runtime_out = step.action({"delta_x": 0.1, "delta_y": 0.2, "delta_z": 0.3, "gripper": 1.0})

    # The feature schema must match the runtime output 1:1.
    assert declared_keys == set(runtime_out)
    # Consumed inputs are gone; the emitted gripper_vel is present.
    assert "gripper" not in declared_keys
    assert "delta_x" not in declared_keys
    assert "gripper_vel" in declared_keys


def test_map_delta_to_robot_features_removes_gripper_when_present():
    """The "gripper" feature registered upstream must be removed by this step."""
    features = _empty_features()
    features[PipelineFeatureType.ACTION] = {
        "gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        "some_other_feature": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }

    features = MapDeltaActionToRobotActionStep().transform_features(features)
    action_features = features[PipelineFeatureType.ACTION]

    assert "gripper" not in action_features
    # Unrelated features are left untouched.
    assert "some_other_feature" in action_features
