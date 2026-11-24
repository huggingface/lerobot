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

import torch

from lerobot.configs.types import PipelineFeatureType
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)


def test_disabled_gripper_stays_instead_of_raising():
    """A 3-vector action must not raise when the gripper is turned off.

    ``MapTensorToDeltaActionDictStep(use_gripper=False)`` used to omit the
    gripper key, and the next step popped it unconditionally.
    """
    tensor_step = MapTensorToDeltaActionDictStep(use_gripper=False)
    delta_step = MapDeltaActionToRobotActionStep()

    transition = tensor_step(create_transition(action=torch.tensor([0.2, -0.1, 0.0])))
    transition = delta_step(transition)

    action = transition[TransitionKey.ACTION]
    assert action["gripper_vel"] == 1.0
    assert action["enabled"] is True
    assert action["target_x"] == torch.tensor(0.2).item()


def test_disabled_gripper_still_declares_the_gripper_feature():
    """The feature map keeps gripper so downstream steps can pop it."""
    features = {PipelineFeatureType.ACTION: {}}
    transformed = MapTensorToDeltaActionDictStep(use_gripper=False).transform_features(features)
    assert "gripper" in transformed[PipelineFeatureType.ACTION]
    assert "delta_x" in transformed[PipelineFeatureType.ACTION]


def test_enabled_gripper_reads_the_fourth_component():
    """With the gripper enabled, the fourth tensor value is the gripper command."""
    tensor_step = MapTensorToDeltaActionDictStep(use_gripper=True)
    delta_step = MapDeltaActionToRobotActionStep()

    transition = tensor_step(create_transition(action=torch.tensor([0.0, 0.0, 0.0, 0.0])))
    transition = delta_step(transition)

    assert transition[TransitionKey.ACTION]["gripper_vel"] == 0.0
    assert transition[TransitionKey.ACTION]["enabled"] is False
