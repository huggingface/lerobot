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

from dataclasses import asdict, dataclass
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ActionProcessorStep, PolicyAction, ProcessorStepRegistry, RobotAction
from lerobot.utils.constants import ACTION


@dataclass
@ProcessorStepRegistry.register("robot_action_to_policy_action_processor")
class RobotActionToPolicyActionProcessorStep(ActionProcessorStep):
    """Processor step to map a dictionary to a tensor action."""

    motor_names: list[str]

    def action(self, action: RobotAction) -> PolicyAction:
        if len(self.motor_names) != len(action):
            raise ValueError(f"Action must have {len(self.motor_names)} elements, got {len(action)}")
        return torch.tensor([action[f"{name}.pos"] for name in self.motor_names])

    def get_config(self) -> dict[str, Any]:
        return asdict(self)

    def transform_features(self, features):
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(self.motor_names),)
        )
        return features


@dataclass
@ProcessorStepRegistry.register("policy_action_to_robot_action_processor")
class PolicyActionToRobotActionProcessorStep(ActionProcessorStep):
    """Processor step to map a policy action to a robot action."""

    motor_names: list[str]

    def action(self, action: PolicyAction) -> RobotAction:
        if len(self.motor_names) != len(action):
            raise ValueError(f"Action must have {len(self.motor_names)} elements, got {len(action)}")
        return {f"{name}.pos": action[i] for i, name in enumerate(self.motor_names)}

    def get_config(self) -> dict[str, Any]:
        return asdict(self)

    def transform_features(self, features):
        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features
