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

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ActionProcessorStep, PolicyAction, ProcessorStepRegistry, RobotAction, ProcessorStep, EnvTransition, TransitionKey
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


@ProcessorStepRegistry.register("direct_joint_to_policy_action")
@dataclass
class DirectJointToPolicyActionProcessorStep(ProcessorStep):
    """Convert direct joint control to policy action."""

    motor_names: list[str] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Check if we have direct joint control action
        robot_action = complementary_data.get("robot_action")
        if robot_action is not None:
            # Use direct joint control action
            action_values = []

            # Add arm joint positions
            for motor_name in self.motor_names:
                joint_key = f"{motor_name}.pos"
                if joint_key in robot_action:
                    action_values.append(robot_action[joint_key])
                else:
                    action_values.append(0.0)  # Default value

            # Add gripper if present
            if "gripper.pos" in robot_action:
                action_values.append(robot_action["gripper.pos"])
            elif "gripper" in robot_action:  # Fallback to non-.pos format
                action_values.append(robot_action["gripper"])

            # Convert to tensor
            action_tensor = torch.tensor(action_values, dtype=torch.float32)
            transition[TransitionKey.ACTION] = action_tensor

            # Store control mode in info for debugging
            info = transition.get(TransitionKey.INFO, {})
            info["control_mode"] = "direct_joint"
            transition[TransitionKey.INFO] = info

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step doesn't change the feature definitions
        return features
