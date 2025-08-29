# !/usr/bin/env python

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

from dataclasses import dataclass

from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ActionProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDict(ActionProcessorStep):
    """
    Map a tensor to a delta action dictionary.
    """

    def action(self, action: Tensor) -> dict:
        if isinstance(action, dict):
            return action
        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            "action.delta_x": action[0],
            "action.delta_y": action[1],
            "action.delta_z": action[2],
        }
        if action.shape[0] > 3:
            delta_action["action.gripper"] = action[3]
        return delta_action


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotAction(ActionProcessorStep):
    """
    Map delta actions from teleoperators (gamepad, keyboard) to robot target actions
    for use with inverse kinematics processors.

    Expected input ACTION keys:
    {
        "action.delta_x": float,
        "action.delta_y": float,
        "action.delta_z": float,
        "action.gripper": float (optional),
    }

    Output ACTION keys:
    {
        "action.enabled": bool,
        "action.displacement_x": float,
        "action.displacement_y": float,
        "action.displacement_z": float,
        "action.displacement_wx": float,
        "action.displacement_wy": float,
        "action.displacement_wz": float,
        "action.gripper": float,
    }
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    rotation_scale: float = 0.0  # No rotation deltas for gamepad/keyboard
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: dict) -> dict:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("action.delta_x", 0.0)
        delta_y = action.pop("action.delta_y", 0.0)
        delta_z = action.pop("action.delta_z", 0.0)
        gripper = action.pop("action.gripper", 1.0)  # Default to "stay" (1.0)

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement delta is detected
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # Use Euclidean norm for position
        enabled = position_magnitude > self.noise_threshold  # Small threshold to avoid noise

        # Scale the deltas appropriately
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # For gamepad/keyboard, we don't have rotation input, so set to 0
        # These could be extended in the future for more sophisticated teleoperators
        displacement_wx = 0.0
        displacement_wy = 0.0
        displacement_wz = 0.0

        # Update action with robot displacement format
        action = {
            "action.enabled": enabled,
            "action.displacement_x": scaled_delta_x,
            "action.displacement_y": scaled_delta_y,
            "action.displacement_z": scaled_delta_z,
            "action.displacement_wx": displacement_wx,
            "action.displacement_wy": displacement_wy,
            "action.displacement_wz": displacement_wz,
            "action.gripper": float(gripper),
        }

        return action

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transform features to match output format."""
        # Update features to reflect the new action format
        features.update(
            {
                "action.enabled": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.displacement_x": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.displacement_y": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.displacement_z": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.displacement_wx": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.displacement_wy": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.displacement_wz": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
            }
        )
        return features
