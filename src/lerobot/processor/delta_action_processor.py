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
from lerobot.processor.pipeline import ActionProcessor, ProcessorStepRegistry


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDict(ActionProcessor):
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
class MapDeltaActionToRobotAction(ActionProcessor):
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
        "action.target_x": float,
        "action.target_y": float,
        "action.target_z": float,
        "action.target_wx": float,
        "action.target_wy": float,
        "action.target_wz": float,
        "action.gripper": float,
    }
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    rotation_scale: float = 0.0  # No rotation deltas for gamepad/keyboard

    def action(self, action: dict) -> dict:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("action.delta_x", 0.0)
        delta_y = action.pop("action.delta_y", 0.0)
        delta_z = action.pop("action.delta_z", 0.0)
        gripper = action.pop("action.gripper", 1.0)  # Default to "stay" (1.0)

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement delta is detected
        position_magnitude = abs(delta_x) + abs(delta_y) + abs(delta_z)
        enabled = position_magnitude > 1e-6  # Small threshold to avoid noise

        # Scale the deltas appropriately
        scaled_delta_x = float(delta_x) * self.position_scale
        scaled_delta_y = float(delta_y) * self.position_scale
        scaled_delta_z = float(delta_z) * self.position_scale

        # For gamepad/keyboard, we don't have rotation input, so set to 0
        # These could be extended in the future for more sophisticated teleoperators
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # Update action with robot target format
        action = {
            "action.enabled": enabled,
            "action.target_x": scaled_delta_x,
            "action.target_y": scaled_delta_y,
            "action.target_z": scaled_delta_z,
            "action.target_wx": target_wx,
            "action.target_wy": target_wy,
            "action.target_wz": target_wz,
            "action.gripper": float(gripper),
        }

        return action

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transform features to match output format."""
        # Update features to reflect the new action format
        features.update(
            {
                "action.enabled": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.target_x": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.target_y": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.target_z": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.target_wx": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.target_wy": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.target_wz": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
                "action.gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
            }
        )
        return features
