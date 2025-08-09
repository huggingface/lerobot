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

from dataclasses import dataclass, field

from torch import Tensor

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import ActionProcessor, ProcessorStepRegistry


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
    gripper_deadzone: float = 0.1  # Threshold for gripper activation
    _prev_enabled: bool = field(default=False, init=False, repr=False)

    def action(self, action: dict | Tensor | None) -> dict:
        if action is None:
            return {}

        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        if isinstance(action, dict):
            delta_x = action.pop("action.delta_x", 0.0)
            delta_y = action.pop("action.delta_y", 0.0)
            delta_z = action.pop("action.delta_z", 0.0)
            gripper = action.pop("action.gripper", 1.0)  # Default to "stay" (1.0)
        else:
            delta_x = action[0].item()
            delta_y = action[1].item()
            delta_z = action[2].item()
            gripper = action[3].item()

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

        self._prev_enabled = enabled
        return action

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transform features to match output format."""
        # Update features to reflect the new action format
        features.update(
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
        )
        return features
