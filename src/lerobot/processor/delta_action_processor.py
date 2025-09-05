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

from dataclasses import dataclass

from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION

from .pipeline import ActionProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    Maps a flat action tensor from a policy to a structured delta action dictionary.

    This step is typically used after a policy outputs a continuous action vector.
    It decomposes the vector into named components for delta movements of the
    end-effector (x, y, z) and optionally the gripper.

    Attributes:
        use_gripper: If True, assumes the 4th element of the tensor is the
                     gripper action.
    """

    use_gripper: bool = True

    def action(self, action: Tensor) -> dict:
        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            f"{ACTION}.delta_x": action[0],
            f"{ACTION}.delta_y": action[1],
            f"{ACTION}.delta_z": action[2],
        }
        if self.use_gripper:
            delta_action[f"{ACTION}.gripper"] = action[3]
        return delta_action

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features[f"{ACTION}.delta_x"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.delta_y"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.delta_z"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        if self.use_gripper:
            features[f"{ACTION}.gripper"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(ActionProcessorStep):
    """
    Maps delta actions from teleoperators to robot target actions for inverse kinematics.

    This step converts a dictionary of delta movements (e.g., from a gamepad)
    into a target action format that includes an "enabled" flag and target
    end-effector positions. It also handles scaling and noise filtering.

    Attributes:
        position_scale: A factor to scale the delta position inputs.
        rotation_scale: A factor to scale the delta rotation inputs (currently unused).
        noise_threshold: The magnitude below which delta inputs are considered noise
                         and do not trigger an "enabled" state.
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    rotation_scale: float = 0.0  # No rotation deltas for gamepad/keyboard
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: dict) -> dict:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop(f"{ACTION}.delta_x", 0.0)
        delta_y = action.pop(f"{ACTION}.delta_y", 0.0)
        delta_z = action.pop(f"{ACTION}.delta_z", 0.0)
        gripper = action.pop(f"{ACTION}.gripper", 1.0)  # Default to "stay" (1.0)

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
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # Update action with robot target format
        action = {
            f"{ACTION}.enabled": enabled,
            f"{ACTION}.target_x": scaled_delta_x,
            f"{ACTION}.target_y": scaled_delta_y,
            f"{ACTION}.target_z": scaled_delta_z,
            f"{ACTION}.target_wx": target_wx,
            f"{ACTION}.target_wy": target_wy,
            f"{ACTION}.target_wz": target_wz,
            f"{ACTION}.gripper": float(gripper),
        }

        return action

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transform features to match output format."""
        features.pop(f"{ACTION}.delta_x", None)
        features.pop(f"{ACTION}.delta_y", None)
        features.pop(f"{ACTION}.delta_z", None)
        features.pop(f"{ACTION}.gripper", None)

        features[f"{ACTION}.enabled"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.target_x"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.target_y"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.target_z"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.target_wx"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.target_wy"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.target_wz"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        features[f"{ACTION}.gripper"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features
