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

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics

from .core import TransitionKey
from .hil_processor import TELEOP_ACTION_KEY
from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry

logger = logging.getLogger(__name__)


@ProcessorStepRegistry.register("teleop_convert_joint_to_delta")
@dataclass
class TeleopConvertJointToDeltaStep(ComplementaryDataProcessorStep):
    """
    Converts teleop_action from joint space to delta x, y, z using forward kinematics.

    This step reads teleop_action (joint positions) from complementary_data, computes forward
    kinematics for both current and teleop joint positions, and converts to delta format
    expected by InterventionActionProcessorStep.

    The conversion process:
    1. Reads teleop_action (joint space: {"shoulder_pan.pos": float, ...}) from complementary_data
    2. Reads current joint positions from observation
    3. Computes forward kinematics for both current and teleop joint positions
    4. Computes delta = teleop_ee_pos - current_ee_pos
    5. Updates teleop_action in complementary_data to {"delta_x": float, "delta_y": float, "delta_z": float, "gripper": float}

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics computation.
        motor_names: List of motor/joint names (excluding gripper) to use for FK computation.
        use_gripper: Whether to include gripper in the output delta action.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    use_gripper: bool = False
    gripper_threshold: float = 5
    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Converts teleop_action from joint space to delta x, y, z format.

        Args:
            complementary_data: The incoming complementary data dictionary containing teleop_action.

        Returns:
            A new complementary data dictionary with teleop_action converted to delta format.
        """
        new_complementary_data = dict(complementary_data)

        # Get teleop_action from complementary_data
        teleop_action = new_complementary_data.get(TELEOP_ACTION_KEY)
        if teleop_action is None:
            # No teleop action available, return unchanged
            return new_complementary_data

        # Check if teleop_action is already in delta format (dict with delta_x, delta_y, delta_z)
        if isinstance(teleop_action, dict) and "delta_x" in teleop_action:
            # Already converted, return unchanged
            return new_complementary_data

        # Get current joint positions from observation
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            logger.warning("No observation available for teleop conversion. Returning unchanged.")
            return new_complementary_data

        # Extract joint positions from teleop_action and current observation
        try:
            # Get teleop joint positions
            teleop_joint_positions = []
            for motor_name in self.motor_names:
                joint_key = f"{motor_name}.pos"
                if isinstance(teleop_action, dict):
                    teleop_pos = teleop_action.get(joint_key)
                else:
                    # If teleop_action is not a dict, we can't process it
                    logger.warning(f"teleop_action is not a dict: {type(teleop_action)}. Returning unchanged.")
                    return new_complementary_data

                if teleop_pos is None:
                    logger.warning(f"Missing joint {joint_key} in teleop_action. Returning unchanged.")
                    return new_complementary_data

                teleop_joint_positions.append(float(teleop_pos))

            # Get current joint positions from observation
            current_joint_positions = []
            for motor_name in self.motor_names:
                joint_key = f"{motor_name}.pos"
                current_pos = observation.get(joint_key)
                if current_pos is None:
                    logger.warning(f"Missing joint {joint_key} in observation. Returning unchanged.")
                    return new_complementary_data

                # Handle both float and tensor values
                if hasattr(current_pos, "item"):
                    current_pos = current_pos.item()
                current_joint_positions.append(float(current_pos))

            # Convert to numpy arrays (assuming degrees)
            teleop_joints_deg = np.array(teleop_joint_positions, dtype=float)
            current_joints_deg = np.array(current_joint_positions, dtype=float)

            # Compute forward kinematics for both
            teleop_ee_transform = self.kinematics.forward_kinematics(teleop_joints_deg)
            current_ee_transform = self.kinematics.forward_kinematics(current_joints_deg)

            # Extract positions (x, y, z) from transformation matrices
            teleop_ee_pos = teleop_ee_transform[:3, 3]
            current_ee_pos = current_ee_transform[:3, 3]

            # Compute delta
            delta_pos = teleop_ee_pos - current_ee_pos

            # Extract gripper position if available
            gripper_value = 1.0  # Default gripper value

            
            def gripp(gripper_raw):
                if gripper_raw == self.gripper_threshold:
                    return 1
                elif gripper_raw > self.gripper_threshold:
                    return 2
                elif gripper_raw < self.gripper_threshold:
                    return 0
                else:
                    return 1.0  # default fallback
            if isinstance(teleop_action, dict):
                gripper_key = "gripper.pos"
                if gripper_key in teleop_action:
                    gripper_raw = float(teleop_action[gripper_key])
                    gripper_value = gripp(gripper_raw)
                elif "gripper" in teleop_action:
                    gripper_raw = float(teleop_action["gripper"])
                    gripper_value = gripp(gripper_raw)


            # Create delta action dict
            delta_action = {
                "delta_x": float(delta_pos[0]),
                "delta_y": float(delta_pos[1]),
                "delta_z": float(delta_pos[2]),
            }

            if self.use_gripper:
                delta_action["gripper"] = gripper_value

            # Update teleop_action in complementary_data
            new_complementary_data[TELEOP_ACTION_KEY] = delta_action

            logger.debug(
                f"Converted teleop_action from joint space to delta: "
                f"delta_x={delta_pos[0]:.4f}, delta_y={delta_pos[1]:.4f}, delta_z={delta_pos[2]:.4f}"
            )

        except Exception as e:
            logger.error(f"Error converting teleop_action to delta format: {e}. Returning unchanged.")
            return new_complementary_data

        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "motor_names": self.motor_names,
            "use_gripper": self.use_gripper,
        }

    def transform_features(
        self, features: dict[Any, dict[str, Any]]
    ) -> dict[Any, dict[str, Any]]:
        """
        No feature transformation needed - this step only modifies complementary_data.

        Args:
            features: The policy features dictionary.

        Returns:
            The unchanged features dictionary.
        """
        return features
