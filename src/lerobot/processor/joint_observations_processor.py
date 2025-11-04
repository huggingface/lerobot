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
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STATE


@dataclass
@ProcessorStepRegistry.register("joint_velocity_processor")
class JointVelocityProcessorStep(ObservationProcessorStep):
    """
    Calculates and appends joint velocity information to the observation state.

    This step computes the velocity of each joint by calculating the finite
    difference between the current and the last observed joint positions. The
    resulting velocity vector is then concatenated to the original state vector.

    Attributes:
        dt: The time step (delta time) in seconds between observations, used for
            calculating velocity.
        last_joint_positions: Stores the joint positions from the previous step
                              to enable velocity calculation.
    """

    dt: float = 0.1

    last_joint_positions: torch.Tensor | None = None

    def observation(self, observation: dict) -> dict:
        """
        Computes joint velocities and adds them to the observation state.

        Args:
            observation: The input observation dictionary, expected to contain
                         an `observation.state` key with joint positions.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include joint velocities.

        Raises:
            ValueError: If `observation.state` is not found in the observation.
        """
        # Get current joint positions (assuming they're in observation.state)
        current_positions = observation.get(OBS_STATE)
        if current_positions is None:
            raise ValueError(f"{OBS_STATE} is not in observation")

        # Initialize last joint positions if not already set
        if self.last_joint_positions is None:
            self.last_joint_positions = current_positions.clone()
            joint_velocities = torch.zeros_like(current_positions)
        else:
            # Compute velocities
            joint_velocities = (current_positions - self.last_joint_positions) / self.dt

        self.last_joint_positions = current_positions.clone()

        # Extend observation with velocities
        extended_state = torch.cat([current_positions, joint_velocities], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the time step `dt`.
        """
        return {
            "dt": self.dt,
        }

    def reset(self) -> None:
        """Resets the internal state, clearing the last known joint positions."""
        self.last_joint_positions = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added velocities.

        This method doubles the size of the first dimension of the `observation.state`
        shape to account for the concatenation of position and velocity vectors.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Double the shape to account for positions + velocities
            new_shape = (original_feature.shape[0] * 2,) + original_feature.shape[1:]

            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features


@dataclass
@ProcessorStepRegistry.register("current_processor")
class MotorCurrentProcessorStep(ObservationProcessorStep):
    """
    Reads motor currents from a robot and appends them to the observation state.

    This step queries the robot's hardware interface to get the present current
    for each motor and concatenates this information to the existing state vector.

    Attributes:
        robot: An instance of a `lerobot` Robot class that provides access to
               the hardware bus.
    """

    robot: Robot | None = None

    def observation(self, observation: dict) -> dict:
        """
        Fetches motor currents and adds them to the observation state.

        Args:
            observation: The input observation dictionary.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include motor currents.

        Raises:
            ValueError: If the `robot` attribute has not been set.
        """
        # Get current values from robot state
        if self.robot is None:
            raise ValueError("Robot is not set")

        present_current_dict = self.robot.bus.sync_read("Present_Current")  # type: ignore[attr-defined]
        motor_currents = torch.tensor(
            [present_current_dict[name] for name in self.robot.bus.motors],  # type: ignore[attr-defined]
            dtype=torch.float32,
        ).unsqueeze(0)

        current_state = observation.get(OBS_STATE)
        if current_state is None:
            return observation

        extended_state = torch.cat([current_state, motor_currents], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added motor currents.

        This method increases the size of the first dimension of the `observation.state`
        shape by the number of motors in the robot.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and self.robot is not None:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Add motor current dimensions to the original state shape
            num_motors = 0
            if hasattr(self.robot, "bus") and hasattr(self.robot.bus, "motors"):  # type: ignore[attr-defined]
                num_motors = len(self.robot.bus.motors)  # type: ignore[attr-defined]

            if num_motors > 0:
                new_shape = (original_feature.shape[0] + num_motors,) + original_feature.shape[1:]
                features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                    type=original_feature.type, shape=new_shape
                )
        return features
