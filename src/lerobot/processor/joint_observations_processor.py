from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import (
    ObservationProcessor,
    ProcessorStepRegistry,
)
from lerobot.robots import Robot


@dataclass
@ProcessorStepRegistry.register("joint_velocity_processor")
class JointVelocityProcessor(ObservationProcessor):
    """Add joint velocity information to observations."""

    dt: float = 0.1

    last_joint_positions: torch.Tensor | None = None

    def observation(self, observation: dict) -> dict:
        # Get current joint positions (assuming they're in observation.state)
        current_positions = observation.get("observation.state")

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
        new_observation["observation.state"] = extended_state

        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
        }

    def reset(self) -> None:
        self.last_joint_positions = None

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        new_feature = features.copy()
        if "observation.state" in features:
            original_feature = features["observation.state"]
            # Double the shape to account for positions + velocities
            new_shape = (original_feature.shape[0] * 2,) + original_feature.shape[1:]

            new_feature["observation.state"] = PolicyFeature(type=original_feature.type, shape=new_shape)
        return new_feature


@dataclass
@ProcessorStepRegistry.register("current_processor")
class MotorCurrentProcessor(ObservationProcessor):
    """Add motor current information to observations."""

    robot: Robot | None = None

    def observation(self, observation: dict) -> dict:
        # Get current values from robot state
        if self.robot is None:
            raise ValueError("Robot is not set")

        present_current_dict = self.robot.bus.sync_read("Present_Current")  # type: ignore[attr-defined]
        motor_currents = torch.tensor(
            [present_current_dict[name] for name in self.robot.bus.motors],  # type: ignore[attr-defined]
            dtype=torch.float32,
        ).unsqueeze(0)

        current_state = observation.get("observation.state")
        if current_state is None:
            return observation

        extended_state = torch.cat([current_state, motor_currents], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation["observation.state"] = extended_state

        return new_observation

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        new_features = features.copy()
        if "observation.state" in features and self.robot is not None:
            original_feature = features["observation.state"]
            # Add motor current dimensions to the original state shape
            num_motors = 0
            if hasattr(self.robot, "bus") and hasattr(self.robot.bus, "motors"):  # type: ignore[attr-defined]
                num_motors = len(self.robot.bus.motors)  # type: ignore[attr-defined]

            if num_motors > 0:
                new_shape = (original_feature.shape[0] + num_motors,) + original_feature.shape[1:]
                new_features["observation.state"] = PolicyFeature(type=original_feature.type, shape=new_shape)
        return new_features
