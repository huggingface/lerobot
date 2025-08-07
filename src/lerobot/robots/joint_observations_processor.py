from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey


@dataclass
@ProcessorStepRegistry.register("joint_velocity_processor")
class JointVelocityProcessor:
    """Add joint velocity information to observations.

    Computes joint velocities by tracking changes in joint positions over time.
    """

    joint_velocity_limits: float = 100.0
    dt: float = 1.0 / 10

    last_joint_positions: torch.Tensor | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        # Get current joint positions (assuming they're in observation.state)
        current_positions = observation.get("observation.state")
        if current_positions is None:
            return transition

        # Initialize last joint positions if not already set
        if self.last_joint_positions is None:
            self.last_joint_positions = current_positions.clone()

        # Compute velocities
        joint_velocities = (current_positions - self.last_joint_positions) / self.dt
        self.last_joint_positions = current_positions.clone()

        # Extend observation with velocities
        extended_state = torch.cat([current_positions, joint_velocities], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation["observation.state"] = extended_state

        # Return new transition
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "joint_velocity_limits": self.joint_velocity_limits,
            "dt": self.dt,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        self.last_joint_positions = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("current_processor")
class MotorCurrentProcessor:
    """Add motor current information to observations."""

    env: gym.Env = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        # Get current values from complementary_data (where robot state would be stored)
        present_current_dict = self.env.unwrapped.robot.bus.sync_read("Present_Current")
        motor_currents = torch.tensor(
            [present_current_dict[name] for name in self.env.unwrapped.robot.bus.motors],
            dtype=torch.float32,
        ).unsqueeze(0)

        current_state = observation.get("observation.state")
        if current_state is None:
            return transition

        extended_state = torch.cat([current_state, motor_currents], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation["observation.state"] = extended_state

        # Return new transition
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
