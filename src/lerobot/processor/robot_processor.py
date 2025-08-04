from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.model.kinematics import RobotKinematics
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


@dataclass
@ProcessorStepRegistry.register("inverse_kinematics_processor")
class InverseKinematicsProcessor:
    """Convert end-effector space actions to joint space using inverse kinematics.

    This processor transforms delta commands in end-effector space (delta_x, delta_y, delta_z)
    to joint space commands using forward and inverse kinematics. It maintains the current
    end-effector pose and joint positions to compute the transformations.
    """

    urdf_path: str
    target_frame_name: str = "gripper_link"
    end_effector_step_sizes: dict[str, float] = field(default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0})
    end_effector_bounds: dict[str, list[float]] | None = None
    max_gripper_pos: float = 30.0

    # State tracking
    current_ee_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    current_joint_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    kinematics: RobotKinematics | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the kinematics module after dataclass initialization."""
        if self.urdf_path:
            self.kinematics = RobotKinematics(
                urdf_path=self.urdf_path,
                target_frame_name=self.target_frame_name,
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        action_np = action.detach().cpu().numpy().squeeze()

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        raw_joint_positions = complementary_data.get("raw_joint_positions")
        current_gripper_pos = raw_joint_positions[-1]
        if self.current_joint_pos is None:
            self.current_joint_pos = raw_joint_positions

        # Initialize end-effector position if not available
        if self.current_joint_pos is None:
            return transition  # Cannot proceed without joint positions

        # Calculate current end-effector position using forward kinematics
        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos)

        # Scale deltas by step sizes
        delta_ee = np.array(
            [
                action_np[0] * self.end_effector_step_sizes["x"],
                action_np[1] * self.end_effector_step_sizes["y"],
                action_np[2] * self.end_effector_step_sizes["z"],
            ],
            dtype=np.float32,
        )

        # Set desired end-effector position by adding delta
        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, :3] = self.current_ee_pos[:3, :3]  # Keep orientation

        # Add delta to position and clip to bounds
        desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + delta_ee
        if self.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # Compute inverse kinematics to get joint positions
        target_joint_values = self.kinematics.inverse_kinematics(self.current_joint_pos, desired_ee_pos)

        # Update current state
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = target_joint_values.copy()

        # Create new action with joint space commands
        gripper_action = current_gripper_pos
        if len(action_np) > 3:
            # Handle gripper command separately
            gripper_command = action_np[3]

            # Process gripper command (convert from [0,2] to delta) and discretize
            gripper_delta = np.round(gripper_command - 1.0).astype(int) * self.max_gripper_pos
            gripper_action = np.clip(current_gripper_pos + gripper_delta, 0, self.max_gripper_pos)

        # Combine joint positions and gripper
        target_joint_values[-1] = gripper_action

        converted_action = torch.from_numpy(target_joint_values).to(action.device).to(action.dtype)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = converted_action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "urdf_path": self.urdf_path,
            "target_frame_name": self.target_frame_name,
            "end_effector_step_sizes": self.end_effector_step_sizes,
            "end_effector_bounds": self.end_effector_bounds,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        """Reset the processor state."""
        self.current_ee_pos = None
        self.current_joint_pos = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
