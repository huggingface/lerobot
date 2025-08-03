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

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.types import PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import EnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    DeviceProcessor,
    ImageProcessor,
    RobotProcessor,
    StateProcessor,
)
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.robots.robot import Robot
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,  # noqa: F401
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.INFO)


def create_transition(
    observation=None, action=None, reward=0.0, done=False, truncated=False, info=None, complementary_data=None
):
    """Helper to create an EnvTransition dictionary."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray):
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        busy_wait(0.015)


class RobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.
    """

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] = None,
        reset_time_s: float = 5.0,
    ):
        """
        Initialize the RobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        Args:
            robot: The robot interface object used to connect and interact with the physical robot.
            display_cameras: If True, the robot's camera feeds will be displayed during execution.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.current_observation = None
        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._setup_spaces()

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Helper to convert a dictionary from bus.sync_read to an ordered numpy array."""
        obs_dict = self.robot.get_observation()
        joint_positions = np.array([obs_dict[name] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}
        self.current_observation = {"agent_pos": joint_positions, "pixels": images}

    def _setup_spaces(self):
        """
        Dynamically configure the observation and action spaces based on the robot's capabilities.

        Observation Space:
            - For keys with "image": A Box space with pixel values ranging from 0 to 255.
            - For non-image keys: A nested Dict space is created under 'observation.state' with a suitable range.

        Action Space:
            - The action space is defined as a Box space representing joint position commands. It is defined as relative (delta)
              or absolute, based on the configuration.
        """
        self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if "pixels" in self.current_observation:
            prefix = "observation.images"
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=self.current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in self.current_observation["pixels"]
            }

        observation_spaces["observation.state"] = gym.spaces.Box(
            low=0,
            high=10,
            shape=self.current_observation["agent_pos"].shape,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        Args:
            seed: A seed for random number generation to ensure reproducibility.
            options: Additional options to influence the reset behavior.

        Returns:
            A tuple containing:
                - observation (dict): The initial sensor observation.
                - info (dict): A dictionary with supplementary information, including the key "is_intervention".
        """
        # Reset the robot
        # self.robot.reset()
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, self.reset_pose)
            log_say("Reset the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        self.current_observation = None
        self._get_observation()
        return self.current_observation, {
            "is_intervention": False,
            "raw_joint_positions": self.current_observation["agent_pos"],
        }

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        self.robot.send_action(joint_targets_dict)

        self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            self.current_observation,
            reward,
            terminated,
            truncated,
            {"is_intervention": False, "raw_joint_positions": self.current_observation["agent_pos"]},
        )

    def render(self):
        """
        Render the current state of the environment by displaying the robot's camera feeds.
        """
        import cv2

        image_keys = [key for key in self.current_observation if "image" in key]

        for key in image_keys:
            cv2.imshow(key, cv2.cvtColor(self.current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()


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
            "fps": self.fps,
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
@ProcessorStepRegistry.register("image_crop_resize_processor")
class ImageCropResizeProcessor:
    """Crop and resize image observations."""

    crop_params_dict: dict[str, tuple[int, int, int, int]]
    resize_size: tuple[int, int] = (128, 128)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        if self.resize_size is None and not self.crop_params_dict:
            return transition

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            # Always resize
            image = F.resize(image, self.resize_size)
            image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessor:
    """Track episode time and enforce time limits."""

    max_episode_steps: int
    current_step: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        truncated = transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            return transition

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        new_transition = transition.copy()
        new_transition[TransitionKey.TRUNCATED] = truncated
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        self.current_step = 0

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessor:
    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Calculate gripper penalty and add to complementary data."""
        action = transition.get(TransitionKey.ACTION)
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)

        if complementary_data is None or action is None:
            return transition

        current_gripper_pos = complementary_data.get("raw_joint_positions", None)[-1]
        if current_gripper_pos is None:
            return transition

        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos
        gripper_action_normalized = gripper_action - 1.0

        # Calculate penalty boolean as in original
        gripper_penalty_bool = (gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5
        )

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Add penalty information to complementary data
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Create new complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data["discrete_penalty"] = gripper_penalty

        # Create new transition with updated complementary data
        new_transition = transition.copy()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        """Reset the processor state."""
        self.last_gripper_state = None

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessor:
    """Handle action intervention based on signals in the transition.

    This processor checks for intervention signals in the transition's complementary data
    and overrides agent actions when intervention is active.
    """

    use_gripper: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        # Get intervention signals from complementary data
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get("teleop_action", {})
        is_intervention = complementary_data.get("is_intervention", False)
        terminate_episode = complementary_data.get("terminate_episode", False)
        success = complementary_data.get("success", False)
        rerecord_episode = complementary_data.get("rerecord_episode", False)

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action:
            # Convert teleop_action dict to tensor format
            action_list = [
                teleop_action.get("delta_x", 0.0),
                teleop_action.get("delta_y", 0.0),
                teleop_action.get("delta_z", 0.0),
            ]
            if self.use_gripper:
                action_list.append(teleop_action.get("gripper", 1.0))

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode)
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info["is_intervention"] = is_intervention
        info["rerecord_episode"] = rerecord_episode
        info["next.success"] = success if terminate_episode else info.get("next.success", False)
        new_transition[TransitionKey.INFO] = info
        new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] = new_transition[
            TransitionKey.ACTION
        ]

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "use_gripper": self.use_gripper,
        }

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
    env: gym.Env = None  # Environment reference to get current state

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


def make_robot_env(cfg: EnvConfig) -> tuple[gym.Env, Any]:
    """
    Factory function to create a robot environment.

    This function builds a robot environment with all necessary wrappers
    based on the provided configuration.

    Args:
        cfg: Configuration object containing environment parameters.

    Returns:
        A tuple containing:
            - A gym environment with all necessary wrappers applied.
            - The teleoperation device for use in action processors.
    """
    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment
    env = RobotEnv(
        robot=robot,
        use_gripper=cfg.processor.use_gripper,
        display_cameras=cfg.processor.display_cameras,
        reset_pose=cfg.processor.fixed_reset_joint_positions,
    )

    return env, teleop_device


def make_processors(env, cfg):
    """
    Factory function to create environment and action processors.

    Args:
        env: The robot environment
        cfg: Configuration object containing processor parameters

    Returns:
        tuple: (env_processor, action_processor)
    """
    env_pipeline_steps = [
        ImageProcessor(),
        StateProcessor(),
        JointVelocityProcessor(dt=1.0 / cfg.fps),
        MotorCurrentProcessor(env=env),
        ImageCropResizeProcessor(
            crop_params_dict=cfg.processor.crop_params_dict, resize_size=cfg.processor.resize_size
        ),
        TimeLimitProcessor(max_episode_steps=int(cfg.processor.control_time_s * cfg.fps)),
    ]
    if cfg.processor.use_gripper:
        env_pipeline_steps.append(
            GripperPenaltyProcessor(
                penalty=cfg.processor.gripper_penalty, max_gripper_pos=cfg.processor.max_gripper_pos
            )
        )
    env_pipeline_steps.append(DeviceProcessor(device=cfg.device))

    env_processor = RobotProcessor(steps=env_pipeline_steps)

    action_pipeline_steps = [
        InterventionActionProcessor(
            use_gripper=cfg.processor.use_gripper,
        ),
        InverseKinematicsProcessor(
            urdf_path=cfg.processor.urdf_path,
            target_frame_name=cfg.processor.target_frame_name,
            end_effector_step_sizes=cfg.processor.end_effector_step_sizes,
            end_effector_bounds=cfg.processor.end_effector_bounds,
            max_gripper_pos=cfg.processor.max_gripper_pos,
            env=env,
        ),
    ]
    action_processor = RobotProcessor(steps=action_pipeline_steps)

    return env_processor, action_processor


def step_env_and_process_transition(
    env,
    transition,
    action,
    teleop_device,
    env_processor,
    action_processor,
):
    """
    Execute one step with processors handling intervention and observation processing.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute (will be replaced by neutral action in gym_manipulator mode)
        teleop_device: Teleoperator device for getting intervention signals
        env_processor: Environment processor for observations
        action_processor: Action processor for handling interventions

    Returns:
        tuple: (new_transition, terminate_episode)
    """
    # Get teleoperation action and events
    teleop_action = teleop_device.get_action()
    teleop_events = teleop_device.get_teleop_events()

    # Create action transition
    action_transition = dict(transition)
    action_transition[TransitionKey.ACTION] = action

    # Add teleoperation data to complementary data
    action_complementary_data = action_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).copy()
    action_complementary_data["teleop_action"] = teleop_action
    action_complementary_data.update(teleop_events)
    action_transition[TransitionKey.COMPLEMENTARY_DATA] = action_complementary_data

    # Process action through action pipeline (handles intervention)
    processed_action_transition = action_processor(action_transition)

    # Extract processed action and metadata
    processed_action = processed_action_transition[TransitionKey.ACTION]
    terminate_episode = processed_action_transition.get(TransitionKey.DONE, False)

    # Step environment with processed action
    obs, reward, terminated, truncated, info = env.step(processed_action)

    # Combine rewards from environment and action processor
    reward = reward + processed_action_transition[TransitionKey.REWARD]

    # Process new observation
    complementary_data = {
        "raw_joint_positions": info.pop("raw_joint_positions"),
        **processed_action_transition[TransitionKey.COMPLEMENTARY_DATA],
    }
    info.update(processed_action_transition[TransitionKey.INFO])

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated or terminate_episode,
        truncated=truncated,
        info=info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition, terminate_episode


def control_loop(env, env_processor, action_processor, teleop_device, cfg: EnvConfig):
    dt = 1.0 / cfg.fps

    print(f"Starting control loop at {cfg.fps} FPS")
    print("Controls:")
    print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
    obs, info = env.reset()
    complementary_data = {"raw_joint_positions": info.pop("raw_joint_positions")}
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(transition)

    if cfg.mode == "record":
        action_features = teleop_device.action_features
        features = {
            "action": action_features,
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }
        if cfg.processor.use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == "observation.state":
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.dataset_root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    while episode_idx < cfg.num_episodes:
        step_start_time = time.perf_counter()

        # Create a neutral action (no movement)
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if hasattr(env, "use_gripper") and env.use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

        # Use the new step function
        transition, terminate_episode = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            teleop_device=teleop_device,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if cfg.mode == "record":
            observations = {k: v.squeeze(0).cpu() for k, v in transition[TransitionKey.OBSERVATION].items()}
            frame = {
                **observations,
                "action": transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"].cpu(),
                "next.reward": np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                "next.done": np.array([terminated or truncated], dtype=bool),
            }
            if cfg.processor.use_gripper:
                frame["complementary_info.discrete_penalty"] = np.array(
                    [transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"]], dtype=np.float32
                )
            dataset.add_frame(frame, task=cfg.task)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated or terminate_episode:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            episode_step = 0
            episode_idx += 1

            if cfg.mode == "record":
                if transition[TransitionKey.INFO].get("rerecord_episode", False):
                    logging.info(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    logging.info(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            complementary_data = {"raw_joint_positions": info.pop("raw_joint_positions")}
            env_processor.reset()
            action_processor.reset()

            transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
            transition = env_processor(transition)

        # Maintain fps timing
        busy_wait(dt - (time.perf_counter() - step_start_time))

    if cfg.mode == "record" and cfg.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(env, action_processor, cfg):
    dataset = LeRobotDataset(
        cfg.repo_id, root=cfg.dataset_root, episodes=[cfg.episode], download_videos=False
    )
    dataset_actions = dataset.hf_dataset.select_columns(["action"])
    _, info = env.reset()

    for _, action in enumerate(dataset_actions):
        start_time = time.perf_counter()
        transition = create_transition(
            action=action["action"], complementary_data={"raw_joint_positions": info["raw_joint_positions"]}
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        busy_wait(1 / cfg.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: EnvConfig):
    env, teleop_device = make_robot_env(cfg)
    env_processor, action_processor = make_processors(env, cfg)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
