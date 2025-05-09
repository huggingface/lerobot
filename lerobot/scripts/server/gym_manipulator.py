#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import sys
import time
from collections import deque
from threading import Lock
from typing import Annotated, Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    is_headless,
    reset_follower_position,
)
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser
from lerobot.scripts.server.kinematics import RobotKinematics

logging.basicConfig(level=logging.INFO)
MAX_GRIPPER_COMMAND = 30


class TorchBox(gym.spaces.Box):
    """
    A version of gym.spaces.Box that handles PyTorch tensors.

    This class extends gym.spaces.Box to work with PyTorch tensors,
    providing compatibility between NumPy arrays and PyTorch tensors.
    """

    def __init__(
        self,
        low: float | Sequence[float] | np.ndarray,
        high: float | Sequence[float] | np.ndarray,
        shape: Sequence[int] | None = None,
        np_dtype: np.dtype | type = np.float32,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the PyTorch-compatible Box space.

        Args:
            low: Lower bounds of the space.
            high: Upper bounds of the space.
            shape: Shape of the space. If None, inferred from low and high.
            np_dtype: NumPy data type for internal storage.
            torch_dtype: PyTorch data type for tensor conversion.
            device: PyTorch device for returned tensors.
            seed: Random seed for sampling.
        """
        super().__init__(low, high, shape=shape, dtype=np_dtype, seed=seed)
        self.torch_dtype = torch_dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        """
        Sample a random point from the space.

        Returns:
            A PyTorch tensor within the space bounds.
        """
        arr = super().sample()
        return torch.as_tensor(arr, dtype=self.torch_dtype, device=self.device)

    def contains(self, x: torch.Tensor) -> bool:
        """
        Check if a tensor is within the space bounds.

        Args:
            x: The PyTorch tensor to check.

        Returns:
            Boolean indicating whether the tensor is within bounds.
        """
        # Move to CPU/numpy and cast to the internal dtype
        arr = x.detach().cpu().numpy().astype(self.dtype, copy=False)
        return super().contains(arr)

    def seed(self, seed: int | np.random.Generator | None = None):
        """
        Set the random seed for sampling.

        Args:
            seed: The random seed to use.

        Returns:
            List containing the seed.
        """
        super().seed(seed)
        return [seed]

    def __repr__(self) -> str:
        """
        Return a string representation of the space.

        Returns:
            Formatted string with space details.
        """
        return (
            f"TorchBox({self.low_repr}, {self.high_repr}, {self.shape}, "
            f"np={self.dtype.name}, torch={self.torch_dtype}, device={self.device})"
        )


class TorchActionWrapper(gym.Wrapper):
    """
    Wrapper that changes the action space to use PyTorch tensors.

    This wrapper modifies the action space to return PyTorch tensors when sampled
    and handles converting PyTorch actions to NumPy when stepping the environment.
    """

    def __init__(self, env: gym.Env, device: str):
        """
        Initialize the PyTorch action space wrapper.

        Args:
            env: The environment to wrap.
            device: The PyTorch device to use for tensor operations.
        """
        super().__init__(env)
        self.action_space = TorchBox(
            low=env.action_space.low,
            high=env.action_space.high,
            shape=env.action_space.shape,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def step(self, action: torch.Tensor):
        """
        Step the environment with a PyTorch tensor action.

        This method handles conversion from PyTorch tensors to NumPy arrays
        for compatibility with the underlying environment.

        Args:
            action: PyTorch tensor action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if action.dim() == 2:
            action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        return self.env.step(action)


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
        display_cameras: bool = False,
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

        self.current_joint_positions = self.robot.follower_arms["main"].read("Present_Position")

        self._setup_spaces()

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
        example_obs = self.robot.capture_observation()

        # Define observation spaces for images and other states.
        image_keys = [key for key in example_obs if "image" in key]
        observation_spaces = {
            key: gym.spaces.Box(low=0, high=255, shape=example_obs[key].shape, dtype=np.uint8)
            for key in image_keys
        }
        observation_spaces["observation.state"] = gym.spaces.Box(
            low=0,
            high=10,
            shape=example_obs["observation.state"].shape,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = len(self.robot.follower_arms["main"].read("Present_Position"))
        bounds = {}
        bounds["min"] = np.ones(action_dim) * -1000
        bounds["max"] = np.ones(action_dim) * 1000

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
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
        super().reset(seed=seed, options=options)

        # Capture the initial observation.
        observation = self.robot.capture_observation()

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        return observation, {"is_intervention": False}

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is processed and sent to the robot as joint position commands
        that may be either absolute values or deltas based on the environment configuration.

        Args:
            action: The commanded joint positions as a numpy array or torch tensor.

        Returns:
            A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including intervention status.
        """
        self.current_joint_positions = self.robot.follower_arms["main"].read("Present_Position")

        self.robot.send_action(torch.from_numpy(action))
        observation = self.robot.capture_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            {"is_intervention": False},
        )

    def render(self):
        """
        Render the current state of the environment by displaying the robot's camera feeds.
        """
        import cv2

        observation = self.robot.capture_observation()
        image_keys = [key for key in observation if "image" in key]

        for key in image_keys:
            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()


class AddJointVelocityToObservation(gym.ObservationWrapper):
    """
    Wrapper that adds joint velocity information to the observation.

    This wrapper computes joint velocities by tracking changes in joint positions over time,
    and extends the observation space to include these velocities.
    """

    def __init__(self, env, joint_velocity_limits=100.0, fps=30, num_dof=6):
        """
        Initialize the joint velocity wrapper.

        Args:
            env: The environment to wrap.
            joint_velocity_limits: Maximum expected joint velocity for space bounds.
            fps: Frames per second used to calculate velocity (position delta / time).
            num_dof: Number of degrees of freedom (joints) in the robot.
        """
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        self.last_joint_positions = np.zeros(num_dof)

        new_low = np.concatenate([old_low, np.ones(num_dof) * -joint_velocity_limits])
        new_high = np.concatenate([old_high, np.ones(num_dof) * joint_velocity_limits])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

        self.dt = 1.0 / fps

    def observation(self, observation):
        """
        Add joint velocity information to the observation.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation with joint velocities.
        """
        joint_velocities = (observation["observation.state"] - self.last_joint_positions) / self.dt
        self.last_joint_positions = observation["observation.state"].clone()
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], joint_velocities], dim=-1
        )
        return observation


class AddCurrentToObservation(gym.ObservationWrapper):
    """
    Wrapper that adds motor current information to the observation.

    This wrapper extends the observation space to include the current values
    from each motor, providing information about the forces being applied.
    """

    def __init__(self, env, max_current=500, num_dof=6):
        """
        Initialize the current observation wrapper.

        Args:
            env: The environment to wrap.
            max_current: Maximum expected current for space bounds.
            num_dof: Number of degrees of freedom (joints) in the robot.
        """
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        new_low = np.concatenate([old_low, np.zeros(num_dof)])
        new_high = np.concatenate([old_high, np.ones(num_dof) * max_current])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        """
        Add current information to the observation.

        Args:
            observation: The original observation from the environment.

        Returns:
            The modified observation with current values.
        """
        present_current = (
            self.unwrapped.robot.follower_arms["main"].read("Present_Current").astype(np.float32)
        )
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], torch.from_numpy(present_current)], dim=-1
        )
        return observation


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device="cuda"):
        """
        Wrapper to add reward prediction to the environment using a trained classifier.

        Args:
            env: The environment to wrap.
            reward_classifier: The reward classifier model.
            device: The device to run the model on.
        """
        self.env = env

        self.device = device

        self.reward_classifier = torch.compile(reward_classifier)
        self.reward_classifier.to(self.device)

    def step(self, action):
        """
        Execute a step and compute the reward using the classifier.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        observation, _, terminated, truncated, info = self.env.step(action)

        images = {}
        for key in observation:
            if "image" in key:
                images[key] = observation[key].to(self.device, non_blocking=(self.device == "cuda"))
                if images[key].dim() == 3:
                    images[key] = images[key].unsqueeze(0)

        start_time = time.perf_counter()
        with torch.inference_mode():
            success = (
                self.reward_classifier.predict_reward(images, threshold=0.7)
                if self.reward_classifier is not None
                else 0.0
            )
        info["Reward classifier frequency"] = 1 / (time.perf_counter() - start_time)

        reward = 0.0
        if success == 1.0:
            terminated = True
            reward = 1.0

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            The initial observation and info from the wrapped environment.
        """
        return self.env.reset(seed=seed, options=options)


class TimeLimitWrapper(gym.Wrapper):
    """
    Wrapper that adds a time limit to episodes and tracks execution time.

    This wrapper terminates episodes after a specified time has elapsed, providing
    better control over episode length.
    """

    def __init__(self, env, control_time_s, fps):
        """
        Initialize the time limit wrapper.

        Args:
            env: The environment to wrap.
            control_time_s: Maximum episode duration in seconds.
            fps: Frames per second for calculating the maximum number of steps.
        """
        self.env = env
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

        self.max_episode_steps = int(self.control_time_s * self.fps)

        self.current_step = 0

    def step(self, action):
        """
        Step the environment and track time elapsed.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()
        self.current_step += 1
        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step < self.fps:
            logging.debug(f"Current timestep exceeded expected fps {self.fps}")

        if self.current_step >= self.max_episode_steps:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment and time tracking.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            The initial observation and info from the wrapped environment.
        """
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    """
    Wrapper that crops and resizes image observations.

    This wrapper processes image observations to focus on relevant regions by
    cropping and then resizing to a standard size.
    """

    def __init__(
        self,
        env,
        crop_params_dict: Dict[str, Annotated[Tuple[int], 4]],
        resize_size=None,
    ):
        """
        Initialize the image crop and resize wrapper.

        Args:
            env: The environment to wrap.
            crop_params_dict: Dictionary mapping image observation keys to crop parameters
                             (top, left, height, width).
            resize_size: Target size for resized images (height, width). Defaults to (128, 128).
        """
        super().__init__(env)
        self.env = env
        self.crop_params_dict = crop_params_dict
        print(f"obs_keys , {self.env.observation_space}")
        print(f"crop params dict {crop_params_dict.keys()}")
        for key_crop in crop_params_dict:
            if key_crop not in self.env.observation_space.keys():  # noqa: SIM118
                raise ValueError(f"Key {key_crop} not in observation space")
        for key in crop_params_dict:
            new_shape = (3, resize_size[0], resize_size[1])
            self.observation_space[key] = gym.spaces.Box(low=0, high=255, shape=new_shape)

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        """
        Step the environment and process image observations.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) with processed images.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if obs[k].dim() >= 3:
                # Reshape to combine height and width dimensions for easier calculation
                batch_size = obs[k].size(0)
                channels = obs[k].size(1)
                flattened_spatial_dims = obs[k].view(batch_size, channels, -1)

                # Calculate standard deviation across spatial dimensions (H, W)
                # If any channel has std=0, all pixels in that channel have the same value
                # This is helpful if one camera mistakenly covered or the image is black
                std_per_channel = torch.std(flattened_spatial_dims, dim=2)
                if (std_per_channel <= 0.02).any():
                    logging.warning(
                        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                    )

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            # TODO (michel-aractingi): Bug in resize, it returns values outside [0, 1]
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment and process image observations.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) with processed images.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    """
    Wrapper that converts standard observations to LeRobot format.

    This wrapper processes observations to match the expected format for LeRobot,
    including normalizing image values and moving tensors to the specified device.
    """

    def __init__(self, env, device: str = "cpu"):
        """
        Initialize the LeRobot observation converter.

        Args:
            env: The environment to wrap.
            device: Target device for the observation tensors.
        """
        super().__init__(env)

        self.device = torch.device(device)

    def observation(self, observation):
        """
        Convert observations to LeRobot format.

        Args:
            observation: The original observation from the environment.

        Returns:
            The processed observation with normalized images and proper tensor formats.
        """
        for key in observation:
            observation[key] = observation[key].float()
            if "image" in key:
                observation[key] = observation[key].permute(2, 0, 1)
                observation[key] /= 255.0
        observation = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
        }

        return observation


class ResetWrapper(gym.Wrapper):
    """
    Wrapper that handles environment reset procedures.

    This wrapper provides additional functionality during environment reset,
    including the option to reset to a fixed pose or allow manual reset.
    """

    def __init__(
        self,
        env: RobotEnv,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
    ):
        """
        Initialize the reset wrapper.

        Args:
            env: The environment to wrap.
            reset_pose: Fixed joint positions to reset to. If None, manual reset is used.
            reset_time_s: Time in seconds to wait after reset or allowed for manual reset.
        """
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose
        self.robot = self.unwrapped.robot

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment with either fixed or manual reset procedure.

        If reset_pose is provided, the robot will move to that position.
        Otherwise, manual teleoperation control is allowed for reset_time_s seconds.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            The initial observation and info from the wrapped environment.
        """
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot.follower_arms["main"], self.reset_pose)
            log_say("Reset the environment done.", play_sounds=True)

            if len(self.robot.leader_arms) > 0:
                self.robot.leader_arms["main"].write("Torque_Enable", 1)
                log_say("Reset the leader robot.", play_sounds=True)
                reset_follower_position(self.robot.leader_arms["main"], self.reset_pose)
                log_say("Reset the leader robot done.", play_sounds=True)
        else:
            log_say(
                f"Manually reset the environment for {self.reset_time_s} seconds.",
                play_sounds=True,
            )
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                self.robot.teleop_step()

            log_say("Manual reset of the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        return super().reset(seed=seed, options=options)


class BatchCompatibleWrapper(gym.ObservationWrapper):
    """
    Wrapper that ensures observations are compatible with batch processing.

    This wrapper adds a batch dimension to observations that don't already have one,
    making them compatible with models that expect batched inputs.
    """

    def __init__(self, env):
        """
        Initialize the batch compatibility wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add batch dimensions to observations if needed.

        Args:
            observation: Dictionary of observation tensors.

        Returns:
            Dictionary of observation tensors with batch dimensions.
        """
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class GripperPenaltyWrapper(gym.RewardWrapper):
    """
    Wrapper that adds penalties for inefficient gripper commands.

    This wrapper modifies rewards to discourage excessive gripper movement
    or commands that attempt to move the gripper beyond its physical limits.
    """

    def __init__(self, env, penalty: float = -0.1):
        """
        Initialize the gripper penalty wrapper.

        Args:
            env: The environment to wrap.
            penalty: Negative reward value to apply for inefficient gripper actions.
        """
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_state = None

    def reward(self, reward, action):
        """
        Apply penalties to reward based on gripper actions.

        Args:
            reward: The original reward from the environment.
            action: The action that was taken.

        Returns:
            Modified reward with penalty applied if necessary.
        """
        gripper_state_normalized = self.last_gripper_state / MAX_GRIPPER_COMMAND

        action_normalized = action - 1.0  # action / MAX_GRIPPER_COMMAND

        gripper_penalty_bool = (gripper_state_normalized < 0.5 and action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and action_normalized < -0.5
        )

        return reward + self.penalty * int(gripper_penalty_bool)

    def step(self, action):
        """
        Step the environment and apply gripper penalties.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info) with penalty applied.
        """
        self.last_gripper_state = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        gripper_action = action[-1]
        obs, reward, terminated, truncated, info = self.env.step(action)
        gripper_penalty = self.reward(reward, gripper_action)

        info["discrete_penalty"] = gripper_penalty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and penalty tracking.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info with gripper penalty initialized.
        """
        self.last_gripper_state = None
        obs, info = super().reset(**kwargs)
        info["gripper_penalty"] = 0.0
        return obs, info


class GripperActionWrapper(gym.ActionWrapper):
    """
    Wrapper that processes gripper control commands.

    This wrapper quantizes and processes gripper commands, adding a sleep time between
    consecutive gripper actions to prevent rapid toggling.
    """

    def __init__(self, env, quantization_threshold: float = 0.2, gripper_sleep: float = 0.0):
        """
        Initialize the gripper action wrapper.

        Args:
            env: The environment to wrap.
            quantization_threshold: Threshold below which gripper commands are quantized to zero.
            gripper_sleep: Minimum time in seconds between consecutive gripper commands.
        """
        super().__init__(env)
        self.quantization_threshold = quantization_threshold
        self.gripper_sleep = gripper_sleep
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None

    def action(self, action):
        """
        Process gripper commands in the action.

        Args:
            action: The original action from the agent.

        Returns:
            Modified action with processed gripper command.
        """
        if self.gripper_sleep > 0.0:
            if (
                self.last_gripper_action is not None
                and time.perf_counter() - self.last_gripper_action_time < self.gripper_sleep
            ):
                action[-1] = self.last_gripper_action
            else:
                self.last_gripper_action_time = time.perf_counter()
                self.last_gripper_action = action[-1]

        gripper_command = action[-1]
        # Gripper actions are between 0, 2
        # we want to quantize them to -1, 0 or 1
        gripper_command = gripper_command - 1.0

        if self.quantization_threshold is not None:
            # Quantize gripper command to -1, 0 or 1
            gripper_command = (
                np.sign(gripper_command) if abs(gripper_command) > self.quantization_threshold else 0.0
            )
        gripper_command = gripper_command * MAX_GRIPPER_COMMAND
        gripper_state = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        gripper_action = np.clip(gripper_state + gripper_command, 0, MAX_GRIPPER_COMMAND)
        action[-1] = gripper_action.item()
        return action

    def reset(self, **kwargs):
        """
        Reset the gripper action tracking.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        obs, info = super().reset(**kwargs)
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None
        return obs, info


class EEActionWrapper(gym.ActionWrapper):
    """
    Wrapper that converts end-effector space actions to joint space actions.

    This wrapper takes actions defined in cartesian space (x, y, z, gripper) and
    converts them to joint space actions using inverse kinematics.
    """

    def __init__(self, env, ee_action_space_params=None, use_gripper=False):
        """
        Initialize the end-effector action wrapper.

        Args:
            env: The environment to wrap.
            ee_action_space_params: Parameters defining the end-effector action space.
            use_gripper: Whether to include gripper control in the action space.
        """
        super().__init__(env)
        self.ee_action_space_params = ee_action_space_params
        self.use_gripper = use_gripper

        # Initialize kinematics instance for the appropriate robot type
        robot_type = getattr(env.unwrapped.robot.config, "type", "so100")
        self.kinematics = RobotKinematics(robot_type)
        self.fk_function = self.kinematics.fk_gripper_tip

        action_space_bounds = np.array(
            [
                ee_action_space_params.x_step_size,
                ee_action_space_params.y_step_size,
                ee_action_space_params.z_step_size,
            ]
        )
        if self.use_gripper:
            # gripper actions open at 2.0, and closed at 0.0
            min_action_space_bounds = np.concatenate([-action_space_bounds, [0.0]])
            max_action_space_bounds = np.concatenate([action_space_bounds, [2.0]])
        else:
            min_action_space_bounds = -action_space_bounds
            max_action_space_bounds = action_space_bounds

        self.action_space = gym.spaces.Box(
            low=min_action_space_bounds,
            high=max_action_space_bounds,
            shape=(3 + int(self.use_gripper),),
            dtype=np.float32,
        )

        self.bounds = ee_action_space_params.bounds

    def action(self, action):
        """
        Convert end-effector action to joint space action.

        Args:
            action: End-effector action in cartesian space.

        Returns:
            Converted action in joint space.
        """
        desired_ee_pos = np.eye(4)

        if self.use_gripper:
            gripper_command = action[-1]
            action = action[:-1]

        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = self.fk_function(current_joint_pos)
        desired_ee_pos[:3, 3] = np.clip(
            current_ee_pos[:3, 3] + action,
            self.bounds["min"],
            self.bounds["max"],
        )
        target_joint_pos = self.kinematics.ik(
            current_joint_pos,
            desired_ee_pos,
            position_only=True,
            fk_func=self.fk_function,
        )
        if self.use_gripper:
            target_joint_pos[-1] = gripper_command

        return target_joint_pos


class EEObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that adds end-effector pose information to observations.

    This wrapper computes the end-effector pose using forward kinematics
    and adds it to the observation space.
    """

    def __init__(self, env, ee_pose_limits):
        """
        Initialize the end-effector observation wrapper.

        Args:
            env: The environment to wrap.
            ee_pose_limits: Dictionary with 'min' and 'max' keys containing limits for EE pose.
        """
        super().__init__(env)

        # Extend observation space to include end effector pose
        prev_space = self.observation_space["observation.state"]

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.concatenate([prev_space.low, ee_pose_limits["min"]]),
            high=np.concatenate([prev_space.high, ee_pose_limits["max"]]),
            shape=(prev_space.shape[0] + 3,),
            dtype=np.float32,
        )

        # Initialize kinematics instance for the appropriate robot type
        robot_type = getattr(env.unwrapped.robot.config, "robot_type", "so100")
        self.kinematics = RobotKinematics(robot_type)
        self.fk_function = self.kinematics.fk_gripper_tip

    def observation(self, observation):
        """
        Add end-effector pose to the observation.

        Args:
            observation: Original observation from the environment.

        Returns:
            Enhanced observation with end-effector pose information.
        """
        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = self.fk_function(current_joint_pos)
        observation["observation.state"] = torch.cat(
            [
                observation["observation.state"],
                torch.from_numpy(current_ee_pos[:3, 3]),
            ],
            dim=-1,
        )
        return observation


###########################################################
# Wrappers related to human intervention and input devices
###########################################################


class BaseLeaderControlWrapper(gym.Wrapper):
    """
    Base class for leader-follower robot control wrappers.

    This wrapper enables human intervention through a leader-follower robot setup,
    where the human can control a leader robot to guide the follower robot's movements.
    """

    def __init__(
        self, env, use_geared_leader_arm: bool = False, ee_action_space_params=None, use_gripper=False
    ):
        """
        Initialize the base leader control wrapper.

        Args:
            env: The environment to wrap.
            use_geared_leader_arm: Whether to use a geared leader arm setup.
            ee_action_space_params: Parameters defining the end-effector action space.
            use_gripper: Whether to include gripper control.
        """
        super().__init__(env)
        self.robot_leader = env.unwrapped.robot.leader_arms["main"]
        self.robot_follower = env.unwrapped.robot.follower_arms["main"]
        self.use_geared_leader_arm = use_geared_leader_arm
        self.ee_action_space_params = ee_action_space_params
        self.use_ee_action_space = ee_action_space_params is not None
        self.use_gripper: bool = use_gripper

        # Set up keyboard event tracking
        self._init_keyboard_events()
        self.event_lock = Lock()  # Thread-safe access to events

        # Initialize robot control
        robot_type = getattr(env.unwrapped.robot.config, "robot_type", "so100")
        self.kinematics = RobotKinematics(robot_type)
        self.prev_leader_ee = None
        self.prev_leader_pos = None
        self.leader_torque_enabled = True

        # Configure leader arm
        # NOTE: Lower the gains of leader arm for automatic take-over
        # With lower gains we can manually move the leader arm without risk of injury to ourselves or the robot
        # With higher gains, it would be dangerous and difficult to modify the leader's pose while torque is enabled
        # Default value for P_coeff is 32
        self.robot_leader.write("Torque_Enable", 1)
        self.robot_leader.write("P_Coefficient", 4)
        self.robot_leader.write("I_Coefficient", 0)
        self.robot_leader.write("D_Coefficient", 4)

        self._init_keyboard_listener()

    def _init_keyboard_events(self):
        """
        Initialize the keyboard events dictionary.

        This method sets up tracking for keyboard events used for intervention control.
        It should be overridden in subclasses to add additional events.
        """
        self.keyboard_events = {
            "episode_success": False,
            "episode_end": False,
            "rerecord_episode": False,
        }

    def _handle_key_press(self, key, keyboard):
        """
        Handle key press events.

        Args:
            key: The key that was pressed.
            keyboard: The keyboard module with key definitions.

        This method should be overridden in subclasses for additional key handling.
        """
        try:
            if key == keyboard.Key.esc:
                self.keyboard_events["episode_end"] = True
                return
            if key == keyboard.Key.left:
                self.keyboard_events["rerecord_episode"] = True
                return
            if hasattr(key, "char") and key.char == "s":
                logging.info("Key 's' pressed. Episode success triggered.")
                self.keyboard_events["episode_success"] = True
                return
        except Exception as e:
            logging.error(f"Error handling key press: {e}")

    def _init_keyboard_listener(self):
        """
        Initialize the keyboard listener for intervention control.

        This method sets up keyboard event handling if not in headless mode.
        """
        if is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
            )
            return
        try:
            from pynput import keyboard

            def on_press(key):
                with self.event_lock:
                    self._handle_key_press(key, keyboard)

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        except ImportError:
            logging.warning("Could not import pynput. Keyboard interface will not be available.")
            self.listener = None

    def _check_intervention(self):
        """
        Check if human intervention is needed.

        Returns:
            Boolean indicating whether intervention is needed.

        This method should be overridden in subclasses with specific intervention logic.
        """
        return False

    def _handle_intervention(self, action):
        """
        Process actions during intervention mode.

        Args:
            action: The original action from the agent.

        Returns:
            Tuple of (modified_action, intervention_action).
        """
        if self.leader_torque_enabled:
            self.robot_leader.write("Torque_Enable", 0)
            self.leader_torque_enabled = False

        leader_pos = self.robot_leader.read("Present_Position")
        follower_pos = self.robot_follower.read("Present_Position")

        # [:3, 3] Last column of the transformation matrix corresponds to the xyz translation
        leader_ee = self.kinematics.fk_gripper_tip(leader_pos)[:3, 3]
        follower_ee = self.kinematics.fk_gripper_tip(follower_pos)[:3, 3]

        if self.prev_leader_ee is None:
            self.prev_leader_ee = leader_ee

        # NOTE: Using the leader's position delta for teleoperation is too noisy
        # Instead, we move the follower to match the leader's absolute position,
        # and record the leader's position changes as the intervention action
        action = leader_ee - follower_ee
        action_intervention = leader_ee - self.prev_leader_ee
        self.prev_leader_ee = leader_ee

        if self.use_gripper:
            # Get gripper action delta based on leader pose
            leader_gripper = leader_pos[-1]
            follower_gripper = follower_pos[-1]
            gripper_delta = leader_gripper - follower_gripper

            # Normalize by max angle and quantize to {0,1,2}
            normalized_delta = gripper_delta / MAX_GRIPPER_COMMAND
            if normalized_delta > 0.3:
                gripper_action = 2
            elif normalized_delta < -0.3:
                gripper_action = 0
            else:
                gripper_action = 1

            action = np.append(action, gripper_action)
            action_intervention = np.append(action_intervention, gripper_delta)

        return action, action_intervention

    def _handle_leader_teleoperation(self):
        """
        Handle leader teleoperation in non-intervention mode.

        This method synchronizes the leader robot position with the follower.
        """
        if not self.leader_torque_enabled:
            self.robot_leader.write("Torque_Enable", 1)
            self.leader_torque_enabled = True

        follower_pos = self.robot_follower.read("Present_Position")
        self.robot_leader.write("Goal_Position", follower_pos)

    def step(self, action):
        """
        Execute a step with possible human intervention.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        is_intervention = self._check_intervention()
        action_intervention = None

        # NOTE:
        if is_intervention:
            action, action_intervention = self._handle_intervention(action)
        else:
            self._handle_leader_teleoperation()

        # NOTE:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add intervention info
        info["is_intervention"] = is_intervention
        info["action_intervention"] = action_intervention if is_intervention else None

        # Check for success or manual termination
        success = self.keyboard_events["episode_success"]
        terminated = terminated or self.keyboard_events["episode_end"] or success

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and intervention state.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        self.prev_leader_ee = None
        self.prev_leader_pos = None
        self.keyboard_events = dict.fromkeys(self.keyboard_events, False)
        return super().reset(**kwargs)

    def close(self):
        """
        Clean up resources, including stopping keyboard listener.

        Returns:
            Result of closing the wrapped environment.
        """
        if hasattr(self, "listener") and self.listener is not None:
            self.listener.stop()
        return self.env.close()


class GearedLeaderControlWrapper(BaseLeaderControlWrapper):
    """
    Wrapper that enables manual intervention via keyboard.

    This wrapper extends the BaseLeaderControlWrapper to allow explicit toggling
    of human intervention mode with keyboard controls.
    """

    def _init_keyboard_events(self):
        """
        Initialize keyboard events including human intervention flag.

        Extends the base class dictionary with an additional flag for tracking
        intervention state toggled by keyboard.
        """
        super()._init_keyboard_events()
        self.keyboard_events["human_intervention_step"] = False

    def _handle_key_press(self, key, keyboard):
        """
        Handle key presses including space for intervention toggle.

        Args:
            key: The key that was pressed.
            keyboard: The keyboard module with key definitions.

        Extends the base handler to respond to space key for toggling intervention.
        """
        super()._handle_key_press(key, keyboard)
        if key == keyboard.Key.space:
            if not self.keyboard_events["human_intervention_step"]:
                logging.info(
                    "Space key pressed. Human intervention required.\n"
                    "Place the leader in similar pose to the follower and press space again."
                )
                self.keyboard_events["human_intervention_step"] = True
                log_say("Human intervention step.", play_sounds=True)
            else:
                self.keyboard_events["human_intervention_step"] = False
                logging.info("Space key pressed for a second time.\nContinuing with policy actions.")
                log_say("Continuing with policy actions.", play_sounds=True)

    def _check_intervention(self):
        """
        Check if human intervention is active based on keyboard toggle.

        Returns:
            Boolean indicating whether intervention mode is active.
        """
        return self.keyboard_events["human_intervention_step"]


class GearedLeaderAutomaticControlWrapper(BaseLeaderControlWrapper):
    """
    Wrapper with automatic intervention based on error thresholds.

    This wrapper monitors the error between leader and follower positions
    and automatically triggers intervention when error exceeds thresholds.
    """

    def __init__(
        self,
        env,
        ee_action_space_params=None,
        use_gripper=False,
        intervention_threshold=1.7,
        release_threshold=0.01,
        queue_size=10,
    ):
        """
        Initialize the automatic intervention wrapper.

        Args:
            env: The environment to wrap.
            ee_action_space_params: Parameters defining the end-effector action space.
            use_gripper: Whether to include gripper control.
            intervention_threshold: Error threshold to trigger intervention.
            release_threshold: Error threshold to release intervention.
            queue_size: Number of error measurements to track for smoothing.
        """
        super().__init__(env, ee_action_space_params=ee_action_space_params, use_gripper=use_gripper)

        # Error tracking parameters
        self.intervention_threshold = intervention_threshold  # Threshold to trigger intervention
        self.release_threshold = release_threshold  # Threshold to release intervention
        self.queue_size = queue_size  # Number of error measurements to keep

        # Error tracking variables
        self.error_queue = deque(maxlen=self.queue_size)
        self.error_over_time_queue = deque(maxlen=self.queue_size)
        self.previous_error = 0.0
        self.is_intervention_active = False
        self.start_time = time.perf_counter()

    def _check_intervention(self):
        """
        Determine if intervention should occur based on leader-follower error.

        This method monitors the error rate between leader and follower positions
        and automatically triggers intervention when the error rate exceeds
        the intervention threshold, releasing when it falls below the release threshold.

        Returns:
            Boolean indicating whether intervention should be active.
        """
        # Skip intervention logic for the first few steps to collect data
        if time.perf_counter() - self.start_time < 1.0:  # Wait 1 second before enabling
            return False

        # Get current positions
        leader_positions = self.robot_leader.read("Present_Position")
        follower_positions = self.robot_follower.read("Present_Position")

        # Calculate error and error rate
        error = np.linalg.norm(leader_positions - follower_positions)
        error_over_time = np.abs(error - self.previous_error)

        # Add to queue for running average
        self.error_queue.append(error)
        self.error_over_time_queue.append(error_over_time)

        # Update previous error
        self.previous_error = error

        # Calculate averages if we have enough data
        if len(self.error_over_time_queue) >= self.queue_size:
            avg_error_over_time = np.mean(self.error_over_time_queue)

            # Debug info
            if self.is_intervention_active:
                logging.debug(f"Error rate during intervention: {avg_error_over_time:.4f}")

            # Determine if intervention should start or stop
            if not self.is_intervention_active and avg_error_over_time > self.intervention_threshold:
                # Transition to intervention mode
                self.is_intervention_active = True
                logging.info(f"Starting automatic intervention: error rate {avg_error_over_time:.4f}")

            elif self.is_intervention_active and avg_error_over_time < self.release_threshold:
                # End intervention mode
                self.is_intervention_active = False
                logging.info(f"Ending automatic intervention: error rate {avg_error_over_time:.4f}")

        return self.is_intervention_active

    def reset(self, **kwargs):
        """
        Reset error tracking on environment reset.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        self.error_queue.clear()
        self.error_over_time_queue.clear()
        self.previous_error = 0.0
        self.is_intervention_active = False
        self.start_time = time.perf_counter()
        return super().reset(**kwargs)


class GamepadControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
    ):
        """
        Initialize the gamepad controller wrapper.

        Args:
            env: The environment to wrap.
            x_step_size: Base movement step size for X axis in meters.
            y_step_size: Base movement step size for Y axis in meters.
            z_step_size: Base movement step size for Z axis in meters.
            use_gripper: Whether to include gripper control.
            auto_reset: Whether to auto reset the environment when episode ends.
            input_threshold: Minimum movement delta to consider as active input.
        """
        super().__init__(env)
        from lerobot.scripts.server.end_effector_control_utils import (
            GamepadController,
            GamepadControllerHID,
        )

        # use HidApi for macos
        if sys.platform == "darwin":
            self.controller = GamepadControllerHID(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        else:
            self.controller = GamepadController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

        logging.info("Gamepad control wrapper initialized")
        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick: Move in Z axis (up/down)")
        print("  X/Square button: End episode (FAILURE)")
        print("  Y/Triangle button: End episode (SUCCESS)")
        print("  B/Circle button: Exit program")

    def get_gamepad_action(
        self,
    ) -> Tuple[bool, np.ndarray, bool, bool, bool]:
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple containing:
            - is_active: Whether gamepad input is active
            - action: The action derived from gamepad input
            - terminate_episode: Whether episode termination was requested
            - success: Whether episode success was signaled
            - rerecord_episode: Whether episode rerecording was requested
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [2.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [0.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [1.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        Args:
            action: Original action from agent.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        # Only override the action if gamepad is active
        action = gamepad_action if is_intervention else action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        info["is_intervention"] = is_intervention
        info["action_intervention"] = action
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Clean up resources when environment closes.

        Returns:
            Result of closing the wrapped environment.
        """
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class GymHilDeviceWrapper(gym.Wrapper):
    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.device = device

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in obs:
            obs[k] = obs[k].to(self.device)
        if "action_intervention" in info:
            info["action_intervention"] = torch.from_numpy(info["action_intervention"]).to(self.device)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for k in obs:
            obs[k] = obs[k].to(self.device)
        if "action_intervention" in info:
            info["action_intervention"] = torch.from_numpy(info["action_intervention"]).to(self.device)
        return obs, info


class GymHilObservationProcessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        prev_space = self.observation_space
        new_space = {}

        for key in prev_space:
            if "pixels" in key:
                for k in prev_space["pixels"]:
                    new_space[f"observation.images.{k}"] = gym.spaces.Box(
                        0.0, 255.0, shape=(3, 128, 128), dtype=np.uint8
                    )

            if key == "agent_pos":
                new_space["observation.state"] = prev_space["agent_pos"]

        self.observation_space = gym.spaces.Dict(new_space)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return preprocess_observation(observation)


###########################################################
# Factory functions
###########################################################


def make_robot_env(cfg) -> gym.vector.VectorEnv:
    """
    Factory function to create a vectorized robot environment.

    This function builds a robot environment with all necessary wrappers
    based on the provided configuration.

    Args:
        cfg: Configuration object containing environment parameters.

    Returns:

        A vectorized gym environment with all necessary wrappers applied.
    """
    if cfg.type == "hil":
        import gym_hil  # noqa: F401

        # TODO (azouitine)
        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            step_size=cfg.wrapper.ee_action_space_params.x_step_size,
            use_gripper=cfg.wrapper.use_gripper,
            gripper_penalty=cfg.wrapper.gripper_penalty,
        )
        env = GymHilObservationProcessorWrapper(env=env)
        env = GymHilDeviceWrapper(env=env, device=cfg.device)
        env = BatchCompatibleWrapper(env=env)
        env = TorchActionWrapper(env=env, device=cfg.device)
        return env

    robot = make_robot_from_config(cfg.robot)
    # Create base environment
    env = RobotEnv(
        robot=robot,
        display_cameras=cfg.wrapper.display_cameras,
    )

    # Add observation and image processing
    if cfg.wrapper.add_joint_velocity_to_observation:
        env = AddJointVelocityToObservation(env=env, fps=cfg.fps)
    if cfg.wrapper.add_current_to_observation:
        env = AddCurrentToObservation(env=env)
    if cfg.wrapper.add_ee_pose_to_observation:
        env = EEObservationWrapper(env=env, ee_pose_limits=cfg.wrapper.ee_action_space_params.bounds)

    env = ConvertToLeRobotObservation(env=env, device=cfg.device)

    if cfg.wrapper.crop_params_dict is not None:
        env = ImageCropResizeWrapper(
            env=env,
            crop_params_dict=cfg.wrapper.crop_params_dict,
            resize_size=cfg.wrapper.resize_size,
        )

    # Add reward computation and control wrappers
    reward_classifier = init_reward_classifier(cfg)
    if reward_classifier is not None:
        env = RewardWrapper(env=env, reward_classifier=reward_classifier, device=cfg.device)
    env = TimeLimitWrapper(env=env, control_time_s=cfg.wrapper.control_time_s, fps=cfg.fps)
    if cfg.wrapper.use_gripper:
        env = GripperActionWrapper(env=env, quantization_threshold=cfg.wrapper.gripper_quantization_threshold)
        if cfg.wrapper.gripper_penalty is not None:
            env = GripperPenaltyWrapper(
                env=env,
                penalty=cfg.wrapper.gripper_penalty,
            )

    env = EEActionWrapper(
        env=env,
        ee_action_space_params=cfg.wrapper.ee_action_space_params,
        use_gripper=cfg.wrapper.use_gripper,
    )

    if cfg.wrapper.ee_action_space_params.control_mode == "gamepad":
        env = GamepadControlWrapper(
            env=env,
            x_step_size=cfg.wrapper.ee_action_space_params.x_step_size,
            y_step_size=cfg.wrapper.ee_action_space_params.y_step_size,
            z_step_size=cfg.wrapper.ee_action_space_params.z_step_size,
            use_gripper=cfg.wrapper.use_gripper,
        )
    elif cfg.wrapper.ee_action_space_params.control_mode == "leader":
        env = GearedLeaderControlWrapper(
            env=env,
            ee_action_space_params=cfg.wrapper.ee_action_space_params,
            use_gripper=cfg.wrapper.use_gripper,
        )
    elif cfg.wrapper.ee_action_space_params.control_mode == "leader_automatic":
        env = GearedLeaderAutomaticControlWrapper(
            env=env,
            ee_action_space_params=cfg.wrapper.ee_action_space_params,
            use_gripper=cfg.wrapper.use_gripper,
        )
    else:
        raise ValueError(f"Invalid control mode: {cfg.wrapper.ee_action_space_params.control_mode}")

    env = ResetWrapper(
        env=env,
        reset_pose=cfg.wrapper.fixed_reset_joint_positions,
        reset_time_s=cfg.wrapper.reset_time_s,
    )
    env = BatchCompatibleWrapper(env=env)
    env = TorchActionWrapper(env=env, device=cfg.device)

    return env


def init_reward_classifier(cfg):
    """
    Load a reward classifier policy from a pretrained path if configured.

    Args:
        cfg: The environment configuration containing classifier paths.

    Returns:
        The loaded classifier model or None if not configured.
    """
    if cfg.reward_classifier_pretrained_path is None:
        return None

    from lerobot.common.policies.reward_model.modeling_classifier import Classifier

    # Get device from config or default to CUDA
    device = getattr(cfg, "device", "cpu")

    # Load the classifier directly using from_pretrained
    classifier = Classifier.from_pretrained(
        pretrained_name_or_path=cfg.reward_classifier_pretrained_path,
    )

    # Ensure model is on the correct device
    classifier.to(device)
    classifier.eval()  # Set to evaluation mode

    return classifier


###########################################################
# Record and replay functions
###########################################################


def record_dataset(env, policy, cfg):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    This function runs episodes in the environment and records the observations,
    actions, and results for dataset creation.

    Args:
        env: The environment to record from.
        policy: Optional policy to generate actions (if None, uses teleop).
        cfg: Configuration object containing recording parameters like:
            - repo_id: Repository ID for dataset storage
            - dataset_root: Local root directory for dataset
            - num_episodes: Number of episodes to record
            - fps: Frames per second for recording
            - push_to_hub: Whether to push dataset to Hugging Face Hub
            - task: Name/description of the task being recorded
            - number_of_steps_after_success: Number of additional steps to continue recording after
                                  a success (reward=1) is detected. This helps collect
                                  more positive examples for reward classifier training.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Setup initial action (zero action if using teleop)
    action = env.action_space.sample() * 0.0

    action_names = ["delta_x_ee", "delta_y_ee", "delta_z_ee"]
    if cfg.wrapper.use_gripper:
        action_names.append("gripper_delta")

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
    }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
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

    # Record episodes
    episode_index = 0
    recorded_action = None
    while episode_index < cfg.num_episodes:
        obs, _ = env.reset()
        start_episode_t = time.perf_counter()
        log_say(f"Recording episode {episode_index}", play_sounds=True)

        # Track success state collection
        success_detected = False
        success_steps_collected = 0

        # Run episode steps
        while time.perf_counter() - start_episode_t < cfg.wrapper.control_time_s:
            start_loop_t = time.perf_counter()

            # Get action from policy if available
            if cfg.pretrained_policy_name_or_path is not None:
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            # For teleop, get action from intervention
            recorded_action = {
                "action": info["action_intervention"].cpu().squeeze(0).float() if policy is None else action
            }

            # Process observation for dataset
            obs_processed = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Check if we've just detected success
            if reward == 1.0 and not success_detected:
                success_detected = True
                logging.info("Success detected! Collecting additional success states.")

            # Add frame to dataset - continue marking as success even during extra collection steps
            frame = {**obs_processed, **recorded_action}

            # If we're in the success collection phase, keep marking rewards as 1.0
            if success_detected:
                frame["next.reward"] = np.array([1.0], dtype=np.float32)
            else:
                frame["next.reward"] = np.array([reward], dtype=np.float32)

            # Only mark as done if we're truly done (reached end or collected enough success states)
            really_done = terminated or truncated
            if success_detected:
                success_steps_collected += 1
                really_done = success_steps_collected >= cfg.number_of_steps_after_success

            frame["next.done"] = np.array([really_done], dtype=bool)
            frame["task"] = cfg.task
            frame["complementary_info.discrete_penalty"] = torch.tensor(
                [info.get("discrete_penalty", 0.0)], dtype=torch.float32
            )
            dataset.add_frame(frame)

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            # Check if we should end the episode
            if (terminated or truncated) and not success_detected:
                # Regular termination without success
                break
            elif success_detected and success_steps_collected >= cfg.number_of_steps_after_success:
                # We've collected enough success states
                logging.info(f"Collected {success_steps_collected} additional success states")
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode_index}")
            continue

        dataset.save_episode()
        episode_index += 1

    # Finalize dataset
    # dataset.consolidate(run_compute_stats=True)
    if cfg.push_to_hub:
        dataset.push_to_hub()


def replay_episode(env, cfg):
    """
    Replay a recorded episode in the environment.

    This function loads actions from a previously recorded episode
    and executes them in the environment.

    Args:
        env: The environment to replay in.
        cfg: Configuration object containing replay parameters:
            - repo_id: Repository ID for dataset
            - dataset_root: Local root directory for dataset
            - episode: Episode ID to replay
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.dataset_root, episodes=[cfg.episode])
    env.reset()

    actions = dataset.hf_dataset.select_columns("action")

    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        env.step(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / 10 - dt_s)


@parser.wrap()
def main(cfg: EnvConfig):
    """
    Main entry point for the robot environment script.

    This function runs the robot environment in one of several modes
    based on the provided configuration.

    Args:
        cfg: Configuration object defining the run parameters,
             including mode (record, replay, random) and other settings.
    """
    env = make_robot_env(cfg)

    if cfg.mode == "record":
        policy = None
        if cfg.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy

            policy = SACPolicy.from_pretrained(cfg.pretrained_policy_name_or_path)
            policy.to(cfg.device)
            policy.eval()

        record_dataset(
            env,
            policy=policy,
            cfg=cfg,
        )
        exit()

    if cfg.mode == "replay":
        replay_episode(
            env,
            cfg=cfg,
        )
        exit()

    env.reset()

    # Initialize the smoothed action as a random sample.
    smoothed_action = env.action_space.sample()

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 1.0

    num_episode = 0
    successes = []
    while num_episode < 10:
        start_loop_s = time.perf_counter()
        # Sample a new random action from the robot's action space.
        new_random_action = env.action_space.sample()
        # Update the smoothed action using an exponential moving average.
        smoothed_action = alpha * new_random_action + (1 - alpha) * smoothed_action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step(smoothed_action)
        if terminated or truncated:
            successes.append(reward)
            env.reset()
            num_episode += 1

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / cfg.fps - dt_s)

    logging.info(f"Success after 20 steps {successes}")
    logging.info(f"success rate {sum(successes) / len(successes)}")


if __name__ == "__main__":
    main()
