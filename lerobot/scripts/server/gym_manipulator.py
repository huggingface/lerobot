import logging
import sys
import time
from threading import Lock
from typing import Annotated, Any, Dict, Tuple

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
MAX_GRIPPER_COMMAND = 40


class HILSerlRobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.
    """

    def __init__(
        self,
        robot,
        use_delta_action_space: bool = True,
        display_cameras: bool = False,
    ):
        """
        Initialize the HILSerlRobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        cfg.
            robot: The robot interface object used to connect and interact with the physical robot.
            use_delta_action_space (bool): If True, uses a delta (relative) action space for joint control. Otherwise, absolute
                joint positions are used.
            display_cameras (bool): If True, the robot's camera feeds will be displayed during execution.
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

        self.use_delta_action_space = use_delta_action_space
        self.current_joint_positions = self.robot.follower_arms["main"].read("Present_Position")

        # Retrieve the size of the joint position interval bound.

        self.relative_bounds_size = None
        # (
        #     (
        #         self.robot.config.joint_position_relative_bounds["max"]
        #         - self.robot.config.joint_position_relative_bounds["min"]
        #     )
        #     if self.robot.config.joint_position_relative_bounds is not None
        #     else None
        # )
        self.robot.config.joint_position_relative_bounds = None

        self.robot.config.max_relative_target = (
            self.relative_bounds_size.float() if self.relative_bounds_size is not None else None
        )

        # Dynamically configure the observation and action spaces.
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
        if self.use_delta_action_space:
            bounds = (
                self.relative_bounds_size
                if self.relative_bounds_size is not None
                else np.ones(action_dim) * 1000
            )
            self.action_space = gym.spaces.Box(
                low=-bounds,
                high=bounds,
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            bounds_min = (
                self.robot.config.joint_position_relative_bounds["min"].cpu().numpy()
                if self.robot.config.joint_position_relative_bounds is not None
                else np.ones(action_dim) * -1000
            )
            bounds_max = (
                self.robot.config.joint_position_relative_bounds["max"].cpu().numpy()
                if self.robot.config.joint_position_relative_bounds is not None
                else np.ones(action_dim) * 1000
            )
            self.action_space = gym.spaces.Box(
                low=bounds_min,
                high=bounds_max,
                shape=(action_dim,),
                dtype=np.float32,
            )

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        cfg.
            seed (Optional[int]): A seed for random number generation to ensure reproducibility.
            options (Optional[dict]): Additional options to influence the reset behavior.

        Returns:
            A tuple containing:
                - observation (dict): The initial sensor observation.
                - info (dict): A dictionary with supplementary information, including the key "initial_position".
        """
        super().reset(seed=seed, options=options)

        # Capture the initial observation.
        observation = self.robot.capture_observation()

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        return observation, {}

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is processed and sent to the robot as joint position commands
        that may be either absolute values or deltas based on the environment configuration.

        cfg.
            action (np.ndarray or torch.Tensor): The commanded joint positions.

        Returns:
            tuple: A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including:
        """
        self.current_joint_positions = self.robot.follower_arms["main"].read("Present_Position")
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.use_delta_action_space:
            target_joint_positions = self.current_joint_positions + action
        else:
            target_joint_positions = action

        self.robot.send_action(torch.from_numpy(target_joint_positions))
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
            {},
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
    def __init__(self, env, joint_velocity_limits=100.0, fps=30, num_dof=6):
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
        joint_velocities = (observation["observation.state"] - self.last_joint_positions) / self.dt
        self.last_joint_positions = observation["observation.state"].clone()
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], joint_velocities], dim=-1
        )
        return observation


class AddCurrentToObservation(gym.ObservationWrapper):
    def __init__(self, env, max_current=500, num_dof=6):
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
        present_current = (
            self.unwrapped.robot.follower_arms["main"].read("Present_Current").astype(np.float32)
        )
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], torch.from_numpy(present_current)], dim=-1
        )
        return observation


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, nb_repeat: int = 1):
        super().__init__(env)
        self.nb_repeat = nb_repeat

    def step(self, action):
        for _ in range(self.nb_repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            if done or truncated:
                break
        return obs, reward, done, truncated, info


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device: torch.device = "cuda"):
        """
        Wrapper to add reward prediction to the environment, it use a trained classifier.

        cfg.
            env: The environment to wrap
            reward_classifier: The reward classifier model
            device: The device to run the model on
        """
        self.env = env

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.reward_classifier = torch.compile(reward_classifier)
        self.reward_classifier.to(self.device)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        images = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
            if "image" in key
        }
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = (
                self.reward_classifier.predict_reward(images, threshold=0.8)
                if self.reward_classifier is not None
                else 0.0
            )
        info["Reward classifier frequency"] = 1 / (time.perf_counter() - start_time)

        if success == 1.0:
            terminated = True
            reward = 1.0

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, control_time_s, fps):
        self.env = env
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

        self.max_episode_steps = int(self.control_time_s * self.fps)

        self.current_step = 0

    def step(self, action):
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
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        crop_params_dict: Dict[str, Annotated[Tuple[int], 4]],
        resize_size=None,
    ):
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
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if obs[k].dim() >= 3:
                # Reshape to combine height and width dimensions for easier calculation
                batch_size = obs[k].size(0)
                channels = obs[k].size(1)
                flattened_spatial_dims = obs[k].view(batch_size, channels, -1)

                # Calculate standard deviation across spatial dimensions (H, W)
                std_per_channel = torch.std(flattened_spatial_dims, dim=2)

                # If any channel has std=0, all pixels in that channel have the same value
                if (std_per_channel <= 0.02).any():
                    logging.warning(
                        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                    )
            # Check for NaNs before processing
            if torch.isnan(obs[k]).any():
                logging.error(f"NaN values detected in observation {k} before crop and resize")

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            # TODO(michel-aractingi): Bug in resize, it returns values outside [0, 1]
            obs[k] = obs[k].clamp(0.0, 1.0)

            # import cv2
            # cv2.imwrite(f"tmp_img/{k}.jpg", obs[k].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)

            # Check for NaNs after processing
            if torch.isnan(obs[k]).any():
                logging.error(f"NaN values detected in observation {k} after crop and resize")

            obs[k] = obs[k].to(device)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
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
    def __init__(self, env, device):
        super().__init__(env)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def observation(self, observation):
        observation = preprocess_observation(observation)

        observation = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
        }

        return observation


class KeyboardInterfaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.listener = None
        self.events = {
            "exit_early": False,
            "pause_policy": False,
            "reset_env": False,
            "human_intervention_step": False,
            "episode_success": False,
        }
        self.event_lock = Lock()  # Thread-safe access to events
        self._init_keyboard_listener()

    def _init_keyboard_listener(self):
        """Initialize keyboard listener if not in headless mode"""

        if is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
            )
            return
        try:
            from pynput import keyboard

            def on_press(key):
                with self.event_lock:
                    try:
                        if key == keyboard.Key.right or key == keyboard.Key.esc:
                            print("Right arrow key pressed. Exiting loop...")
                            self.events["exit_early"] = True
                            return
                        if hasattr(key, "char") and key.char == "s":
                            print("Key 's' pressed. Episode success triggered.")
                            self.events["episode_success"] = True
                            return
                        if key == keyboard.Key.space and not self.events["exit_early"]:
                            if not self.events["pause_policy"]:
                                print(
                                    "Space key pressed. Human intervention required.\n"
                                    "Place the leader in similar pose to the follower and press space again."
                                )
                                self.events["pause_policy"] = True
                                log_say(
                                    "Human intervention stage. Get ready to take over.",
                                    play_sounds=True,
                                )
                                return
                            if self.events["pause_policy"] and not self.events["human_intervention_step"]:
                                self.events["human_intervention_step"] = True
                                print("Space key pressed. Human intervention starting.")
                                log_say("Starting human intervention.", play_sounds=True)
                                return
                            if self.events["pause_policy"] and self.events["human_intervention_step"]:
                                self.events["pause_policy"] = False
                                self.events["human_intervention_step"] = False
                                print("Space key pressed for a third time.")
                                log_say("Continuing with policy actions.", play_sounds=True)
                                return
                    except Exception as e:
                        print(f"Error handling key press: {e}")

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except ImportError:
            logging.warning("Could not import pynput. Keyboard interface will not be available.")
            self.listener = None

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        terminated_by_keyboard = False

        # Check the event flags without holding the lock for too long.
        with self.event_lock:
            if self.events["exit_early"]:
                terminated_by_keyboard = True
            pause_policy = self.events["pause_policy"]

        if pause_policy:
            # Now, wait for human_intervention_step without holding the lock
            while True:
                with self.event_lock:
                    if self.events["human_intervention_step"]:
                        is_intervention = True
                        break
                time.sleep(0.1)  # Check frequently for intervention step

        # Execute the step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step((policy_action, is_intervention))

        # Override reward and termination if episode success event triggered
        with self.event_lock:
            if self.events["episode_success"]:
                reward = 1
                terminated_by_keyboard = True

        return obs, reward, terminated or terminated_by_keyboard, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset the environment and clear any pending events
        """
        with self.event_lock:
            self.events = dict.fromkeys(self.events, False)
        return self.env.reset(**kwargs)

    def close(self):
        """
        Properly clean up the keyboard listener when the environment is closed
        """
        if self.listener is not None:
            self.listener.stop()
        super().close()


class ResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env: HILSerlRobotEnv,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
        open_gripper_on_reset: bool = False,
    ):
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose
        self.robot = self.unwrapped.robot
        self.open_gripper_on_reset = open_gripper_on_reset

    def reset(self, *, seed=None, options=None):
        if self.reset_pose is not None:
            start_time = time.perf_counter()
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, self.reset_pose)
            busy_wait(self.reset_time_s - (time.perf_counter() - start_time))
            log_say("Reset the environment done.", play_sounds=True)
            if self.open_gripper_on_reset:
                current_joint_pos = self.robot.follower_arms["main"].read("Present_Position")
                current_joint_pos[-1] = MAX_GRIPPER_COMMAND
                self.robot.send_action(torch.from_numpy(current_joint_pos))
                busy_wait(0.1)
                current_joint_pos[-1] = 0.0
                self.robot.send_action(torch.from_numpy(current_joint_pos))
                busy_wait(0.2)
        else:
            log_say(
                f"Manually reset the environment for {self.reset_time_s} seconds.",
                play_sounds=True,
            )
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                self.robot.teleop_step()

            log_say("Manual reset of the environment done.", play_sounds=True)
        return super().reset(seed=seed, options=options)


class BatchCompatibleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty: float = -0.1, gripper_penalty_in_reward: bool = True):
        super().__init__(env)
        self.penalty = penalty
        self.gripper_penalty_in_reward = gripper_penalty_in_reward
        self.last_gripper_state = None

    def reward(self, reward, action):
        gripper_state_normalized = self.last_gripper_state / MAX_GRIPPER_COMMAND

        action_normalized = action - 1.0  # action / MAX_GRIPPER_COMMAND

        gripper_penalty_bool = (gripper_state_normalized < 0.5 and action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and action_normalized < -0.5
        )

        return reward + self.penalty * int(gripper_penalty_bool)

    def step(self, action):
        self.last_gripper_state = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        gripper_action = action[-1]
        obs, reward, terminated, truncated, info = self.env.step(action)
        gripper_penalty = self.reward(reward, gripper_action)

        if self.gripper_penalty_in_reward:
            reward += gripper_penalty
        else:
            info["discrete_penalty"] = gripper_penalty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_gripper_state = None
        obs, info = super().reset(**kwargs)
        if self.gripper_penalty_in_reward:
            info["gripper_penalty"] = 0.0
        return obs, info


class GripperActionWrapper(gym.ActionWrapper):
    def __init__(self, env, quantization_threshold: float = 0.2, gripper_sleep: float = 0.0):
        super().__init__(env)
        self.quantization_threshold = quantization_threshold
        self.gripper_sleep = gripper_sleep
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None

    def action(self, action):
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
        obs, info = super().reset(**kwargs)
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None
        return obs, info


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_space_params=None, use_gripper=False):
        super().__init__(env)
        self.ee_action_space_params = ee_action_space_params
        self.use_gripper = use_gripper

        # Initialize kinematics instance for the appropriate robot type
        robot_type = getattr(env.unwrapped.robot.config, "robot_type", "so100")
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
        desired_ee_pos = np.eye(4)

        if self.use_gripper:
            gripper_command = action[-1]
            action = action[:-1]

        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = self.fk_function(current_joint_pos)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
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
    def __init__(self, env, ee_pose_limits):
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


class GearedLeaderControlWrapper(gym.Wrapper):
    def __init__(self, env, use_geared_leader_arm: bool = False, ee_action_space_params=None):
        super().__init__(env)
        self.robot_leader = env.unwrapped.robot.leader_arms["main"]
        self.robot_follower = env.unwrapped.robot.follower_arms["main"]
        self.use_geared_leader_arm = use_geared_leader_arm
        self.ee_action_space_params = ee_action_space_params

        self.keyboard_events = {
            "pause_policy": False,
            "episode_success": False,
            "episode_end": False,
            "rerecord_episode": False,
        }
        self.event_lock = Lock()  # Thread-safe access to events
        self._init_keyboard_listener()
        self.kinematics = RobotKinematics(env.unwrapped.robot.config.robot_type)
        self.prev_leader_ee = None

    def _init_keyboard_listener(self):
        """Initialize keyboard listener if not in headless mode"""

        if is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
            )
            return
        try:
            from pynput import keyboard

            def on_press(key):
                with self.event_lock:
                    try:
                        if key == keyboard.Key.esc:
                            self.events["episode_end"] = True
                            return
                        if key == keyboard.Key.left:
                            self.events["rerecord_episode"] = True
                            return
                        if hasattr(key, "char") and key.char == "s":
                            logging.info("Key 's' pressed. Episode success triggered.")
                            self.events["episode_success"] = True
                            return
                        if key == keyboard.Key.space:
                            if not self.events["pause_policy"]:
                                logging.info(
                                    "Space key pressed. Human intervention required.\n"
                                    "Place the leader in similar pose to the follower and press space again."
                                )
                                self.events["pause_policy"] = True
                                log_say(
                                    "Human intervention step.",
                                    play_sounds=True,
                                )
                                return
                            else:
                                self.events["pause_policy"] = False
                                logging.info(
                                    "Space key pressed for a second time.\nContinuing with policy actions."
                                )
                                log_say("Continuing with policy actions.", play_sounds=True)
                                return

                    except Exception as e:
                        print(f"Error handling key press: {e}")

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        except ImportError:
            logging.warning("Could not import pynput. Keyboard interface will not be available.")
            self.listener = None

    def step(self, action):
        leader_pos = self.robot_leader.read("Present_Position")
        is_intervention = False
        if self.use_ee_action_space:
            leader_ee = self.kinematics.fk_gripper_tip(leader_pos)
            if self.prev_leader_ee is None:
                self.prev_leader_ee = leader_ee
            action = leader_ee - self.prev_leader_ee
            self.prev_leader_ee = leader_ee
        else:
            action = self.robot_leader.read("Present_Position")

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["is_intervention"] = is_intervention

        return self.env.step(action)

    def reset(self, **kwargs):
        self.prev_leader_ee = None
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

        cfg.
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            vendor_id: USB vendor ID of the gamepad (default: Logitech)
            product_id: USB product ID of the gamepad (default: RumblePad 2)
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
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

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
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

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
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
        final_action = torch.from_numpy(gamepad_action) if is_intervention else action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(final_action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        if isinstance(final_action, np.ndarray):
            action_intervention = torch.from_numpy(final_action)
        else:
            action_intervention = final_action

        info["is_intervention"] = is_intervention
        info["action_intervention"] = action_intervention
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
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


def make_robot_env(cfg) -> gym.vector.VectorEnv:
    """
    Factory function to create a vectorized robot environment.

    cfg.
        robot: Robot instance to control
        reward_classifier: Classifier model for computing rewards
        cfg: Configuration object containing environment parameters

    Returns:
        A vectorized gym environment with all the necessary wrappers applied.
    """
    if "maniskill" in cfg.name:
        from lerobot.scripts.server.maniskill_manipulator import make_maniskill

        logging.warning("WE SHOULD REMOVE THE MANISKILL BEFORE THE MERGE INTO MAIN")
        env = make_maniskill(
            cfg=cfg,
            n_envs=1,
        )
        return env
    robot = make_robot_from_config(cfg.robot)
    # Create base environment
    env = HILSerlRobotEnv(
        robot=robot,
        display_cameras=cfg.wrapper.display_cameras,
        use_delta_action_space=cfg.wrapper.use_relative_joint_positions
        and cfg.wrapper.ee_action_space_params is None,
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
                gripper_penalty_in_reward=cfg.wrapper.gripper_penalty_in_reward,
            )

    if cfg.wrapper.ee_action_space_params is not None:
        env = EEActionWrapper(
            env=env,
            ee_action_space_params=cfg.wrapper.ee_action_space_params,
            use_gripper=cfg.wrapper.use_gripper,
        )

    if cfg.wrapper.ee_action_space_params is not None:
        if cfg.wrapper.ee_action_space_params.use_gamepad:
            env = GamepadControlWrapper(
                env=env,
                x_step_size=cfg.wrapper.ee_action_space_params.x_step_size,
                y_step_size=cfg.wrapper.ee_action_space_params.y_step_size,
                z_step_size=cfg.wrapper.ee_action_space_params.z_step_size,
                use_gripper=cfg.wrapper.use_gripper,
            )
        elif cfg.wrapper.ee_action_space_params.use_leader_control:
            env = GearedLeaderControlWrapper(
                env=env,
                ee_action_space_params=cfg.wrapper.ee_action_space_params,
            )
    else:
        env = KeyboardInterfaceWrapper(env=env)

    env = ResetWrapper(
        env=env,
        reset_pose=cfg.wrapper.fixed_reset_joint_positions,
        reset_time_s=cfg.wrapper.reset_time_s,
        open_gripper_on_reset=cfg.wrapper.open_gripper_on_reset,
    )
    env = BatchCompatibleWrapper(env=env)

    return env


def init_reward_classifier(cfg):
    """
    Load a reward classifier policy from a pretrained path if configured.

    Args:
        cfg: The environment configuration containing classifier paths

    Returns:
        The loaded classifier model or None if not configured
    """
    if cfg.reward_classifier_pretrained_path is None:
        return None

    from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier

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


def record_dataset(env, policy, cfg):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    cfg.
        env: The environment to record from
        repo_id: Repository ID for dataset storage
        root: Local root directory for dataset (optional)
        num_episodes: Number of episodes to record
        control_time_s: Maximum episode length in seconds
        fps: Frames per second for recording
        push_to_hub: Whether to push dataset to Hugging Face Hub
        task_description: Description of the task being recorded
        policy: Optional policy to generate actions (if None, uses teleop)
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Setup initial action (zero action if using teleop)
    dummy_action = env.action_space.sample() * 0.0
    if isinstance(dummy_action, np.ndarray):
        dummy_action = torch.from_numpy(dummy_action)
    action = dummy_action

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": env.action_space.shape,
            "names": None,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
                "names": None,
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
            obs = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Add frame to dataset
            frame = {**obs, **recorded_action}
            frame["next.reward"] = np.array([reward], dtype=np.float32)
            frame["next.done"] = np.array([terminated or truncated], dtype=bool)
            frame["task"] = cfg.task
            dataset.add_frame(frame)

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            if terminated or truncated:
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode_index}")
            continue

        dataset.save_episode(cfg.task)
        episode_index += 1

    # Finalize dataset
    # dataset.consolidate(run_compute_stats=True)
    if cfg.push_to_hub:
        dataset.push_to_hub()


def replay_episode(env, cfg):
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
    while num_episode < 20:
        start_loop_s = time.perf_counter()
        # Sample a new random action from the robot's action space.
        new_random_action = env.action_space.sample()
        # Update the smoothed action using an exponential moving average.
        smoothed_action = alpha * new_random_action + (1 - alpha) * smoothed_action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(smoothed_action))
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
