import argparse
import logging
import time
from threading import Lock
from typing import Annotated, Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import busy_wait, is_headless, reset_follower_position
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, log_say

logging.basicConfig(level=logging.INFO)


class HILSerlRobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.

    The environment can switch between executing actions from a policy or using teleoperated actions (human intervention) during
    each step. When teleoperation is used, the override action is captured and returned in the `info` dict along with a flag
    `is_intervention`.
    """

    def __init__(
        self,
        robot,
        use_delta_action_space: bool = True,
        delta: float | None = None,
        display_cameras: bool = False,
    ):
        """
        Initialize the HILSerlRobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        Args:
            robot: The robot interface object used to connect and interact with the physical robot.
            use_delta_action_space (bool): If True, uses a delta (relative) action space for joint control. Otherwise, absolute
                joint positions are used.
            delta (float or None): A scaling factor for the relative adjustments applied to joint positions. Should be a value between
                0 and 1 when using a delta action space.
            display_cameras (bool): If True, the robot's camera feeds will be displayed during execution.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        self.initial_follower_position = robot.follower_arms["main"].read("Present_Position")

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self.delta = delta
        self.use_delta_action_space = use_delta_action_space
        self.current_joint_positions = self.robot.follower_arms["main"].read("Present_Position")

        # Retrieve the size of the joint position interval bound.
        self.relative_bounds_size = (
            self.robot.config.joint_position_relative_bounds["max"]
            - self.robot.config.joint_position_relative_bounds["min"]
        )

        self.delta_relative_bounds_size = self.relative_bounds_size * self.delta

        self.robot.config.max_relative_target = self.delta_relative_bounds_size.float()

        # Dynamically configure the observation and action spaces.
        self._setup_spaces()

    def _setup_spaces(self):
        """
        Dynamically configure the observation and action spaces based on the robot's capabilities.

        Observation Space:
            - For keys with "image": A Box space with pixel values ranging from 0 to 255.
            - For non-image keys: A nested Dict space is created under 'observation.state' with a suitable range.

        Action Space:
            - The action space is defined as a Tuple where:
                • The first element is a Box space representing joint position commands. It is defined as relative (delta)
                  or absolute, based on the configuration.
                • The second element is a Discrete space (with 2 values) serving as a flag for intervention (teleoperation).
        """
        example_obs = self.robot.capture_observation()

        # Define observation spaces for images and other states.
        image_keys = [key for key in example_obs if "image" in key]
        state_keys = [key for key in example_obs if "image" not in key]
        observation_spaces = {
            key: gym.spaces.Box(low=0, high=255, shape=example_obs[key].shape, dtype=np.uint8)
            for key in image_keys
        }
        observation_spaces["observation.state"] = gym.spaces.Dict(
            {
                key: gym.spaces.Box(low=0, high=10, shape=example_obs[key].shape, dtype=np.float32)
                for key in state_keys
            }
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = len(self.robot.follower_arms["main"].read("Present_Position"))
        if self.use_delta_action_space:
            action_space_robot = gym.spaces.Box(
                low=-self.relative_bounds_size.cpu().numpy(),
                high=self.relative_bounds_size.cpu().numpy(),
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            action_space_robot = gym.spaces.Box(
                low=self.robot.config.joint_position_relative_bounds["min"].cpu().numpy(),
                high=self.robot.config.joint_position_relative_bounds["max"].cpu().numpy(),
                shape=(action_dim,),
                dtype=np.float32,
            )

        self.action_space = gym.spaces.Tuple(
            (
                action_space_robot,
                gym.spaces.Discrete(2),
            ),
        )

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        Args:
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

        return observation, {"initial_position": self.initial_follower_position}

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is a tuple comprised of:
            • A policy action (joint position commands) that may be either in absolute values or as a delta.
            • A boolean flag indicating whether teleoperation (human intervention) should be used for this step.

        Behavior:
            - When the intervention flag is False, the environment processes and sends the policy action to the robot.
            - When True, a teleoperation step is executed. If using a delta action space, an absolute teleop action is converted
              to relative change based on the current joint positions.

        Args:
            action (tuple): A tuple with two elements:
                - policy_action (np.ndarray or torch.Tensor): The commanded joint positions.
                - intervention_bool (bool): True if the human operator intervenes by providing a teleoperation input.

        Returns:
            tuple: A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including:
                    ◦ "action_intervention": The teleop action if intervention was used.
                    ◦ "is_intervention": Flag indicating whether teleoperation was employed.
        """
        policy_action, intervention_bool = action
        teleop_action = None
        self.current_joint_positions = self.robot.follower_arms["main"].read("Present_Position")
        if isinstance(policy_action, torch.Tensor):
            policy_action = policy_action.cpu().numpy()
            policy_action = np.clip(policy_action, self.action_space[0].low, self.action_space[0].high)
        if not intervention_bool:
            if self.use_delta_action_space:
                target_joint_positions = self.current_joint_positions + self.delta * policy_action
            else:
                target_joint_positions = policy_action
            self.robot.send_action(torch.from_numpy(target_joint_positions))
            observation = self.robot.capture_observation()
        else:
            observation, teleop_action = self.robot.teleop_step(record_data=True)
            teleop_action = teleop_action["action"]  # Convert tensor to appropriate format

            # When applying the delta action space, convert teleop absolute values to relative differences.
            if self.use_delta_action_space:
                teleop_action = teleop_action - self.current_joint_positions
                if torch.any(teleop_action < -self.delta_relative_bounds_size * self.delta) and torch.any(
                    teleop_action > self.delta_relative_bounds_size
                ):
                    print(
                        f"Relative teleop delta exceeded bounds {self.delta_relative_bounds_size}, teleop_action {teleop_action}\n"
                        f"lower bounds condition {teleop_action < -self.delta_relative_bounds_size}\n"
                        f"upper bounds condition {teleop_action > self.delta_relative_bounds_size}"
                    )

                    teleop_action = torch.clamp(
                        teleop_action, -self.delta_relative_bounds_size, self.delta_relative_bounds_size
                    )
            # NOTE: To mimic the shape of a neural network output, we add a batch dimension to the teleop action.
            if teleop_action.dim() == 1:
                teleop_action = teleop_action.unsqueeze(0)

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            {"action_intervention": teleop_action, "is_intervention": teleop_action is not None},
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
        Wrapper to add reward prediction to the environment, it use a trained classifer.

        Args:
            env: The environment to wrap
            reward_classifier: The reward classifier model
            device: The device to run the model on
        """
        self.env = env

        # NOTE: We got 15% speedup by compiling the model
        self.reward_classifier = torch.compile(reward_classifier)
        self.device = device

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        images = [
            observation[key].to(self.device, non_blocking=True) for key in observation if "image" in key
        ]
        start_time = time.perf_counter()
        with torch.inference_mode():
            reward = (
                self.reward_classifier.predict_reward(images) if self.reward_classifier is not None else 0.0
            )
        info["Reward classifer frequency"] = 1 / (time.perf_counter() - start_time)
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
        # logging.warning(f"Current timestep is lower than the expected fps {self.fps}")
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()
        self.current_step += 1
        # check if last timestep took more time than the expected fps
        # if 1.0 / time_since_last_step < self.fps:
        #     logging.warning(f"Current timestep exceeded expected fps {self.fps}")

        if self.episode_time_in_s > self.control_time_s:
            # if self.current_step >= self.max_episode_steps:
            # Terminated = True
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    def __init__(self, env, crop_params_dict: Dict[str, Annotated[Tuple[int], 4]], resize_size=None):
        super().__init__(env)
        self.env = env
        self.crop_params_dict = crop_params_dict
        print(f"obs_keys , {self.env.observation_space}")
        print(f"crop params dict {crop_params_dict.keys()}")
        for key_crop in crop_params_dict:
            if key_crop not in self.env.observation_space.keys():
                raise ValueError(f"Key {key_crop} not in observation space")
        for key in crop_params_dict:
            top, left, height, width = crop_params_dict[key]
            new_shape = (top + height, left + width)
            self.observation_space[key] = gym.spaces.Box(low=0, high=255, shape=new_shape)

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].to(device)
            # print(f"observation with key {k} with size {obs[k].size()}")
            # cv2.imshow(k, cv2.cvtColor(obs[k].cpu().squeeze(0).permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, observation):
        observation = preprocess_observation(observation)

        observation = {key: observation[key].to(self.device, non_blocking=True) for key in observation}
        observation = {k: torch.tensor(v, device=self.device) for k, v in observation.items()}
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
                        elif key == keyboard.Key.space:
                            if not self.events["pause_policy"]:
                                print(
                                    "Space key pressed. Human intervention required.\n"
                                    "Place the leader in similar pose to the follower and press space again."
                                )
                                self.events["pause_policy"] = True
                                log_say("Human intervention stage. Get ready to take over.", play_sounds=True)
                            elif self.events["pause_policy"] and not self.events["human_intervention_step"]:
                                self.events["human_intervention_step"] = True
                                print("Space key pressed. Human intervention starting.")
                                log_say("Starting human intervention.", play_sounds=True)
                            else:
                                self.events["pause_policy"] = False
                                self.events["human_intervention_step"] = False
                                print("Space key pressed for a third time.")
                                log_say("Continuing with policy actions.", play_sounds=True)
                    except Exception as e:
                        print(f"Error handling key press: {e}")

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except ImportError:
            logging.warning("Could not import pynput. Keyboard interface will not be available.")
            self.listener = None

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        is_intervention = False
        terminated_by_keyboard = False

        # Extract policy_action if needed
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            policy_action = action[0]

        # Check the event flags without holding the lock for too long.
        with self.event_lock:
            if self.events["exit_early"]:
                terminated_by_keyboard = True
            # If we need to wait for human intervention, we note that outside the lock.
            pause_policy = self.events["pause_policy"]

        if pause_policy:
            # Now, wait for human_intervention_step without holding the lock
            while True:
                with self.event_lock:
                    if self.events["human_intervention_step"]:
                        is_intervention = True
                        break
                time.sleep(0.1)  # Check more frequently if desired

        # Execute the step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step((policy_action, is_intervention))
        return obs, reward, terminated or terminated_by_keyboard, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset the environment and clear any pending events
        """
        with self.event_lock:
            self.events = {k: False for k in self.events}
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
        self, env: HILSerlRobotEnv, reset_fn: Optional[Callable[[], None]] = None, reset_time_s: float = 5
    ):
        super().__init__(env)
        self.reset_fn = reset_fn
        self.reset_time_s = reset_time_s

        self.robot = self.unwrapped.robot
        self.init_pos = self.unwrapped.initial_follower_position

    def reset(self, *, seed=None, options=None):
        if self.reset_fn is not None:
            self.reset_fn(self.env)
        else:
            log_say(f"Manually reset the environment for {self.reset_time_s} seconds.", play_sounds=True)
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                self.robot.teleop_step()

            log_say("Manual reseting of the environment done.", play_sounds=True)
        return super().reset(seed=seed, options=options)


class BatchCompitableWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


def make_robot_env(
    robot,
    reward_classifier,
    cfg,
    n_envs: int = 1,
) -> gym.vector.VectorEnv:
    """
    Factory function to create a vectorized robot environment.

    Args:
        robot: Robot instance to control
        reward_classifier: Classifier model for computing rewards
        cfg: Configuration object containing environment parameters
        n_envs: Number of environments to create in parallel. Defaults to 1.

    Returns:
        A vectorized gym environment with all the necessary wrappers applied.
    """

    # Create base environment
    env = HILSerlRobotEnv(
        robot=robot,
        display_cameras=cfg.wrapper.display_cameras,
        delta=cfg.wrapper.delta_action,
        use_delta_action_space=cfg.wrapper.use_relative_joint_positions,
    )

    # Add observation and image processing
    env = ConvertToLeRobotObservation(env=env, device=cfg.device)
    if cfg.wrapper.crop_params_dict is not None:
        env = ImageCropResizeWrapper(
            env=env, crop_params_dict=cfg.wrapper.crop_params_dict, resize_size=cfg.wrapper.resize_size
        )

    # Add reward computation and control wrappers
    env = RewardWrapper(env=env, reward_classifier=reward_classifier, device=cfg.device)
    env = TimeLimitWrapper(env=env, control_time_s=cfg.wrapper.control_time_s, fps=cfg.fps)
    env = KeyboardInterfaceWrapper(env=env)
    env = ResetWrapper(env=env, reset_fn=None, reset_time_s=cfg.wrapper.reset_time_s)
    env = BatchCompitableWrapper(env=env)

    return env

    # batched version of the env that returns an observation of shape (b, c)


def get_classifier(pretrained_path, config_path, device="mps"):
    if pretrained_path is None or config_path is None:
        return

    from lerobot.common.policies.factory import _policy_cfg_from_hydra_cfg
    from lerobot.common.policies.hilserl.classifier.configuration_classifier import ClassifierConfig
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import Classifier

    cfg = init_hydra_config(config_path)

    classifier_config = _policy_cfg_from_hydra_cfg(ClassifierConfig, cfg)
    classifier_config.num_cameras = len(cfg.training.image_keys)  # TODO automate these paths
    model = Classifier(classifier_config)
    model.load_state_dict(Classifier.from_pretrained(pretrained_path).state_dict())
    model = model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30, help="control frequency")
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument(
        "--display-cameras", help=("Whether to display the camera feed while the rollout is happening")
    )
    parser.add_argument(
        "--reward-classifier-pretrained-path",
        type=str,
        default=None,
        help="Path to the pretrained classifier weights.",
    )
    parser.add_argument(
        "--reward-classifier-config-file",
        type=str,
        default=None,
        help="Path to a yaml config file that is necessary to build the reward classifier model.",
    )
    parser.add_argument("--env-path", type=str, default=None, help="Path to the env yaml file")
    parser.add_argument("--env-overrides", type=str, default=None, help="Overrides for the env yaml file")
    parser.add_argument("--control-time-s", type=float, default=20, help="Maximum episode length in seconds")
    parser.add_argument("--reset-follower-pos", type=int, default=1, help="Reset follower between episodes")
    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)

    reward_classifier = get_classifier(
        args.reward_classifier_pretrained_path, args.reward_classifier_config_file
    )
    user_relative_joint_positions = True

    cfg = init_hydra_config(args.env_path, args.env_overrides)
    env = make_robot_env(
        robot,
        reward_classifier,
        cfg.wrapper,
    )

    env.reset()

    # Retrieve the robot's action space for joint commands.
    action_space_robot = env.action_space.spaces[0]

    # Initialize the smoothed action as a random sample.
    smoothed_action = action_space_robot.sample()

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 0.4

    while True:
        start_loop_s = time.perf_counter()
        # Sample a new random action from the robot's action space.
        new_random_action = action_space_robot.sample()
        # Update the smoothed action using an exponential moving average.
        smoothed_action = alpha * new_random_action + (1 - alpha) * smoothed_action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step((torch.from_numpy(smoothed_action), False))
        if terminated or truncated:
            env.reset()

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / args.fps - dt_s)
