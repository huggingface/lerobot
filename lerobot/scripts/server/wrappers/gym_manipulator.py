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
from lerobot.common.robot_devices.control_utils import is_headless, reset_follower_position
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, log_say

logging.basicConfig(level=logging.INFO)


class HILSerlRobotEnv(gym.Env):
    """
    Gym-like environment wrapper for robot policy evaluation.

    This wrapper provides a consistent interface for interacting with the robot,
    following the OpenAI Gym environment conventions.
    """

    def __init__(
        self,
        robot,
        display_cameras=False,
    ):
        """
        Initialize the robot environment.

        Args:
            robot: The robot interface object
            reward_classifier: Optional reward classifier
            fps: Frames per second for control
            control_time_s: Total control time for each episode
            display_cameras: Whether to display camera feeds
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # connect robot
        if not self.robot.is_connected:
            self.robot.connect()

        # Dynamically determine observation and action spaces
        self._setup_spaces()

        self.initial_follower_position = robot.follower_arms["main"].read("Present_Position")

        # Episode tracking
        self.current_step = 0
        self.episode_data = None

    def _setup_spaces(self):
        """
        Dynamically determine observation and action spaces based on robot capabilities.

        This method should be customized based on the specific robot's observation
        and action representations.
        """
        # Example space setup - you'll need to adapt this to your specific robot
        example_obs = self.robot.capture_observation()

        # Observation space (assuming image-based observations)
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

        # Action space (assuming joint positions)
        action_dim = len(self.robot.follower_arms["main"].read("Present_Position"))
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32),
                gym.spaces.Discrete(2),
            ),
        )

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation (dict): Initial observation
            info (dict): Additional information
        """
        super().reset(seed=seed, options=options)

        # Capture initial observation
        observation = self.robot.capture_observation()

        # Reset tracking variables
        self.current_step = 0
        self.episode_data = None

        return observation, {"initial_position": self.initial_follower_position}

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action tuple(np.ndarray, bool):
                    Policy action to be executed on the robot and boolean to determine
                    whether to choose policy action or expert action.

        Returns:
            observation (dict): Next observation
            reward (float): Reward for this step
            terminated (bool): Whether the episode has terminated
            truncated (bool): Whether the episode was truncated
            info (dict): Additional information
        """
        # The actions recieved are the in form of a tuple containing the policy action and an intervention bool
        # The boolean inidicated whether we will use the expert's actions (through teleoperation) or the policy actions
        policy_action, intervention_bool = action
        teleop_action = None
        if not intervention_bool:
            self.robot.send_action(policy_action.cpu())
            observation = self.robot.capture_observation()
        else:
            observation, teleop_action = self.robot.teleop_step(record_data=True)
            teleop_action = teleop_action["action"]  # teleop step returns torch tensors but in a dict

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
        Render the environment (in this case, display camera feeds).
        """
        import cv2

        observation = self.robot.capture_observation()
        image_keys = [key for key in observation if "image" in key]

        for key in image_keys:
            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))

        cv2.waitKey(1)

    def close(self):
        """
        Close the environment and disconnect the robot.
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


class RelativeJointPositionActionWrapper(gym.Wrapper):
    def __init__(self, env: HILSerlRobotEnv, delta: float = 0.1):
        super().__init__(env)
        self.joint_positions = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        self.delta = delta

    def step(self, action):
        action_joint = action
        self.joint_positions = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            action_joint = action[0]
        joint_positions = self.joint_positions + (self.delta * action_joint)
        # clip the joint positions to the joint limits with the action space
        joint_positions = np.clip(joint_positions, self.action_space.low, self.action_space.high)

        if isinstance(self.env.action_space, gym.spaces.Tuple):
            return self.env.step((joint_positions, action[1]))

        obs, reward, terminated, truncated, info = self.env.step(joint_positions)
        if info["is_intervention"]:
            # teleop actions are returned in absolute joint space
            # If we are using a relative joint position action space,
            # there will be a mismatch between the spaces of the policy and teleop actions
            # Solution is to transform the teleop actions into relative space.
            teleop_action = info["action_intervention"]  # teleop actions are in absolute joint space
            relative_teleop_action = (teleop_action - self.joint_positions) / self.delta
            info["action_intervention"] = relative_teleop_action

        return self.env.step(joint_positions)


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier: Optional[None], device: torch.device = "cuda"):
        self.env = env
        self.reward_classifier = reward_classifier
        self.device = device

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        images = [
            observation[key].to(self.device, non_blocking=True) for key in observation if "image" in key
        ]
        reward = self.reward_classifier.predict_reward(images) if self.reward_classifier is not None else 0.0
        reward = reward.item()
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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()

        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step < self.fps:
            logging.warning(f"Current timestep exceeded expected fps {self.fps}")

        if self.episode_time_in_s > self.control_time_s:
            # Terminated = True
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    def __init__(self, env, crop_params_dict: Dict[str, Annotated[Tuple[int], 4]], resize_size=None):
        self.env = env
        self.crop_params_dict = crop_params_dict
        for key in crop_params_dict:
            assert key in self.env.observation_space, f"Key {key} not in observation space"
            top, left, height, width = crop_params_dict[key]
            new_shape = (top + height, left + width)
            self.observation_space[key] = gym.spaces.Box(low=0, high=255, shape=new_shape)

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
        return obs, reward, terminated, truncated, info


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


def make_robot_env(
    robot,
    reward_classifier,
    crop_params_dict=None,
    fps=30,
    control_time_s=20,
    reset_follower_pos=True,
    display_cameras=False,
    device="cuda:0",
    resize_size=None,
    reset_time_s=10,
    delta_action=0.1,
    nb_repeats=1,
    use_relative_joint_positions=False,
):
    """
    Factory function to create the robot environment.

    Mimics gym.make() for consistent environment creation.
    """
    env = HILSerlRobotEnv(robot, display_cameras)
    env = ConvertToLeRobotObservation(env, device)
    # if crop_params_dict is not None:
    #     env = ImageCropResizeWrapper(env, crop_params_dict, resize_size=resize_size)
    # env = RewardWrapper(env, reward_classifier)
    env = TimeLimitWrapper(env, control_time_s, fps)
    # if use_relative_joint_positions:
    #     env = RelativeJointPositionActionWrapper(env, delta=delta_action)
    # env = ActionRepeatWrapper(env, nb_repeat=nb_repeats)
    env = KeyboardInterfaceWrapper(env)
    env = ResetWrapper(env, reset_fn=None, reset_time_s=reset_time_s)
    return env


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
    parser.add_argument("--control-time-s", type=float, default=20, help="Maximum episode length in seconds")
    parser.add_argument("--reset-follower-pos", type=int, default=1, help="Reset follower between episodes")
    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)

    reward_classifier = get_classifier(
        args.reward_classifier_pretrained_path, args.reward_classifier_config_file
    )

    env = make_robot_env(
        robot,
        reward_classifier,
        None,
        args.fps,
        args.control_time_s,
        args.reset_follower_pos,
        args.display_cameras,
        device="mps",
        resize_size=None,
        reset_time_s=10,
        delta_action=0.1,
        nb_repeats=1,
        use_relative_joint_positions=False,
    )

    env.reset()
    init_pos = env.unwrapped.initial_follower_position
    goal_pos = init_pos

    right_goal = init_pos.copy()
    right_goal[0] += 50

    left_goal = init_pos.copy()
    left_goal[0] -= 50

    # Michel is a beast
    pitch_angle = np.linspace(left_goal[0], right_goal[0], 1000)

    while True:
        for i in range(len(pitch_angle)):
            goal_pos[0] = pitch_angle[i]
            obs, reward, terminated, truncated, info = env.step((torch.from_numpy(goal_pos), False))
            if terminated or truncated:
                logging.info("Max control time reached, reset environment.")
                env.reset()

        for i in reversed(range(len(pitch_angle))):
            goal_pos[0] = pitch_angle[i]
            obs, reward, terminated, truncated, info = env.step((torch.from_numpy(goal_pos), False))
            if terminated or truncated:
                logging.info("Max control time reached, reset environment.")
                env.reset()
