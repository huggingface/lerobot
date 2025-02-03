import argparse
import logging
import time
from typing import Annotated, Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import reset_follower_position
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

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
        reset_follower_position=True,
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

        self._initial_follower_position = robot.follower_arms["main"].read("Present_Position")
        self.reset_follower_position = reset_follower_position

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

        if self.reset_follower_position:
            reset_follower_position(self.robot, target_position=self._initial_follower_position)

        # Capture initial observation
        observation = self.robot.capture_observation()

        # Reset tracking variables
        self.current_step = 0
        self.episode_data = None

        return observation, {}

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
            self.robot.send_action(policy_action.cpu().numpy())
            observation = self.robot.capture_observation()
        else:
            observation, teleop_action = self.robot.teleop_step(record_data=True)
            teleop_action = teleop_action["action"]  # teleop step returns torch tensors but in a dict

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, {"action": teleop_action}

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


class HILSerlTimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, control_time_s, fps):
        self.env = env
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

    def step(self, action):
        ret = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()

        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step > self.fps:
            logging.warning(f"Current timestep exceeded expected fps {self.fps}")

        if self.episode_time_in_s > self.control_time_s:
            # Terminated = True
            ret[2] = True
        return ret

    def reset(self, seed=None, options=None):
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        return self.env.reset(seed, options=None)


class HILSerlRewardWrapper(gym.Wrapper):
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


class HILSerlImageCropResizeWrapper(gym.Wrapper):
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
):
    """
    Factory function to create the robot environment.

    Mimics gym.make() for consistent environment creation.
    """
    env = HILSerlRobotEnv(robot, reset_follower_pos, display_cameras)
    env = ConvertToLeRobotObservation(env, device)
    if crop_params_dict is not None:
        env = HILSerlImageCropResizeWrapper(env, crop_params_dict, resize_size=resize_size)
    env = HILSerlRewardWrapper(env, reward_classifier)
    env = HILSerlTimeLimitWrapper(env, control_time_s, fps)
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
    )

    env.reset()
    while True:
        intervention_action = (None, True)
        obs, reward, terminated, truncated, info = env.step(intervention_action)
        if terminated or truncated:
            logging.info("Max control time reached, reset environment.")
            env.reset()
