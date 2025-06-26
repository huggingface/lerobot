import argparse
import json
import logging
import logging.handlers
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import torch

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.constants import OBS_STATE
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features

# NOTE: Configs need to be loaded for the client to be able to instantiate the policy config
from lerobot.common.policies.act.configuration_act import ACTConfig  # noqa: F401
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: F401
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig  # noqa: F401
from lerobot.common.robots.robot import Robot
from lerobot.common.robots.so100_follower import SO100FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.configs.types import PolicyFeature
from lerobot.scripts.server.constants import supported_robots


def visualize_action_queue_size(action_queue_size: list[int]) -> None:
    fig, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.show()


def validate_robot_cameras_for_policy(
    lerobot_observation_features: dict[str, dict], policy_image_features: dict[str, PolicyFeature]
) -> None:
    image_keys = list(filter(is_image_key, lerobot_observation_features))
    assert set(image_keys) == set(policy_image_features.keys()), (
        f"Policy image features must match robot cameras! Received {list(policy_image_features.keys())} != {image_keys}"
    )


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, "observation", use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith("observation.images")


def map_image_key_to_smolvla_base_key(idx: int) -> str:
    """Dataset contain image features keys named as observation.images.<camera_name>, but SmolVLA adapts this
    to observation.image, observation.image2, ..."""
    idx_text = str(idx + 1) if idx != 0 else ""
    return f"observation.image{idx_text}"


def resize_robot_observation_image(image: torch.tensor, resize_dims: tuple[int, int, int]) -> torch.tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def prepare_observation_for_policy(
    robot_obs: dict, lerobot_features: dict[str, dict], policy_image_features: dict[str, PolicyFeature]
) -> dict[str, torch.tensor]:
    """First, turns the {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} into
    {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}"""
    lerobot_obs = build_dataset_frame(lerobot_features, robot_obs, prefix="observation")

    """1. Greps all observation.images.<> keys"""
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: torch.tensor(lerobot_obs[OBS_STATE]).unsqueeze(0)}
    image_dict = {image_k: torch.tensor(lerobot_obs[image_k]) for image_k in image_keys}

    # turning image features to (C, H, W) with H, W matching the policy image features
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    return {**state_dict, **image_dict}


def make_default_camera_config(
    index_or_path: int = 1, fps: int = 30, width: int = 1920, height: int = 1080
) -> OpenCVCameraConfig:
    # NOTE(fracapuano): This will be removed when moving to draccus parser
    return OpenCVCameraConfig(index_or_path=index_or_path, fps=fps, width=width, height=height)


def make_robot(args: argparse.Namespace) -> Robot:
    if args.robot not in supported_robots:
        raise ValueError(f"Robot {args.robot} not yet supported!")

    if args.robot == "so100":
        config = SO100FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras={k: make_default_camera_config(**v) for k, v in json.loads(args.robot_cameras).items()},
        )

    return make_robot_from_config(config)


def setup_logging(prefix: str, info_bracket: str):
    """Sets up logging"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Delete any existing prefix_* log files
    for old_log_file in os.listdir("logs"):
        if old_log_file.startswith(prefix) and old_log_file.endswith(".log"):
            try:
                os.remove(os.path.join("logs", old_log_file))
                print(f"Deleted old log file: {old_log_file}")
            except Exception as e:
                print(f"Failed to delete old log file {old_log_file}: {e}")

    # Set up logging with both console and file output
    logger = logging.getLogger(prefix)
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d [{info_bracket}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    # File handler - creates a new log file for each run
    file_handler = logging.handlers.RotatingFileHandler(
        f"logs/policy_server_{int(time.time())}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d [{info_bracket}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger


class TimedData:
    def __init__(self, timestamp: float, data: Any, timestep: int):
        """Initialize a TimedData object.

        Args:
            timestamp: Unix timestamp relative to data's creation.
            data: The actual data to wrap a timestamp around.
            timestep: The timestep of the data.
        """
        self.timestamp = timestamp
        self.data = data
        self.timestep = timestep

    def get_data(self):
        return self.data

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


class TimedAction(TimedData):
    def __init__(self, timestamp: float, action: torch.Tensor, timestep: int):
        super().__init__(timestamp=timestamp, data=action, timestep=timestep)

    def get_action(self):
        return self.get_data()


class TimedObservation(TimedData):
    def __init__(
        self,
        timestamp: float,
        observation: dict[str, torch.Tensor],
        timestep: int,
        transfer_state: int = 0,
        must_go: bool = False,
    ):
        super().__init__(timestamp=timestamp, data=observation, timestep=timestep)
        self.transfer_state = transfer_state
        self.must_go = must_go

    def get_observation(self):
        return self.get_data()


class TinyPolicyConfig:
    def __init__(
        self,
        policy_type: str = "act",
        pretrained_name_or_path: str = "fracapuano/act_so100_test",
        device: str = "cpu",
    ):
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    return torch.linalg.norm(obs1_state - obs2_state) < atol


def observations_similar(obs1: TimedObservation, obs2: TimedObservation, atol: float = 1) -> bool:
    """Check if two observations are similar, under a tolerance threshold"""
    obs1_state = obs1.get_observation()["observation.state"]
    obs2_state = obs2.get_observation()["observation.state"]

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
