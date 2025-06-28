import argparse
import io
import json
import logging
import logging.handlers
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
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
from lerobot.common.transport import async_inference_pb2
from lerobot.common.transport.utils import bytes_buffer_size
from lerobot.common.utils.utils import init_logging
from lerobot.configs.types import PolicyFeature
from lerobot.scripts.server.constants import supported_robots

Observation = dict[str, torch.Tensor]
Action = torch.Tensor

# Additional type to distinguish between the raw observation and the observations ready for inference
RawObservation = dict[str, torch.Tensor]


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


def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    device: str,
) -> Observation:
    observation = {}

    raw_observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for k, v in raw_observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0).to(device)
            else:
                observation[k] = v.to(device)
        else:
            observation[k] = v

    return observation


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> RawObservation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = build_dataset_frame(lerobot_features, robot_obs, prefix="observation")

    # 2. Greps all observation.images.<> keys
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: torch.tensor(lerobot_obs[OBS_STATE]).unsqueeze(0)}
    image_dict = {image_k: torch.tensor(lerobot_obs[image_k]) for image_k in image_keys}

    # Turns the image features to (C, H, W) with H, W matching the policy image features.
    # This reduces the resolution of the images
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

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


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Get a logger using the standardized logging setup from utils.py.

    Args:
        name: Logger name (e.g., 'policy_server', 'robot_client')
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # Initialize the standardized logging
    init_logging(log_file=log_file, display_pid=False)

    # Return a named logger
    return logging.getLogger(name)


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class FPSTracker:
    """Utility class to track FPS metrics over time."""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Calculate average FPS vs target"""
        self.total_obs_count += 1

        # Initialize first observation time
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # Calculate overall average FPS (since start)
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """Reset the FPS tracker state"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class TinyPolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    device: str = "cpu"


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    return torch.linalg.norm(obs1_state - obs2_state) < atol


def observations_similar(obs1: TimedObservation, obs2: TimedObservation, atol: float = 1) -> bool:
    """Check if two observations are similar, under a tolerance threshold"""
    obs1_state = obs1.get_observation()["observation.state"]
    obs2_state = obs2.get_observation()["observation.state"]

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)


def send_bytes_in_chunks(
    buffer: bytes,
    message_class: Any,
    log_prefix: str = "",
    silent: bool = True,
    chunk_size: int = 3 * 1024 * 1024,
):
    # NOTE(fracapuano): Partially copied from lerobot.common.transport.utils.send_bytes_in_chunks. Duplication can't be avoided if we
    # don't use a unique class for messages sent (due to the different transfer states sent). Also, I'd want more control over the
    # chunk size as I am using it to send image observations.
    buffer = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = async_inference_pb2.TransferState.TRANSFER_MIDDLE

        if sent_bytes + chunk_size >= size_in_bytes:
            transfer_state = async_inference_pb2.TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = async_inference_pb2.TransferState.TRANSFER_BEGIN

        size_to_read = min(chunk_size, size_in_bytes - sent_bytes)
        chunk = buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(iterator, continue_receiving: Event, log_prefix: str = ""):  # type: ignore
    # NOTE(fracapuano): Partially copied from lerobot.common.transport.utils.receive_bytes_in_chunks. Duplication can't be avoided if we
    # don't use a unique class for messages sent (due to the different transfer states sent). Also, on the server side the logic for receiving
    # is opposite then the HIL-SERL design (my event showcases keeping on running instead of shutdown)
    bytes_buffer = io.BytesIO()
    step = 0

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if not continue_receiving.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == async_inference_pb2.TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0

        elif item.transfer_state == async_inference_pb2.TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")

        elif item.transfer_state == async_inference_pb2.TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            complete_bytes = bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
            return complete_bytes

        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")
