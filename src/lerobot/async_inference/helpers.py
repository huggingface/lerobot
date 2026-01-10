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
import logging.handlers
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.robot import Robot
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.utils import init_logging

Action = Any

# observation as received from the robot
RawObservation = dict[str, Any]

# observation as those recorded in LeRobot dataset (keys are different)
LeRobotObservation = dict[str, Any]

# observation, ready for policy inference (image keys resized)
Observation = dict[str, Any]


def _validate_feature_names(features: dict[str, dict]) -> None:
    """Validate that feature names do not contain invalid characters.

    We keep this local to avoid importing `lerobot.datasets.utils` (which is heavyweight).
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Lightweight version of `lerobot.datasets.utils.hw_to_dataset_features`.

    The async inference client only needs a small subset of dataset feature logic, and importing
    the full dataset stack (datasets/pandas/pyarrow/torchvision/...) is very expensive on small
    devices like a Raspberry Pi.
    """
    features: dict[str, dict] = {}

    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def build_dataset_frame(ds_features: dict[str, dict], values: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    """Lightweight version of `lerobot.datasets.utils.build_dataset_frame`."""
    frame: dict[str, np.ndarray] = {}
    for key, ft in ds_features.items():
        if not key.startswith(prefix):
            continue
        if ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
    return frame


def visualize_action_queue_size(action_queue_size: list[int]) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.set_ylim(0, max(action_queue_size) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.show()


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: Any, resize_dims: tuple[int, int, int]) -> Any:
    import torch

    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


# TODO(Steven): Consider implementing a pipeline step for this
def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    import torch

    observation = {}

    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0)
        else:
            observation[k] = v

    return observation


def prepare_image(image: Any) -> Any:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    import torch

    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> Any:
    """Extract the state from a raw observation."""
    import torch

    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> Any:
    """Extract the images from a raw observation."""
    import torch

    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """Make a lerobot observation from a raw observation."""
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    import torch

    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. Greps all observation.images.<> keys
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    # Turns the image features to (C, H, W) with H, W matching the policy image features.
    # This reduces the resolution of the images
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


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
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    # Client-driven RTC configuration (optional; server may ignore if policy doesn't support RTC)
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float | None = None  # None = use num_flow_matching_steps (Alex Soare opt)
    rtc_prefix_attention_schedule: str = "linear"
    rtc_sigma_d: float = 1.0  # Prior variance (0.2 = stronger guidance, 1.0 = original RTC)
    rtc_full_trajectory_alignment: bool = False  # Skip gradient for faster/smoother transitions

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Back-compat for pickles created before RTC fields existed."""
        self.__dict__.update(state)
        self.__dict__.setdefault("rtc_enabled", False)
        self.__dict__.setdefault("rtc_execution_horizon", 10)
        self.__dict__.setdefault("rtc_max_guidance_weight", None)  # Default to auto (Alex Soare opt)
        self.__dict__.setdefault("rtc_prefix_attention_schedule", "linear")
        self.__dict__.setdefault("rtc_sigma_d", 1.0)
        self.__dict__.setdefault("rtc_full_trajectory_alignment", False)


def _compare_observation_states(obs1_state: Any, obs2_state: Any, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    import torch

    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """Check if two observations are similar, under a tolerance threshold. Measures distance between
    observations as the difference in joint-space between the two observations.

    NOTE(fracapuano): This is a very simple check, and it is enough for the current use case.
    An immediate next step is to use (fast) perceptual difference metrics comparing some camera views,
    to surpass this joint-space similarity check.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
