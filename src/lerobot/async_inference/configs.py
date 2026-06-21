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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Supervisor monitor configuration (optional event-triggered replanning)
    supervisor_enabled: bool = field(
        default=False, metadata={"help": "Enable supervisor camera monitor for event-triggered replanning"}
    )
    supervisor_camera: str = field(
        default="overall", metadata={"help": "Camera key used by the supervisor monitor"}
    )
    supervisor_poll_fps: int = field(default=20, metadata={"help": "Supervisor monitor polling rate (Hz)"})
    supervisor_cooldown_s: float = field(
        default=1.0, metadata={"help": "Minimum seconds between supervisor triggers"}
    )
    supervisor_motion_threshold: float = field(
        default=0.02, metadata={"help": "Fraction of frame pixels in motion to fire a trigger"}
    )
    supervisor_detector_type: str = field(
        default="motion", metadata={"help": "Supervisor detector type. Options: motion, red_cube_speed"}
    )
    supervisor_slow_speed_px_s: float = field(
        default=40.0,
        metadata={"help": "Red cube speed mapped to the minimum adaptive replan threshold"},
    )
    supervisor_fast_speed_px_s: float = field(
        default=200.0,
        metadata={"help": "Red cube speed mapped to the maximum adaptive replan threshold"},
    )
    supervisor_urgent_speed_px_s: float = field(
        default=250.0, metadata={"help": "Red cube speed that triggers immediate replanning"}
    )
    supervisor_min_chunk_size_threshold: float = field(
        default=0.25, metadata={"help": "Adaptive threshold used when the red cube is slow"}
    )
    supervisor_max_chunk_size_threshold: float = field(
        default=0.75, metadata={"help": "Adaptive threshold used when the red cube is fast"}
    )
    supervisor_red_hue_tolerance_deg: float = field(
        default=20.0, metadata={"help": "HSV hue tolerance around red for the cube mask"}
    )
    supervisor_red_saturation_min: float = field(
        default=0.45, metadata={"help": "Minimum HSV saturation for the red cube mask"}
    )
    supervisor_red_value_min: float = field(
        default=0.25, metadata={"help": "Minimum HSV value for the red cube mask"}
    )
    supervisor_red_min_area_ratio: float = field(
        default=0.001, metadata={"help": "Minimum frame area occupied by the red cube mask"}
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.supervisor_enabled:
            if self.supervisor_poll_fps <= 0:
                raise ValueError(f"supervisor_poll_fps must be positive, got {self.supervisor_poll_fps}")
            if self.supervisor_cooldown_s < 0:
                raise ValueError(
                    f"supervisor_cooldown_s must be non-negative, got {self.supervisor_cooldown_s}"
                )
            if not 0 < self.supervisor_motion_threshold <= 1:
                raise ValueError(
                    f"supervisor_motion_threshold must be in (0, 1], got {self.supervisor_motion_threshold}"
                )
            if self.supervisor_detector_type not in {"motion", "red_cube_speed"}:
                raise ValueError(
                    "supervisor_detector_type must be one of {'motion', 'red_cube_speed'}, "
                    f"got {self.supervisor_detector_type}"
                )
            if self.supervisor_slow_speed_px_s < 0:
                raise ValueError(
                    f"supervisor_slow_speed_px_s must be non-negative, got {self.supervisor_slow_speed_px_s}"
                )
            if self.supervisor_fast_speed_px_s <= self.supervisor_slow_speed_px_s:
                raise ValueError(
                    "supervisor_fast_speed_px_s must be greater than supervisor_slow_speed_px_s, "
                    f"got {self.supervisor_fast_speed_px_s} <= {self.supervisor_slow_speed_px_s}"
                )
            if self.supervisor_urgent_speed_px_s < self.supervisor_slow_speed_px_s:
                raise ValueError(
                    "supervisor_urgent_speed_px_s must be at least supervisor_slow_speed_px_s, "
                    f"got {self.supervisor_urgent_speed_px_s} < {self.supervisor_slow_speed_px_s}"
                )
            if not 0 <= self.supervisor_min_chunk_size_threshold <= 1:
                raise ValueError(
                    "supervisor_min_chunk_size_threshold must be between 0 and 1, "
                    f"got {self.supervisor_min_chunk_size_threshold}"
                )
            if not 0 <= self.supervisor_max_chunk_size_threshold <= 1:
                raise ValueError(
                    "supervisor_max_chunk_size_threshold must be between 0 and 1, "
                    f"got {self.supervisor_max_chunk_size_threshold}"
                )
            if self.supervisor_min_chunk_size_threshold > self.supervisor_max_chunk_size_threshold:
                raise ValueError(
                    "supervisor_min_chunk_size_threshold must be <= supervisor_max_chunk_size_threshold, "
                    f"got {self.supervisor_min_chunk_size_threshold} > "
                    f"{self.supervisor_max_chunk_size_threshold}"
                )
            if not 0 <= self.supervisor_red_hue_tolerance_deg <= 180:
                raise ValueError(
                    "supervisor_red_hue_tolerance_deg must be between 0 and 180, "
                    f"got {self.supervisor_red_hue_tolerance_deg}"
                )
            if not 0 <= self.supervisor_red_saturation_min <= 1:
                raise ValueError(
                    "supervisor_red_saturation_min must be between 0 and 1, "
                    f"got {self.supervisor_red_saturation_min}"
                )
            if not 0 <= self.supervisor_red_value_min <= 1:
                raise ValueError(
                    f"supervisor_red_value_min must be between 0 and 1, got {self.supervisor_red_value_min}"
                )
            if not 0 < self.supervisor_red_min_area_ratio <= 1:
                raise ValueError(
                    f"supervisor_red_min_area_ratio must be in (0, 1], got {self.supervisor_red_min_area_ratio}"
                )

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
            "supervisor_enabled": self.supervisor_enabled,
            "supervisor_camera": self.supervisor_camera,
            "supervisor_poll_fps": self.supervisor_poll_fps,
            "supervisor_cooldown_s": self.supervisor_cooldown_s,
            "supervisor_motion_threshold": self.supervisor_motion_threshold,
            "supervisor_detector_type": self.supervisor_detector_type,
            "supervisor_slow_speed_px_s": self.supervisor_slow_speed_px_s,
            "supervisor_fast_speed_px_s": self.supervisor_fast_speed_px_s,
            "supervisor_urgent_speed_px_s": self.supervisor_urgent_speed_px_s,
            "supervisor_min_chunk_size_threshold": self.supervisor_min_chunk_size_threshold,
            "supervisor_max_chunk_size_threshold": self.supervisor_max_chunk_size_threshold,
            "supervisor_red_hue_tolerance_deg": self.supervisor_red_hue_tolerance_deg,
            "supervisor_red_saturation_min": self.supervisor_red_saturation_min,
            "supervisor_red_value_min": self.supervisor_red_value_min,
            "supervisor_red_min_area_ratio": self.supervisor_red_min_area_ratio,
        }
