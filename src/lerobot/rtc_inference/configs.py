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

from lerobot.configs.types import RTCAttentionSchedule
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

    # Optional observation key remapping (e.g. camera1 -> image)
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.camera1": "observation.images.image",
            "observation.images.camera2": "observation.images.image2",
            "observation.images.camera3": "observation.images.image3",
            "observation.images.camera4": "observation.images.image4",
        },
        metadata={"help": "Optional mapping to align robot observation keys with model input_features"},
    )

    # RTC configuration (used by rtc_inference.policy_server)
    rtc_enabled: bool = field(default=True, metadata={"help": "Enable RTC on server-side policy"})
    rtc_execution_horizon: int = field(
        default=10,
        metadata={"help": "RTC execution horizon for chunk continuity"},
    )
    rtc_max_guidance_weight: float = field(
        default=10.0,
        metadata={"help": "RTC guidance strength"},
    )
    rtc_prefix_attention_schedule: RTCAttentionSchedule = field(
        default=RTCAttentionSchedule.EXP,
        metadata={"help": "RTC prefix attention schedule"},
    )
    rtc_debug: bool = field(default=False, metadata={"help": "Enable RTC debug tracking"})
    rtc_debug_maxlen: int = field(default=100, metadata={"help": "RTC debug buffer size"})

    # Fixed inference delay in timesteps for RTC.
    # If None, server estimates from observed inference latency.
    inference_delay_steps: int | None = field(
        default=None,
        metadata={"help": "Fixed inference delay steps for RTC (optional)"},
    )

    # Optional XVLA domain override at inference.
    xvla_domain_id: int | None = field(
        default=None,
        metadata={"help": "Optional XVLA domain_id override"},
    )

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
    obs_timestep_independent: bool = field(
        default=True,
        metadata={"help": "Use a monotonic observation timestep independent from action timestep"},
    )
    image_compress_enable: bool = field(
        default=False,
        metadata={"help": "Enable JPEG compression for transport only (no resize)"},
    )
    image_compress_quality: int = field(
        default=90,
        metadata={"help": "JPEG quality for transport compression (1-100)"},
    )
    interpolation_multiplier: int = field(
        default=1,
        metadata={"help": "Control-rate multiplier for action interpolation (1 = disabled)"},
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

        if not isinstance(self.obs_timestep_independent, bool):
            raise ValueError("obs_timestep_independent must be a boolean")

        if not isinstance(self.image_compress_enable, bool):
            raise ValueError("image_compress_enable must be a boolean")

        if not 1 <= self.image_compress_quality <= 100:
            raise ValueError(
                f"image_compress_quality must be between 1 and 100, got {self.image_compress_quality}"
            )

        if self.interpolation_multiplier <= 0:
            raise ValueError(
                f"interpolation_multiplier must be positive, got {self.interpolation_multiplier}"
            )

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.rtc_execution_horizon <= 0:
            raise ValueError(f"rtc_execution_horizon must be positive, got {self.rtc_execution_horizon}")

        if self.rtc_max_guidance_weight <= 0:
            raise ValueError(f"rtc_max_guidance_weight must be positive, got {self.rtc_max_guidance_weight}")

        if self.rtc_debug_maxlen <= 0:
            raise ValueError(f"rtc_debug_maxlen must be positive, got {self.rtc_debug_maxlen}")

        if self.inference_delay_steps is not None and self.inference_delay_steps < 0:
            raise ValueError(f"inference_delay_steps must be >= 0, got {self.inference_delay_steps}")

        if not isinstance(self.rename_map, dict):
            raise ValueError("rename_map must be a dictionary")

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
            "obs_timestep_independent": self.obs_timestep_independent,
            "image_compress_enable": self.image_compress_enable,
            "image_compress_quality": self.image_compress_quality,
            "interpolation_multiplier": self.interpolation_multiplier,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "rename_map": self.rename_map,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
            "rtc_enabled": self.rtc_enabled,
            "rtc_execution_horizon": self.rtc_execution_horizon,
            "rtc_max_guidance_weight": self.rtc_max_guidance_weight,
            "rtc_prefix_attention_schedule": self.rtc_prefix_attention_schedule,
            "rtc_debug": self.rtc_debug,
            "rtc_debug_maxlen": self.rtc_debug_maxlen,
            "inference_delay_steps": self.inference_delay_steps,
            "xvla_domain_id": self.xvla_domain_id,
        }
