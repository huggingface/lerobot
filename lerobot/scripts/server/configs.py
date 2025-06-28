from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from lerobot.common.robots.robot import Robot
from lerobot.configs.types import PolicyFeature
from lerobot.scripts.server.constants import (
    DEFAULT_IDLE_WAIT,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Action chunking configuration
    actions_per_chunk: int = field(default=20, metadata={"help": "Number of actions in each chunk"})

    # Timing configuration
    fps: int = field(default=30, metadata={"help": "Frames per second"})
    idle_wait: float = field(default=DEFAULT_IDLE_WAIT, metadata={"help": "Idle wait time in seconds"})
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

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.idle_wait < 0:
            raise ValueError(f"idle_wait must be non-negative, got {self.idle_wait}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

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
            "actions_per_chunk": self.actions_per_chunk,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "idle_wait": self.idle_wait,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Robot to wrap with async inference capabilities
    robot: Robot = field(metadata={"help": "Robot instance to use"})
    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    # robot.get_observation() returns dict with keys different from the ones expected for recording a dataset/inference
    # The following field helps map these keys into the expected ones through the `build_dataset_frame` dataset's util
    lerobot_features: dict[str, dict] = field(metadata={"help": "Features for dataset recording/inference, in the LeRobot format"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=30, metadata={"help": "Frames per second"})

    aggregate_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = field(
        default=None, metadata={"help": "Function to aggregate actions on overlapping sections"}
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

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if not self.robot:
            raise ValueError("robot cannot be empty")

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
            "chunk_size_threshold": self.chunk_size_threshold,
            "control_frequency": self.control_frequency,
            "robot": self.robot,
            "camera_activation_delay": self.camera_activation_delay,
        }
