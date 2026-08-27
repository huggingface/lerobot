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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.robots.config import RobotConfig
from lerobot.rollout.inference.factory import (
    InferenceEngineConfig,
    RTCInferenceConfig,
    SyncInferenceConfig,
)

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
class RecordingRGBEncoderConfig:
    """Dependency-free CLI mirror of the RGB video encoder configuration.

    The real encoder is constructed only after recording is enabled, so a normal
    async client does not require PyAV or the dataset extra merely to parse flags.
    """

    vcodec: str = "auto"
    pix_fmt: str = "yuv420p"
    g: int | None = 2
    crf: int | float | None = 30
    preset: int | str | None = None
    fast_decode: int = 0
    video_backend: str = "pyav"
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationRecordingConfig:
    """Client-side camera recording configuration for asynchronous inference."""

    enabled: bool = field(default=False, metadata={"help": "Record RGB camera observations locally"})
    repo_id: str = field(
        default="",
        metadata={"help": "Dataset identifier, for example '<user>/async_camera_recording'"},
    )
    root: str | Path | None = field(
        default=None,
        metadata={"help": "Local dataset directory. Defaults to $HF_LEROBOT_HOME/<repo_id>"},
    )
    resume: bool = field(
        default=False,
        metadata={"help": "Append a new episode to an existing local recording dataset"},
    )
    streaming_encoding: bool = field(
        default=True,
        metadata={"help": "Encode camera frames continuously while the client is running"},
    )
    encoder_queue_maxsize: int = field(
        default=30,
        metadata={"help": "Maximum buffered frames per camera for streaming encoding"},
    )
    encoder_threads: int | None = field(
        default=None,
        metadata={"help": "Threads used by each video encoder. None lets the codec decide"},
    )
    num_image_writer_processes: int = field(
        default=0,
        metadata={"help": "Image-writer processes used when streaming encoding is disabled"},
    )
    num_image_writer_threads_per_camera: int = field(
        default=4,
        metadata={"help": "Image-writer threads per camera when streaming encoding is disabled"},
    )
    rgb_encoder: RecordingRGBEncoderConfig = field(default_factory=RecordingRGBEncoderConfig)

    def __post_init__(self) -> None:
        if self.enabled and not self.repo_id:
            raise ValueError("recording.repo_id cannot be empty when recording is enabled")
        if self.enabled and self.resume and not self.root:
            raise ValueError("recording.root is required when recording.resume is enabled")
        if self.encoder_queue_maxsize <= 0:
            raise ValueError(
                f"recording.encoder_queue_maxsize must be positive, got {self.encoder_queue_maxsize}"
            )
        if self.encoder_threads is not None and self.encoder_threads <= 0:
            raise ValueError(f"recording.encoder_threads must be positive, got {self.encoder_threads}")
        if self.num_image_writer_processes < 0:
            raise ValueError(
                "recording.num_image_writer_processes must be non-negative, "
                f"got {self.num_image_writer_processes}"
            )
        if self.num_image_writer_threads_per_camera < 0:
            raise ValueError(
                "recording.num_image_writer_threads_per_camera must be non-negative, "
                f"got {self.num_image_writer_threads_per_camera}"
            )


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

    # Inference mode. Reuse rollout's config hierarchy so the async CLI exposes
    # the same --inference.type and --inference.rtc.* flags.
    inference: InferenceEngineConfig = field(default_factory=SyncInferenceConfig)

    # Observation transport configuration
    observation_image_compression: str = field(
        default="jpeg",
        metadata={"help": "Image transport codec. Supported values: 'jpeg' and 'none'"},
    )
    jpeg_quality: int = field(
        default=85,
        metadata={"help": "JPEG quality for camera observations (1-100)"},
    )

    # Optional local recording. Keeping the default as None avoids importing
    # dataset/video runtime dependencies for clients that only stream observations.
    recording: ObservationRecordingConfig | None = None

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

        if isinstance(self.inference, RTCInferenceConfig):
            if self.inference.queue_threshold < 0:
                raise ValueError(
                    f"inference.queue_threshold must be non-negative, got {self.inference.queue_threshold}"
                )
            if self.inference.rtc.execution_horizon <= 0:
                raise ValueError(
                    "inference.rtc.execution_horizon must be positive, "
                    f"got {self.inference.rtc.execution_horizon}"
                )
            if self.inference.rtc.execution_horizon > self.actions_per_chunk:
                raise ValueError(
                    "inference.rtc.execution_horizon cannot exceed actions_per_chunk, "
                    f"got {self.inference.rtc.execution_horizon} > {self.actions_per_chunk}"
                )

        if self.observation_image_compression not in {"jpeg", "none"}:
            raise ValueError(
                "observation_image_compression must be either 'jpeg' or 'none', "
                f"got {self.observation_image_compression!r}"
            )

        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError(f"jpeg_quality must be between 1 and 100, got {self.jpeg_quality}")

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
            "inference": {
                "type": self.inference.type,
                **asdict(self.inference),
            },
            "observation_image_compression": self.observation_image_compression,
            "jpeg_quality": self.jpeg_quality,
            "recording": asdict(self.recording) if self.recording is not None else None,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
        }
