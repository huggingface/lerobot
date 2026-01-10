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

"""Configuration classes for the improved async inference implementation.

These configurations follow the latency-adaptive async inference algorithm
with proper SPSC mailboxes, Jacobson-Karels latency estimation,
cool-down mechanism, and freshest-observation-wins merging.
"""

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from .constants import DEFAULT_FPS, DEFAULT_OBS_QUEUE_TIMEOUT


# =============================================================================
# Robot Client Configuration
# =============================================================================


@dataclass
class RobotClientImprovedConfig:
    """Configuration for the improved latency-adaptive robot client.

    This configuration follows the latency-adaptive async inference algorithm
    from the paper, with proper SPSC mailboxes, Jacobson-Karels latency estimation,
    cool-down mechanism, and freshest-observation-wins merging.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use (e.g., 'act', 'smolvla')"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Actions per chunk (should be <= policy's max action horizon)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk (H in the paper)"})

    # Task instruction for the robot
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration (for policy inference on server)
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Control frequency
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Control loop frequency in Hz"})

    # Latency-adaptive parameters
    epsilon: int = field(default=5, metadata={"help": "Safety margin in action steps (ε)"})
    latency_estimator_type: str = field(
        default="jk",
        metadata={"help": "Latency estimator type: 'jk' (Jacobson-Karels) or 'max_last_10'"},
    )
    latency_alpha: float = field(
        default=0.125, metadata={"help": "Jacobson-Karels smoothing factor for RTT mean"}
    )
    latency_beta: float = field(
        default=0.25, metadata={"help": "Jacobson-Karels smoothing factor for RTT deviation"}
    )
    latency_k: float = field(
        default=4.0, metadata={"help": "Jacobson-Karels scaling factor for deviation (K)"}
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size after stopping"}
    )

    # RTC (client-driven, server-side inpainting; flow policies only)
    rtc_enabled: bool = field(
        default=True,
        metadata={"help": "Enable RTC-style inpainting on the policy server (flow policies only)"},
    )
    rtc_execution_horizon: int = field(
        default=10,
        metadata={"help": "RTC execution horizon (prefix blending horizon)"},
    )
    rtc_max_guidance_weight: float = field(
        default=1.0,
        metadata={"help": "RTC max guidance weight (clamp)"},
    )
    rtc_prefix_attention_schedule: str = field(
        default="linear",
        metadata={"help": "RTC prefix attention schedule: zeros|ones|linear|exp"},
    )

    # Diagnostics configuration (off by default)
    diagnostics_enabled: bool = field(
        default=False,
        metadata={"help": "Enable periodic diagnostics logs (timing, latency jitter, action deltas)"},
    )
    diagnostics_interval_s: float = field(
        default=2.0, metadata={"help": "How often to emit a diagnostics summary (seconds)"}
    )
    diagnostics_window_s: float = field(
        default=10.0, metadata={"help": "Rolling window for diagnostics stats (seconds)"}
    )

    # Control-loop clocking (optional)
    control_use_deadline_clock: bool = field(
        default=True,
        metadata={"help": "Use a deadline-based control clock (reduces jitter under overruns)"},
    )

    # Observation sender robustness
    obs_fallback_on_failure: bool = field(
        default=True,
        metadata={
            "help": "If robot observation capture fails, reuse the last good observation to avoid stalling"
        },
    )
    obs_fallback_max_age_s: float = field(
        default=2.0,
        metadata={"help": "Max age (seconds) of the last good observation that may be reused on failure"},
    )

    # Simulation mode (for experiments)
    simulation_mode: bool = field(
        default=False,
        metadata={"help": "Use mock robot instead of real hardware (for experiments)"},
    )
    cooldown_enabled: bool = field(
        default=True,
        metadata={"help": "Enable cooldown mechanism (set False for classic async baseline)"},
    )

    # Drop injection (for experiments)
    drop_obs_p: float = field(
        default=0.0,
        metadata={"help": "Random probability of dropping an observation (0.0-1.0)"},
    )
    drop_obs_burst_pattern: str | None = field(
        default=None,
        metadata={"help": "Deterministic drop burst pattern, e.g. '1s@20s' = drop for 1s every 20s"},
    )
    drop_action_p: float = field(
        default=0.0,
        metadata={"help": "Random probability of dropping an action chunk (0.0-1.0)"},
    )
    drop_action_burst_pattern: str | None = field(
        default=None,
        metadata={"help": "Deterministic drop burst pattern, e.g. '1s@20s' = drop for 1s every 20s"},
    )

    # Experiment metrics (CSV export)
    experiment_metrics_path: str | None = field(
        default=None,
        metadata={"help": "Path to write experiment metrics CSV (None = disabled)"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1.0 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.latency_estimator_type not in ("jk", "max_last_10"):
            raise ValueError(
                f"latency_estimator_type must be 'jk' or 'max_last_10', got {self.latency_estimator_type}"
            )
        if self.diagnostics_interval_s <= 0:
            raise ValueError(f"diagnostics_interval_s must be positive, got {self.diagnostics_interval_s}")
        if self.diagnostics_window_s <= 0:
            raise ValueError(f"diagnostics_window_s must be positive, got {self.diagnostics_window_s}")
        if self.obs_fallback_max_age_s <= 0:
            raise ValueError(f"obs_fallback_max_age_s must be positive, got {self.obs_fallback_max_age_s}")
        if self.rtc_execution_horizon <= 0:
            raise ValueError(f"rtc_execution_horizon must be positive, got {self.rtc_execution_horizon}")
        if self.rtc_max_guidance_weight <= 0:
            raise ValueError(f"rtc_max_guidance_weight must be positive, got {self.rtc_max_guidance_weight}")


# =============================================================================
# Policy Server Configuration
# =============================================================================


@dataclass
class PolicyServerImprovedConfig:
    """Configuration for the improved PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    following the 2-thread model from the latency-adaptive async inference paper.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second (control frequency)"})

    # Observation queue timeout
    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT,
        metadata={"help": "Timeout for observation queue in seconds"},
    )

    # Mock policy configuration (for simulation experiments)
    mock_policy: bool = field(
        default=False,
        metadata={"help": "Use mock policy instead of real model (for experiments)"},
    )
    mock_inference_delay_ms: float = field(
        default=0.0,
        metadata={"help": "Fixed delay in milliseconds to add to mock inference"},
    )
    mock_inference_spike_pattern: str | None = field(
        default=None,
        metadata={"help": "Spike pattern e.g. '+2000ms@30s/1s' = +2s spike every 30s lasting 1s"},
    )
    mock_action_dim: int = field(
        default=6,
        metadata={"help": "Action dimension for mock policy output"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1.0 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")
