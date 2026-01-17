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
from .utils.simulation import DropConfig, SpikeDelayConfig


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
    s_min: int = field(
        default=30,
        metadata={
            "help": "Minimum execution horizon in action steps (s_min from RTC paper). "
            "Trigger inference when schedule_size <= s_min. "
            "Effective execution horizon is max(s_min, latency_steps)."
        },
    )
    epsilon: int = field(
        default=1,
        metadata={
            "help": "Cooldown buffer in action steps. "
            "After triggering inference, cooldown is set to latency_steps + epsilon. "
            "Small values (1-2) prevent over-triggering without adding significant delay."
        },
    )
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
        default=1.5, metadata={"help": "Jacobson-Karels scaling factor for deviation (K)"}
    )
    latency_prime_count: int = field(
        default=5,
        metadata={"help": "Number of priming rounds for latency estimation (0 to disable)"},
    )
    latency_prime_timeout_s: float = field(
        default=5.0,
        metadata={"help": "Timeout in seconds for each latency priming round"},
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
    rtc_max_guidance_weight: float | None = field(
        default=None,
        metadata={
            "help": "RTC max guidance weight (clamp). If None, uses num_flow_matching_steps "
            "(Alex Soare optimization: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)"
        },
    )
    rtc_prefix_attention_schedule: str = field(
        default="linear",
        metadata={"help": "RTC prefix attention schedule: zeros|ones|linear|exp"},
    )
    rtc_sigma_d: float = field(
        default=0.2,
        metadata={
            "help": "RTC prior variance σ_d. Lower values (e.g., 0.2) give stronger guidance "
            "and smoother transitions. 1.0 = original RTC behavior. "
            "(Alex Soare optimization: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)"
        },
    )
    rtc_full_trajectory_alignment: bool = field(
        default=False,
        metadata={
            "help": "Skip gradient computation in RTC and use error directly. "
            "Faster and smoother when distance between chunks is small."
        },
    )
    num_flow_matching_steps: int | None = field(
        default=8,
        metadata={
            "help": "Override for number of flow matching denoising steps. "
            "If None, uses the policy's default (e.g., 10 for PI0/SmolVLA). "
            "Higher values = smoother but slower inference. "
            "(Alex Soare optimization: Beta should scale with n)"
        },
    )

    # Diagnostics configuration (off by default)
    diagnostics_enabled: bool = field(
        default=True,
        metadata={"help": "Enable periodic diagnostics logs (timing, latency jitter, action deltas)"},
    )
    diagnostics_interval_s: float = field(
        default=2.0, metadata={"help": "How often to emit a diagnostics summary (seconds)"}
    )
    diagnostics_window_s: float = field(
        default=10.0, metadata={"help": "Rolling window for diagnostics stats (seconds)"}
    )

    # Trajectory visualization (sends data to policy server via gRPC)
    trajectory_viz_enabled: bool = field(
        default=True,
        metadata={"help": "Enable sending trajectory data to policy server for visualization"},
    )
    trajectory_viz_ws_url: str = field(
        default="ws://localhost:8089",
        metadata={"help": "WebSocket URL for trajectory visualization server (for executed actions)"},
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
    inference_reset_mode: str = field(
        default="cooldown",
        metadata={
            "help": "Mode for resetting inference readiness: "
            "'cooldown' (default) decrements each tick and allows recovery from drops; "
            "'merge_reset' resets only when actions are merged (RTC-style, stalls on drops)"
        },
    )

    # Drop injection (for experiments)
    drop_obs_config: DropConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for observation drop injection. "
            "Example: DropConfig(random_drop_p=0.05) or DropConfig(burst_period_s=20, burst_duration_s=1)"
        },
    )
    drop_action_config: DropConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for action chunk drop injection. "
            "Example: DropConfig(random_drop_p=0.05) or DropConfig(burst_period_s=20, burst_duration_s=1)"
        },
    )

    # Spike injection (for experiments, passed to server)
    # List of dicts: [{"start_s": 5.0, "delay_ms": 2000}, ...]
    spikes: list[dict] = field(
        default_factory=list,
        metadata={
            "help": "Explicit spike events as list of dicts. "
            "Example: [{'start_s': 5, 'delay_ms': 2000}, {'start_s': 15, 'delay_ms': 1000}]"
        },
    )

    # Experiment metrics (CSV export)
    experiment_metrics_path: str | None = field(
        default=None,
        metadata={"help": "Path to write experiment metrics CSV (None = disabled)"},
    )
    # Action smoothing to reduce policy jitter / servo hunting
    # Modes: "none", "adaptive_lowpass", "hold_stable"
    action_filter_mode: str = field(
        default="none",
        metadata={
            "help": "Action filtering mode: "
            "'none' = no filtering, "
            "'adaptive_lowpass' = IIR filter with adaptive alpha based on delta magnitude, "
            "'hold_stable' = hold previous action when delta is below threshold (eliminates jitter)"
        },
    )
    action_filter_alpha_min: float = field(
        default=0.1,
        metadata={
            "help": "Low-pass filter alpha for small deltas (heavy smoothing). "
            "Used when action delta is below deadband threshold. Range: (0, 1]. "
            "Lower = more smoothing. 0.1 gives strong attenuation of high-freq jitter."
        },
    )
    action_filter_alpha_max: float = field(
        default=0.5,
        metadata={
            "help": "Low-pass filter alpha for large deltas (faster response). "
            "Used when action delta exceeds deadband threshold. Range: (0, 1]"
        },
    )
    action_filter_deadband: float = field(
        default=0.05,
        metadata={
            "help": "Deadband threshold in action units (radians for joints). "
            "For 'adaptive_lowpass': deltas below this get alpha_min, above get alpha_max. "
            "For 'hold_stable': deltas below this are ignored entirely. "
            "Default 0.05 rad ≈ 3 degrees."
        },
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
        if self.s_min <= 0:
            raise ValueError(f"s_min must be positive, got {self.s_min}")
        if self.s_min >= self.actions_per_chunk:
            raise ValueError(
                f"s_min must be < actions_per_chunk, got {self.s_min} >= {self.actions_per_chunk}"
            )
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.latency_estimator_type not in ("jk", "max_last_10", "fixed"):
            raise ValueError(
                f"latency_estimator_type must be 'jk', 'max_last_10', or 'fixed', got {self.latency_estimator_type}"
            )
        if self.inference_reset_mode not in ("cooldown", "merge_reset"):
            raise ValueError(
                f"inference_reset_mode must be 'cooldown' or 'merge_reset', got {self.inference_reset_mode}"
            )
        if self.latency_prime_count < 0:
            raise ValueError(f"latency_prime_count must be non-negative, got {self.latency_prime_count}")
        if self.latency_prime_timeout_s <= 0:
            raise ValueError(f"latency_prime_timeout_s must be positive, got {self.latency_prime_timeout_s}")
        if self.diagnostics_interval_s <= 0:
            raise ValueError(f"diagnostics_interval_s must be positive, got {self.diagnostics_interval_s}")
        if self.diagnostics_window_s <= 0:
            raise ValueError(f"diagnostics_window_s must be positive, got {self.diagnostics_window_s}")
        if self.obs_fallback_max_age_s <= 0:
            raise ValueError(f"obs_fallback_max_age_s must be positive, got {self.obs_fallback_max_age_s}")
        if self.rtc_max_guidance_weight is not None and self.rtc_max_guidance_weight <= 0:
            raise ValueError(f"rtc_max_guidance_weight must be positive or None, got {self.rtc_max_guidance_weight}")
        if self.rtc_sigma_d <= 0:
            raise ValueError(f"rtc_sigma_d must be positive, got {self.rtc_sigma_d}")
        if self.num_flow_matching_steps is not None and self.num_flow_matching_steps <= 0:
            raise ValueError(f"num_flow_matching_steps must be positive or None, got {self.num_flow_matching_steps}")
        if self.action_filter_mode not in ("none", "adaptive_lowpass", "hold_stable"):
            raise ValueError(
                f"action_filter_mode must be 'none', 'adaptive_lowpass', or 'hold_stable', "
                f"got {self.action_filter_mode}"
            )
        if self.action_filter_alpha_min <= 0 or self.action_filter_alpha_min > 1:
            raise ValueError(f"action_filter_alpha_min must be in (0, 1], got {self.action_filter_alpha_min}")
        if self.action_filter_alpha_max <= 0 or self.action_filter_alpha_max > 1:
            raise ValueError(f"action_filter_alpha_max must be in (0, 1], got {self.action_filter_alpha_max}")
        if self.action_filter_deadband < 0:
            raise ValueError(f"action_filter_deadband must be non-negative, got {self.action_filter_deadband}")


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
    mock_spike_config: SpikeDelayConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for mock inference latency spikes. "
            "Example: SpikeDelayConfig.from_dicts([{'start_s': 5, 'delay_ms': 2000}])"
        },
    )
    mock_action_dim: int = field(
        default=6,
        metadata={"help": "Action dimension for mock policy output"},
    )

    # RTC action chunk cache (for server-side inpainting with raw actions)
    rtc_chunk_cache_size: int = field(
        default=10,
        metadata={"help": "Number of recent raw action chunks to cache for RTC inpainting"},
    )

    # Trajectory visualization (receives data from robot client via gRPC)
    trajectory_viz_enabled: bool = field(
        default=True,
        metadata={"help": "Enable trajectory visualization server (HTTP + WebSocket)"},
    )
    trajectory_viz_http_port: int = field(
        default=8088,
        metadata={"help": "HTTP port for trajectory visualization web page"},
    )
    trajectory_viz_ws_port: int = field(
        default=8089,
        metadata={"help": "WebSocket port for trajectory data streaming"},
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
