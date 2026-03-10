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

"""
Data classes for remote RTC policy inference.

These classes define the communication protocol between RTC policy servers
and clients (robots, simulations, or evaluation scripts).
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig


@dataclass
class RTCRemotePolicyConfig:
    """Configuration sent by client to initialize policy on server.

    This is sent from clients to the server when establishing a connection,
    telling the server which policy to load and how to configure it.
    """

    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, Any]
    rtc_config: RTCConfig | None = None
    device: str = "cuda"
    rename_map: dict[str, str] = field(default_factory=dict)
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    chunk_size: int | None = None
    n_action_steps: int | None = None


@dataclass
class RTCObservationData:
    """Observation data with RTC parameters sent from client to server.

    Contains the observation dict along with RTC-specific parameters
    needed for inference:
    - inference_delay: Number of steps the inference is expected to take
    - prev_chunk_left_over: Unconsumed actions from previous chunk for RTC guidance
    - execution_horizon: How far into the future to plan
    """

    observation: dict[str, Any]
    timestamp: float
    timestep: int
    inference_delay: int
    prev_chunk_left_over: torch.Tensor | None
    execution_horizon: int


@dataclass
class RTCActionData:
    """Action data returned from server to client.

    Contains both the postprocessed actions (ready for robot execution)
    and the original actions (for RTC left-over tracking in the action queue).
    """

    actions: torch.Tensor  # Postprocessed actions for robot
    original_actions: torch.Tensor  # Original actions for RTC left-over tracking
    timestamp: float
    timestep: int
    timing: "RTCTimingData | None" = None


@dataclass
class RTCTimingData:
    """Timing breakdown for one remote inference request."""

    queue_wait_ms: float | None = None
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    pickle_ms: float | None = None
    total_ms: float | None = None
