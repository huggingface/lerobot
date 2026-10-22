# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from math import degrees, isfinite

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("asimov")
@dataclass
class AsimovConfig(RobotConfig):
    """Asimov 1 locomotion over the Menlo SDK's UDP transport."""

    host: str
    command_port: int = 8850
    # Receive LAN telemetry on all local interfaces; the SDK filters by robot source IP.
    state_host: str = "0.0.0.0"  # nosec B104
    state_port: int = 8851
    version_port: int = 3000
    connect_timeout_s: float = 5.0
    # Reject old observations, independently of the SDK's lost-link detection.
    max_state_age_s: float = 0.5
    # Each command expires unless refreshed. The SDK sends zero when this expires.
    command_timeout_s: float = 0.25
    # Body-frame forward/left limits (m/s), and counter-clockwise yaw (deg/s).
    max_linear_velocity: float = 0.3
    max_lateral_velocity: float = 0.2
    max_angular_velocity: float = 30.0
    # Cameras use the standard LeRobot backends; UDP does not carry onboard video.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.host.strip():
            raise ValueError("host must be the Asimov robot's address.")
        for name in ("command_port", "state_port", "version_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be an integer between 1 and 65535.")
        for name in ("connect_timeout_s", "max_state_age_s", "command_timeout_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        # Keep recorded actions within the firmware's caps, not just the SDK's limits.
        for name, cap in (
            ("max_linear_velocity", 0.4),
            ("max_lateral_velocity", 0.4),
            ("max_angular_velocity", degrees(0.8)),
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= cap:
                raise ValueError(f"{name} must be finite and between 0 and {cap}.")
