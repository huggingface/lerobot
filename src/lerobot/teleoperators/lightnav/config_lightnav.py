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

"""Configuration for driving a mobile base from a LightNav-0 navigation server."""

from dataclasses import dataclass
from math import isfinite

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lightnav")
@dataclass
class LightNavTeleopConfig(TeleoperatorConfig):
    """Stream one camera to ``lightnav-serve`` and follow the waypoints it predicts.

    Linear speeds are in m/s and angular speeds in deg/s, like ``keyboard_base``.
    """

    # WebSocket URL of a running `lightnav-serve` (started with `--task vln` for navigation).
    server_url: str = "ws://127.0.0.1:8050"
    # Natural-language navigation instruction sent with every frame.
    instruction: str = ""
    # Observation key of the RGB camera the model sees, e.g. LeKiwi's `front`.
    camera_key: str = "front"
    # Duration of one predicted waypoint row. LightNav's reference controllers use 0.1 s.
    waypoint_dt_s: float = 0.1
    # Speed caps. A faster plan is slowed down uniformly, so the robot keeps the planned path.
    max_linear_speed: float = 0.3
    max_angular_speed: float = 90.0
    # Command lateral motion. Set to false for a base that cannot strafe.
    holonomic: bool = True
    jpeg_quality: int = 90
    connect_timeout_s: float = 10.0
    response_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("LightNav needs a navigation instruction: set --base_teleop.instruction.")
        if not self.server_url.startswith(("ws://", "wss://")):
            raise ValueError(f"LightNav server_url must start with ws:// or wss://, got {self.server_url!r}")
        for name in (
            "waypoint_dt_s",
            "max_linear_speed",
            "max_angular_speed",
            "connect_timeout_s",
            "response_timeout_s",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"LightNav {name} must be finite and positive, got {value}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"LightNav jpeg_quality must be in [1, 100], got {self.jpeg_quality}")
