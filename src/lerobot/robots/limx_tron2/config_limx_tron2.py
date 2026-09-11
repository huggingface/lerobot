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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

DEFAULT_INIT_JOINTS = (
    0.026899,
    0.2612,
    -0.02709991,
    -1.5477003,
    0.265,
    0.0180999,
    -0.0614999,
    0.008999,
    -0.269,
    0.02069998,
    -1.5567001,
    -0.254,
    -0.02309972,
    0.06469989,
)
DEFAULT_INIT_HEAD = (1.0467, -0.0139998)


@RobotConfig.register_subclass("limx_tron2")
@dataclass
class LimxTron2RobotConfig(RobotConfig):
    robot_ip: str = "127.0.0.1"
    port: int = 5000
    state_queue_maxlen: int = 7
    polling_rate: float = 200.0
    connection_timeout: float = 5.0
    state_timeout: float = 1.0
    publish_rate: float = 300.0
    control_frequency: float = 30.0
    init_joints: list[float] = field(default_factory=lambda: list(DEFAULT_INIT_JOINTS))
    init_head: list[float] = field(default_factory=lambda: list(DEFAULT_INIT_HEAD))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    observation_source: str = "legacy"
    bridge_host: str = ""
    bridge_ws_path: str = "/bridge/ws"
    bridge_image_max_fps: int = 0
    bridge_align_max_delay_ms: int = 200
    bridge_verify_tls: bool = False
    bridge_connection_timeout: float = 5.0
    bridge_timeout: float = 1.0
    bridge_camera_width: int = 640
    bridge_camera_height: int = 480

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("state_timeout", "publish_rate", "control_frequency"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if len(self.init_joints) != 14:
            raise ValueError(f"init_joints must have 14 elements, got {len(self.init_joints)}")
        if len(self.init_head) != 2:
            raise ValueError(f"init_head must have 2 elements, got {len(self.init_head)}")
        if self.observation_source not in {"legacy", "bridge"}:
            raise ValueError("observation_source must be 'legacy' or 'bridge'")
        if self.observation_source == "bridge" and (
            not self.bridge_host or "BRIDGE_HOST" in self.bridge_host
        ):
            raise ValueError("bridge_host must be set to the real Bridge WebSocket URL")
        if self.bridge_connection_timeout <= 0 or self.bridge_timeout <= 0:
            raise ValueError("bridge timeouts must be greater than zero")
        if self.bridge_camera_width <= 0 or self.bridge_camera_height <= 0:
            raise ValueError("bridge camera dimensions must be greater than zero")
