#!/usr/bin/env python

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


@RobotConfig.register_subclass("unitree_go2")
@dataclass
class UnitreeGo2Config(RobotConfig):
    """Configuration for the Unitree Go2 quadruped (EDU).

    The host machine talks DDS directly to the dog over Ethernet/WiFi via
    ``unitree_sdk2py`` — no onboard companion computer is required. Actions
    are high-level sport-mode body velocities; observations are sport-mode
    odometry plus the dog's built-in front camera.
    """

    # Network interface on the host that is wired/bridged to the Go2
    # (the dog lives on 192.168.123.x when connected over Ethernet).
    network_interface: str = "eth0"

    # DDS domain id (0 for a stock Go2).
    domain_id: int = 0

    # Safety clamps applied in send_action() before commands reach the dog.
    # The Go2 accepts far more (vx up to ~3.7 m/s) — keep indoor-sane defaults.
    max_x_vel: float = 1.0  # m/s, body forward
    max_y_vel: float = 0.5  # m/s, body left
    max_theta_vel: float = 1.5  # rad/s, CCW about z-up

    # Built-in front camera, served through the SDK VideoClient.
    use_front_camera: bool = True
    front_camera_width: int = 1280
    front_camera_height: int = 720

    # Send BalanceStand once on connect so the dog is ready to walk.
    stand_on_connect: bool = True

    # Additional external cameras (standard LeRobot camera configs).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
