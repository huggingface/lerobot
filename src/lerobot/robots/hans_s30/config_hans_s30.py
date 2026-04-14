#!/usr/bin/env python

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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class HansS30Config:
    """Base configuration for Hans Robot S30 6-DOF arm.

    The S30 communicates via TCP socket (port 10003) and XML-RPC (port 20000).
    No USB serial port is required.

    Args:
        ip: IPv4 address of the Hans controller box.
        port: TCP port for CPS motion commands (default: 10003).
        box_id: Controller box ID (default: 0).
        robot_id: Robot ID inside the controller (default: 0).
        velocity: Default joint-motion velocity in deg/s (default: 50).
            Must be strictly less than ``acc`` (controller enforces this).
        acc: Default joint-motion acceleration in deg/s² (default: 100).
            Must be strictly greater than ``velocity``.
        speed_override: Global speed override ratio, range [0.01, 1.0] (default: 0.5).
        tcp_name: Name of the tool-coordinate frame configured on the controller.
        ucs_name: Name of the user-coordinate frame configured on the controller.
        electrify_wait_s: Seconds to wait after powering on before connecting the
            EtherCAT master. Increase if the power supply is slow.
        controller_init_wait_s: Seconds to wait after starting the EtherCAT master
            before enabling the servo group.
        max_relative_target: Maximum allowed joint displacement (degrees) per command.
            Set to a positive value to add a safety clamp, or None to disable.
        cameras: Optional dict of camera configs keyed by camera name.
    """

    ip: str = "192.168.115.11"
    port: int = 10003
    box_id: int = 0
    robot_id: int = 0

    velocity: float = 50.0
    acc: float = 100.0
    speed_override: float = 0.5

    tcp_name: str = "TCP"
    ucs_name: str = "Base"

    electrify_wait_s: float = 15.0
    controller_init_wait_s: float = 20.0

    max_relative_target: float | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("hans_s30")
@dataclass
class HansS30RobotConfig(RobotConfig, HansS30Config):
    """LeRobot-registered configuration for the Hans Robot S30."""
