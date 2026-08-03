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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("roarm_m3_follower")
@dataclass
class RoarmM3FollowerConfig(RobotConfig):
    """Configuration for a single Waveshare RoArm-M3 follower arm and its cameras.

    The RoArm-M3 is driven over a JSON-over-serial link through its on-board ESP32
    (via ``roarm_sdk``), not over a Feetech/Dynamixel binary bus, so this config carries a
    serial ``port`` and ``baudrate`` rather than a motors map.
    """

    # Serial device of the follower arm, e.g. "/dev/ttyUSB0". Prefer a stable
    # /dev/serial/by-id/... path; ttyUSBx numbers can swap on reboot.
    port: str = "/dev/ttyUSB0"

    # Serial baudrate. Use 1_000_000 (1 Mbps) on firmware with the baud patch applied
    # (USB Serial.begin(1000000), matching the internal servo bus); this lifts the
    # state-read rate from ~23 Hz to ~42 Hz and enables smooth 30 fps teleop. Stock
    # firmware is 115200.
    baudrate: int = 1_000_000

    # `max_relative_target` limits the magnitude of the relative positional target vector
    # for safety (degrees). float => same value for all joints; dict => per joint (keys
    # must be joint names); None => disabled.
    max_relative_target: float | dict[str, float] | None = None

    # Quantization applied to the action before sending it to the firmware (which takes
    # integer degrees). "floor" truncates toward zero (firmware default), "round" rounds to
    # the nearest integer (avoids the downward bias of truncation), "float" forwards the raw
    # float for firmware builds that accept non-integer joint targets.
    action_quantization: Literal["floor", "round", "float"] = "floor"

    # Opt-in Waveshare Gripper B (CF-3512, force-control) support. Default False =
    # position-only, a single bundled write per step (right for the stock gripper / Gripper
    # A). When True, the driver adds a second, gripper-only write each step so the CF-3512
    # receives its constant-current torque; see the robot docstring and roarm_m3.mdx.
    force_control_gripper: bool = False

    # Send {"T":605,"cmd":0} on connect to stop the firmware feedback stream (fresh reads).
    disable_info_flow: bool = True

    # Release follower torque on disconnect so the arm can be moved by hand.
    disable_torque_on_disconnect: bool = True

    # Motion speed / acceleration passed to the firmware joint commands.
    motion_speed: int = 500
    motion_acc: int = 50

    # Cameras keyed by short logical name ("wrist", "front", ...). Empty by default; the
    # caller injects camera configs (USB indices vary per boot).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
