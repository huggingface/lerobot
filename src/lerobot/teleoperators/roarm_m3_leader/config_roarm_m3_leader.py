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

from dataclasses import dataclass
from typing import Literal

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("roarm_m3_leader")
@dataclass
class RoarmM3LeaderConfig(TeleoperatorConfig):
    """Configuration for a Waveshare RoArm-M3 used as a passive leader arm.

    The leader is a second RoArm-M3 held torque-off and moved by hand; its joint angles
    drive a ``roarm_m3_follower``. Like the follower it talks JSON-over-serial through its
    ESP32 (``roarm_sdk``). The produced action is in the follower's joint space (same fixed
    joint order, in degrees).
    """

    # Serial device of the leader arm. Prefer a stable /dev/serial/by-id/... path (ttyUSBx
    # numbers can swap on reboot), and a port distinct from the follower; do not open the
    # same serial port from another process at the same time.
    port: str = "/dev/ttyUSB1"
    baudrate: int = 1_000_000

    # Exponential-moving-average factor on the raw leader read, to smooth gearbox jitter.
    # 0 => max smoothing, 1 => raw.
    ema_alpha: float = 0.5

    # Quantization of the produced action; must match the follower's so the recorded label
    # equals what the robot executes (see RoarmM3FollowerConfig.action_quantization).
    action_quantization: Literal["floor", "round", "float"] = "floor"

    # Send {"T":605,"cmd":0} on connect to stop the firmware feedback stream (fresh reads).
    disable_info_flow: bool = True

    # Remap the gripper from the leader's range to the follower's range with a quadratic
    # curve. Default False => the gripper passes through like any other joint (the right
    # choice when leader and follower share the same gripper - stock / identical arms). Set
    # True only when the two grippers differ (e.g. a Gripper B / CF-3512 follower); then the
    # four gripper_* angles below are used, otherwise they are ignored.
    gripper_remap: bool = False

    # Leader gripper raw range (degrees). Only used when gripper_remap is True.
    gripper_leader_open: float = 100.0
    gripper_leader_close: float = 0.0

    # Follower gripper calibrated range (degrees). Only used when gripper_remap is True.
    # Rig-specific (these are Gripper B / CF-3512 values).
    gripper_follower_open: float = 115.0
    gripper_follower_close: float = 73.0
