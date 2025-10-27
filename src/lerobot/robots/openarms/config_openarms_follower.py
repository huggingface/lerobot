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
from typing import Dict, List, Optional

from lerobot.cameras import CameraConfig
from lerobot.motors.damiao.tables import MotorType

from ..config import RobotConfig


@RobotConfig.register_subclass("openarms_follower")
@dataclass
class OpenArmsFollowerConfig(RobotConfig):
    """Configuration for the OpenArms follower robot with Damiao motors."""
    
    # CAN interface to connect to
    # Linux: "can0", "can1", etc.
    # macOS: "/dev/cu.usbmodem*" (serial device)
    port: str = "can0"
    
    # CAN interface type: "socketcan" (Linux), "slcan" (macOS/serial), or "auto" (auto-detect)
    can_interface: str = "auto"
    
    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True
    
    # Safety limit for relative target positions
    # Set to a positive scalar for all motors, or a dict mapping motor names to limits
    max_relative_target: Optional[float | Dict[str, float]] = None
    
    # Camera configurations
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    
    # Motor configuration for OpenArms (7 DOF per arm)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on: https://docs.openarm.dev/software/setup/configure-test
    # OpenArms uses 4 types of motors:
    # - DM8009 (DM-J8009P-2EC) for shoulders (high torque)
    # - DM4340P and DM4340 for shoulder rotation and elbow
    # - DM4310 (DM-J4310-2EC V1.1) for wrist and gripper
    motor_config: Dict[str, tuple[int, int, str]] = field(default_factory=lambda: {
        "joint_1": (0x01, 0x11, "dm8009"),   # J1 - Shoulder pan (DM8009)
        "joint_2": (0x02, 0x12, "dm8009"),   # J2 - Shoulder lift (DM8009)
        "joint_3": (0x03, 0x13, "dm4340"),   # J3 - Shoulder rotation (DM4340)
        "joint_4": (0x04, 0x14, "dm4340"),   # J4 - Elbow flex (DM4340)
        "joint_5": (0x05, 0x15, "dm4310"),   # J5 - Wrist roll (DM4310)
        "joint_6": (0x06, 0x16, "dm4310"),   # J6 - Wrist pitch (DM4310)
        "joint_7": (0x07, 0x17, "dm4310"),   # J7 - Wrist rotation (DM4310)
        "gripper": (0x08, 0x18, "dm4310"),   # J8 - Gripper (DM4310)
    })
    
    # MIT control parameters for position control
    position_kp: float = 10.0  # Position gain
    position_kd: float = 0.5   # Velocity damping
    
    # Calibration parameters
    calibration_mode: str = "manual"  # "manual" or "auto"
    zero_position_on_connect: bool = False  # Set zero position on connect
