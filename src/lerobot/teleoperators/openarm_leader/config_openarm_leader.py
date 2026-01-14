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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarms_leader")
@dataclass
class OpenArmsLeaderConfig(TeleoperatorConfig):
    """Configuration for the OpenArms leader/teleoperator with Damiao motors."""

    # CAN interfaces - one per arm
    # Right arm CAN interface (e.g., "can2")
    # Left arm CAN interface (e.g., "can3")
    # Linux: "can0", "can1", etc.
    # macOS: "/dev/cu.usbmodem*" (serial device)
    port_right: str = "can2"  # CAN interface for right arm
    port_left: str = "can3"  # CAN interface for left arm

    # CAN interface type: "socketcan" (Linux), "slcan" (macOS/serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Motor configuration for OpenArms (7 DOF per arm)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on: https://docs.openarm.dev/software/setup/configure-test
    # OpenArms uses 4 types of motors:
    # - DM8009 (DM-J8009P-2EC) for shoulders (high torque)
    # - DM4340P and DM4340 for shoulder rotation and elbow
    # - DM4310 (DM-J4310-2EC V1.1) for wrist and gripper
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - Shoulder pan (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - Shoulder lift (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - Shoulder rotation (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - Elbow flex (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - Wrist roll (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - Wrist pitch (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - Wrist rotation (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - Gripper (DM4310)
        }
    )

    # Torque mode settings for manual control
    # When enabled, motors have torque disabled for manual movement
    manual_control: bool = True

    # MIT control parameters (used when manual_control=False for torque control)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])

    # Damping gains for stability when applying torque compensation (gravity/friction)
    # Used when kp=0 and only torque is applied
    damping_kd: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1])

    # Friction model parameters: τ_fric(ω) = F_o + F_v·ω + F_c·tanh(k·ω)
    # From OpenArms config/leader.yaml (note: Fc[5] is slightly different: 0.083 vs 0.093)
    friction_f_c: list[float] = field(
        default_factory=lambda: [0.306, 0.306, 0.40, 0.166, 0.050, 0.083, 0.172, 0.0512]
    )  # Coulomb friction [Nm]
    friction_k: list[float] = field(
        default_factory=lambda: [28.417, 28.417, 29.065, 130.038, 151.771, 242.287, 7.888, 4.000]
    )  # tanh steepness
    friction_f_v: list[float] = field(
        default_factory=lambda: [0.063, 0.0630, 0.604, 0.813, 0.029, 0.072, 0.084, 0.084]
    )  # Viscous friction [Nm·s/rad]
    friction_f_o: list[float] = field(
        default_factory=lambda: [0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059, -0.050]
    )  # Offset torque [Nm]
