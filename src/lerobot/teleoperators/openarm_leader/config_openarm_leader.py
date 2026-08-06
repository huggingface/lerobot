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


@dataclass
class OpenArmLeaderConfigBase:
    """Base configuration for the OpenArms leader/teleoperator with Damiao motors."""

    # CAN interfaces - one per arm
    # Arm CAN interface (e.g., "can3")
    # Linux: "can0", "can1", etc.
    port: str

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
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

    # When True, expose `.vel` and `.torque` per motor in action features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # TODO(Steven, Pepijn): Not used ... ?
    # MIT control parameters (used when manual_control=False for torque control)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])


@TeleoperatorConfig.register_subclass("openarm_leader")
@dataclass
class OpenArmLeaderConfig(TeleoperatorConfig, OpenArmLeaderConfigBase):
    """Configuration for the OpenArm leader/teleoperator arm (CAN bus, Damiao motors).

    Args:
        port (`str`):
            CAN interface the arm is connected to, e.g. `"can0"` on Linux.
        can_interface (`str`, *optional*, defaults to `"socketcan"`):
            CAN backend type: `"socketcan"` (Linux), `"slcan"` (serial), or `"auto"` (auto-detect).
        use_can_fd (`bool`, *optional*, defaults to `True`):
            Whether to use CAN FD, which OpenArm uses by default.
        can_bitrate (`int`, *optional*, defaults to 1000000):
            Nominal CAN bus bitrate, in bits per second.
        can_data_bitrate (`int`, *optional*, defaults to 5000000):
            CAN FD data-phase bitrate, in bits per second. Only used when `use_can_fd` is `True`.
        motor_config (`dict[str, tuple[int, int, str]]`, *optional*):
            Maps motor name to `(send_can_id, recv_can_id, motor_type)`. Defaults to the standard 7-DOF
            plus gripper OpenArm layout, using DM8009 (shoulder), DM4340 (shoulder rotation, elbow), and
            DM4310 (wrist, gripper) Damiao motors.
        manual_control (`bool`, *optional*, defaults to `True`):
            Whether motors have torque disabled for manual movement. Required for a leader arm that is
            moved by hand.
        use_velocity_and_torque (`bool`, *optional*, defaults to `False`):
            Whether to expose `.vel` and `.torque` per motor in [`~teleoperators.Teleoperator.action_features`],
            in addition to `.pos`.
        position_kp (`list[float]`, *optional*):
            Per-joint position gain, used for MIT torque control when `manual_control` is `False`.
            Defaults to the standard 8-value OpenArm gain set (one value per joint, plus gripper).
        position_kd (`list[float]`, *optional*):
            Per-joint velocity gain, used for MIT torque control when `manual_control` is `False`.
            Defaults to the standard 8-value OpenArm damping set (one value per joint, plus gripper).
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    pass
