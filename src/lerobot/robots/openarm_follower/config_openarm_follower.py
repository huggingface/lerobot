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

LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-90.0, 9.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}

RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-9.0, 90.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class OpenArmFollowerConfigBase:
    """Field definitions for the OpenArm follower, a 7-DOF arm plus gripper on Damiao CAN motors.

    This class only carries the fields. The registered configuration users instantiate is
    [`OpenArmFollowerConfig`], which documents them all in one place — doc-builder renders only a class's
    own docstring, never its bases'. It is also used directly as the per-arm config of
    [`~robots.bi_openarm_follower.BiOpenArmFollowerConfig`].
    """

    # CAN interfaces - one per arm
    # arm CAN interface (e.g., "can1")
    # Linux: "can0", "can1", etc.
    port: str

    # side of the arm: "left" or "right". If "None" default values will be used
    side: str | None = None

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # When True, expose `.vel` and `.torque` per motor in observation features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # Safety limit for relative target positions
    # Set to a positive scalar for all motors, or a dict mapping motor names to limits
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

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

    # MIT control parameters for position control (used in send_action)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 25.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3, 0.3])

    # Values for joint limits. Can be overridden via CLI (for custom values) or by setting config.side to either 'left' or 'right'.
    # If config.side is left set to None and no CLI values are passed, the default joint limit values are small for safety.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-5.0, 5.0),
            "joint_2": (-5.0, 5.0),
            "joint_3": (-5.0, 5.0),
            "joint_4": (0.0, 5.0),
            "joint_5": (-5.0, 5.0),
            "joint_6": (-5.0, 5.0),
            "joint_7": (-5.0, 5.0),
            "gripper": (-5.0, 0.0),
        }
    )


@RobotConfig.register_subclass("openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig, OpenArmFollowerConfigBase):
    """Configuration for a single OpenArm follower arm.

    OpenArm is a 7-DOF arm plus gripper on Damiao CAN motors, so `port` names a CAN interface rather than a
    serial device. Calibration follows the usual LeRobot flow and is stored per `id`.

    > [!WARNING]
    > `joint_limits` defaults to a deliberately tiny range so an uncalibrated arm cannot swing. Set `side`
    > to `"left"` or `"right"` to get the real limits for that arm, or pass your own.

    The per-joint lists — `position_kp`, `position_kd` — hold 8 values in motor order: `joint_1` through
    `joint_7`, then `gripper`.

    Args:
        port (`str`):
            CAN interface the arm is on, e.g. `"can0"` on Linux.
        side (`str`, *optional*):
            Which arm this is, `"left"` or `"right"`. Selects that side's joint limits. Leaving it `None`
            keeps the small safety defaults.
        can_interface (`str`, *optional*, defaults to `"socketcan"`):
            CAN backend: `"socketcan"` on Linux, `"slcan"` for a serial adapter, or `"auto"` to detect.
        use_can_fd (`bool`, *optional*, defaults to `True`):
            Whether to use CAN FD. OpenArm uses it by default.
        can_bitrate (`int`, *optional*, defaults to 1000000):
            Nominal CAN bitrate, 1 Mbps.
        can_data_bitrate (`int`, *optional*, defaults to 5000000):
            CAN FD data bitrate, 5 Mbps. Only used when `use_can_fd` is `True`.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Whether to release the motors on disconnect. Leave `True` unless the arm is holding a load it
            must not drop.
        use_velocity_and_torque (`bool`, *optional*, defaults to `False`):
            Whether to expose `.vel` and `.torque` per motor in the observation features. Kept `False` by
            default for compatibility with the position-only `openarm_mini` teleoperator.
        max_relative_target (`float | dict[str, float]`, *optional*):
            Caps how far a single action may move the arm from its present position, as a safety limit. A
            scalar applies to every motor; a dict maps motor name to a per-motor cap. `None` disables
            clipping.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the arm's joint positions.
        motor_config (`dict[str, tuple[int, int, str]]`, *optional*):
            Maps motor name to `(send_can_id, recv_can_id, motor_type)`. Defaults to the stock OpenArm
            layout; change it only if you have rewired or re-addressed the motors.
        position_kp (`list[float]`, *optional*):
            MIT-mode proportional gains used by `send_action`, 8 values in motor order.
        position_kd (`list[float]`, *optional*):
            MIT-mode derivative gains used by `send_action`, 8 values in motor order.
        joint_limits (`dict[str, tuple[float, float]]`, *optional*):
            Soft `(min, max)` limits in degrees per joint, clipped against on every action.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.
    """

    pass
