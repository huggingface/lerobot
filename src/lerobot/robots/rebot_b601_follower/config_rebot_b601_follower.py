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
from .motor_family import MOTOR_PROFILES, ArmControlMode, GripperControlMode, MotorFamily


@dataclass
class RebotB601FollowerConfig:
    """Configuration shared by the Damiao and RobStride B601 follower arms."""

    # Serial device or native CAN channel.
    port: str

    # Motor family: "dm" or "rs".
    motor_family: MotorFamily = MotorFamily.DM

    # CAN transport: "damiao" (serial bridge) or "socketcan" (native CAN).
    can_adapter: str | None = None

    # Damiao serial bridge baud rate.
    dm_serial_baud: int = 921600

    # Disable motor torque before disconnecting.
    disable_torque_on_disconnect: bool = True

    # Maximum position change per command in degrees. None disables the limit.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint name to (send ID, receive ID).
    motor_can_ids: dict[str, tuple[int, int]] | None = None

    # Arm mode: "mit" or "pos_vel".
    control_mode: ArmControlMode = ArmControlMode.MIT

    # Gripper mode: "mit", "force_pos", or "mit_impedance".
    gripper_control_mode: GripperControlMode | None = None

    # MIT gains shared by all joints or keyed by joint name.
    mit_kp: float | dict[str, float] | None = None
    mit_kd: float | dict[str, float] | None = None

    # POS_VEL and FORCE_POS speed limit in degrees per second.
    pos_vel_velocity: float | dict[str, float] | None = None

    # FORCE_POS gripper force as a fraction of peak torque.
    gripper_torque_ratio: float | None = None

    # Impedance gripper moving and holding torque limits in N.m.
    gripper_torque_limit: float | None = None
    gripper_hold_torque_limit: float | None = None

    # Soft limits in public joint degrees.
    joint_limits: dict[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        self.motor_family = MotorFamily(self.motor_family)
        self.control_mode = ArmControlMode(self.control_mode)
        profile = MOTOR_PROFILES[self.motor_family]
        joints = tuple(profile.motor_models)

        if self.can_adapter is None:
            self.can_adapter = profile.can_adapter

        if self.motor_can_ids is None:
            self.motor_can_ids = dict(profile.motor_can_ids)

        if self.gripper_control_mode is None:
            self.gripper_control_mode = profile.gripper_control_mode
        else:
            self.gripper_control_mode = GripperControlMode(self.gripper_control_mode)

        for name in ("mit_kp", "mit_kd", "pos_vel_velocity"):
            value = getattr(self, name)
            default = getattr(profile, name)
            if value is None:
                value = default
            if value is not None:
                if isinstance(value, (int, float)):
                    value = dict.fromkeys(joints, float(value))
                else:
                    value = {**(default or {}), **value}
                setattr(self, name, value)

        if self.joint_limits is None:
            self.joint_limits = dict(profile.joint_limits)
        else:
            self.joint_limits = {**profile.joint_limits, **self.joint_limits}

        if self.gripper_torque_ratio is None:
            self.gripper_torque_ratio = profile.gripper_torque_ratio

        for name in ("gripper_torque_limit", "gripper_hold_torque_limit"):
            if getattr(self, name) is None:
                setattr(self, name, getattr(profile, name))


@RobotConfig.register_subclass("rebot_b601_follower")
@dataclass
class RebotB601FollowerRobotConfig(RobotConfig, RebotB601FollowerConfig):
    """Registered configuration for the reBot B601 follower robot."""

    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        RebotB601FollowerConfig.__post_init__(self)
