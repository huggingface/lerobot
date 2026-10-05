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

"""Configuration shared by single-arm and bimanual YAM followers."""

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

JOINT_NAMES = tuple(f"joint_{i}" for i in range(6))
MOTOR_NAMES = (*JOINT_NAMES, "gripper")
YAM_FEATURE_NAMES = tuple(f"{name}.pos" for name in MOTOR_NAMES)
BI_YAM_FEATURE_NAMES = tuple(f"{side}_{name}.pos" for side in ("left", "right") for name in MOTOR_NAMES)
# I2RT yam/v1/yam.xml; radians, without widening the manufacturer's limits.
JOINT_LIMITS = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)


@dataclass
class YamArmConfig:
    port: str
    # Linux USB serial verification prevents can0/can1 enumeration swapping arms.
    expected_adapter_serial: str | None = None
    # Measured raw MOTOR radians at the physical closed and open stops.
    # Either polarity is supported; no nominal stroke or automatic homing guess.
    gripper_closed_rad: float | None = None
    gripper_open_rad: float | None = None
    joint_signs: list[float] = field(default_factory=lambda: [1.0] * 6)
    joint_offsets_rad: list[float] = field(default_factory=lambda: [0.0] * 6)
    # An assertion, NOT a motion command. Place the supported arm in this pose
    # before enabling control. Zero is the manufacturer's folded reference pose.
    # None accepts the current valid measured pose instead of a fixed startup pose.
    initial_position_rad: list[float] | None = field(default_factory=lambda: [0.0] * 6)
    initial_tolerance_rad: float = 0.2
    initial_gripper_position: float | None = None
    initial_gripper_tolerance: float = 0.1
    kp: list[float] = field(default_factory=lambda: [80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
    kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
    gripper_kp: float = 5.0
    gripper_kd: float = 0.005
    gripper_torque_limit: float = 0.5
    max_joint_speed_rad_s: float = 0.3
    max_gripper_speed_s: float = 12.0
    max_tracking_error_rad: float = 0.15
    # Feedforward for the standard linear_4310 hardware. Camera/payload changes
    # require revalidation. This model is not a collision avoidance system.
    gravity_compensation: bool = True
    gravity_factors: list[float] = field(default_factory=lambda: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0])

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("A YAM CAN interface is required")
        for name in ("joint_signs", "joint_offsets_rad", "kp", "kd", "gravity_factors"):
            values = getattr(self, name)
            if len(values) != 6 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must contain six finite values")
        if any(s not in (-1, 1) for s in self.joint_signs):
            raise ValueError("joint_signs must contain only -1 or +1")
        for name, maximum in (("kp", 500), ("kd", 5)):
            if any(not 0 < x <= maximum for x in getattr(self, name)):
                raise ValueError(f"{name} outside MIT gain limits")
        for name in (
            "initial_tolerance_rad",
            "initial_gripper_tolerance",
            "gripper_kp",
            "gripper_kd",
            "gripper_torque_limit",
            "max_joint_speed_rad_s",
            "max_gripper_speed_s",
            "max_tracking_error_rad",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gripper_kp > 500 or self.gripper_kd > 5 or self.gripper_torque_limit > 1:
            raise ValueError("Gripper gains/torque exceed supported limits (maximum 1 Nm)")
        if self.initial_gripper_position is not None and not 0 <= self.initial_gripper_position <= 1:
            raise ValueError("initial_gripper_position must be between 0 closed and 1 open")
        if self.initial_position_rad is not None:
            if len(self.initial_position_rad) != 6 or not all(
                math.isfinite(v) for v in self.initial_position_rad
            ):
                raise ValueError("initial_position_rad must contain six finite values or be null")
            for value, limits in zip(self.initial_position_rad, JOINT_LIMITS, strict=True):
                if not limits[0] <= value <= limits[1]:
                    raise ValueError("initial_position_rad is outside YAM joint limits")
        ends = (self.gripper_closed_rad, self.gripper_open_rad)
        if any(x is not None for x in ends):
            if any(x is None or not math.isfinite(x) or abs(x) > 12.5 for x in ends):
                raise ValueError("Provide both finite raw gripper endpoints within motor limits")
            closed, opened = ends
            assert closed is not None and opened is not None
            if abs(opened - closed) < 0.01:
                raise ValueError("Gripper endpoints must be distinct")


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerConfig(RobotConfig, YamArmConfig):
    """Registered configuration for a single YAM follower arm."""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    read_only: bool = True
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.2
    command_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        YamArmConfig.__post_init__(self)
        for name in ("control_frequency", "feedback_timeout_s", "command_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 20 <= self.control_frequency <= 250:
            raise ValueError("control_frequency must be between 20 and 250 Hz")
        if set(self.cameras) & set(YAM_FEATURE_NAMES):
            raise ValueError("Camera names collide with YAM motor features")
