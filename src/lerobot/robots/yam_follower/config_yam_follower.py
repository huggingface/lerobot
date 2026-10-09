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

JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7))
MOTOR_NAMES = (*JOINT_NAMES, "gripper")
YAM_FEATURE_NAMES = tuple(f"{name}.pos" for name in MOTOR_NAMES)
# I2RT yam/v1/yam.xml; radians, without widening the manufacturer's limits.
JOINT_LIMITS_RAD = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)
JOINT_LIMITS_DEG = tuple((math.degrees(lower), math.degrees(upper)) for lower, upper in JOINT_LIMITS_RAD)
DM_MIT_POSITION_LIMIT_RAD = 12.5
# Plausible raw motor travel between the two gripper stops (0.5 to 10 rad).
GRIPPER_STROKE_RANGE_DEG = (math.degrees(0.5), math.degrees(10.0))


def motor_feature_names(use_velocity_and_torque: bool = False) -> tuple[str, ...]:
    """Return the motor feature names of one arm, grouped per motor."""
    suffixes = (".pos", ".vel", ".torque") if use_velocity_and_torque else (".pos",)
    return tuple(f"{name}{suffix}" for name in MOTOR_NAMES for suffix in suffixes)


@dataclass
class YamFollowerConfig:
    """Settings of one YAM follower arm, shared by the single-arm and bimanual robots."""

    port: str
    # Linux USB serial verification prevents can0/can1 enumeration swapping arms.
    expected_adapter_serial: str | None = None
    # Raw motor angles at the physical closed and open stops, normally loaded from calibration.
    # Either polarity is supported; no nominal stroke or automatic homing guess.
    gripper_closed_deg: float | None = None
    gripper_open_deg: float | None = None
    joint_signs: list[float] = field(default_factory=lambda: [1.0] * 6)
    joint_offsets_deg: list[float] = field(default_factory=lambda: [0.0] * 6)
    # A check, NOT a motion command. Place the supported arm in this pose before enabling
    # control. Zero is the manufacturer's folded reference pose; None accepts any valid pose.
    initial_position_deg: list[float] | None = field(default_factory=lambda: [0.0] * 6)
    initial_tolerance_deg: float = 11.5
    # Gripper opening from 0 (closed) to 100 (open); None skips the check.
    initial_gripper_position: float | None = None
    initial_gripper_tolerance: float = 10.0
    kp: list[float] = field(default_factory=lambda: [80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
    kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
    # I2RT's gains for the linear DM4310 gripper.
    gripper_kp: float = 20.0
    gripper_kd: float = 0.5
    # Finger force once the gripper is blocked on an object, and the finger travel between stops.
    gripper_force_limit_n: float = 50.0
    gripper_stroke_m: float = 0.096
    max_joint_speed_deg_s: float = 17.0
    max_gripper_speed_s: float = 12.0
    max_tracking_error_deg: float = 8.5
    # Feedforward for the standard linear_4310 hardware. Camera/payload changes
    # require revalidation. This model is not a collision avoidance system.
    gravity_compensation: bool = True
    gravity_factors: list[float] = field(default_factory=lambda: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    # Before the first action and after command_timeout_s: "hold" the pose or "float" (movable by hand).
    idle_mode: str = "hold"
    # I2RT's float-mode damping and Coulomb friction for the standard YAM v1 arm.
    float_kd: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.3, 0.05, 0.05])
    friction_compensation: bool = False
    coulomb_friction: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 0.06, 0.06, 0.06])
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Opt in only after verifying the CAN port, encoder frame and gripper calibration.
    read_only: bool = True
    # Refuse torque unless every motor has a CAN timeout of at most 400 ms.
    require_motor_can_timeout: bool = True
    # Degrees and a 0-100 gripper; False gives radians and a 0-1 gripper (I2RT and MolmoAct2 data).
    use_degrees: bool = True
    use_velocity_and_torque: bool = False
    # Leave classic CAN bandwidth for both refresh and MIT command feedback.
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.2
    # Time without actions before entering the configured idle mode.
    command_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("A YAM CAN interface is required")
        for name in (
            "joint_signs",
            "joint_offsets_deg",
            "kp",
            "kd",
            "gravity_factors",
            "float_kd",
            "coulomb_friction",
        ):
            values = getattr(self, name)
            if len(values) != 6 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must contain six finite values")
        if any(s not in (-1, 1) for s in self.joint_signs):
            raise ValueError("joint_signs must contain only -1 or +1")
        for name, maximum in (("kp", 500), ("kd", 5)):
            if any(not 0 < x <= maximum for x in getattr(self, name)):
                raise ValueError(f"{name} outside MIT gain limits")
        for name in (
            "initial_tolerance_deg",
            "initial_gripper_tolerance",
            "gripper_kp",
            "gripper_kd",
            "gripper_force_limit_n",
            "gripper_stroke_m",
            "max_joint_speed_deg_s",
            "max_gripper_speed_s",
            "max_tracking_error_deg",
            "control_frequency",
            "feedback_timeout_s",
            "command_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if any(not 0 <= x <= 5 for x in self.float_kd):
            raise ValueError("float_kd outside MIT gain limits")
        if any(x < 0 for x in self.coulomb_friction):
            raise ValueError("coulomb_friction must not be negative")
        if self.idle_mode not in ("hold", "float"):
            raise ValueError('idle_mode must be "hold" or "float"')
        if self.idle_mode == "float" and not self.gravity_compensation:
            raise ValueError("idle_mode='float' needs gravity_compensation, or the arm would fall")
        if self.gripper_kp > 500 or self.gripper_kd > 5:
            raise ValueError("Gripper gains exceed the MIT limits")
        if not 20 <= self.control_frequency <= 250:
            raise ValueError("control_frequency must be between 20 and 250 Hz")
        if self.initial_gripper_position is not None and not 0 <= self.initial_gripper_position <= 100:
            raise ValueError("initial_gripper_position must be between 0 closed and 100 open")
        if self.initial_position_deg is not None:
            if len(self.initial_position_deg) != 6 or not all(
                math.isfinite(v) for v in self.initial_position_deg
            ):
                raise ValueError("initial_position_deg must contain six finite values or be null")
            for value, (lower, upper) in zip(self.initial_position_deg, JOINT_LIMITS_DEG, strict=True):
                if not lower <= value <= upper:
                    raise ValueError("initial_position_deg is outside YAM joint limits")
        ends = (self.gripper_closed_deg, self.gripper_open_deg)
        if any(x is not None for x in ends):
            motor_limit_deg = math.degrees(DM_MIT_POSITION_LIMIT_RAD)
            if any(x is None or not math.isfinite(x) or abs(x) > motor_limit_deg for x in ends):
                raise ValueError("Provide both finite raw gripper endpoints within motor limits")
            closed, opened = ends
            assert closed is not None and opened is not None
            low, high = GRIPPER_STROKE_RANGE_DEG
            if not low <= abs(opened - closed) <= high:
                raise ValueError(f"Gripper stroke must be between {low:.0f} and {high:.0f} raw motor degrees")
        if set(self.cameras) & set(motor_feature_names(use_velocity_and_torque=True)):
            raise ValueError("Camera names collide with YAM motor features")


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerRobotConfig(RobotConfig, YamFollowerConfig):
    """Configuration for a single YAM follower."""

    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        YamFollowerConfig.__post_init__(self)
