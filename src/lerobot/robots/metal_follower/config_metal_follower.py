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

"""Configuration shared by single-arm and bimanual Metal followers."""

import math
from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
MOTOR_NAMES = (*JOINT_NAMES, "gripper")
METAL_FEATURE_NAMES = tuple(f"{name}.pos" for name in MOTOR_NAMES)
# Joint limits in degrees, from the Metal Arm v1 profile. Zero is the folded rest pose.
JOINT_LIMITS_DEG = (
    (-160.0, 160.0),
    (-180.0, 0.0),
    (0.0, 180.0),
    (-123.0, 81.0),
    (-85.0, 85.0),
    (-145.0, 145.0),
)
# Fastest joint command; a supply drop-out rebooted every motor at 321 deg/s.
MAX_JOINT_SPEED_DEG_S = 280.0


@dataclass
class MetalFollowerConfig:
    """Settings of one Metal arm, shared by the single-arm and bimanual robots and the leader."""

    port: str
    can_interface: Literal["auto", "socketcan", "slcan"] = "auto"
    initial_position_deg: list[float] | None = field(default_factory=lambda: [0.0] * 6)
    initial_tolerance_deg: float = 11.5
    kp: list[float] = field(default_factory=lambda: [100.0, 100.0, 100.0, 50.0, 20.0, 20.0])
    kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 5.0, 0.6, 0.6])
    gripper_kp: float = 20.0
    gripper_kd: float = 0.6
    gripper_hold_torque_nm: float = 0.5
    max_joint_speed_deg_s: float = 90.0
    max_gripper_speed: float = 2.4
    max_tracking_error_deg: float = 8.5
    gravity_compensation: bool = True
    gravity_factors: list[float] = field(default_factory=lambda: [1.2, 1.15, 1.1, 1.15, 1.0, 1.0])
    urdf_path: str | None = None
    idle_mode: Literal["hold", "float"] = "hold"
    float_kd: list[float] = field(default_factory=lambda: [0.1] * 6)
    coulomb_friction: list[float] = field(default_factory=lambda: [0.0] * 6)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    use_velocity_and_torque: bool = False
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.3
    command_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        # A short list would shift the gripper's command onto an arm joint.
        for name in ("kp", "kd", "gravity_factors", "float_kd", "coulomb_friction"):
            if len(getattr(self, name)) != 6:
                raise ValueError(f"{name} must contain six values; got {getattr(self, name)}")
        # A speed or band below zero inverts the controller's clipping.
        for name in ("max_joint_speed_deg_s", "max_gripper_speed", "max_tracking_error_deg"):
            if (value := getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        if self.max_joint_speed_deg_s > MAX_JOINT_SPEED_DEG_S:
            raise ValueError(f"max_joint_speed_deg_s must be at most {MAX_JOINT_SPEED_DEG_S:g}")
        # A negative factor flips the gravity torque.
        if any(factor < 0 for factor in self.gravity_factors):
            raise ValueError(f"gravity_factors must not be negative; got {self.gravity_factors}")
        # Freshness is judged against it: zero, negative or NaN would accept any feedback or none.
        if not (math.isfinite(self.feedback_timeout_s) and self.feedback_timeout_s > 0):
            raise ValueError(f"feedback_timeout_s must be positive; got {self.feedback_timeout_s}")
        if self.idle_mode == "float" and not self.gravity_compensation:
            raise ValueError("idle_mode='float' needs gravity_compensation")


@RobotConfig.register_subclass("metal_follower")
@dataclass
class MetalFollowerRobotConfig(RobotConfig, MetalFollowerConfig):
    """Configuration for a single Metal follower.

    Metal needs no calibration file: the joint zeros are stored in the motors, at the folded rest pose with
    the gripper closed, and the gripper stroke is fixed.

    Args:
        port (`str`):
            CAN interface of the arm: a SocketCAN name such as `can0`, or the serial port of an slcan
            adapter such as `/dev/ttyACM0`.
        can_interface (`Literal["auto", "socketcan", "slcan"]`, *optional*, defaults to `"auto"`):
            python-can interface; `auto` picks `slcan` for ports under `/dev/` and `socketcan` otherwise.
        initial_position_deg (`list[float] | None`, *optional*):
            Position, in degrees, the arm must be in before torque is enabled; the arm isn't moved there.
            Zeros are the folded rest pose; `None` skips this check.
        initial_tolerance_deg (`float`, *optional*, defaults to 11.5):
            Allowed distance from `initial_position_deg`, in degrees.
        kp (`list[float]`, *optional*):
            MIT position gains of the six arm joints.
        kd (`list[float]`, *optional*):
            MIT damping gains of the six arm joints.
        gripper_kp (`float`, *optional*, defaults to 20.0):
            MIT position gain of the gripper.
        gripper_kd (`float`, *optional*, defaults to 0.6):
            MIT damping gain of the gripper.
        gripper_hold_torque_nm (`float`, *optional*, defaults to 0.5):
            Gripper motor torque, in newton-metres, once the gripper is blocked on an object.
        max_joint_speed_deg_s (`float`, *optional*, defaults to 90.0):
            Maximum joint speed, in degrees per second, at most 280.
        max_gripper_speed (`float`, *optional*, defaults to 2.4):
            Maximum gripper speed, in full strokes per second.
        max_tracking_error_deg (`float`, *optional*, defaults to 8.5):
            Maximum difference between the measured and commanded joint positions, in degrees.
        gravity_compensation (`bool`, *optional*, defaults to `True`):
            Whether to compensate gravity with the Metal model, downloaded on first use.
        gravity_factors (`list[float]`, *optional*):
            Per-joint multipliers of the gravity torques. Check them again after adding cameras or a payload,
            which the model doesn't include.
        urdf_path (`str | None`, *optional*):
            Local Metal URDF for gravity compensation, instead of the one downloaded from the Hub.
        idle_mode (`Literal["hold", "float"]`, *optional*, defaults to `"hold"`):
            What the arm does before the first action and after `command_timeout_s` without one: `hold`
            keeps the position, `float` lets you move the arm by hand.
        float_kd (`list[float]`, *optional*):
            Damping gains in float mode.
        coulomb_friction (`list[float]`, *optional*):
            Coulomb friction of each joint, in newton-metres, compensated in float mode; zeros disable it.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras attached to this arm.
        use_degrees (`bool`, *optional*, defaults to `True`):
            Whether joints use degrees and the gripper 0 to 100. `False` gives radians and 0 to 1.
        use_velocity_and_torque (`bool`, *optional*, defaults to `False`):
            Whether observations include motor velocity and torque.
        control_frequency (`float`, *optional*, defaults to 100.0):
            Control loop frequency in hertz.
        feedback_timeout_s (`float`, *optional*, defaults to 0.3):
            Maximum age of each motor's feedback; older feedback stops control.
        command_timeout_s (`float`, *optional*, defaults to 1.0):
            Time without an action before the arm goes idle, see `idle_mode`.
        id (`str | None`, *optional*):
            Robot identifier.
        calibration_dir (`pathlib.Path | None`, *optional*):
            Unused: Metal has no calibration file.
    """

    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        MetalFollowerConfig.__post_init__(self)
