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
    """Configuration of a single YAM v1 follower arm on Linux SocketCAN.

    Run `lerobot-calibrate` once per arm to measure its gripper stops; connecting for control
    requires that calibration.

    Args:
        port (`str`): SocketCAN interface of the arm, e.g. `can0`.
        expected_adapter_serial (`str | None`, *optional*): USB serial of the CAN adapter. When set, connecting fails if `port` belongs to another adapter, so a `can0`/`can1` enumeration swap cannot swap arms.
        gripper_closed_rad (`float | None`, *optional*): Raw motor radians at the closed gripper stop. Normally loaded from the calibration file; set it only to override.
        gripper_open_rad (`float | None`, *optional*): Raw motor radians at the open gripper stop. Normally loaded from the calibration file; set it only to override.
        joint_signs (`list`, *optional*): Sign (+1 or -1) mapping each of the six motor directions to the joint frame.
        joint_offsets_rad (`list`, *optional*): Offset in radians added to each of the six joints after the sign.
        initial_position_rad (`list[float] | None`, *optional*): Joint pose, in radians, the arm must be in before torque is enabled. It is a check, not a motion command; the default zeros are the folded reference pose, and `None` accepts any valid pose.
        initial_tolerance_rad (`float`, *optional*, defaults to 0.2): Allowed per-joint deviation from `initial_position_rad`.
        initial_gripper_position (`float | None`, *optional*): Gripper opening (0 closed, 1 open) required before torque is enabled; `None` skips the check.
        initial_gripper_tolerance (`float`, *optional*, defaults to 0.1): Allowed deviation from `initial_gripper_position`.
        kp (`list`, *optional*): MIT position gains of the six joints.
        kd (`list`, *optional*): MIT damping gains of the six joints.
        gripper_kp (`float`, *optional*, defaults to 5.0): MIT position gain of the gripper.
        gripper_kd (`float`, *optional*, defaults to 0.005): MIT damping gain of the gripper.
        gripper_torque_limit (`float`, *optional*, defaults to 0.5): Maximum gripper torque in Nm (at most 1), enforced by bounding the gripper position error.
        max_joint_speed_rad_s (`float`, *optional*, defaults to 0.3): Fastest the commanded joint positions move toward a new target.
        max_gripper_speed_s (`float`, *optional*, defaults to 12.0): Fastest the commanded gripper opening moves, in full strokes per second.
        max_tracking_error_rad (`float`, *optional*, defaults to 0.15): Furthest a commanded joint may lead its measured position, which limits force when the arm is blocked or pushed.
        gravity_compensation (`bool`, *optional*, defaults to `True`): Add gravity feed-forward torques from the bundled model of the standard arm with a linear gripper. Payloads or added cameras need revalidation.
        gravity_factors (`list`, *optional*): Per-joint scale applied to the model gravity torques.
        cameras (`dict`, *optional*): Cameras read with each observation, keyed by name.
        read_only (`bool`, *optional*, defaults to `True`): Read feedback without ever enabling torque; `send_action` raises. Disable only after checking the CAN port, encoder frame and gripper calibration.
        control_frequency (`float`, *optional*, defaults to 100.0): Rate of the background servo loop in Hz, between 20 and 250.
        feedback_timeout_s (`float`, *optional*, defaults to 0.2): Longest motor feedback may be missing before the servo stops and disables torque.
        command_timeout_s (`float`, *optional*, defaults to 1.0): When no action arrives for this long, the arm holds its current pose.
        id (`str | None`, *optional*): Name of this arm; it selects the calibration file.
        calibration_dir (`pathlib.Path | None`, *optional*): Directory of calibration files. Calibration (`lerobot-calibrate`) measures only the gripper's closed and open stops and never changes the joint zeros.
    """

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
