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
class YamFollowerConfigBase:
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
    # Damping of the six joints after a fault (kp = 0, gravity of the last valid pose kept).
    fault_damping_kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
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
    # Refuse torque when a motor's CAN timeout is off, as nothing then stops it if the host dies.
    require_motor_can_timeout: bool = True
    # Degrees and a 0-100 gripper; False gives radians and a 0-1 gripper (I2RT and MolmoAct2 data).
    use_degrees: bool = True
    use_velocity_and_torque: bool = False
    # Leave classic CAN bandwidth for both refresh and MIT command feedback.
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.2
    command_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("A YAM CAN interface is required")
        for name in (
            "joint_signs",
            "joint_offsets_deg",
            "kp",
            "kd",
            "fault_damping_kd",
            "gravity_factors",
            "float_kd",
            "coulomb_friction",
        ):
            values = getattr(self, name)
            if len(values) != 6 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} must contain six finite values")
        if any(s not in (-1, 1) for s in self.joint_signs):
            raise ValueError("joint_signs must contain only -1 or +1")
        for name, maximum in (("kp", 500), ("kd", 5), ("fault_damping_kd", 5)):
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
class YamFollowerConfig(RobotConfig, YamFollowerConfigBase):
    """Configuration of a single YAM v1 follower arm on Linux SocketCAN.

    Run `lerobot-calibrate` once per arm to measure its gripper stops; connecting for control
    requires that calibration.

    Args:
        port (`str`): SocketCAN interface of the arm, e.g. `can0`.
        expected_adapter_serial (`str | None`, *optional*): USB serial of the CAN adapter. When set, connecting fails if `port` belongs to another adapter, so a `can0`/`can1` enumeration swap cannot swap arms.
        gripper_closed_deg (`float | None`, *optional*): Raw motor angle in degrees at the closed gripper stop. Normally loaded from the calibration file; set it only to override.
        gripper_open_deg (`float | None`, *optional*): Raw motor angle in degrees at the open gripper stop. Normally loaded from the calibration file; set it only to override.
        joint_signs (`list`, *optional*): Sign (+1 or -1) mapping each of the six motor directions to the joint frame.
        joint_offsets_deg (`list`, *optional*): Offset in degrees added to each of the six joints after the sign.
        initial_position_deg (`list[float] | None`, *optional*): Joint pose in degrees the arm must be in before torque is enabled. It is a check, not a motion command; the default zeros are the folded reference pose, and `None` accepts any valid pose.
        initial_tolerance_deg (`float`, *optional*, defaults to 11.5): Allowed per-joint deviation from `initial_position_deg`, in degrees.
        initial_gripper_position (`float | None`, *optional*): Gripper opening (0 closed, 100 open) required before torque is enabled; `None` skips the check.
        initial_gripper_tolerance (`float`, *optional*, defaults to 10.0): Allowed deviation from `initial_gripper_position`, on the same 0-100 scale.
        kp (`list`, *optional*): MIT position gains of the six joints.
        kd (`list`, *optional*): MIT damping gains of the six joints.
        fault_damping_kd (`list`, *optional*): Damping of the six joints after a fault, applied with zero stiffness while keeping the gravity torque of the last valid pose, until `disconnect()` disables torque.
        gripper_kp (`float`, *optional*, defaults to 20.0): MIT position gain of the gripper (I2RT's value).
        gripper_kd (`float`, *optional*, defaults to 0.5): MIT damping gain of the gripper (I2RT's value).
        gripper_force_limit_n (`float`, *optional*, defaults to 50.0): Finger force applied once the gripper is blocked on an object, as in I2RT's gripper force limiter. The gripper moves freely until then.
        gripper_stroke_m (`float`, *optional*, defaults to 0.096): Finger travel between the two gripper stops, in m, used to convert `gripper_force_limit_n` into motor torque.
        max_joint_speed_deg_s (`float`, *optional*, defaults to 17.0): Fastest the commanded joint positions move toward a new target, in degrees per second.
        max_gripper_speed_s (`float`, *optional*, defaults to 12.0): Fastest the commanded gripper opening moves, in full strokes per second.
        max_tracking_error_deg (`float`, *optional*, defaults to 8.5): Furthest a commanded joint may lead its measured position, in degrees, which limits force when the arm is blocked or pushed.
        gravity_compensation (`bool`, *optional*, defaults to `True`): Add gravity feed-forward torques from the bundled model of the standard arm with a linear gripper. Payloads or added cameras need revalidation.
        gravity_factors (`list`, *optional*): Per-joint scale applied to the model gravity torques.
        idle_mode (`str`, *optional*, defaults to `"hold"`): What the arm does before the first action and after `command_timeout_s`: `"hold"` keeps the measured pose stiffly, `"float"` lets it be moved by hand while gravity is compensated (the gripper keeps holding). Float needs `gravity_compensation`.
        float_kd (`list`, *optional*): Damping of the six joints in float mode, with zero stiffness. The defaults are I2RT's values for the standard arm.
        friction_compensation (`bool`, *optional*, defaults to `False`): Add Coulomb friction compensation in float mode, in the direction each joint moves.
        coulomb_friction (`list`, *optional*): Coulomb friction of the six joints in Nm, used when `friction_compensation` is on. The defaults are I2RT's values for the standard arm.
        cameras (`dict`, *optional*): Cameras read with each observation, keyed by name.
        read_only (`bool`, *optional*, defaults to `True`): Read feedback without ever enabling torque; `send_action` raises. Disable only after checking the CAN port, encoder frame and gripper calibration.
        require_motor_can_timeout (`bool`, *optional*, defaults to `True`): Refuse to enable torque when a motor's CAN loss-of-communication timeout is off, since nothing then stops that motor if this process dies. Set it to `False` to only warn.
        use_degrees (`bool`, *optional*, defaults to `True`): Report and accept joints in degrees and the gripper from 0 to 100. Set it to `False` for radians and a 0-1 gripper, the units of I2RT and MolmoAct2 data.
        use_velocity_and_torque (`bool`, *optional*, defaults to `False`): Add `.vel` and `.torque` features for each motor to observations.
        control_frequency (`float`, *optional*, defaults to 100.0): Rate of the background servo loop in Hz, between 20 and 250.
        feedback_timeout_s (`float`, *optional*, defaults to 0.2): Maximum age of each motor reply. The foreground check allows two such intervals plus one servo period; long Python scheduling stalls can still stop the servo.
        command_timeout_s (`float`, *optional*, defaults to 1.0): When no action arrives for this long, the arm holds its current pose.
        id (`str | None`, *optional*): Name of this arm; it selects the calibration file.
        calibration_dir (`pathlib.Path | None`, *optional*): Directory of calibration files. Each file stores the raw motor angles of the gripper stops in whole degrees; calibration never changes joint zeros.
    """

    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        YamFollowerConfigBase.__post_init__(self)
