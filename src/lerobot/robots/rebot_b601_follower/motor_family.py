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

"""Family-specific hardware facts and defaults for the reBot B601.

The B601-DM (Damiao) and B601-RS (RobStride) share one software interface but
differ in actuator models, wiring, mounting directions and validated modes.
Those integration facts live here; user preferences remain on the robot config.
"""

from dataclasses import dataclass
from enum import StrEnum

# Motor order. Per-joint config fields are validated against the joints actually
# declared in `motor_can_ids`, so this tuple only fixes the default layout.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

# The three high-torque base joints, which carry a different motor model (and
# therefore different torque and MIT-gain scaling) than the four distal ones.
PROXIMAL_JOINTS: frozenset[str] = frozenset({"shoulder_pan", "shoulder_lift", "elbow_flex"})

GRIPPER_MOTOR = "gripper"


def _by_segment[T](proximal: T, distal: T) -> dict[str, T]:
    """Assign one value to the proximal joints and another to the distal ones."""
    return {joint: (proximal if joint in PROXIMAL_JOINTS else distal) for joint in JOINT_NAMES}


class MotorFamily(StrEnum):
    """Actuator family of a reBot B601 arm."""

    DM = "dm"
    RS = "rs"


# Arm control modes. Only MIT carries a feedforward torque term, so it is the only
# mode a torque-based feature (gravity compensation, force limiting) can build on.
ARM_MODE_MIT = "mit"
ARM_MODE_POS_VEL = "pos_vel"

# Gripper control modes.
GRIPPER_MODE_MIT = "mit"
GRIPPER_MODE_FORCE_POS = "force_pos"
GRIPPER_MODE_MIT_IMPEDANCE = "mit_impedance"


@dataclass(frozen=True)
class MotorFamilyProfile:
    """Hardware facts and default tuning for one motor family."""

    family: MotorFamily

    # --- hardware facts (never user-overridable) ---

    # Vendor model string per joint, passed to the matching motorbridge factory.
    motor_models: dict[str, str]
    # Control modes supported by this B601 integration. A vendor may expose more
    # protocol modes that have not been validated on this arm.
    arm_modes: frozenset[str]
    gripper_modes: frozenset[str]
    # --- defaults for the matching config fields ---

    # CAN transports supported by this motor family. The Damiao serial bridge is
    # vendor-specific; MotorBridge's native CAN transport can carry both families.
    can_adapters: frozenset[str]
    can_adapter: str
    control_mode: str
    gripper_control_mode: str
    motor_can_ids: dict[str, tuple[int, int]]
    # MIT gains per joint, including the gripper: the gripper's MIT gains live here
    # rather than in dedicated fields so each joint has a single source of truth.
    mit_kp: dict[str, float]
    mit_kd: dict[str, float]
    joint_limits: dict[str, tuple[float, float]]
    # Sign converting between the public robot coordinate frame and the raw motor
    # frame. It is applied in both directions so observations and actions share
    # one convention.
    joint_directions: dict[str, float]
    # Speed cap (deg/s) per joint for POS_VEL arm joints and the FORCE_POS gripper.
    # None on families without those modes.
    pos_vel_velocity: dict[str, float] | None
    # FORCE_POS gripper: grip force as a fraction of peak torque, in [0, 1].
    gripper_torque_ratio: float | None
    # Impedance gripper: max |feedforward torque| (N.m) while moving, and the
    # gentler cap applied at near-zero speed so a grasp neither crushes nor
    # overcurrents.
    gripper_torque_limit: float | None
    gripper_hold_torque_limit: float | None


DM_PROFILE = MotorFamilyProfile(
    family=MotorFamily.DM,
    motor_models=_by_segment("4340P", "4310"),
    arm_modes=frozenset({ARM_MODE_MIT, ARM_MODE_POS_VEL}),
    gripper_modes=frozenset({GRIPPER_MODE_FORCE_POS, GRIPPER_MODE_MIT}),
    can_adapters=frozenset({"damiao", "socketcan"}),
    can_adapter="damiao",
    control_mode=ARM_MODE_MIT,
    gripper_control_mode=GRIPPER_MODE_FORCE_POS,
    motor_can_ids={
        "shoulder_pan": (0x01, 0x11),
        "shoulder_lift": (0x02, 0x12),
        "elbow_flex": (0x03, 0x13),
        "wrist_flex": (0x04, 0x14),
        "wrist_yaw": (0x05, 0x15),
        "wrist_roll": (0x06, 0x16),
        "gripper": (0x07, 0x17),
    },
    mit_kp={
        "shoulder_pan": 45.0,
        "shoulder_lift": 45.0,
        "elbow_flex": 45.0,
        "wrist_flex": 8.0,
        "wrist_yaw": 9.0,
        "wrist_roll": 8.0,
        "gripper": 8.0,
    },
    mit_kd={
        "shoulder_pan": 12.0,
        "shoulder_lift": 12.0,
        "elbow_flex": 12.0,
        "wrist_flex": 1.0,
        "wrist_yaw": 1.0,
        "wrist_roll": 1.0,
        # The gripper is softer than the arm joints when it runs in MIT mode.
        "gripper": 0.3,
    },
    joint_limits={
        "shoulder_pan": (-150.0, 150.0),
        "shoulder_lift": (-200.0, 1.0),
        "elbow_flex": (-200.0, 1.0),
        "wrist_flex": (-80.0, 90.0),
        "wrist_yaw": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
        "gripper": (-270.0, 0.0),
    },
    joint_directions=dict.fromkeys(JOINT_NAMES, 1.0),
    pos_vel_velocity={**dict.fromkeys(JOINT_NAMES, 150.0), GRIPPER_MOTOR: 900.0},
    gripper_torque_ratio=0.07,
    gripper_torque_limit=None,
    gripper_hold_torque_limit=None,
)

RS_PROFILE = MotorFamilyProfile(
    family=MotorFamily.RS,
    motor_models=_by_segment("rs-06", "rs-00"),
    # RobStride motors have no FORCE_POS equivalent. MotorBridge exposes
    # RobStride position modes, but Seeed's current B601 integration uses MIT
    # because position-velocity operation was unstable. Supporting position mode
    # here requires a dedicated command path and hardware validation.
    arm_modes=frozenset({ARM_MODE_MIT}),
    gripper_modes=frozenset({GRIPPER_MODE_MIT_IMPEDANCE}),
    can_adapters=frozenset({"socketcan"}),
    # "socketcan" selects MotorBridge's native, platform-specific CAN transport.
    can_adapter="socketcan",
    control_mode=ARM_MODE_MIT,
    gripper_control_mode=GRIPPER_MODE_MIT_IMPEDANCE,
    # RobStride motors all reply on the host id rather than a per-motor recv id.
    motor_can_ids={joint: (i, 0xFD) for i, joint in enumerate(JOINT_NAMES, start=1)},
    mit_kp={
        "shoulder_pan": 50.0,
        "shoulder_lift": 150.0,
        "elbow_flex": 150.0,
        "wrist_flex": 50.0,
        "wrist_yaw": 50.0,
        "wrist_roll": 50.0,
        "gripper": 12.0,
    },
    mit_kd={
        "shoulder_pan": 3.0,
        "shoulder_lift": 10.0,
        "elbow_flex": 10.0,
        "wrist_flex": 5.0,
        "wrist_yaw": 4.0,
        "wrist_roll": 4.0,
        "gripper": 0.05,
    },
    # RS motors are installed opposite to the DM build, so these are the
    # positive-physical travel ranges. Incoming targets are mapped into them by
    # `joint_directions` before clipping.
    joint_limits={
        "shoulder_pan": (-145.0, 145.0),
        "shoulder_lift": (0.0, 170.0),
        "elbow_flex": (0.0, 200.0),
        "wrist_flex": (-80.0, 90.0),
        "wrist_yaw": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
        "gripper": (0.0, 270.0),
    },
    joint_directions=dict.fromkeys(JOINT_NAMES, -1.0),
    pos_vel_velocity=None,
    gripper_torque_ratio=None,
    gripper_torque_limit=3.5,
    gripper_hold_torque_limit=1.0,
)

PROFILES: dict[MotorFamily, MotorFamilyProfile] = {
    MotorFamily.DM: DM_PROFILE,
    MotorFamily.RS: RS_PROFILE,
}


def profile_for(family: MotorFamily | str) -> MotorFamilyProfile:
    """Return the hardware profile for a motor family.

    Raises:
        ValueError: if `family` is not a known motor family.
    """
    try:
        return PROFILES[MotorFamily(family)]
    except ValueError:
        known = ", ".join(f.value for f in MotorFamily)
        raise ValueError(f"Unknown motor_family '{family}'. Available families: {known}.") from None
