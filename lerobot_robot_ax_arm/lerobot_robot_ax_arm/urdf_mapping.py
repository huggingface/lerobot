"""Mapping between AX motor ticks and the URDF joint convention.

The AX-12A reports raw ticks (0-1023) over ~300 deg of travel, with a zero and direction
unrelated to the URDF. We map ticks <-> URDF joint degrees with a per-joint affine transform
anchored at a reference pose captured during calibration::

    q_urdf_deg = q_ref + sign * (tick - tick_ref) * SCALE

- ``SCALE``    : AX-12A mechanical travel per tick (300 deg / 1023 ticks).
- ``q_ref``    : URDF angle of the reference pose (:data:`REFERENCE_URDF_DEG`).
- ``tick_ref`` : tick captured at that reference pose, stored as ``MotorCalibration.homing_offset``.
- ``sign``     : mounting direction (+1/-1), stored as ``MotorCalibration.drive_mode`` (0 -> +1, 1 -> -1).

The ``homing_offset``/``drive_mode`` fields are unused by the Dynamixel Protocol-1.0 normalization
path, so repurposing them here does not affect anything else.
"""

from __future__ import annotations

import numpy as np

from lerobot.motors import MotorCalibration

# Arm joints in URDF order (base yaw, shoulder pitch, elbow pitch). The gripper is not remapped.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
URDF_JOINT_NAMES = ["robot_joint_1", "robot_joint_2", "robot_joint_3"]

AX_TRAVEL_DEG = 300.0  # AX-12A mechanical travel over the full 0-1023 tick range
AX_MAX_TICK = 1023.0
SCALE = AX_TRAVEL_DEG / AX_MAX_TICK  # URDF degrees per motor tick

# URDF joint angle (deg) of each arm joint at the calibration reference pose.
REFERENCE_URDF_DEG = {"shoulder_pan": 0.0, "shoulder_lift": 45.0, "elbow_flex": 90.0}
# URDF joint limits (deg) from ax_arm.urdf, used to guide the lower/upper jog during calibration.
URDF_LIMITS_DEG = {
    "shoulder_pan": (-45.0, 45.0),
    "shoulder_lift": (0.0, 90.0),
    "elbow_flex": (0.0, 90.0),
}


def _sign(calib: MotorCalibration) -> float:
    return -1.0 if calib.drive_mode else 1.0


def tick_to_urdf_deg(joint: str, tick: float, calib: MotorCalibration) -> float:
    return REFERENCE_URDF_DEG[joint] + _sign(calib) * (tick - calib.homing_offset) * SCALE


def urdf_deg_to_tick(joint: str, q_deg: float, calib: MotorCalibration) -> int:
    return int(round(calib.homing_offset + _sign(calib) * (q_deg - REFERENCE_URDF_DEG[joint]) / SCALE))


def ticks_to_urdf_vector(ticks: dict[str, float], calibration: dict[str, MotorCalibration]) -> np.ndarray:
    """Arm joint ticks -> URDF joint angles (deg), in :data:`ARM_JOINTS` order."""
    return np.array([tick_to_urdf_deg(j, ticks[j], calibration[j]) for j in ARM_JOINTS], dtype=float)


def urdf_vector_to_ticks(q_deg: np.ndarray, calibration: dict[str, MotorCalibration]) -> dict[str, int]:
    """URDF joint angles (deg), in :data:`ARM_JOINTS` order -> arm joint ticks."""
    return {j: urdf_deg_to_tick(j, float(q_deg[i]), calibration[j]) for i, j in enumerate(ARM_JOINTS)}
