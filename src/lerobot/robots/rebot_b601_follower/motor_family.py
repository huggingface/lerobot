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

"""Motor-family hardware facts and defaults for the reBot B601."""

from dataclasses import dataclass
from enum import StrEnum

# Joint order shared by both builds.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

GRIPPER_MOTOR = "gripper"


class MotorFamily(StrEnum):
    """Actuator family of a reBot B601 arm."""

    DM = "dm"
    RS = "rs"


class ArmControlMode(StrEnum):
    """Control modes supported by the arm joints."""

    MIT = "mit"
    POS_VEL = "pos_vel"


class GripperControlMode(StrEnum):
    """Control modes supported by the gripper."""

    MIT = "mit"
    FORCE_POS = "force_pos"
    MIT_IMPEDANCE = "mit_impedance"


@dataclass
class MotorFamilyProfile:
    """Hardware facts and default tuning for one motor family."""

    motor_models: dict[str, str]
    # Sign from public joint positions to raw motor positions.
    joint_directions: dict[str, float]
    can_adapter: str
    gripper_control_mode: GripperControlMode
    motor_can_ids: dict[str, tuple[int, int]]
    mit_kp: dict[str, float]
    mit_kd: dict[str, float]
    # Soft limits in the public joint coordinate frame.
    joint_limits: dict[str, tuple[float, float]]
    # POS_VEL and FORCE_POS speed limits in deg/s.
    pos_vel_velocity: dict[str, float] | None
    gripper_torque_ratio: float | None
    # Impedance gripper moving and holding torque limits in N.m.
    gripper_torque_limit: float | None
    gripper_hold_torque_limit: float | None


DM_PROFILE = MotorFamilyProfile(
    motor_models={
        "shoulder_pan": "4340P",
        "shoulder_lift": "4340P",
        "elbow_flex": "4340P",
        "wrist_flex": "4310",
        "wrist_yaw": "4310",
        "wrist_roll": "4310",
        "gripper": "4310",
    },
    can_adapter="damiao",
    gripper_control_mode=GripperControlMode.FORCE_POS,
    motor_can_ids={joint: (motor_id, motor_id + 0x10) for motor_id, joint in enumerate(JOINT_NAMES, start=1)},
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
    motor_models={
        "shoulder_pan": "rs-06",
        "shoulder_lift": "rs-06",
        "elbow_flex": "rs-06",
        "wrist_flex": "rs-00",
        "wrist_yaw": "rs-00",
        "wrist_roll": "rs-00",
        "gripper": "rs-00",
    },
    # MotorBridge native CAN transport.
    can_adapter="socketcan",
    gripper_control_mode=GripperControlMode.MIT_IMPEDANCE,
    # All motors reply to host ID 0xFD.
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
    # Public limits corresponding to the reversed raw motor ranges.
    joint_limits={
        "shoulder_pan": (-145.0, 145.0),
        "shoulder_lift": (-170.0, 0.0),
        "elbow_flex": (-200.0, 0.0),
        "wrist_flex": (-90.0, 80.0),
        "wrist_yaw": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
        "gripper": (-270.0, 0.0),
    },
    joint_directions=dict.fromkeys(JOINT_NAMES, -1.0),
    pos_vel_velocity=None,
    gripper_torque_ratio=None,
    gripper_torque_limit=3.5,
    gripper_hold_torque_limit=1.0,
)

MOTOR_PROFILES: dict[MotorFamily, MotorFamilyProfile] = {
    MotorFamily.DM: DM_PROFILE,
    MotorFamily.RS: RS_PROFILE,
}
