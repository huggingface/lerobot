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

"""
OpenLoong "Qinglong" humanoid robot utilities.

OpenLoong is a humanoid robot from Shanghai Humanoid Robotics Innovation Center.
It uses MPC (Model Predictive Control) and WBC (Whole-Body Control) for locomotion.
Reference: https://github.com/loongOpen/OpenLoong-Dyn-Control

Joint Configuration:
- 6 DOF per leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
- 3 DOF waist (yaw, roll, pitch)
- 7 DOF per arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
- Total: 29 joints (similar to Unitree G1 29-DOF configuration)
"""

from enum import IntEnum

# ruff: noqa: N801, N815

NUM_MOTORS = 29


class OpenLoongJointIndex(IntEnum):
    """
    Joint indices for OpenLoong humanoid robot.
    
    The robot has a total of 29 joints:
    - Left leg: 6 joints (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    - Right leg: 6 joints
    - Waist: 3 joints (yaw, roll, pitch)
    - Left arm: 7 joints (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
    - Right arm: 7 joints
    """
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    # Waist
    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class OpenLoongArmJointIndex(IntEnum):
    """Arm joint indices for OpenLoong."""
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class OpenLoongLegJointIndex(IntEnum):
    """Leg joint indices for OpenLoong."""
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11


# Joint limits (in radians) - based on typical humanoid robot limits
# These should be calibrated for the specific robot
OPENLOONG_JOINT_LIMITS = {
    # Left leg
    "kLeftHipPitch": (-0.87, 0.87),
    "kLeftHipRoll": (-0.52, 0.52),
    "kLeftHipYaw": (-0.87, 0.87),
    "kLeftKnee": (-0.17, 2.10),
    "kLeftAnklePitch": (-0.87, 0.52),
    "kLeftAnkleRoll": (-0.35, 0.35),
    # Right leg
    "kRightHipPitch": (-0.87, 0.87),
    "kRightHipRoll": (-0.52, 0.52),
    "kRightHipYaw": (-0.87, 0.87),
    "kRightKnee": (-0.17, 2.10),
    "kRightAnklePitch": (-0.87, 0.52),
    "kRightAnkleRoll": (-0.35, 0.35),
    # Waist
    "kWaistYaw": (-1.22, 1.22),
    "kWaistRoll": (-0.26, 0.26),
    "kWaistPitch": (-0.52, 0.52),
    # Left arm
    "kLeftShoulderPitch": (-2.97, 2.97),
    "kLeftShoulderRoll": (-0.52, 3.14),
    "kLeftShoulderYaw": (-2.97, 2.97),
    "kLeftElbow": (-0.17, 2.97),
    "kLeftWristRoll": (-0.79, 0.79),
    "kLeftWristPitch": (-0.79, 0.79),
    "kLeftWristYaw": (-0.79, 0.79),
    # Right arm
    "kRightShoulderPitch": (-2.97, 2.97),
    "kRightShoulderRoll": (-3.14, 0.52),
    "kRightShoulderYaw": (-2.97, 2.97),
    "kRightElbow": (-0.17, 2.97),
    "kRightWristRoll": (-0.79, 0.79),
    "kRightWristPitch": (-0.79, 0.79),
    "kRightWristYaw": (-0.79, 0.79),
}

# Default control gains for OpenLoong
# These are starting values and should be tuned for specific tasks
OPENLOONG_DEFAULT_GAINS = {
    "left_leg": {
        "kp": [150, 150, 150, 300, 40, 40],
        "kd": [2, 2, 2, 4, 2, 2],
    },  # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    "right_leg": {
        "kp": [150, 150, 150, 300, 40, 40],
        "kd": [2, 2, 2, 4, 2, 2],
    },
    "waist": {
        "kp": [250, 250, 250],
        "kd": [5, 5, 5],
    },  # yaw, roll, pitch
    "left_arm": {
        "kp": [80, 80, 80, 80, 40, 40, 40],
        "kd": [3, 3, 3, 3, 1.5, 1.5, 1.5],
    },  # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    "right_arm": {
        "kp": [80, 80, 80, 80, 40, 40, 40],
        "kd": [3, 3, 3, 3, 1.5, 1.5, 1.5],
    },
}

# Default standing position (all joints in neutral position)
OPENLOONG_DEFAULT_STANDING_POSITION = [
    # Left leg
    0.0, 0.0, 0.0, 0.3, -0.15, 0.0,
    # Right leg
    0.0, 0.0, 0.0, 0.3, -0.15, 0.0,
    # Waist
    0.0, 0.0, 0.0,
    # Left arm
    0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
    # Right arm
    0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
]

# Joint names for display/logging
OPENLOONG_JOINT_NAMES = [joint.name for joint in OpenLoongJointIndex]
