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

"""G1 29-DOF joint limits (rad) in MuJoCo / G1_29_JointIndex order — from g1_29dof.xml.

SONIC encoder mode 0 expects Isaac Lab order; the whole-body controller remaps on ingest.
"""

import numpy as np

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

# (low, high) per joint index 0..28
_G1_LIMITS = np.array(
    [
        (-2.5307, 2.8798),  # kLeftHipPitch
        (-0.5236, 2.9671),  # kLeftHipRoll
        (-2.7576, 2.7576),  # kLeftHipYaw
        (-0.087267, 2.8798),  # kLeftKnee
        (-0.87267, 0.5236),  # kLeftAnklePitch
        (-0.2618, 0.2618),  # kLeftAnkleRoll
        (-2.5307, 2.8798),  # kRightHipPitch
        (-2.9671, 0.5236),  # kRightHipRoll
        (-2.7576, 2.7576),  # kRightHipYaw
        (-0.087267, 2.8798),  # kRightKnee
        (-0.87267, 0.5236),  # kRightAnklePitch
        (-0.2618, 0.2618),  # kRightAnkleRoll
        (-2.618, 2.618),  # kWaistYaw
        (-0.52, 0.52),  # kWaistRoll
        (-0.52, 0.52),  # kWaistPitch
        (-3.0892, 2.6704),  # kLeftShoulderPitch
        (-1.5882, 2.2515),  # kLeftShoulderRoll
        (-2.618, 2.618),  # kLeftShoulderYaw
        (-1.0472, 2.0944),  # kLeftElbow
        (-1.97222, 1.97222),  # kLeftWristRoll
        (-1.61443, 1.61443),  # kLeftWristPitch
        (-1.61443, 1.61443),  # kLeftWristYaw
        (-3.0892, 2.6704),  # kRightShoulderPitch
        (-2.2515, 1.5882),  # kRightShoulderRoll
        (-2.618, 2.618),  # kRightShoulderYaw
        (-1.0472, 2.0944),  # kRightElbow
        (-1.97222, 1.97222),  # kRightWristRoll
        (-1.61443, 1.61443),  # kRightWristPitch
        (-1.61443, 1.61443),  # kRightWristYaw
    ],
    dtype=np.float32,
)

JOINT_NAMES = [m.name for m in G1_29_JointIndex]
JOINT_LO = _G1_LIMITS[:, 0]
JOINT_HI = _G1_LIMITS[:, 1]
