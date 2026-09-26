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

from enum import IntEnum

import numpy as np

# ruff: noqa: N801, N815

NUM_MOTORS = 29

# Joint-order permutation between IsaacLab and Mujoco convention
ISAACLAB_TO_MUJOCO = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int32,
)
MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO).astype(np.int32)

REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")
REMOTE_BUTTONS = tuple(f"remote.button.{i}" for i in range(16))
REMOTE_KEYS = REMOTE_AXES + REMOTE_BUTTONS


def default_remote_input() -> dict[str, float]:
    """Return a zeroed-out remote input dict (axes + buttons)."""
    return dict.fromkeys(REMOTE_KEYS, 0.0)


def get_gravity_orientation(quaternion: list[float] | np.ndarray) -> np.ndarray:
    """Get gravity orientation from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def make_ort_session_options(
    intra_op_num_threads: int | None = None, inter_op_num_threads: int | None = None
):
    """Build quiet ONNX Runtime SessionOptions, optionally capping the CPU thread pool.

    These tiny MLP policies are latency-bound, not throughput-bound, so letting ORT grab
    every core starves the real-time control loop / torch policy and causes stutter. Pass
    1 intra + 1 inter thread for lowest-latency per-step inference.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if intra_op_num_threads is not None:
        so.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        so.inter_op_num_threads = inter_op_num_threads
    return so


class G1_29_JointArmIndex(IntEnum):
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


class G1_29_JointIndex(IntEnum):
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
