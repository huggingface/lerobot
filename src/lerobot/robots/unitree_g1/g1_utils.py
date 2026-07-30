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

import importlib
from enum import IntEnum

import numpy as np

# ruff: noqa: N801, N815

NUM_MOTORS = 29

# Joint-order permutations between the two 29-DoF layouts used across the G1 stack:
# IsaacLab (policy/training order) and MuJoCo (deploy order). ``a[ISAACLAB_TO_MUJOCO]``
# reorders an IsaacLab-ordered vector into MuJoCo order, and vice-versa.
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
MUJOCO_TO_ISAACLAB = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int32,
)

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


# Unitree motor-model parameters shared by the controllers that derive their PD gains
# from motor physics rather than hand-tuning (SONIC decoder, Holosoma). NATURAL_FREQ is
# the target closed-loop stiffness bandwidth (rad/s); MOTOR_ARMATURE is per-model rotor
# inertia (keys are Unitree motor model names). From these: kp = armature * w**2 and
# kd = 4 * armature * w, with an optional x2 factor on stiff joints (ankles/waist).
NATURAL_FREQ = 10.0 * 2.0 * np.pi
MOTOR_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}


def compute_pd_gains(motor_models, double_indices=()) -> tuple[np.ndarray, np.ndarray]:
    """Derive per-joint PD gains (kp, kd) from motor armature and target bandwidth.

    ``motor_models`` is a per-joint sequence of Unitree motor model names (in the
    controller's own joint order); joints whose index is in ``double_indices`` get a
    x2 stiffness/damping factor. Returns two (N,) float32 arrays in that same order.
    """
    double = set(double_indices)

    def s(k):
        return MOTOR_ARMATURE[k] * NATURAL_FREQ**2

    def d(k):
        return 4.0 * MOTOR_ARMATURE[k] * NATURAL_FREQ

    kp = np.array([2 * s(k) if i in double else s(k) for i, k in enumerate(motor_models)], dtype=np.float32)
    kd = np.array([2 * d(k) if i in double else d(k) for i, k in enumerate(motor_models)], dtype=np.float32)
    return kp, kd


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


def make_locomotion_controller(name: str | None):
    """Instantiate a locomotion controller by class name. Returns None if name is None."""
    if name is None:
        return None
    controllers = {
        "GrootLocomotionController": "lerobot.robots.unitree_g1.controllers.gr00t_locomotion",
        "HolosomaLocomotionController": "lerobot.robots.unitree_g1.controllers.holosoma_locomotion",
        "SonicWholeBodyController": "lerobot.robots.unitree_g1.controllers.sonic_whole_body",
    }
    module_path = controllers.get(name)
    if module_path is None:
        raise ValueError(f"Unknown controller: {name!r}. Available: {list(controllers)}")
    module = importlib.import_module(module_path)
    return getattr(module, name)()


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
