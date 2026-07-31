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
from collections.abc import Iterable, Sequence
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

# ── G1 PD-gain derivation from Unitree motor physics (shared by SONIC + Holosoma) ──
# Rather than hand-tuning, the PD gains (kp/kd) and the residual action_scale are derived
# from motor physics: each joint is one of four Unitree motor models with a known rotor
# armature (inertia) and peak effort. For a target closed-loop natural frequency w, a
# critically-damped second-order response gives kp = armature * w**2 and kd = 4 * armature
# * w; stiff joints (ankles/waist) optionally get a 2x factor. Callers pass their own
# controller's per-joint motor-model sequence (SONIC uses IsaacLab order, Holosoma its
# own), so the returned arrays come back in that same joint order.

# Target closed-loop natural frequency (rad/s), i.e. 10 Hz.
NATURAL_FREQ = 10.0 * 2.0 * np.pi

# Rotor armature (kg·m²) and peak effort (N·m) per Unitree motor model.
G1_MOTOR_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
G1_MOTOR_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}

# Per-joint Unitree motor model in standard G1_29_JointIndex order (legs, waist, arms), and
# the stiff joints (ankles + waist roll/pitch) that receive a 2x gain factor. Controllers may
# override individual joints (e.g. Holosoma softens the hip-pitch joints) on top of these.
G1_MOTOR_MODELS = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]  # left leg
    + ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]  # right leg
    + ["7520_14", "5020", "5020"]  # waist: yaw, roll, pitch
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"]  # left arm
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"]  # right arm
)
STIFF_JOINT_INDICES = frozenset({4, 5, 10, 11, 13, 14})


def compute_pd_gains(
    motor_models: Sequence[str], double_indices: Iterable[int] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Derive PD gains (kp, kd) from Unitree motor physics.

    ``motor_models`` is a per-joint sequence of Unitree motor-model names in the caller's
    own joint order; joints whose index is in ``double_indices`` get a 2x stiffness/damping
    factor (the stiff ankle/waist joints). ``kp = armature * w**2`` and ``kd = 4 * armature
    * w`` (critically damped). Returns ``(kp, kd)``, each an (N,) float32 array in the same
    order as ``motor_models``.
    """
    double = set(double_indices)
    armature = np.array([G1_MOTOR_ARMATURE[m] for m in motor_models], dtype=np.float32)
    factor = np.array([2.0 if i in double else 1.0 for i in range(len(motor_models))], dtype=np.float32)
    kp = factor * armature * NATURAL_FREQ**2
    kd = factor * 4.0 * armature * NATURAL_FREQ
    return kp, kd


def compute_action_scale(motor_models: Sequence[str]) -> np.ndarray:
    """Derive the residual action scale from Unitree motor physics.

    ``action_scale = 0.25 * effort / (armature * w**2)``, returned as an (N,) float32 array
    in the same order as ``motor_models``. Maps a policy's residual output to a joint-angle
    delta added on top of the standing pose (``default_angles``).
    """
    armature = np.array([G1_MOTOR_ARMATURE[m] for m in motor_models], dtype=np.float32)
    effort = np.array([G1_MOTOR_EFFORT[m] for m in motor_models], dtype=np.float32)
    return 0.25 * effort / (armature * NATURAL_FREQ**2)


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
