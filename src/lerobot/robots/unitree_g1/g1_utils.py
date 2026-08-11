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
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class RobotController(Protocol):
    """Interface for the controllers driving ``UnitreeG1``'s background control thread.

    Covers both locomotion controllers (GR00T, Holosoma) and whole-body ones (SONIC).

    Each tick the robot hands the controller the latest lowstate plus a snapshot of the
    incoming action, and publishes the absolute joint targets it returns, keyed
    ``<joint>.q``. It lives in this module rather than in ``controllers/`` so that
    importing the robot does not pull in the controller implementations and their
    onnxruntime dependency.

    Controllers may also expose any of the following, which the robot picks up when present:

    - ``kp`` / ``kd``: ``(29,)`` PD gains published with the targets, overriding the config.
    - ``default_angles``: ``(29,)`` home pose that residual actions are applied onto.
    - ``action_ft`` / ``observation_ft``: feature dicts that take over the robot's default
      29-DoF action space and proprioceptive state (SONIC's 64-D latent token).
    - ``observation_state()``: current values for the keys advertised in ``observation_ft``.
    """

    control_dt: float
    """Control period in seconds; sets the rate of the robot's controller thread."""

    def run_step(self, action: dict, lowstate) -> dict:
        """Map one lowstate plus action into absolute joint targets keyed ``<joint>.q``."""
        ...

    def reset(self) -> None:
        """Drop per-episode state, e.g. history buffers and held commands."""
        ...


def make_robot_controller(name: str | None) -> RobotController | None:
    """Instantiate a robot controller by class name. Returns None if name is None."""
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
