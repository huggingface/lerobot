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

# Reserved action-dict field used to forward the set of currently-pressed keyboard
# keys from a KeyboardTeleop through the standard action pipeline to the SONIC
# whole-body controller (see SonicWholeBodyController._process_keyboard).
KEYBOARD_KEYS_FIELD = "keyboard.keys"

# ── Dense whole-body joint reference (SONIC encode_mode 0, OpenHLM / pi0.5) ──────
# A single 34-D whole-body command per tick, in the OpenHLM action layout:
#   [L-arm(7), L-grip(1), R-arm(7), R-grip(1), L-leg(6), R-leg(6), waist(3),
#    root roll/pitch + yaw-rate(3)]
# Fed as flat scalars ``wb.0.pos .. wb.33.pos``. The ``.pos`` suffix makes these
# behave like ordinary joint-position action features so ``lerobot-rollout`` routes
# them straight from a 34-D VLA (OpenHLM / pi0.5) onto the robot.
WB_ACTION_PREFIX = "wb."
WB_ACTION_DIM = 34


def wb_action_key(i: int) -> str:
    """Action-dict key for the ``i``-th whole-body command scalar (``wb.{i}.pos``)."""
    return f"{WB_ACTION_PREFIX}{i}.pos"


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


def lowstate_to_obs(lowstate) -> dict:
    """Build a robot observation dict from a Unitree lowstate.

    Shared by ``UnitreeG1.get_observation`` and the SONIC pipeline so the
    lowstate -> obs mapping lives in exactly one place. Keys match the
    ``<joint>.q``/``imu.*`` schema consumed across the controllers.
    """
    obs: dict = {}

    for motor in G1_29_JointIndex:
        idx = motor.value
        obs[f"{motor.name}.q"] = lowstate.motor_state[idx].q
        obs[f"{motor.name}.dq"] = lowstate.motor_state[idx].dq
        obs[f"{motor.name}.tau"] = lowstate.motor_state[idx].tau_est

    imu = lowstate.imu_state
    if imu.gyroscope:
        obs["imu.gyro.x"] = imu.gyroscope[0]
        obs["imu.gyro.y"] = imu.gyroscope[1]
        obs["imu.gyro.z"] = imu.gyroscope[2]
    if imu.accelerometer:
        obs["imu.accel.x"] = imu.accelerometer[0]
        obs["imu.accel.y"] = imu.accelerometer[1]
        obs["imu.accel.z"] = imu.accelerometer[2]
    if imu.quaternion:
        obs["imu.quat.w"] = imu.quaternion[0]
        obs["imu.quat.x"] = imu.quaternion[1]
        obs["imu.quat.y"] = imu.quaternion[2]
        obs["imu.quat.z"] = imu.quaternion[3]
    if imu.rpy:
        obs["imu.rpy.roll"] = imu.rpy[0]
        obs["imu.rpy.pitch"] = imu.rpy[1]
        obs["imu.rpy.yaw"] = imu.rpy[2]

    wr = getattr(lowstate, "wireless_remote", None)
    if wr:
        obs["wireless_remote"] = bytes(wr) if not isinstance(wr, (bytes, bytearray)) else wr

    return obs


def obs_to_wb34_state(obs: dict) -> np.ndarray:
    """Build the 34-D OpenHLM / pi0.5 proprio state from a G1 observation dict.

    Mirrors the whole-body *action* layout so the policy sees state and action in
    the same coordinates::

        [L-arm(7), L-grip(1), R-arm(7), R-grip(1),
         L-leg(6), R-leg(6), waist(3), root roll/pitch + yaw-rate(3)]

    Joint positions come from the ``<joint>.q`` obs keys, which are already in
    MuJoCo / Unitree-SDK order — the same body-part grouping OpenHLM uses
    ([L-leg 0:6, R-leg 6:12, waist 12:15, L-arm 15:22, R-arm 22:29]) — so they are
    regrouped directly (no IsaacLab permutation). The G1 has no grippers in its
    29-DoF body, so both gripper slots are 0. Root roll/pitch are the IMU RPY and
    the last slot is the IMU yaw rate (gyro z).
    """
    q_mj = np.array(
        [float(obs.get(f"{m.name}.q", 0.0)) for m in G1_29_JointIndex],
        dtype=np.float32,
    )
    lleg, rleg, waist = q_mj[0:6], q_mj[6:12], q_mj[12:15]
    larm, rarm = q_mj[15:22], q_mj[22:29]

    state = np.zeros(34, dtype=np.float32)
    state[0:7] = larm
    # state[7] left gripper — none on 29-DoF G1
    state[8:15] = rarm
    # state[15] right gripper — none on 29-DoF G1
    state[16:22] = lleg
    state[22:28] = rleg
    state[28:31] = waist
    state[31] = float(obs.get("imu.rpy.roll", 0.0))
    state[32] = float(obs.get("imu.rpy.pitch", 0.0))
    state[33] = float(obs.get("imu.gyro.z", 0.0))
    return state


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
