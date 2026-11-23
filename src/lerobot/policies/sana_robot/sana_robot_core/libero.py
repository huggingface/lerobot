# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/data/utils/convert_libero.py. Import paths are rewritten to the LeRobot package;
# the robot rows of a LIBERO observation and the frames of the simulated Panda; the dataset converter is not vendored, and hand_pose is the first half of the upstream grip_site_pose.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import math

import numpy as np

from lerobot.policies.sana_robot.sana_robot_core.kinematics import flange_poses
from lerobot.policies.sana_robot.sana_robot_core.robot80 import (
    gripper_closedness,
    pack_robot_rows,
    rotation_to_rot6d,
)

GRIPPER_OPEN, GRIPPER_CLOSED = 0.08, 0.0
FRANKA_JOINTS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (-1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -0.316, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0825, 0.0, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((-0.0825, 0.384, 0.0), (-1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.088, 0.0, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
)
FRANKA_FLANGE = ((0.0, 0.0, 0.107), (0.0, 0.0, 0.0))
FRANKA_FLANGE_AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

HAND_HEIGHT, HAND_QUATERNION = 0.1065, (0.0, 0.0, -0.383, 0.924)
GRIP_SITE_HEIGHT, GRIP_SITE_TURN = 0.097, -math.pi / 2


def quaternion_to_rotation(quaternion) -> np.ndarray:
    """Return the rotation matrices [..., 3, 3] of unit quaternions [..., 4] in the order (x, y, z, w)."""
    x, y, z, w = np.moveaxis(np.asarray(quaternion, dtype=np.float64), -1, 0)
    rows = (
        (x * x - y * y - z * z + w * w, 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), -x * x + y * y - z * z + w * w, 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), -x * x - y * y + z * z + w * w),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def hand_pose() -> np.ndarray:
    """Return the fixed [4, 4] pose of the hand of the simulated Panda, the body whose rotation LIBERO observes, in the flange frame of ``state_rows``: the hand sits 0.1065 m after the last joint."""
    axes, hand = np.eye(4), np.eye(4)
    axes[:3, :3] = FRANKA_FLANGE_AXES
    hand[:3, :3] = quaternion_to_rotation(np.asarray(HAND_QUATERNION) / np.linalg.norm(HAND_QUATERNION))
    hand[2, 3] = HAND_HEIGHT - FRANKA_FLANGE[0][2]
    return axes.T @ hand


def grip_site_pose() -> np.ndarray:
    """Return the fixed [4, 4] pose of the grip site, the point LIBERO's controller moves, in the flange frame of ``state_rows``: the hand of the simulated Panda sits 0.1065 m after the last joint and the grip site 0.097 m after the hand."""
    site = np.eye(4)
    site[:3, :3] = quaternion_to_rotation(
        (0.0, 0.0, math.sin(GRIP_SITE_TURN / 2), math.cos(GRIP_SITE_TURN / 2))
    )
    site[2, 3] = GRIP_SITE_HEIGHT
    return hand_pose() @ site


def state_rows(joints: np.ndarray, fingers: np.ndarray) -> np.ndarray:
    """Return the state rows [T, 80] of joint angles [T, 7] and two finger positions [T, 2] in metres: the joints, the flange pose in the arm's base frame and the gripper closedness."""
    pose = flange_poses(FRANKA_JOINTS, joints, FRANKA_FLANGE, FRANKA_FLANGE_AXES)
    fingers = np.asarray(fingers, dtype=np.float64)
    state, _ = pack_robot_rows(
        {
            "right_joints": joints,
            "right_eef_position": pose[:, :3, 3],
            "right_eef_rotation": rotation_to_rot6d(pose[:, :3, :3]),
            "right_gripper": gripper_closedness(fingers[:, 0] - fingers[:, 1], GRIPPER_OPEN, GRIPPER_CLOSED),
        }
    )
    return state
