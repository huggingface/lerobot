# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/data/utils/kinematics.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import numpy as np

from lerobot.policies.sana_robot.sana_robot_core.robot80 import (
    gripper_closedness,
    pack_robot_rows,
    rotation_to_rot6d,
)

FLANGE_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def joint_origin(position, roll_pitch_yaw) -> np.ndarray:
    """Return the fixed [4, 4] transform from a joint's parent link to the joint."""
    roll, pitch, yaw = roll_pitch_yaw
    about_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]]
    )
    about_y = np.array(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]]
    )
    about_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    origin = np.eye(4)
    origin[:3, :3] = about_z @ about_y @ about_x
    origin[:3, 3] = position
    return origin


def joint_rotation(axis, angles: np.ndarray) -> np.ndarray:
    """Return the [T, 4, 4] transforms of a revolute joint turned by ``angles`` [T] about its axis."""
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = (
        np.eye(3)
        + np.sin(angles)[:, None, None] * cross
        + (1.0 - np.cos(angles))[:, None, None] * (cross @ cross)
    )
    motion = np.tile(np.eye(4), (len(angles), 1, 1))
    motion[:, :3, :3] = rotation
    return motion


def flange_poses(chain, joints: np.ndarray, tip=None, axes: np.ndarray = FLANGE_AXES) -> np.ndarray:
    """Return the [T, 4, 4] poses of an arm's flange in the arm's base frame for joint angles [T, J]; ``chain`` gives the origin position, the origin roll-pitch-yaw and the axis of every revolute joint, and ``tip`` the origin of a fixed link after the last joint. The columns of ``axes`` are the axes of the pose in the frame of the last link, by default +X approach, +Y down and +Z left."""
    joints = np.asarray(joints, dtype=np.float64)
    pose = np.tile(np.eye(4), (len(joints), 1, 1))
    for (position, roll_pitch_yaw, axis), angles in zip(chain, joints.T, strict=False):
        pose = pose @ joint_origin(position, roll_pitch_yaw) @ joint_rotation(axis, angles)
    if tip is not None:
        pose = pose @ joint_origin(*tip)
    pose[:, :3, :3] = pose[:, :3, :3] @ axes
    return pose


def dual_arm_rows(
    values: np.ndarray, chain, gripper_open: float, gripper_closed: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the [T, 80] robot rows and their slot mask for rows [T, 2 * (J + 1)] of a dual-arm robot, per arm J joint angles and the gripper reading: the joints, the flange pose from the forward kinematics of ``chain`` and the gripper closedness."""
    arm_joints = len(chain)
    fields = {}
    for side, start in (("left", 0), ("right", arm_joints + 1)):
        joints = values[:, start : start + arm_joints]
        pose = flange_poses(chain, joints)
        fields[f"{side}_joints"] = joints
        fields[f"{side}_eef_position"] = pose[:, :3, 3]
        fields[f"{side}_eef_rotation"] = rotation_to_rot6d(pose[:, :3, :3])
        fields[f"{side}_gripper"] = gripper_closedness(
            values[:, start + arm_joints], gripper_open, gripper_closed
        )
    return pack_robot_rows(fields)
