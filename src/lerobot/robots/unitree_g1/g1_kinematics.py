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

"""Arm IK and gravity compensation of the G1-29, built on `lerobot.model`."""

import logging
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from lerobot.envs.configs import G1EndEffector
from lerobot.model import GravityCompensation, RobotKinematics

logger = logging.getLogger(__name__)

G1_MODEL_REPO_ID = "lerobot/unitree-g1-mujoco"
_ARM_JOINT_PARTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
# URDF names of the arm joints, in G1_29_JointArmIndex order.
G1_29_ARM_JOINTS = tuple(f"{side}_{part}_joint" for side in ("left", "right") for part in _ARM_JOINT_PARTS)
# URDF names of the waist joints, in G1_29_JointIndex order.
G1_29_WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
# Models of the G1 carrying each end effector; there is none with bare wrists.
_URDF_BY_END_EFFECTOR = {
    G1EndEffector.DEX1: "g1_29dof_with_dex1_1.urdf",
    G1EndEffector.DEX3: "g1_body29_hand14.urdf",
}
_IK_URDF = _URDF_BY_END_EFFECTOR[G1EndEffector.DEX3]
_WRIST_FRAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# The IK's end effector sits 5 cm along each wrist yaw link's x axis.
_WRIST_TO_EE = np.eye(4)
_WRIST_TO_EE[0, 3] = 0.05
_EE_TO_WRIST = np.linalg.inv(_WRIST_TO_EE)
# Orientation and posture weights relative to position, as in the earlier IPOPT solver's costs (translation
# 50, rotation 1, distance of the joints from zero 0.02).
_IK_ORIENTATION_WEIGHT = 0.02
_IK_POSTURE_WEIGHT = 0.0004
# Weights of the last solutions, newest first, that smooth the IK output.
_IK_SMOOTHING_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


def g1_assets_dir() -> Path:
    """Download the G1 models, if needed, and return their directory."""
    return Path(snapshot_download(G1_MODEL_REPO_ID)) / "assets"


def make_g1_gravity_compensation(end_effector: G1EndEffector) -> GravityCompensation:
    """Build the gravity model of the G1-29's arms carrying `end_effector`, with the pelvis fixed.

    Its torques follow `G1_29_ARM_JOINTS`; pass the measured `G1_29_WAIST_JOINTS` as extra joints.

    Raises:
        ValueError: If there is no model of the G1 with this end effector.
    """
    urdf = _URDF_BY_END_EFFECTOR.get(G1EndEffector(end_effector))
    if urdf is None:
        raise ValueError(
            f"Gravity compensation has no G1 model with end_effector={end_effector}; "
            f"use one of {[str(effector) for effector in _URDF_BY_END_EFFECTOR]}"
        )
    return GravityCompensation(g1_assets_dir() / urdf, G1_29_ARM_JOINTS, base_frame="pelvis")


class WeightedMovingFilter:
    """Weighted average of the last `len(weights)` samples, newest first; repeated samples are skipped."""

    def __init__(self, weights: Sequence[float], data_size: int) -> None:
        self._weights = np.asarray(weights, dtype=float)
        self._data_size = data_size
        self._filtered_data = np.zeros(data_size)
        self._data_queue: deque[np.ndarray] = deque(maxlen=len(weights))

    def add_data(self, new_data: np.ndarray) -> None:
        if len(new_data) != self._data_size:
            raise ValueError(f"Expected {self._data_size} values; got {len(new_data)}")
        if self._data_queue and np.array_equal(new_data, self._data_queue[0]):
            return
        self._data_queue.appendleft(np.asarray(new_data, dtype=float))
        if len(self._data_queue) < len(self._weights):
            self._filtered_data = self._data_queue[0]
        else:
            self._filtered_data = np.asarray(self._data_queue).T @ self._weights

    @property
    def filtered_data(self) -> np.ndarray:
        return self._filtered_data


class G1ArmKinematics:
    """Two-arm IK of the G1-29: poses of both end effectors to the 14 arm joints.

    Joint positions are in radians and `G1_29_ARM_JOINTS` order. Each solve starts from the previous one,
    and its output is smoothed over the last four solves, for teleoperation.

    Args:
        frozen_joints (`Sequence[str]`, *optional*):
            Arm joints, by URDF name, that the IK keeps at zero.
    """

    def __init__(self, frozen_joints: Sequence[str] = ()) -> None:
        unknown = set(frozen_joints) - set(G1_29_ARM_JOINTS)
        if unknown:
            logger.warning(f"Ignoring frozen joints that are not G1 arm joints: {sorted(unknown)}")
        self.urdf_path = g1_assets_dir() / _IK_URDF
        self._moving = np.asarray([name not in frozen_joints for name in G1_29_ARM_JOINTS])
        self.kinematics = RobotKinematics(
            self.urdf_path,
            _WRIST_FRAMES,
            [name for name in G1_29_ARM_JOINTS if name not in frozen_joints],
            posture_weight=_IK_POSTURE_WEIGHT,
        )
        self.q = np.zeros(len(G1_29_ARM_JOINTS))
        self._filter = WeightedMovingFilter(_IK_SMOOTHING_WEIGHTS, len(G1_29_ARM_JOINTS))

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the left and right end-effector poses, as 4x4 matrices, for arm joint positions `q`."""
        joint_pos_deg = np.rad2deg(np.asarray(q, dtype=float)[self._moving])
        left, right = (
            self.kinematics.forward_kinematics(joint_pos_deg, frame) @ _WRIST_TO_EE for frame in _WRIST_FRAMES
        )
        return left, right

    def solve_ik(self, left_ee: np.ndarray, right_ee: np.ndarray) -> np.ndarray:
        """Return arm joint positions that reach both end-effector poses, given as 4x4 matrices."""
        targets = {
            frame: np.asarray(pose) @ _EE_TO_WRIST
            for frame, pose in zip(_WRIST_FRAMES, (left_ee, right_ee), strict=True)
        }
        solution_deg = self.kinematics.inverse_kinematics(
            np.rad2deg(self.q[self._moving]), targets, orientation_weight=_IK_ORIENTATION_WEIGHT
        )
        q = np.zeros(len(G1_29_ARM_JOINTS))
        q[self._moving] = np.deg2rad(solution_deg)
        self._filter.add_data(q)
        self.q = self._filter.filtered_data.copy()
        return self.q.copy()
