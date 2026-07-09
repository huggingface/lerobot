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

"""Standalone SMPL forward kinematics + canonicalization in pure numpy/scipy.

This mirrors the ``rt/smpl`` producer path in ``gear_sonic`` (``compute_from_body_poses``
-> ``process_smpl_joints`` -> ``compute_human_joints``) without depending on torch,
the SMPL mesh model, or ``gear_sonic``. It only needs a small fixed skeleton table
(rest-pose joints + kinematic tree), vendored in ``assets/smpl_skeleton.npz``.

Given the 24 body-joint poses reported by the XRoboToolkit headset SDK
(``xrt.get_body_joints_pose()`` -> (24, 7) of ``[x, y, z, qx, qy, qz, qw]``), it
produces the root-orientation-removed 24x3 SMPL joints the SONIC encoder expects,
plus the root orientation quaternion and pelvis translation.

Quaternions are scalar-first (w, x, y, z) unless noted.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# 24-joint parent tree used by the headset body-pose stream (SMPL-X body subset).
# Matches PoseStreamer.parent_indices in gear_sonic's pico_manager_thread_server.py.
BODY24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22],
    dtype=np.int64,
)

# FK output joints: first 22 SMPL body joints + two thumb tips (SMPL-X indices 39, 54).
OUTPUT_JOINT_INDEX = np.concatenate([np.arange(22), np.array([39, 54])])

_SKELETON_PATH = Path(__file__).parent / "assets" / "smpl_skeleton.npz"


# ── quaternion helpers (scalar-first w, x, y, z) ─────────────────────────────


def aa_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < 1e-8
    safe = np.where(small, 1.0, angle)
    axis = np.where(small, 0.0, aa / safe)
    half = angle * 0.5
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def quat_to_aa(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    w = q[..., 0:1]
    xyz = q[..., 1:]
    n = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, w)
    small = n < 1e-8
    safe = np.where(small, 1.0, n)
    axis = np.where(small, 0.0, xyz / safe)
    return axis * angle


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    c = quat_conj(q)
    return c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-12)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., 1:]
    w = q[..., 0:1]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def smpl_root_ytoz_up(root_quat: np.ndarray) -> np.ndarray:
    """Rotate the root quaternion 90 deg about X to map SMPL Y-up to robot Z-up."""
    base = aa_to_quat(np.array([np.pi / 2.0, 0.0, 0.0]))
    return quat_mul(base, root_quat)


def remove_smpl_base_rot(root_quat: np.ndarray) -> np.ndarray:
    """Conjugate out SMPL's default rest orientation ([0.5, 0.5, 0.5, 0.5])."""
    base_conj = quat_conj(np.array([0.5, 0.5, 0.5, 0.5]))
    return quat_mul(root_quat, base_conj)


# ── forward kinematics ───────────────────────────────────────────────────────


class SmplForwardKinematics:
    """Rest-skeleton SMPL forward kinematics (no mesh, no torch)."""

    def __init__(self, skeleton_path: str | Path = _SKELETON_PATH):
        data = np.load(skeleton_path)
        self.J = data["J"].astype(np.float64)  # (55, 3) rest joint positions
        self.parents = data["parents"].astype(np.int64)  # (55,) kinematic tree
        self.n_joints = self.J.shape[0]

    def _fk(self, full_pose_aa: np.ndarray) -> np.ndarray:
        """full_pose_aa: (n_joints, 3) axis-angle (joint 0 = global). Returns (24, 3)."""
        rot = R.from_rotvec(full_pose_aa).as_matrix()  # (n, 3, 3)
        rel = self.J.copy()
        rel[1:] -= self.J[self.parents[1:]]

        transforms = np.zeros((self.n_joints, 4, 4), np.float64)
        transforms[:, :3, :3] = rot
        transforms[:, :3, 3] = rel
        transforms[:, 3, 3] = 1.0

        chain = [transforms[0]]
        for i in range(1, self.n_joints):
            chain.append(chain[self.parents[i]] @ transforms[i])
        joints = np.stack(chain)[:, :3, 3]
        return joints[OUTPUT_JOINT_INDEX]

    def compute(self, body_poses_np: np.ndarray) -> dict:
        """Convert (24, 7) headset body poses to canonical SMPL joints.

        Args:
            body_poses_np: (24, 7) rows of [x, y, z, qx, qy, qz, qw] (scalar-last).

        Returns:
            dict with:
              - smpl_joints_local: (24, 3) root-orientation-removed joints
              - root_quat: (4,) root/torso orientation (w, x, y, z)
              - root_transl: (3,) pelvis translation
        """
        body_poses_np = np.asarray(body_poses_np, np.float64)
        positions = body_poses_np[:, :3]

        # Global joint rotations from the headset (scalar-last), with the SMPL
        # +180 deg-about-Y frame fix, converted to per-joint local axis-angle.
        global_rots = R.from_quat(body_poses_np[:, 3:7]) * R.from_euler("y", 180, degrees=True)
        gm = global_rots.as_matrix()  # (24, 3, 3)

        local_aa = np.zeros((24, 3), np.float64)
        for i in range(24):
            p = BODY24_PARENTS[i]
            m = gm[i] if p == -1 else gm[p].T @ gm[i]
            local_aa[i] = R.from_matrix(m).as_rotvec()

        global_orient = local_aa[0]
        body_pose = local_aa[1:].reshape(-1)[:63]  # 21 body joints

        # Root: Y-up -> Z-up, then run FK with the transformed root.
        root_quat = aa_to_quat(global_orient)
        root_quat = smpl_root_ytoz_up(root_quat)
        global_orient_new = quat_to_aa(root_quat)

        full_pose = np.concatenate(
            [global_orient_new, body_pose, np.zeros(3 * self.n_joints - 66)]
        ).reshape(self.n_joints, 3)
        joints = self._fk(full_pose)  # (24, 3)

        # Canonicalize: strip SMPL base rot and the root orientation.
        root_quat = remove_smpl_base_rot(root_quat)
        inv = np.broadcast_to(quat_inv(root_quat), (joints.shape[0], 4))
        smpl_joints_local = quat_apply(inv, joints)

        return {
            "smpl_joints_local": smpl_joints_local.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
            "root_transl": positions[0].astype(np.float32),
        }
