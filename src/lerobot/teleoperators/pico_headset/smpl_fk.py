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
(SMPL-X rest-pose joints + kinematic tree), hardcoded below as ``_SKELETON_J`` /
``_SKELETON_PARENTS`` so no external asset download is required.

Given the 24 body-joint poses reported by the XRoboToolkit headset SDK
(``xrt.get_body_joints_pose()`` -> (24, 7) of ``[x, y, z, qx, qy, qz, qw]``), it
produces the root-orientation-removed 24x3 SMPL joints the SONIC encoder expects,
plus the root orientation quaternion and pelvis translation.

Quaternions are scalar-first (w, x, y, z) unless noted.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

# 24-joint parent tree used by the headset body-pose stream (SMPL-X body subset).
# Matches PoseStreamer.parent_indices in gear_sonic's pico_manager_thread_server.py.
BODY24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22],
    dtype=np.int64,
)

# FK output joints: first 22 SMPL body joints + two thumb tips (SMPL-X indices 39, 54).
OUTPUT_JOINT_INDEX = np.concatenate([np.arange(22), np.array([39, 54])])

# SMPL-X rest-pose skeleton (55 joints), vendored inline to avoid an external
# asset. ``_SKELETON_PARENTS[i]`` is joint ``i``'s parent (-1 = root); ``_SKELETON_J``
# holds the (55, 3) rest-pose joint positions.
_SKELETON_PARENTS = np.array(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        15,
        15,
        15,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        20,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
        21,
        52,
        53,
    ],
    dtype=np.int64,
)

_SKELETON_J = np.array(
    [
        [0.0031232605688273907, -0.3514074683189392, 0.012036550790071487],
        [0.06131265312433243, -0.4441709518432617, -0.013964635320007801],
        [-0.06014421582221985, -0.4553154706954956, -0.009213820099830627],
        [0.00036056205863133073, -0.2415168583393097, -0.015581080690026283],
        [0.11600811034440994, -0.8229243755340576, -0.02336069941520691],
        [-0.10435417294502258, -0.8176955580711365, -0.026037702336907387],
        [0.009808260947465897, -0.10966360569000244, -0.02152106538414955],
        [0.07255466282367706, -1.2259838581085205, -0.05523664504289627],
        [-0.08893736451864243, -1.2284233570098877, -0.046229973435401917],
        [-0.0015221529174596071, -0.057428449392318726, 0.006925832014530897],
        [0.11981196701526642, -1.283981204032898, 0.06297968327999115],
        [-0.12774977087974548, -1.2867517471313477, 0.07281902432441711],
        [-0.01368661131709814, 0.10773860663175583, -0.024689510464668274],
        [0.04484200477600098, 0.027515273541212082, -0.0002946509048342705],
        [-0.04921707883477211, 0.026910223066806793, -0.006474069785326719],
        [0.011096873320639133, 0.2681904137134552, -0.003952245227992535],
        [0.16408103704452515, 0.08524329960346222, -0.015755590051412582],
        [-0.15179482102394104, 0.08043467253446579, -0.019142597913742065],
        [0.4182038903236389, 0.01309278141707182, -0.058214444667100906],
        [-0.4229443669319153, 0.04394219070672989, -0.04560968279838562],
        [0.6701906323432922, 0.03631401062011719, -0.06068652495741844],
        [-0.6722118258476257, 0.03940964490175247, -0.06093486770987511],
        [-0.004667762666940689, 0.2676706910133362, -0.009591402485966682],
        [0.03159928321838379, 0.31083211302757263, 0.062195174396038055],
        [-0.031599875539541245, 0.3108319342136383, 0.0621943436563015],
        [0.7720924615859985, 0.02762586995959282, -0.04133538901805878],
        [0.8040408492088318, 0.02984413132071495, -0.03832494467496872],
        [0.8265857696533203, 0.027494050562381744, -0.03827037289738655],
        [0.7795881628990173, 0.029986342415213585, -0.06466733664274216],
        [0.8101993799209595, 0.030793743208050728, -0.0686899945139885],
        [0.8337147831916809, 0.028784994035959244, -0.07280422002077103],
        [0.7542374730110168, 0.02177468128502369, -0.10443613678216934],
        [0.7697082161903381, 0.020643100142478943, -0.11643557250499725],
        [0.7852417230606079, 0.018978532403707504, -0.12765252590179443],
        [0.767634928226471, 0.02704637683928013, -0.08803117275238037],
        [0.7956817150115967, 0.028531836345791817, -0.09329714626073837],
        [0.8185034990310669, 0.02707355096936226, -0.1003914326429367],
        [0.7108263969421387, 0.01833728514611721, -0.03507564589381218],
        [0.7278420925140381, 0.01931309886276722, -0.010097505524754524],
        [0.7483652234077454, 0.01415354385972023, 0.005425570998340845],
        [-0.7720924019813538, 0.027626780793070793, -0.041334930807352066],
        [-0.8040405511856079, 0.029844673350453377, -0.03832409530878067],
        [-0.8265854716300964, 0.027495287358760834, -0.03826868534088135],
        [-0.7795882225036621, 0.029987698420882225, -0.0646686926484108],
        [-0.8101993799209595, 0.030795171856880188, -0.06869153678417206],
        [-0.8337149024009705, 0.02878585271537304, -0.07280556112527847],
        [-0.7542385458946228, 0.021775206550955772, -0.10443780571222305],
        [-0.7697089910507202, 0.02064313367009163, -0.11643654853105545],
        [-0.7852423787117004, 0.01897839829325676, -0.1276528388261795],
        [-0.7676352858543396, 0.02704770304262638, -0.08803359419107437],
        [-0.7956817150115967, 0.028532907366752625, -0.093299500644207],
        [-0.818503737449646, 0.027074117213487625, -0.10039224475622177],
        [-0.7108249664306641, 0.018335221335291862, -0.035073522478342056],
        [-0.7278403043746948, 0.019311318174004555, -0.01009594276547432],
        [-0.7483659386634827, 0.014154116623103619, 0.005425604991614819],
    ],
    dtype=np.float32,
)


# ── fixed frame corrections (shared by FK + canonicalization) ────────────────
# Both mirror the SONIC deploy transform. ``_YTOZ_UP`` maps SMPL's Y-up world to the
# robot's Z-up (a 90 deg rotation about X); ``_SMPL_BASE`` is SMPL's rest-pose base
# orientation, conjugated out during canonicalization.
_YTOZ_UP = R.from_euler("x", 90, degrees=True)
_SMPL_BASE = R.from_quat([0.5, 0.5, 0.5, 0.5])  # scalar-last; symmetric so wxyz==xyzw


# ── forward kinematics ───────────────────────────────────────────────────────


def canonicalize_smpl_joints(smpl_joints: np.ndarray, root_aa: np.ndarray) -> np.ndarray:
    """Remove per-frame root orientation -> SONIC ``smpl_joints_local`` format.

    Mirrors the deploy transform (and ``motion_loader.canonicalize_smpl_joints``):
    reference clips store world-frame joints, but the encoder wants each frame's
    joints with the body root orientation removed.

    Args:
        smpl_joints: (T, 24, 3) world-frame (z-up) SMPL joint positions.
        root_aa: (T, 3) SMPL global-orient axis-angle (y-up convention).

    Returns:
        (T, 24, 3) per-frame root-orientation-removed joints.
    """
    root = _YTOZ_UP * R.from_rotvec(root_aa)
    inv = _SMPL_BASE * root.inv()
    return np.einsum("tij,tkj->tki", inv.as_matrix(), smpl_joints).astype(np.float32)


def root_quats_from_aa(root_aa: np.ndarray) -> np.ndarray:
    """Per-frame root orientation as (T, 4) wxyz, matching the live ``root_quat``.

    Same convention as the headset stream: ytoz-up then base-rotation removed.
    """
    root = (_YTOZ_UP * R.from_rotvec(root_aa)) * _SMPL_BASE.inv()
    return root.as_quat(scalar_first=True).astype(np.float32)  # wxyz


class SmplForwardKinematics:
    """Rest-skeleton SMPL forward kinematics (no mesh, no torch)."""

    def __init__(self, skeleton_path: str | Path | None = None):
        if skeleton_path is not None:
            data = np.load(skeleton_path)
            self.J = data["J"].astype(np.float64)  # (55, 3) rest joint positions
            self.parents = data["parents"].astype(np.int64)  # (55,) kinematic tree
        else:
            self.J = _SKELETON_J.astype(np.float64)
            self.parents = _SKELETON_PARENTS.copy()
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
        root = _YTOZ_UP * R.from_rotvec(global_orient)
        global_orient_new = root.as_rotvec()

        full_pose = np.concatenate([global_orient_new, body_pose, np.zeros(3 * self.n_joints - 66)]).reshape(
            self.n_joints, 3
        )
        joints = self._fk(full_pose)  # (24, 3)

        # Canonicalize: strip SMPL base rot and the root orientation.
        root = root * _SMPL_BASE.inv()
        smpl_joints_local = root.inv().apply(joints)

        return {
            "smpl_joints_local": smpl_joints_local.astype(np.float32),
            "root_quat": root.as_quat(scalar_first=True).astype(np.float32),  # wxyz
            "root_transl": positions[0].astype(np.float32),
        }
