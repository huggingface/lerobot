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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from .smpl_constants import VR3_N_POINTS, VR3_ORN_DIM, VR3_POS_DIM

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


# ── 3-point VR teleop keypoints (SONIC encode_mode 1) ────────────────────────
# SMPL body-joint indices for the 3 tracked keypoints, plus the root/pelvis (0)
# used as the reference frame. Mirrors gear_sonic ``_process_3pt_pose``: neck
# (joint 12) is used rather than head (15) — it is more rigidly coupled to the
# torso and less noisy than the free-looking head.
_VR3_JOINTS = (22, 23, 12)  # left wrist, right wrist, neck

# Per-keypoint rotation offsets aligning each SMPL joint frame to the robot
# convention (root, left wrist, right wrist, neck), ported verbatim from
# gear_sonic ``pico_manager_thread_server.OFFSETS`` (extrinsic xyz euler, degrees).
_VR3_OFFSETS = [
    R.from_euler("xyz", [0, 0, -90], degrees=True),  # root
    R.from_euler("xyz", [90, 0, 0], degrees=True),  # left wrist
    R.from_euler("xyz", [-90, 0, 180], degrees=True),  # right wrist
    R.from_euler("xyz", [0, 0, -90], degrees=True),  # neck
]

# Unity (X-right, Y-up, Z-forward, left-handed) -> robot (X-forward, Y-left,
# Z-up, right-handed) axis remap: Unity [x, y, z] -> robot [-x, z, y].
_UNITY_TO_ROBOT = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def _safe_quat(quats: np.ndarray) -> np.ndarray:
    """Replace zero-norm quaternions with the scalar-last identity.

    The headset reports ``[0, 0, 0, 0]`` for joints it isn't currently tracking;
    ``scipy.Rotation.from_quat`` rejects zero-norm quaternions, so we substitute the
    identity ``[0, 0, 0, 1]`` (no rotation) for those rows to keep FK robust.
    """
    quats = np.asarray(quats, np.float64).copy()
    bad = np.linalg.norm(quats, axis=-1) < 1e-8
    quats[bad] = (0.0, 0.0, 0.0, 1.0)  # scalar-last identity
    return quats


def compute_3point(body_poses_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the SONIC 3-point VR targets from headset body poses.

    Mirrors gear_sonic ``_process_3pt_pose``: transforms the tracked joints from the
    Unity frame to the robot frame, applies the per-keypoint rotation offsets, then
    expresses the left-wrist / right-wrist / neck keypoints relative to the root
    (pelvis) frame. This is the ``encode_mode == 1`` counterpart to :func:`compute`.

    Note: physical wrist/neck position offsets and the operator calibration done in
    gear_sonic's ``ThreePointPose.apply_calibration`` are not applied here; the raw
    tracked joint poses are used.

    Args:
        body_poses_np: (24, 7) rows of ``[x, y, z, qx, qy, qz, qw]`` (scalar-last),
            in the Unity frame, as returned by ``xrt.get_body_joints_pose()``.

    Returns:
        (pos, orn):
          - pos: (9,) float32, root-relative ``[x, y, z]`` for [l-wrist, r-wrist, neck]
          - orn: (12,) float32, root-relative ``[w, x, y, z]`` for the same order
    """
    body = np.asarray(body_poses_np, np.float64)
    q = _UNITY_TO_ROBOT
    # Root (index 0) + the 3 tracked keypoints, each transformed to the robot frame
    # and rotation-offset-corrected.
    positions = np.zeros((4, 3), np.float64)
    rotations: list[R] = []
    quats = _safe_quat(body[:, 3:7])
    for out_i, j in enumerate((0, *_VR3_JOINTS)):
        positions[out_i] = q @ body[j, :3]
        rot = R.from_quat(quats[j]).as_matrix()  # scalar-last input
        rotations.append(R.from_matrix(q @ rot @ q.T) * _VR3_OFFSETS[out_i])

    root_inv = rotations[0].inv()
    root_pos = positions[0]
    pos = np.zeros(VR3_POS_DIM, np.float32)
    orn = np.zeros(VR3_ORN_DIM, np.float32)
    for k in range(VR3_N_POINTS):
        pos[k * 3 : k * 3 + 3] = root_inv.apply(positions[k + 1] - root_pos)
        orn[k * 4 : k * 4 + 4] = (root_inv * rotations[k + 1]).as_quat(scalar_first=True)  # wxyz
    return pos, orn


# ── 3-point VR teleop from raw device poses (no body trackers) ───────────────
# PICO Y-up (X-right, Y-up, Z-back) -> robot Z-up world. Ported verbatim from
# gear_sonic's controller path (``decoupled_wbc`` ``PicoStreamer.R_HEADSET_TO_WORLD``).
_HEADSET_TO_WORLD = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _device_pose_to_world(pose: np.ndarray) -> tuple[np.ndarray, R]:
    """Convert a raw (7,) device pose ``[x, y, z, qx, qy, qz, qw]`` (PICO Y-up frame)
    to a Z-up world ``(position, Rotation)`` pair.

    Handles the all-zero quaternion the SDK emits when a device is momentarily
    untracked by substituting the identity, matching ``PicoStreamer._process_xr_pose``.
    """
    pose = np.asarray(pose, np.float64)
    xyz = _HEADSET_TO_WORLD @ pose[:3]
    quat = pose[3:7]  # scalar-last
    if np.linalg.norm(quat) < 1e-8:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    rot = _HEADSET_TO_WORLD @ R.from_quat(quat).as_matrix() @ _HEADSET_TO_WORLD.T
    return xyz, R.from_matrix(rot)


def compute_3point_from_devices(
    head_pose: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the SONIC 3-point VR targets from raw head + controller poses.

    This is the controller-state path (no PICO Motion Trackers / body tracking
    required): the 3 keypoints are the left controller, right controller, and the
    headset, each expressed relative to the **headset yaw frame** — mirroring
    gear_sonic's ``decoupled_wbc`` ``PicoStreamer._process_xr_pose`` (Y-up -> Z-up,
    then de-headed by the headset yaw). The headset stands in for the "neck" point,
    so its root-relative position is ~0 and its orientation carries pitch/roll.

    Args:
        head_pose, left_pose, right_pose: (7,) ``[x, y, z, qx, qy, qz, qw]`` device
            poses (scalar-last) from ``xrt.get_headset_pose()`` /
            ``xrt.get_left_controller_pose()`` / ``xrt.get_right_controller_pose()``.

    Returns:
        (pos, orn):
          - pos: (9,) float32, headset-yaw-relative ``[x, y, z]`` for [l-wrist, r-wrist, head]
          - orn: (12,) float32, headset-yaw-relative ``[w, x, y, z]`` for the same order
    """
    head_pos, head_rot = _device_pose_to_world(head_pose)
    left_pos, left_rot = _device_pose_to_world(left_pose)
    right_pos, right_rot = _device_pose_to_world(right_pose)

    # De-head: cancel the headset yaw so targets are expressed in a heading-local frame.
    head_yaw = head_rot.as_euler("xyz")[2]
    inv_yaw = R.from_euler("z", -head_yaw)

    points = ((left_pos, left_rot), (right_pos, right_rot), (head_pos, head_rot))
    pos = np.zeros(VR3_POS_DIM, np.float32)
    orn = np.zeros(VR3_ORN_DIM, np.float32)
    for k, (p_pos, p_rot) in enumerate(points):
        pos[k * 3 : k * 3 + 3] = inv_yaw.apply(p_pos - head_pos)
        orn[k * 4 : k * 4 + 4] = (inv_yaw * p_rot).as_quat(scalar_first=True)  # wxyz
    return pos, orn


# ── operator calibration for the 3-point targets ────────────────────────────
# G1 neutral key-frame targets, pelvis-relative [x, y, z] in metres, from MuJoCo FK on
# g1_29dof at the robot's *standing* configuration (``default_angles`` — the pose the
# robot actually holds at calibration time), with gear_sonic's local key-frame offsets
# (``G1_KEY_FRAME_OFFSETS``: wrists +0.18x ∓0.025y, torso +0.35z). These are the poses
# the operator's rest pose is mapped onto so the handoff starts at the robot's neutral
# stance. This is the fixed-``default_angles`` stand-in for gear_sonic's
# ``get_g1_key_frame_poses(q=measured_q)`` (see :meth:`ThreePointCalibrator.capture`):
# since the robot stands at ``default_angles`` after the startup ramp, its measured q
# equals this configuration, so these constants are the measured-q targets for the
# nominal case. (Per-frame *live* measured q would need a reverse controller->publisher
# feedback channel; not wired.)
_G1_NEUTRAL_WRIST_POS = np.array(
    [[0.2232, 0.2177, -0.1555], [0.2232, -0.2177, -0.1555]], np.float64
)
# Wrist neutral orientations (scalar-first w, x, y, z) at ``default_angles`` — NOT
# identity: the wrists are rolled/pitched in the standing pose. Matches gear_sonic
# using ``g1_lwrist_rot`` / ``g1_rwrist_rot`` from FK (not identity) as the rotation
# calibration reference.
_G1_NEUTRAL_WRIST_ROT = [
    R.from_quat([0.9168, 0.0897, 0.3864, 0.0463], scalar_first=True),  # left
    R.from_quat([0.9168, -0.0897, 0.3864, -0.0463], scalar_first=True),  # right
]
# Neck reconstruction chain (mirrors ThreePointPose._apply_calibration): torso link
# +0.05 z, then +0.35 along the neck's local Z.
_NECK_TORSO_OFFSET_Z = 0.05
_NECK_LINK_LENGTH = 0.35


@dataclass
class ThreePointCalibrator:
    """Aligns raw 3-point VR targets to the G1's neutral stance.

    Ports gear_sonic ``ThreePointPose._capture_calibration`` / ``_apply_calibration``:
    on :meth:`capture` (operator holding a neutral rest pose) it records (a) the
    inverse of the head/neck orientation, used to de-tilt all points to upright, and
    (b) per-wrist position + orientation offsets that map the corrected rest pose onto
    the fixed G1 neutral wrist targets. :meth:`apply` then transforms every subsequent
    frame by those offsets, and reconstructs the head/neck position from the calibrated
    neck orientation via the torso->neck kinematic chain.

    All quaternions are scalar-first (w, x, y, z), matching :func:`compute_3point`.
    """

    _neck_quat_inv: R | None = field(default=None, init=False)
    _wrist_pos_offset: np.ndarray | None = field(default=None, init=False)
    _wrist_rot_offset: list[R] = field(default_factory=list, init=False)

    @property
    def is_calibrated(self) -> bool:
        return self._neck_quat_inv is not None and self._wrist_pos_offset is not None

    def reset(self) -> None:
        self._neck_quat_inv = None
        self._wrist_pos_offset = None
        self._wrist_rot_offset = []

    def recalibrate_wrists(self) -> None:
        """Clear only the wrist offsets, preserving the neck calibration.

        Mirrors gear_sonic ``ThreePointPose.reset_with_measured_q``: the next
        :meth:`capture` recomputes the wrist offsets (against the G1 neutral targets)
        while keeping the already-captured neck orientation, so re-aligning the arms
        doesn't force the operator to re-level their head.
        """
        self._wrist_pos_offset = None
        self._wrist_rot_offset = []

    def capture(self, pos: np.ndarray, orn: np.ndarray) -> None:
        """Capture calibration offsets from a neutral-pose frame.

        Neck calibration is captured once and then preserved across subsequent
        captures (matching gear_sonic's ``if self._calibration_neck_quat_inv is
        None``); call :meth:`reset` to clear it or :meth:`recalibrate_wrists` to
        re-align only the arms.

        Args:
            pos: (9,) root-relative ``[x, y, z]`` for [l-wrist, r-wrist, head].
            orn: (12,) root-relative ``[w, x, y, z]`` for the same order.
        """
        pos = np.asarray(pos, np.float64).reshape(3, 3)
        orn = np.asarray(orn, np.float64).reshape(3, 4)
        if self._neck_quat_inv is None:
            self._neck_quat_inv = R.from_quat(orn[2], scalar_first=True).inv()
        neck_inv = self._neck_quat_inv

        self._wrist_pos_offset = np.zeros((2, 3), np.float64)
        self._wrist_rot_offset = []
        for k in range(2):
            corrected_pos = neck_inv.apply(pos[k])
            corrected_rot = neck_inv * R.from_quat(orn[k], scalar_first=True)
            self._wrist_pos_offset[k] = corrected_pos - _G1_NEUTRAL_WRIST_POS[k]
            # rot_offset maps the corrected rest orientation onto the G1 neutral wrist
            # orientation: calibrated = rot_offset * (neck_inv * current).
            self._wrist_rot_offset.append(_G1_NEUTRAL_WRIST_ROT[k] * corrected_rot.inv())

    def apply(self, pos: np.ndarray, orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the stored calibration; returns calibrated ``(pos (9,), orn (12,))``.

        A no-op (returns the inputs unchanged) until :meth:`capture` has been called.
        """
        if self._neck_quat_inv is None or self._wrist_pos_offset is None:
            return (
                np.asarray(pos, np.float32).reshape(-1),
                np.asarray(orn, np.float32).reshape(-1),
            )
        pos = np.asarray(pos, np.float64).reshape(3, 3)
        orn = np.asarray(orn, np.float64).reshape(3, 4)
        neck_inv = self._neck_quat_inv

        out_pos = np.zeros((3, 3), np.float64)
        out_orn = np.zeros((3, 4), np.float64)
        for k in range(2):  # wrists
            out_pos[k] = neck_inv.apply(pos[k]) - self._wrist_pos_offset[k]
            corrected_rot = neck_inv * R.from_quat(orn[k], scalar_first=True)
            out_orn[k] = (self._wrist_rot_offset[k] * corrected_rot).as_quat(scalar_first=True)

        # Head/neck: orientation de-tilted, position from the torso->neck chain.
        neck_rot = neck_inv * R.from_quat(orn[2], scalar_first=True)
        out_orn[2] = neck_rot.as_quat(scalar_first=True)
        neck_z = neck_rot.apply([0.0, 0.0, 1.0])
        out_pos[2] = np.array([0.0, 0.0, _NECK_TORSO_OFFSET_Z]) + _NECK_LINK_LENGTH * neck_z
        return out_pos.reshape(-1).astype(np.float32), out_orn.reshape(-1).astype(np.float32)


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
        global_rots = R.from_quat(_safe_quat(body_poses_np[:, 3:7])) * R.from_euler("y", 180, degrees=True)
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
