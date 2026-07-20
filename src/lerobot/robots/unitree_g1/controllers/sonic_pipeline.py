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

"""SONIC encoder/decoder pipeline for the Unitree G1 whole-body controller.

Pure-Python/ONNX re-implementation of the reference-tracking half of NVIDIA's SONIC
deploy stack (mirrors ``g1_deploy_onnx_ref.cpp``). Given a reference motion buffer
(joint targets + body orientation per frame) it produces 50 Hz joint-position targets
for the robot's PD controller. The upstream *motion planner* is intentionally absent:
here the reference is supplied directly by the caller (e.g. a 34-D OpenHLM / pi0.5 VLA
command per tick, in ``sonic_whole_body.py``).

Two cooperating ONNX models:
  * **encoder** – compresses the reference window into a 64-D latent ``token``
    (refreshed every ``ENCODER_UPDATE_EVERY`` ticks).
  * **decoder** – every tick, maps the token + recent proprioception history to a
    residual action that is scaled and added to ``DEFAULT_ANGLES``.

Index spaces: joints exist in two orderings — **IsaacLab** (policy/training order)
and **MuJoCo** (deploy order). ``ISAACLAB_TO_MUJOCO`` / ``MUJOCO_TO_ISAACLAB`` convert
between them. Quaternions are scalar-first ``(w, x, y, z)``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _onnxruntime_available

from ..g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    G1_29_JointIndex,
    get_gravity_orientation,
)

if TYPE_CHECKING or _onnxruntime_available:
    import onnxruntime as ort
else:
    ort = None

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# Robot/motor physical constants and the joint-order permutation tables. All
# 29-vectors are in IsaacLab joint order unless the name says ``_MUJOCO``.

# Nominal standing pose (rad), 29 joints in IsaacLab order. Actions are residuals
# added on top of this; also used as the planner/encoder standing reference.
DEFAULT_ANGLES = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

# Per-motor-type parameters used to derive action scaling and PD gains. Keys are
# Unitree motor model names; ARMATURE = rotor inertia, EFFORT = torque limit (N·m).
NATURAL_FREQ = 10.0 * 2.0 * np.pi  # target closed-loop stiffness bandwidth (rad/s)
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}


def _action_scale(k):
    """Per-motor residual-action scale (maps policy output to joint-angle delta)."""
    return 0.25 * EFFORT[k] / (ARMATURE[k] * NATURAL_FREQ**2)


# Per-joint motor model (IsaacLab order): legs, waist, then arms. Single source of
# truth for both ACTION_SCALE and compute_kp_kd().
MOTOR_MODELS = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2
    + ["7520_14", "5020", "5020"]
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"] * 2
)
ACTION_SCALE = np.array([_action_scale(k) for k in MOTOR_MODELS], dtype=np.float32)  # (29,) IsaacLab order

CONTROL_DT = 0.02  # 50 Hz control period (s)
DEFAULT_HEIGHT = 0.788740  # nominal pelvis height (m)
TOKEN_DIM = 64  # encoder latent size
ENCODER_UPDATE_EVERY = 5  # refresh the encoder token every N ticks (decoder runs every tick)
DEBUG_PRINT_EVERY = 100  # ticks between debug prints


def _to_mujoco(a):
    """Reorder a 29-vector from IsaacLab order into MuJoCo/deploy order."""
    return a[MUJOCO_TO_ISAACLAB]


DEFAULT_ANGLES_MUJOCO = _to_mujoco(DEFAULT_ANGLES)
ENCODER_STANDING_REF = DEFAULT_ANGLES.copy()

# Joint-index subsets (IsaacLab order) used to slice encoder observations.
LOWER_BODY_IL = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int32)  # 12 leg joints
WRIST_IL = np.array([23, 24, 25, 26, 27, 28], dtype=np.int32)  # 6 wrist joints
VR_TARGET_DEF = np.zeros(9, dtype=np.float32)  # 3-point VR position targets (mode 1)
VR_ORN_DEF = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)  # VR orn targets (mode 1)
SMPL_DEF = np.zeros(720, dtype=np.float32)  # SMPL whole-body window default (mode 2)

# ── PD gains ─────────────────────────────────────────────────────────────────


def compute_kp_kd():
    """Derive per-joint PD gains (kp, kd) from motor armature and target bandwidth.

    Ankle and waist joints get a x2 factor for extra stiffness. Returns two
    (29,) float32 arrays in IsaacLab joint order.
    """

    def s(k):
        return ARMATURE[k] * NATURAL_FREQ**2

    def d(k):
        return 2.0 * 2.0 * ARMATURE[k] * NATURAL_FREQ

    _double = {4, 5, 10, 11, 13, 14}  # ankle + waist indices with factor 2
    kp = np.array([2 * s(k) if i in _double else s(k) for i, k in enumerate(MOTOR_MODELS)], dtype=np.float32)
    kd = np.array([2 * d(k) if i in _double else d(k) for i, k in enumerate(MOTOR_MODELS)], dtype=np.float32)
    return kp, kd


_kp_kd = compute_kp_kd  # backward-compatible alias


# ── Quaternion helpers ────────────────────────────────────────────────────────
# All quaternions are scalar-first (w, x, y, z). "heading" = yaw-only quaternion.


def quat_conj(q):
    """Quaternion conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    """Hamilton product ``q1 ⊗ q2``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_to_6d(q):
    """Quaternion → 6-D rotation representation (first two rotated basis rows)."""
    w, x, y, z = q
    return np.array(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
        ],
        dtype=np.float32,
    )


def calc_heading(q):
    """Extract the yaw (heading) angle in radians from a quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def heading_quat(q, sign=1.0):
    """Yaw-only quaternion for ``q``'s heading (``sign=-1`` gives its inverse)."""
    a = sign * calc_heading(q) / 2.0
    return np.array([np.cos(a), 0, 0, np.sin(a)], dtype=np.float64)


def heading_quat_inv(q):
    """Inverse yaw-only quaternion for ``q``'s heading."""
    return heading_quat(q, -1.0)


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions (scalar ``t``)."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / (np.linalg.norm(r) + 1e-12)
    th = np.arccos(dot)
    st = np.sin(th)
    return (np.sin((1 - t) * th) / st) * q0 + (np.sin(t * th) / st) * q1


def quat_slerp_batch(q0, q1, t):
    """Vectorized slerp over arrays of quaternions with a per-row parameter ``t``."""
    q0 = q0 / (np.linalg.norm(q0, axis=1, keepdims=True) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1, axis=1, keepdims=True) + 1e-12)
    dot = np.sum(q0 * q1, axis=1)
    neg = dot < 0
    q1 = q1.copy()
    q1[neg] = -q1[neg]
    dot[neg] = -dot[neg]
    dot = np.clip(dot, -1, 1)
    lin = dot > 0.9995
    th = np.arccos(dot)
    st = np.where(np.sin(th) == 0, 1, np.sin(th))
    c0 = np.sin((1 - t) * th) / st
    c1 = np.sin(t * th) / st
    c0[lin] = 1 - t[lin]
    c1[lin] = t[lin]
    r = c0[:, None] * q0 + c1[:, None] * q1
    return r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)


def ort_providers(force_cpu: bool = False) -> list[str]:
    """Prefer CUDA for enc/dec/planner (matches deploy when onnxruntime-gpu is installed)."""
    avail = ort.get_available_providers()
    if not force_cpu and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_ort_session_options():
    """Build ONNX Runtime SessionOptions (quiet logging, default threading)."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return so


# ── Encoder / Decoder ─────────────────────────────────────────────────────────


class StandingEncoderDecoder:
    """Runs the encoder + decoder ONNX models and owns the proprioception history.

    Each tick it appends the latest robot state to 10-frame history buffers, builds
    the encoder observation (1762-D, layout depends on ``encode_mode``) to refresh
    the 64-D ``token``, then builds the decoder observation (994-D) and maps
    ``token + history`` to a residual action added onto ``DEFAULT_ANGLES``.

    ``PlannerController`` subclasses this to source the reference from a live,
    planner-generated motion buffer instead of a fixed standing pose.
    """

    def __init__(self, encoder, decoder):
        self.encoder, self.decoder = encoder, decoder
        self.encoder_input = encoder.get_inputs()[0].name
        self.decoder_input = decoder.get_inputs()[0].name
        enc_dim = int(encoder.get_inputs()[0].shape[1])
        dec_dim = int(decoder.get_inputs()[0].shape[1])
        if enc_dim != 1762 or dec_dim != 994:
            raise RuntimeError(f"Unexpected dims encoder={enc_dim}, decoder={dec_dim}")
        self.token = np.zeros(TOKEN_DIM, np.float32)
        self.last_action_mj = np.zeros(29, np.float32)
        self.h_q_mj = [np.zeros(29, np.float32)] * 10
        self.h_dq_mj = [np.zeros(29, np.float32)] * 10
        self.h_ang = [np.zeros(3, np.float32)] * 10
        self.h_act_mj = [np.zeros(29, np.float32)] * 10
        self.h_quat = [np.array([1, 0, 0, 0], np.float32)] * 10
        self.init_base_quat = np.array([1, 0, 0, 0], np.float32)
        self.init_ref_quat = np.array([1, 0, 0, 0], np.float32)
        self._heading_init = False
        self.encode_mode = 0
        self.vr_3point_local_target = VR_TARGET_DEF.copy()
        self.vr_3point_local_orn_target = VR_ORN_DEF.copy()
        self.smpl_joints_10frame_step1 = SMPL_DEF.copy()
        # Optional per-frame SMPL root orientation (wxyz) for the mode-2 anchor.
        # When None, the anchor falls back to the planner reference body quat.
        self.smpl_root_quat = None
        self.set_zero_reference()

    def update_history(self, q, dq, ang, quat):
        """Push the latest proprioception (pos/vel/gyro/orientation) into the 10-frame buffers."""
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        q_mj = _to_mujoco(q)
        dq_mj = _to_mujoco(dq)
        self.h_q_mj = [q_mj - DEFAULT_ANGLES_MUJOCO] + self.h_q_mj[:-1]
        self.h_dq_mj = [dq_mj] + self.h_dq_mj[:-1]
        self.h_ang = [ang.copy()] + self.h_ang[:-1]
        self.h_act_mj = [self.last_action_mj.copy()] + self.h_act_mj[:-1]
        self.h_quat = [quat.copy()] + self.h_quat[:-1]
        if not self._heading_init:
            self.init_base_quat = quat.copy()
            self._heading_init = True

    def _heading_quat(self, q):
        h = calc_heading(q) / 2.0
        return np.array([np.cos(h), 0, 0, np.sin(h)], np.float32)

    def _heading_quat_inv(self, q):
        h = calc_heading(q) / 2.0
        return np.array([np.cos(-h), 0, 0, np.sin(-h)], np.float32)

    def _anchor_6d(self, base_quat, ref_quat=None):
        """6-D orientation error between the robot base and the (heading-aligned) reference."""
        if ref_quat is None:
            ref_quat = self.init_ref_quat
        delta = quat_mul(self._heading_quat(self.init_base_quat), self._heading_quat_inv(self.init_ref_quat))
        new_ref = quat_mul(delta, ref_quat)
        return quat_to_6d(quat_mul(quat_conj(base_quat), new_ref))

    def set_zero_reference(self):
        """Initialize the reference to a single standing frame (used before a plan exists)."""
        self.motion_joint_positions = [ENCODER_STANDING_REF.copy()]
        self.motion_joint_velocities = [np.zeros(29, np.float32)]
        self.motion_body_quats = [np.array([1, 0, 0, 0], np.float32)]
        self.motion_body_z = [DEFAULT_HEIGHT]
        self.motion_timesteps = 1
        self.freeze_ref_frame = 0
        self.init_ref_quat = self.motion_body_quats[0].copy()

    def build_encoder_obs(self):
        """Assemble the 1762-D encoder input; slot layout depends on ``encode_mode``.

        mode 0 = locomotion (ref joint pos + anchor), 1 = 3-point VR teleop
        (lower-body ref + VR targets), 2 = SMPL whole-body window + anchor/wrist.
        """
        obs = np.zeros(1762, np.float32)
        obs[0] = float(self.encode_mode)
        rf = min(self.freeze_ref_frame, self.motion_timesteps - 1)
        ref_pos, ref_quat = self.motion_joint_positions[rf], self.motion_body_quats[rf]
        if self.encode_mode == 0:
            for f in range(10):
                obs[4 + 29 * f : 4 + 29 * (f + 1)] = ref_pos
                obs[601 + 6 * f : 601 + 6 * (f + 1)] = self._anchor_6d(self.h_quat[0], ref_quat)
        elif self.encode_mode == 1:
            ref_lower = ref_pos[LOWER_BODY_IL]
            for f in range(10):
                obs[661 + 12 * f : 661 + 12 * (f + 1)] = ref_lower
            obs[901:910] = self.vr_3point_local_target
            obs[910:922] = self.vr_3point_local_orn_target
            obs[595:601] = self._anchor_6d(self.h_quat[0], ref_quat)
        elif self.encode_mode == 2:
            # Prefer the SMPL clip/stream root orientation for the anchor; fall
            # back to the planner reference body quat when no root is provided.
            anchor_ref = self.smpl_root_quat if self.smpl_root_quat is not None else ref_quat
            obs[922:1642] = self.smpl_joints_10frame_step1
            for f in range(10):
                obs[1642 + 6 * f : 1642 + 6 * (f + 1)] = self._anchor_6d(self.h_quat[0], anchor_ref)
                obs[1702 + 6 * f : 1702 + 6 * (f + 1)] = ref_pos[WRIST_IL]
        else:
            raise RuntimeError(f"Unsupported encoder mode: {self.encode_mode}")
        return obs

    def build_decoder_obs(self):
        """Assemble the 994-D decoder input: token + 10-frame proprioception history + gravity."""
        obs = np.zeros(994, np.float32)
        off = 0
        obs[off : off + 64] = self.token
        off += 64
        for h, sz in [
            (list(reversed(self.h_ang)), 3),
            (list(reversed(self.h_q_mj)), 29),
            (list(reversed(self.h_dq_mj)), 29),
            (list(reversed(self.h_act_mj)), 29),
        ]:
            for f in range(10):
                obs[off : off + sz] = h[f]
                off += sz
        for q in reversed(self.h_quat):
            obs[off : off + 3] = get_gravity_orientation(q)
            off += 3
        assert off == 994, f"Decoder obs mismatch: {off}"
        return obs

    def run_encoder(self):
        """Run the encoder ONNX model and return the fresh 64-D token."""
        return (
            self.encoder.run(None, {self.encoder_input: self.build_encoder_obs().reshape(1, -1)})[0]
            .squeeze()
            .astype(np.float32)
        )

    def step(self, robot_obs, update_encoder, debug=False):
        """One control tick: read robot obs, (optionally) re-encode, decode → joint targets.

        Args:
            robot_obs: dict with ``<joint>.q``/``.dq`` and ``imu.*`` fields.
            update_encoder: refresh the token this tick (else reuse the cached one).
            debug: print action/delta norms.

        Returns:
            dict of ``<joint>.q`` target positions (rad) in IsaacLab joint order.
        """
        jnames = [m.name for m in G1_29_JointIndex]
        q = np.array(
            [
                robot_obs.get(f"{n}.q", DEFAULT_ANGLES[m.value])
                for m, n in zip(G1_29_JointIndex, jnames, strict=False)
            ],
            np.float32,
        )
        dq = np.array([robot_obs.get(f"{n}.dq", 0.0) for n in jnames], np.float32)
        quat = np.array(
            [
                robot_obs.get("imu.quat.w", 1),
                robot_obs.get("imu.quat.x", 0),
                robot_obs.get("imu.quat.y", 0),
                robot_obs.get("imu.quat.z", 0),
            ],
            np.float32,
        )
        ang = np.array([robot_obs.get(f"imu.gyro.{a}", 0) for a in "xyz"], np.float32)
        self.update_history(q, dq, ang, quat)
        if update_encoder:
            self.token = self.run_encoder()
        action_mj = (
            self.decoder.run(None, {self.decoder_input: self.build_decoder_obs().reshape(1, -1)})[0]
            .squeeze()
            .astype(np.float32)
        )
        self.last_action_mj = action_mj.copy()
        target = DEFAULT_ANGLES + action_mj[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        if debug:
            delta = target - q
            logger.debug(
                "token_norm=%.4f action_norm=%.4f delta_max=%.4f delta_rms=%.4f",
                np.linalg.norm(self.token),
                np.linalg.norm(action_mj),
                np.max(np.abs(delta)),
                np.sqrt(np.mean(delta**2)),
            )
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}


class PlannerController(StandingEncoderDecoder):
    """Encoder/decoder driven by a caller-supplied, rolling motion buffer.

    Extends ``StandingEncoderDecoder`` so the reference comes from a motion buffer
    (a lookahead window with per-frame velocities) instead of a single fixed pose,
    and handles heading re-initialization on the first frame / after a reset.
    ``motion_lock`` guards the buffer, which the whole-body controller rewrites each
    tick from the incoming command. The class name is retained for continuity with
    the SONIC reference; no motion planner is involved.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.ref_cursor = 0
        self.motion_timesteps = 0
        self.motion_joint_positions = np.zeros((1500, 29), np.float64)
        self.motion_joint_velocities = np.zeros((1500, 29), np.float64)
        self.motion_body_quats = np.zeros((1500, 4), np.float64)
        self.motion_body_quats[:, 0] = 1.0
        self.motion_body_pos = np.zeros((1500, 3), np.float64)
        self.init_ref_quat = np.array([1, 0, 0, 0], np.float64)
        self.heading_init_base_quat = np.array([1, 0, 0, 0], np.float64)
        self.delta_heading = 0.0
        self.reinit_heading = False
        self.playing = self.first_motion = False
        self.motion_lock = threading.Lock()

    def _heading_apply_delta(self):
        """Heading correction quaternion (init base-vs-ref heading + operator ``delta_heading``)."""
        delta = quat_mul(
            heading_quat(self.heading_init_base_quat).astype(np.float32),
            heading_quat_inv(self.init_ref_quat).astype(np.float32),
        )
        if self.delta_heading:
            h = self.delta_heading / 2.0
            delta = quat_mul(np.array([np.cos(h), 0, 0, np.sin(h)], np.float32), delta)
        return delta

    def _anchor_6d(self, base_quat, ref_quat=None):
        """6-D base-vs-reference orientation error, including the operator heading delta."""
        if ref_quat is None:
            ref_quat = self.init_ref_quat
        new_ref = quat_mul(self._heading_apply_delta(), ref_quat.astype(np.float32))
        return quat_to_6d(quat_mul(quat_conj(base_quat.astype(np.float32)), new_ref))

    def build_encoder_obs(self):
        """Encoder input sourced from the live motion buffer (mode 0/2), lock-protected."""
        obs = np.zeros(1762, np.float32)
        obs[0] = float(self.encode_mode)
        with self.motion_lock:
            if self.encode_mode == 2:
                # SMPL whole-body imitation: the 720-dim SMPL window carries the
                # target pose; the planner reference frame supplies anchor + wrist.
                rf = min(self.ref_cursor, self.motion_timesteps - 1)
                ref_pos = self.motion_joint_positions[rf].astype(np.float32)
                ref_quat = self.motion_body_quats[rf].astype(np.float32)
                # Prefer the SMPL clip/stream root orientation (if provided) so the
                # anchor tracks the operator's/clip's heading; else planner ref.
                if self.smpl_root_quat is not None:
                    ref_quat = np.asarray(self.smpl_root_quat, np.float32)
                anchor = self._anchor_6d(self.h_quat[0], ref_quat)
                wrist = ref_pos[WRIST_IL]
                obs[922:1642] = self.smpl_joints_10frame_step1
                for f in range(10):
                    obs[1642 + 6 * f : 1642 + 6 * (f + 1)] = anchor
                    obs[1702 + 6 * f : 1702 + 6 * (f + 1)] = wrist
                return obs
            if self.encode_mode == 1:
                # 3-point VR teleop: the upper body tracks the VR wrist/neck targets
                # while the planner reference supplies the lower body + anchor. Lower
                # body is per-frame (step 5) like mode 0; the VR targets are current.
                rf = min(self.ref_cursor, self.motion_timesteps - 1)
                obs[595:601] = self._anchor_6d(self.h_quat[0], self.motion_body_quats[rf].astype(np.float32))
                for f in range(10):
                    tf = min(
                        self.ref_cursor + f * 5 if self.playing else self.ref_cursor,
                        self.motion_timesteps - 1,
                    )
                    ref_lower = self.motion_joint_positions[tf].astype(np.float32)[LOWER_BODY_IL]
                    obs[661 + 12 * f : 661 + 12 * (f + 1)] = ref_lower
                obs[901:910] = self.vr_3point_local_target
                obs[910:922] = self.vr_3point_local_orn_target
                return obs
            for f in range(10):
                tf = min(
                    self.ref_cursor + f * 5 if self.playing else self.ref_cursor, self.motion_timesteps - 1
                )
                obs[4 + 29 * f : 4 + 29 * (f + 1)] = self.motion_joint_positions[tf].astype(np.float32)
                if self.playing:
                    obs[294 + 29 * f : 294 + 29 * (f + 1)] = self.motion_joint_velocities[tf].astype(
                        np.float32
                    )
                obs[601 + 6 * f : 601 + 6 * (f + 1)] = self._anchor_6d(
                    self.h_quat[0], self.motion_body_quats[tf].astype(np.float32)
                )
        return obs

    def step(self, robot_obs, update_encoder, debug=False):
        """Re-init the heading reference on first frame / after a reset, then run the base step."""
        if robot_obs and (self.first_motion or self.reinit_heading):
            q = None
            if "imu.quat.w" in robot_obs:
                q = np.array(
                    [
                        robot_obs["imu.quat.w"],
                        robot_obs["imu.quat.x"],
                        robot_obs["imu.quat.y"],
                        robot_obs["imu.quat.z"],
                    ],
                    np.float64,
                )
            else:
                q = robot_obs.get("imu.quaternion")
                if q is not None:
                    q = np.array(q, np.float64)
            if q is not None:
                self.heading_init_base_quat = np.array(q, np.float64)
                with self.motion_lock:
                    rf = min(self.ref_cursor, self.motion_timesteps - 1)
                    if self.encode_mode == 2 and self.smpl_root_quat is not None:
                        # Anchor the heading delta to the SMPL root at init so the
                        # robot turns *relative* to the clip/operator start heading.
                        self.init_ref_quat = np.asarray(self.smpl_root_quat, np.float64)
                    else:
                        self.init_ref_quat = self.motion_body_quats[rf].copy()
                self.delta_heading = 0.0
                self.first_motion = False
                self.reinit_heading = False
                logger.debug("[Heading] init quat: %s", self.heading_init_base_quat)
        return super().step(robot_obs, update_encoder=update_encoder, debug=debug)

    def advance_cursor(self):
        """Advance the reference cursor one frame per 50 Hz tick (no wall-clock catch-up)."""
        if not self.playing:
            return
        with self.motion_lock:
            if self.motion_timesteps > 0:
                self.ref_cursor = min(self.ref_cursor + 1, self.motion_timesteps - 1)
