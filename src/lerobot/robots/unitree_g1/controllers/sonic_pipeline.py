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

"""SONIC planner pipeline for the Unitree G1 whole-body controller.

This module is a pure-Python/ONNX re-implementation of NVIDIA's SONIC deploy stack
(mirrors ``g1_deploy_onnx_ref.cpp``). It turns a high-level movement intent
(walk/run/squat/box/… + speed/height/heading, driven by the joystick) into
50 Hz joint-position targets for the robot's PD controller.

Data flow (one 50 Hz control tick, orchestrated by ``SonicRuntime`` in
``sonic_whole_body.py``):

    intent (MovementState) ──► SonicPlanner ──► PlannerController ──► joint targets
                                   │                    │
                     (planner ONNX, 30 Hz,       (encoder+decoder ONNX,
                      async background thread)     runs every tick)

Three cooperating ONNX models:
  * **planner**  – generates a several-second *reference motion* (body trajectory +
    joint clip) for the current intent. Slow, so it runs asynchronously in a
    background thread (``_planner_worker``) and its 30 Hz output is resampled to
    50 Hz. New motions are cross-faded into the live buffer (``blend_new_motion``).
  * **encoder**  – compresses the reference window into a 64-D latent ``token``
    (refreshed every ``ENCODER_UPDATE_EVERY`` ticks).
  * **decoder**  – every tick, maps the token + recent proprioception history to a
    residual action that is scaled and added to ``DEFAULT_ANGLES``.

Encoder ``encode_mode`` selects what the reference represents:
  * ``0`` – locomotion (planner clip drives lower + upper body).
  * ``1`` – 3-point VR teleop (lower body from planner, arms from VR targets).
  * ``2`` – SMPL whole-body imitation (720-D SMPL window drives the pose).

Index spaces: joints exist in two orderings — **IsaacLab** (policy/training order)
and **MuJoCo** (deploy order). ``ISAACLAB_TO_MUJOCO`` / ``MUJOCO_TO_ISAACLAB`` convert
between them. Quaternions are scalar-first ``(w, x, y, z)``.

Section map: constants & index tables · PD gains · quaternion helpers · locomotion
modes · movement state · encoder/decoder · planner motion buffer · async planner
worker · ``SonicPlanner`` · ``PlannerController`` · joystick input.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
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
MOTION_LOOK_AHEAD_STEPS = 2  # frames ahead used to seed a replan context (hide planner latency)
INITIAL_RANDOM_SEED = 1234
MIN_TOKENS, MAX_TOKENS = 6, 16  # planner prediction-length token range
K = MAX_TOKENS - MIN_TOKENS + 1
DEADZONE = 0.05  # joystick dead zone
BLEND_FRAMES = 8  # cross-fade length when swapping in a freshly planned motion

# Seconds between automatic replans, per motion class (faster for dynamic motions).
REPLAN_INTERVAL = {"running": 0.1, "crawling": 0.2, "boxing": 1.0, "default": 1.0}


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


# ── Locomotion modes ──────────────────────────────────────────────────────────


class LocomotionMode(IntEnum):
    """High-level motion styles understood by the planner (fed as the ``mode`` input)."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    SQUAT = 4
    KNEEL_TWO_LEGS = 5
    KNEEL = 6
    LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19
    LEDGE_WALKING = 20
    OBJECT_CARRYING = 21
    STEALTH_WALK_2 = 22
    HAPPY_DANCE_WALK = 23
    ZOMBIE_WALK = 24
    GUN_WALK = 25
    SCARE_WALK = 26


LM = LocomotionMode

# UI groupings of modes for cycling with the n/p keys; each entry is (label, modes).
MOTION_SETS = [
    ("Standing", [LM.SLOW_WALK, LM.WALK, LM.RUN, LM.FORWARD_JUMP, LM.STEALTH_WALK, LM.INJURED_WALK]),
    ("Squat / Low", [LM.SQUAT, LM.KNEEL_TWO_LEGS, LM.KNEEL, LM.CRAWLING, LM.ELBOW_CRAWLING]),
    (
        "Boxing",
        [
            LM.IDLE_BOXING,
            LM.WALK_BOXING,
            LM.LEFT_PUNCH,
            LM.RIGHT_PUNCH,
            LM.RANDOM_PUNCH,
            LM.LEFT_HOOK,
            LM.RIGHT_HOOK,
        ],
    ),
    (
        "Styled Walks",
        [
            LM.LEDGE_WALKING,
            LM.OBJECT_CARRYING,
            LM.STEALTH_WALK_2,
            LM.HAPPY_DANCE_WALK,
            LM.ZOMBIE_WALK,
            LM.GUN_WALK,
            LM.SCARE_WALK,
        ],
    ),
]

# Mode classifications used by clamping and replan logic.
STATIC_MODES = {LM.IDLE, LM.SQUAT, LM.KNEEL_TWO_LEGS, LM.KNEEL, LM.LYING_FACE_DOWN, LM.IDLE_BOXING}
STANDING_MODES = {
    LM.IDLE,
    LM.SLOW_WALK,
    LM.WALK,
    LM.RUN,
    LM.IDLE_BOXING,
    LM.WALK_BOXING,
    LM.LEFT_PUNCH,
    LM.RIGHT_PUNCH,
    LM.RANDOM_PUNCH,
    LM.LEFT_HOOK,
    LM.RIGHT_HOOK,
    LM.FORWARD_JUMP,
    LM.STEALTH_WALK,
    LM.INJURED_WALK,
    LM.LEDGE_WALKING,
    LM.OBJECT_CARRYING,
    LM.STEALTH_WALK_2,
    LM.HAPPY_DANCE_WALK,
    LM.ZOMBIE_WALK,
    LM.GUN_WALK,
    LM.SCARE_WALK,
}
BOXING_MODES = {LM.WALK_BOXING, LM.LEFT_PUNCH, LM.RIGHT_PUNCH, LM.RANDOM_PUNCH, LM.LEFT_HOOK, LM.RIGHT_HOOK}
SPEED_RANGES = {
    LM.SLOW_WALK: (0.2, 0.8),
    LM.WALK: (0.8, 1.5),
    LM.RUN: (1.5, 3.0),
    LM.CRAWLING: (0.4, 1.0),
    LM.ELBOW_CRAWLING: (0.7, 1.0),
}


def clamp_mode_params(ms):
    """Clamp ``ms.speed``/``ms.height`` into the valid range for its mode in place.

    ``-1.0`` is a sentinel meaning "use the mode's default" (e.g. standing modes
    ignore height; static modes ignore speed).
    """
    m = LM(ms.mode)
    ms.height = -1.0 if m in STANDING_MODES else max(0.1, min(0.8, ms.height if ms.height >= 0 else 0.2))
    if m in STATIC_MODES:
        ms.speed = -1.0
    elif m in SPEED_RANGES:
        lo, hi = SPEED_RANGES[m]
        ms.speed = max(lo, min(hi, ms.speed if ms.speed >= 0 else lo))
    elif m in BOXING_MODES:
        ms.speed = max(0.7, min(1.5, ms.speed if ms.speed >= 0 else 0.7))
    else:
        ms.speed = -1.0


def replan_interval(mode):
    """Seconds between automatic replans for the given mode."""
    m = LM(mode)
    if m == LM.RUN:
        return REPLAN_INTERVAL["running"]
    if m == LM.CRAWLING:
        return REPLAN_INTERVAL["crawling"]
    if m in {LM.LEFT_PUNCH, LM.RIGHT_PUNCH, LM.RANDOM_PUNCH, LM.LEFT_HOOK, LM.RIGHT_HOOK}:
        return REPLAN_INTERVAL["boxing"]
    return REPLAN_INTERVAL["default"]


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


# ── Movement state ────────────────────────────────────────────────────────────


@dataclass
class MovementState:
    """Mutable high-level intent driven by keyboard/joystick and read by the planner.

    Holds the current locomotion ``mode``, target ``speed``/``height`` (``-1`` =
    mode default), facing/movement angles, and the ``needs_replan`` flag the control
    loop watches to decide when to request a fresh motion from the planner.
    """

    mode: int = LM.SLOW_WALK  # not IDLE — walking modes respond to WASD
    speed: float = -1.0
    height: float = -1.0
    facing_angle: float = 0.0
    movement_angle: float = 0.0
    has_movement: bool = False
    motion_set_idx: int = 0
    needs_replan: bool = False
    joy_prev_active: bool = False  # tracks right-stick activity across joystick polls

    @property
    def movement_direction(self):
        """Unit XY movement direction (0 vector when not moving)."""
        if not self.has_movement:
            return (0.0, 0.0, 0.0)
        return (math.cos(self.movement_angle), math.sin(self.movement_angle), 0.0)

    @property
    def facing_direction(self):
        """Unit XY facing direction."""
        return (math.cos(self.facing_angle), math.sin(self.facing_angle), 0.0)

    def status_line(self):
        """Human-readable one-line status for the terminal HUD."""
        return (
            f"[{MOTION_SETS[self.motion_set_idx][0]}] mode={self.mode}({LM(self.mode).name}) "
            f"spd={'default' if self.speed < 0 else f'{self.speed:.1f}'} "
            f"hgt={'default' if self.height < 0 else f'{self.height:.2f}'} "
            f"facing={math.degrees(self.facing_angle):.0f}° "
            f"{'moving' if self.has_movement else 'still'}"
        )


@dataclass
class MovementSnapshot:
    """Immutable copy of the intent at the last replan, for change detection."""

    mode: int = 0
    speed: float = -1.0
    height: float = -1.0
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (1.0, 0.0, 0.0)


def snapshot_ms(ms: MovementState) -> MovementSnapshot:
    """Capture the current movement intent as a comparable snapshot."""
    md, fd = ms.movement_direction, ms.facing_direction
    return MovementSnapshot(ms.mode, ms.speed, ms.height, (md[0], md[1], md[2]), (fd[0], fd[1], fd[2]))


def should_replan_request(ms: MovementState, last: MovementSnapshot, replan_timer: float, step: int) -> bool:
    """Decide whether to request a fresh plan this tick.

    Triggers on an explicit ``needs_replan`` flag, any mode/facing/height change, or
    (for non-static modes) speed/direction changes and periodic timeouts. Mirrors the
    C++ ``G1Deploy::Planner`` replan triggers (``g1_deploy_onnx_ref.cpp``).
    """
    if step <= 0:
        return False
    if ms.needs_replan:
        return True
    md, fd = ms.movement_direction, ms.facing_direction
    facing_changed = fd != last.facing
    height_changed = ms.height != last.height
    mode_changed = ms.mode != last.mode
    speed_changed = ms.speed != last.speed
    dir_changed = md != last.movement
    is_static = LM(ms.mode) in STATIC_MODES
    if mode_changed or facing_changed or height_changed:
        return True
    time_to_replan = replan_timer >= replan_interval(ms.mode)
    return not is_static and (speed_changed or dir_changed or (time_to_replan and ms.speed != 0))


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


# ── Planner motion buffer ─────────────────────────────────────────────────────


class PlannerMotion:
    """Fixed-capacity buffer for a planned motion (joint pos/vel + body pose per frame)."""

    def __init__(self, max_frames=1500):
        self.timesteps = 0
        self.joint_positions = np.zeros((max_frames, 29), np.float64)
        self.joint_velocities = np.zeros((max_frames, 29), np.float64)
        self.body_positions = np.zeros((max_frames, 3), np.float64)
        self.body_quaternions = np.zeros((max_frames, 4), np.float64)
        self.body_quaternions[:, 0] = 1.0


# ── Subprocess planner ────────────────────────────────────────────────────────


def _resample_30_to_50(qpos, n30):
    """Resample planner output (30 Hz MuJoCo qpos) to a 50 Hz IsaacLab-order motion.

    Returns a dict with joint positions/velocities (velocities via finite difference)
    and body position/orientation trajectories at 50 Hz.
    """
    t50 = int(np.floor(n30 / 30.0 * 50))
    f30 = np.arange(t50) / 50.0 * 30.0
    f0 = np.floor(f30).astype(int)
    f1 = np.minimum(f0 + 1, n30 - 1)
    frac, w0 = (f30 - f0).astype(np.float64), None
    w0 = 1.0 - frac
    jp = (w0[:, None] * qpos[f0, 7:36] + frac[:, None] * qpos[f1, 7:36])[:, MUJOCO_TO_ISAACLAB]
    jv = np.zeros_like(jp)
    if t50 >= 2:
        jv[: t50 - 1] = (jp[1:] - jp[:-1]) * 50.0
        jv[-1] = jv[-2]
    return {
        "timesteps": t50,
        "joint_positions": jp,
        "joint_velocities": jv,
        "body_positions": w0[:, None] * qpos[f0, :3] + frac[:, None] * qpos[f1, :3],
        "body_quaternions": quat_slerp_batch(qpos[f0, 3:7], qpos[f1, 3:7], frac),
    }


def _build_planner_inputs(ctx, ms_dict, version, seed):
    """Build the planner ONNX input dict from a context window + movement intent.

    ``version >= 1`` is the TensorRT-style deploy planner with extra height/target
    inputs and a token-count mask; ``version 0`` is the minimal input set.
    """
    inp = {
        "context_mujoco_qpos": ctx.astype(np.float32).reshape(1, 4, 36),
        "target_vel": np.array([ms_dict["speed"]], np.float32),
        "mode": np.array([ms_dict["mode"]], np.int64),
        "movement_direction": np.array(ms_dict["movement_direction"], np.float32).reshape(1, 3),
        "facing_direction": np.array(ms_dict["facing_direction"], np.float32).reshape(1, 3),
        "random_seed": np.array([seed], np.int64),
    }
    if version >= 1:
        # TensorRT deploy: allow 9–11 prediction tokens only (indices 3–5 for MIN_TOKENS=6).
        allowed = np.zeros((1, K), np.int64)
        if K >= 6:
            allowed[0, 3:6] = 1
        inp.update(
            {
                "height": np.array([ms_dict["height"]], np.float32),
                "has_specific_target": np.array([[0]], np.int64),
                "specific_target_positions": np.zeros((1, 4, 3), np.float32),
                "specific_target_headings": np.zeros((1, 4), np.float32),
                "allowed_pred_num_tokens": allowed,
            }
        )
    return inp


def _planner_worker(path, req_q, res_q, stop_evt, version, seed, use_gpu):
    """Background thread: consume replan requests, run the planner ONNX, post motions.

    Loads its own ONNX session, then loops pulling ``(ctx, gen_frame, ms_dict)`` off
    ``req_q``, running inference, resampling to 50 Hz, and putting the newest result
    on ``res_q`` (dropping stale entries). Runs until ``stop_evt`` is set.
    """
    so = make_ort_session_options()
    providers = ort_providers(force_cpu=not use_gpu)
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    while not stop_evt.is_set():
        try:
            ctx, gf, ms_dict = req_q.get(timeout=0.05)
        except queue.Empty:  # nosec B112 - idle poll, nothing queued this tick
            continue
        try:
            inp = _build_planner_inputs(ctx, ms_dict, version, seed)
            t0 = time.time()
            qpos_out, num_pred = sess.run(None, inp)
            t_inf = time.time()
            n = int(num_pred.flat[0])
            qpos = qpos_out[0, :n]
            if np.any(np.isnan(qpos)):
                continue
            motion = _resample_30_to_50(qpos, n)
            motion["gen_frame"] = gf
            logger.debug(
                "[Planner] inf=%.1fms total=%.1fms frames=%d",
                1000 * (t_inf - t0),
                1000 * (time.time() - t0),
                n,
            )
            while not res_q.empty():
                try:
                    res_q.get_nowait()
                except queue.Empty:
                    break
            res_q.put(motion)
        except Exception:
            logger.exception("[Planner] worker error")


# ── SonicPlanner ──────────────────────────────────────────────────────────────


class SonicPlanner:
    """Owns the planner ONNX model and its async background worker.

    Provides the initial motion synchronously (``initialize``), then serves replans
    off-thread: ``request_replan`` enqueues the current context+intent and
    ``try_get_new_motion`` non-blockingly returns a freshly planned motion (which the
    controller cross-fades in). ``version`` selects the planner input schema.
    """

    def __init__(self, session, planner_path):
        self.session = session
        self.planner_path = planner_path
        self.gen_frame = 0
        self.random_seed = INITIAL_RANDOM_SEED
        self.version = 1 if len(session.get_inputs()) >= 11 else 0
        self.motion_50hz = PlannerMotion()
        self._snapshot = PlannerMotion()
        self._req_q = self._res_q = self._stop_evt = self._planner_thread = None
        self._ctrl = None

    def _build_inputs(self, ctx, ms):
        return _build_planner_inputs(
            ctx,
            {
                "mode": ms.mode,
                "speed": ms.speed,
                "height": ms.height,
                "movement_direction": list(ms.movement_direction),
                "facing_direction": list(ms.facing_direction),
            },
            self.version,
            self.random_seed,
        )

    @staticmethod
    def build_initial_context(joint_positions):
        """Build a 4-frame standing context (MuJoCo qpos layout) from a pose."""
        ctx = np.zeros((4, 36), np.float32)
        jp_mj = joint_positions.astype(np.float32)[ISAACLAB_TO_MUJOCO]
        for n in range(4):
            ctx[n, 2] = DEFAULT_HEIGHT
            ctx[n, 3] = 1.0
            ctx[n, 7:36] = jp_mj
        return ctx

    def _context_from_controller(self, current_frame):
        """Sample a 4-frame look-ahead context from the controller's live motion buffer.

        The context starts ``MOTION_LOOK_AHEAD_STEPS`` ahead of ``current_frame`` so a
        replan blends in seamlessly by the time it is ready.
        """
        ctrl = self._ctrl
        gen_frame = current_frame + MOTION_LOOK_AHEAD_STEPS
        t_arr = gen_frame / 50.0 + np.arange(4) / 30.0
        f50 = t_arr * 50.0
        with ctrl.motion_lock:
            ts = ctrl.motion_timesteps
            if ts <= 0:
                return self.build_initial_context(DEFAULT_ANGLES)
            bp, bq, jp = ctrl.motion_body_pos, ctrl.motion_body_quats, ctrl.motion_joint_positions
            f0 = np.minimum(np.floor(f50).astype(int), ts - 1)
            f1 = np.minimum(f0 + 1, ts - 1)
        frac = f50 - f0
        w0 = 1.0 - frac
        ctx = np.zeros((4, 36), np.float32)
        ctx[:, 0:3] = w0[:, None] * bp[f0] + frac[:, None] * bp[f1]
        ctx[:, 3:7] = quat_slerp_batch(bq[f0], bq[f1], frac)
        ij = w0[:, None] * jp[f0] + frac[:, None] * jp[f1]
        ctx[:, 7:36] = ij[:, ISAACLAB_TO_MUJOCO]
        self.gen_frame = gen_frame
        return ctx

    def _load_motion_in_place(self, qpos, n30, target=None):
        """Resample raw planner qpos to 50 Hz and write it into a ``PlannerMotion`` buffer."""
        if target is None:
            target = self.motion_50hz
        r = _resample_30_to_50(qpos, n30)
        n = r["timesteps"]
        target.timesteps = n
        target.joint_positions[:n] = r["joint_positions"]
        target.joint_velocities[:n] = r["joint_velocities"]
        target.body_positions[:n] = r["body_positions"]
        target.body_quaternions[:n] = r["body_quaternions"]
        return target

    def initialize(self, joint_positions, ms):
        """Synchronously run the planner once to produce the first motion buffer."""
        ctx = self.build_initial_context(joint_positions)
        qpos_out, num_pred = self.session.run(None, self._build_inputs(ctx, ms))
        n = int(num_pred.flat[0])
        qpos = qpos_out[0, :n]
        if np.any(np.isnan(qpos)):
            raise RuntimeError("Planner initial output contains NaN")
        logger.info("[Planner] Init: %d frames @ 30 Hz", n)
        self._load_motion_in_place(qpos, n)
        logger.info("[Planner] Resampled to %d frames @ 50 Hz", self.motion_50hz.timesteps)
        return self.motion_50hz

    def request_replan(self, cursor, ms):
        """Enqueue a replan for the worker (drops any pending stale request first)."""
        if self._req_q is None:
            return
        ctx = self._context_from_controller(cursor)
        ms_dict = {
            "mode": ms.mode,
            "speed": ms.speed,
            "height": ms.height,
            "movement_direction": list(ms.movement_direction),
            "facing_direction": list(ms.facing_direction),
        }
        while not self._req_q.empty():
            try:
                self._req_q.get_nowait()
            except queue.Empty:
                break
        self._req_q.put((ctx, self.gen_frame, ms_dict))

    def try_get_new_motion(self):
        """Non-blocking: return ``(snapshot_motion, gen_frame)`` if a new plan is ready, else None."""
        if self._res_q is None:
            return None
        result = None
        while not self._res_q.empty():
            try:
                result = self._res_q.get_nowait()
            except queue.Empty:
                break
        if result is None:
            return None
        n, gf = result["timesteps"], result["gen_frame"]
        s = self._snapshot
        s.timesteps = n
        s.joint_positions[:n] = result["joint_positions"]
        s.joint_velocities[:n] = result["joint_velocities"]
        s.body_positions[:n] = result["body_positions"]
        s.body_quaternions[:n] = result["body_quaternions"]
        return s, gf

    def start_subprocess(self, controller, use_gpu: bool = False):
        """Run planner ONNX in a background thread (avoids mp spawn/fork + CUDA/MuJoCo issues)."""
        self._ctrl = controller
        self._req_q = queue.Queue()
        self._res_q = queue.Queue()
        self._stop_evt = threading.Event()
        self._planner_thread = threading.Thread(
            target=_planner_worker,
            args=(
                self.planner_path,
                self._req_q,
                self._res_q,
                self._stop_evt,
                self.version,
                self.random_seed,
                use_gpu,
            ),
            daemon=True,
            name="sonic-planner",
        )
        self._planner_thread.start()
        logger.info("[Planner] Background thread started (%s)", "GPU" if use_gpu else "CPU")

    def stop_subprocess(self):
        """Signal the planner thread to stop and join it."""
        if self._stop_evt:
            self._stop_evt.set()
        if self._planner_thread is not None:
            self._planner_thread.join(timeout=3.0)
            logger.info("[Planner] Background thread stopped")
        self._planner_thread = None
        self._req_q = self._res_q = self._stop_evt = None


# ── PlannerController ─────────────────────────────────────────────────────────


class PlannerController(StandingEncoderDecoder):
    """Encoder/decoder driven by the planner's live, replannable motion buffer.

    Extends ``StandingEncoderDecoder`` so the reference comes from a rolling motion
    (advanced one frame per tick via ``advance_cursor``) instead of a fixed pose.
    Handles heading re-initialization, cross-fading new plans into the buffer
    (``blend_new_motion``), and the mode-2 SMPL reference. ``motion_lock`` guards the
    buffer against the async planner thread.
    """

    def __init__(self, planner, encoder, decoder):
        super().__init__(encoder, decoder)
        self.planner = planner
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

    def load_initial_motion(self, motion):
        """Copy the planner's first motion into the live buffer and start playback."""
        with self.motion_lock:
            n = motion.timesteps
            self.motion_timesteps = n
            self.motion_joint_positions[:n] = motion.joint_positions[:n]
            self.motion_joint_velocities[:n] = motion.joint_velocities[:n]
            self.motion_body_quats[:n] = motion.body_quaternions[:n]
            self.motion_body_pos[:n] = motion.body_positions[:n]
            self.init_ref_quat = motion.body_quaternions[0].copy()
            self.ref_cursor = 0
            self.first_motion = True
            self.playing = True
            self.delta_heading = 0.0

    def blend_new_motion(self, new_motion, gen_frame):
        """Blend like C++ CurrentFrameAdvancement: 8-frame cross-fade, then copy tail."""
        with self.motion_lock:
            cur = self.ref_cursor
            new_len = gen_frame - cur + new_motion.timesteps
            if new_len <= 0:
                return
            if self.motion_timesteps == 0:
                n = new_motion.timesteps
                self.motion_joint_positions[:n] = new_motion.joint_positions[:n]
                self.motion_joint_velocities[:n] = new_motion.joint_velocities[:n]
                self.motion_body_pos[:n] = new_motion.body_positions[:n]
                self.motion_body_quats[:n] = new_motion.body_quaternions[:n]
                self.motion_timesteps = n
                self.ref_cursor = 0
                self.init_ref_quat = self.motion_body_quats[0].copy()
                self.first_motion = False
                return

            blend_start = max(0, gen_frame - cur)
            blend_end = min(new_len, blend_start + BLEND_FRAMES)

            for f in range(blend_end):
                f_old = min(f + cur, self.motion_timesteps - 1)
                f_new = max(0, min(f + cur - gen_frame, new_motion.timesteps - 1))
                w_new = min(1.0, max(0.0, (f - blend_start) / BLEND_FRAMES))
                w_old = 1.0 - w_new
                self.motion_joint_positions[f] = (
                    w_old * self.motion_joint_positions[f_old] + w_new * new_motion.joint_positions[f_new]
                )
                self.motion_joint_velocities[f] = (
                    w_old * self.motion_joint_velocities[f_old] + w_new * new_motion.joint_velocities[f_new]
                )
                self.motion_body_pos[f] = (
                    w_old * self.motion_body_pos[f_old] + w_new * new_motion.body_positions[f_new]
                )
                self.motion_body_quats[f] = quat_slerp(
                    self.motion_body_quats[f_old], new_motion.body_quaternions[f_new], w_new
                )

            for f in range(blend_end, new_len):
                f_new = max(0, min(f + cur - gen_frame, new_motion.timesteps - 1))
                self.motion_joint_positions[f] = new_motion.joint_positions[f_new]
                self.motion_joint_velocities[f] = new_motion.joint_velocities[f_new]
                self.motion_body_pos[f] = new_motion.body_positions[f_new]
                self.motion_body_quats[f] = new_motion.body_quaternions[f_new].copy()

            self.motion_timesteps = new_len
            self.first_motion = False
            self.ref_cursor = 0
            self.init_ref_quat = self.motion_body_quats[0].copy()

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


# ── Joystick input ────────────────────────────────────────────────────────────


def _parse_wireless(wr):
    """Parse wireless_remote (bytes or int-array) into (lx, ly, rx, ry)."""
    import struct as _st

    if not isinstance(wr, (bytes, bytearray)):
        wr = bytes(wr)
    if len(wr) < 24:
        return None
    lx = _st.unpack("f", wr[4:8])[0]
    rx = _st.unpack("f", wr[8:12])[0]
    ry = _st.unpack("f", wr[12:16])[0]
    ly = _st.unpack("f", wr[20:24])[0]
    return lx, ly, rx, ry


def apply_joystick_axes(lx, ly, rx, ry, ms, controller=None):
    """Map raw stick axes onto ``MovementState`` (left stick=WASD, right stick X=Q/E,
    right stick Y=height).

    Shared by the G1 wireless remote (:func:`process_joystick`) and the PICO
    controller sticks (3-point teleop), so both drive the planner identically. Axes
    are expected pre-negated to the same convention as the parsed wireless remote:
    ``ly`` and ``ry`` already flipped, dead zone not yet applied.
    """
    # Dead zone + negate both Y axes (bridge already flips them once)
    lx = 0.0 if abs(lx) < DEADZONE else lx
    ly = 0.0 if abs(ly) < DEADZONE else -ly
    rx = 0.0 if abs(rx) < DEADZONE else rx
    ry = 0.0 if abs(ry) < DEADZONE else -ry

    left_active = abs(lx) > 0 or abs(ly) > 0

    # Left stick → WASD (movement direction relative to facing)
    if left_active:
        ms.movement_angle = ms.facing_angle + math.atan2(-lx, -ly)
        ms.has_movement = True
        if not ms.joy_prev_active:
            ms.needs_replan = True
        ms.joy_prev_active = True
    elif ms.joy_prev_active and not (abs(rx) > 0 or abs(ry) > 0):
        ms.joy_prev_active = False
        ms.has_movement = False

    # Right stick X → Q/E (facing rotation, ~1 rad/s at full deflection)
    if abs(rx) > 0:
        delta = -0.02 * rx
        ms.facing_angle += delta
        if controller:
            controller.delta_heading += delta

    # Right stick Y → -/= (height adjustment, ~0.25/s at full deflection)
    if abs(ry) > 0:
        step = -0.005 * ry
        ms.height = max(0.1, min(1.0, (ms.height if ms.height >= 0 else DEFAULT_HEIGHT) + step))


def process_joystick(obs, ms, controller=None):
    """Drive ``MovementState`` from the G1 wireless remote in ``obs``."""
    wr = obs.get("wireless_remote")
    if wr is None:
        return
    parsed = _parse_wireless(wr)
    if parsed is None:
        return
    lx, ly, rx, ry = parsed
    apply_joystick_axes(lx, ly, rx, ry, ms, controller)
