#!/usr/bin/env python
"""
SONIC planner with full mode control.

Keyboard controls:
    N / P      - next / previous motion set
    1-8        - select mode within current set
    WASD       - movement direction
    Q / E      - rotate facing left / right
    9 / 0      - decrease / increase speed
    - / =      - decrease / increase height
    R          - force replan
    Space      - emergency stop -> IDLE
    Esc        - quit

Gamepad controls (Unitree wireless controller):
    Left stick Y  - speed (forward = fast, back = stop)
    Left stick X  - movement direction (offset from facing)
    Right stick X - facing direction (incremental rotation)
    Right stick Y - height (up = tall 0.8m, down = low 0.1m)
    Buttons       - unused (mode selection is keyboard-only)
"""

import argparse, gc, math, select, sys, termios, tty
import multiprocessing as mp
import threading, time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

NATURAL_FREQ = 10.0 * 2.0 * np.pi
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT   = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}

def _action_scale(k):
    return 0.25 * EFFORT[k] / (ARMATURE[k] * NATURAL_FREQ**2)

_J = ["7520_22","7520_22","7520_14","7520_22","5020","5020"] * 2 + \
     ["7520_14","5020","5020"] + \
     ["5020","5020","5020","5020","5020","4010","4010"] * 2
ACTION_SCALE = np.array([_action_scale(k) for k in _J], dtype=np.float32)

CONTROL_DT           = 0.02
DEFAULT_HEIGHT       = 0.788740
TOKEN_DIM            = 64
ENCODER_UPDATE_EVERY = 5
DEBUG_PRINT_EVERY    = 100
MOTION_LOOK_AHEAD_STEPS = 2
INITIAL_RANDOM_SEED  = 1234
MIN_TOKENS, MAX_TOKENS = 6, 16
K = MAX_TOKENS - MIN_TOKENS + 1
DEADZONE             = 0.05
BLEND_FRAMES         = 8

REPLAN_INTERVAL = {
    "running": 0.1, "crawling": 0.2, "boxing": 1.0, "default": 1.0
}

ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
], dtype=np.int32)

MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
], dtype=np.int32)

def _to_mujoco(a):   return a[MUJOCO_TO_ISAACLAB]
def _to_runtime(a):  r = np.zeros(29, np.float32); r[MUJOCO_TO_ISAACLAB] = a; return r

DEFAULT_ANGLES_MUJOCO = _to_mujoco(DEFAULT_ANGLES)
ENCODER_STANDING_REF  = DEFAULT_ANGLES.copy()

LOWER_BODY_IL  = np.array([0,3,6,9,13,17,1,4,7,10,14,18], dtype=np.int32)
WRIST_IL       = np.array([23,24,25,26,27,28], dtype=np.int32)
VR_TARGET_DEF  = np.zeros(9, dtype=np.float32)
VR_ORN_DEF     = np.array([1,0,0,0,1,0,0,0,1,0,0,0], dtype=np.float32)
SMPL_DEF       = np.zeros(720, dtype=np.float32)

# ── PD gains ─────────────────────────────────────────────────────────────────

def _kp_kd():
    s = lambda k: ARMATURE[k] * NATURAL_FREQ**2
    d = lambda k: 2.0 * 2.0 * ARMATURE[k] * NATURAL_FREQ
    _kp_keys = ["7520_22","7520_22","7520_14","7520_22","5020","5020"] * 2 + \
               ["7520_14","5020","5020"] + \
               ["5020","5020","5020","5020","5020","4010","4010"] * 2
    _kd_keys = _kp_keys
    _double  = {4,5,10,11,13,14}  # ankle + waist indices with factor 2
    kp = np.array([2*s(k) if i in _double else s(k) for i,k in enumerate(_kp_keys)], dtype=np.float32)
    kd = np.array([2*d(k) if i in _double else d(k) for i,k in enumerate(_kd_keys)], dtype=np.float32)
    return kp, kd

# ── Quaternion helpers ────────────────────────────────────────────────────────

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1;  w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

def gravity_dir(q):
    q = q / (np.linalg.norm(q) + 1e-8)
    qv = np.array([0, 0, 0, -1], dtype=np.float32)
    return quat_mul(quat_mul(quat_conj(q), qv), q)[1:]

def quat_to_6d(q):
    w,x,y,z = q
    return np.array([
        1-2*(y*y+z*z), 2*(x*y-z*w),
        2*(x*y+z*w),   1-2*(x*x+z*z),
        2*(x*z-y*w),   2*(y*z+x*w),
    ], dtype=np.float32)

def calc_heading(q):
    w,x,y,z = q
    return float(np.arctan2(2*(x*y + w*z), 1-2*(y*y+z*z)))

def heading_quat(q, sign=1.0):
    a = sign * calc_heading(q) / 2.0
    return np.array([np.cos(a), 0, 0, np.sin(a)], dtype=np.float64)

heading_quat_inv = lambda q: heading_quat(q, -1.0)

def quat_slerp(q0, q1, t):
    q0 = q0 / (np.linalg.norm(q0)+1e-12);  q1 = q1 / (np.linalg.norm(q1)+1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0: q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q0 + t*(q1-q0); return r/(np.linalg.norm(r)+1e-12)
    th = np.arccos(dot); st = np.sin(th)
    return (np.sin((1-t)*th)/st)*q0 + (np.sin(t*th)/st)*q1

def quat_slerp_batch(q0, q1, t):
    q0 = q0 / (np.linalg.norm(q0,axis=1,keepdims=True)+1e-12)
    q1 = q1 / (np.linalg.norm(q1,axis=1,keepdims=True)+1e-12)
    dot = np.sum(q0*q1, axis=1); neg = dot<0
    q1=q1.copy(); q1[neg]=-q1[neg]; dot[neg]=-dot[neg]; dot=np.clip(dot,-1,1)
    lin = dot>0.9995; th=np.arccos(dot); st=np.where(np.sin(th)==0,1,np.sin(th))
    c0=np.sin((1-t)*th)/st; c1=np.sin(t*th)/st
    c0[lin]=1-t[lin]; c1[lin]=t[lin]
    r = c0[:,None]*q0 + c1[:,None]*q1
    return r / (np.linalg.norm(r,axis=1,keepdims=True)+1e-12)

# ── Locomotion modes ──────────────────────────────────────────────────────────

class LocomotionMode(IntEnum):
    IDLE=0; SLOW_WALK=1; WALK=2; RUN=3; SQUAT=4; KNEEL_TWO_LEGS=5; KNEEL=6
    LYING_FACE_DOWN=7; CRAWLING=8; IDLE_BOXING=9; WALK_BOXING=10
    LEFT_PUNCH=11; RIGHT_PUNCH=12; RANDOM_PUNCH=13; ELBOW_CRAWLING=14
    LEFT_HOOK=15; RIGHT_HOOK=16; FORWARD_JUMP=17; STEALTH_WALK=18
    INJURED_WALK=19; LEDGE_WALKING=20; OBJECT_CARRYING=21; STEALTH_WALK_2=22
    HAPPY_DANCE_WALK=23; ZOMBIE_WALK=24; GUN_WALK=25; SCARE_WALK=26

LM = LocomotionMode

MOTION_SETS = [
    ("Standing",     [LM.SLOW_WALK, LM.WALK, LM.RUN, LM.FORWARD_JUMP, LM.STEALTH_WALK, LM.INJURED_WALK]),
    ("Squat / Low",  [LM.SQUAT, LM.KNEEL_TWO_LEGS, LM.KNEEL, LM.CRAWLING, LM.ELBOW_CRAWLING]),
    ("Boxing",       [LM.IDLE_BOXING, LM.WALK_BOXING, LM.LEFT_PUNCH, LM.RIGHT_PUNCH,
                      LM.RANDOM_PUNCH, LM.LEFT_HOOK, LM.RIGHT_HOOK]),
    ("Styled Walks", [LM.LEDGE_WALKING, LM.OBJECT_CARRYING, LM.STEALTH_WALK_2,
                      LM.HAPPY_DANCE_WALK, LM.ZOMBIE_WALK, LM.GUN_WALK, LM.SCARE_WALK]),
]

STATIC_MODES   = {LM.IDLE, LM.SQUAT, LM.KNEEL_TWO_LEGS, LM.KNEEL, LM.LYING_FACE_DOWN, LM.IDLE_BOXING}
STANDING_MODES = {LM.IDLE, LM.SLOW_WALK, LM.WALK, LM.RUN, LM.IDLE_BOXING, LM.WALK_BOXING,
                  LM.LEFT_PUNCH, LM.RIGHT_PUNCH, LM.RANDOM_PUNCH, LM.LEFT_HOOK, LM.RIGHT_HOOK,
                  LM.FORWARD_JUMP, LM.STEALTH_WALK, LM.INJURED_WALK, LM.LEDGE_WALKING,
                  LM.OBJECT_CARRYING, LM.STEALTH_WALK_2, LM.HAPPY_DANCE_WALK,
                  LM.ZOMBIE_WALK, LM.GUN_WALK, LM.SCARE_WALK}
BOXING_MODES   = {LM.WALK_BOXING, LM.LEFT_PUNCH, LM.RIGHT_PUNCH,
                  LM.RANDOM_PUNCH, LM.LEFT_HOOK, LM.RIGHT_HOOK}
SPEED_RANGES   = {LM.SLOW_WALK:(0.2,0.8), LM.WALK:(0.8,1.5), LM.RUN:(1.5,3.0),
                  LM.CRAWLING:(0.4,1.0),  LM.ELBOW_CRAWLING:(0.7,1.0)}

def clamp_mode_params(ms):
    m = LM(ms.mode)
    ms.height = -1.0 if m in STANDING_MODES else max(0.1, min(0.8, ms.height if ms.height>=0 else 0.2))
    if m in STATIC_MODES:
        ms.speed = -1.0
    elif m in SPEED_RANGES:
        lo, hi = SPEED_RANGES[m]
        ms.speed = max(lo, min(hi, ms.speed if ms.speed>=0 else lo))
    elif m in BOXING_MODES:
        ms.speed = max(0.7, min(1.5, ms.speed if ms.speed>=0 else 0.7))
    else:
        ms.speed = -1.0

def replan_interval(mode):
    m = LM(mode)
    if m == LM.RUN: return REPLAN_INTERVAL["running"]
    if m == LM.CRAWLING: return REPLAN_INTERVAL["crawling"]
    if m in {LM.LEFT_PUNCH, LM.RIGHT_PUNCH, LM.RANDOM_PUNCH, LM.LEFT_HOOK, LM.RIGHT_HOOK}:
        return REPLAN_INTERVAL["boxing"]
    return REPLAN_INTERVAL["default"]

# ── Movement state ────────────────────────────────────────────────────────────

@dataclass
class MovementState:
    mode: int          = 0
    speed: float       = -1.0
    height: float      = -1.0
    facing_angle: float = 0.0
    movement_angle: float = 0.0
    has_movement: bool = False
    motion_set_idx: int = 0
    needs_replan: bool = False

    @property
    def movement_direction(self):
        if not self.has_movement: return (0.0, 0.0, 0.0)
        return (math.cos(self.movement_angle), math.sin(self.movement_angle), 0.0)

    @property
    def facing_direction(self):
        return (math.cos(self.facing_angle), math.sin(self.facing_angle), 0.0)

    def status_line(self):
        return (f"[{MOTION_SETS[self.motion_set_idx][0]}] mode={self.mode}({LM(self.mode).name}) "
                f"spd={'default' if self.speed<0 else f'{self.speed:.1f}'} "
                f"hgt={'default' if self.height<0 else f'{self.height:.2f}'} "
                f"facing={math.degrees(self.facing_angle):.0f}° "
                f"{'moving' if self.has_movement else 'still'}")

# ── Encoder / Decoder ─────────────────────────────────────────────────────────

class StandingEncoderDecoder:
    def __init__(self, encoder, decoder):
        self.encoder, self.decoder = encoder, decoder
        self.encoder_input  = encoder.get_inputs()[0].name
        self.decoder_input  = decoder.get_inputs()[0].name
        enc_dim = int(encoder.get_inputs()[0].shape[1])
        dec_dim = int(decoder.get_inputs()[0].shape[1])
        if enc_dim != 1762 or dec_dim != 994:
            raise RuntimeError(f"Unexpected dims encoder={enc_dim}, decoder={dec_dim}")
        self.token             = np.zeros(TOKEN_DIM, np.float32)
        self.last_action_mj   = np.zeros(29, np.float32)
        self.h_q_mj   = [np.zeros(29, np.float32)] * 10
        self.h_dq_mj  = [np.zeros(29, np.float32)] * 10
        self.h_ang    = [np.zeros(3,  np.float32)] * 10
        self.h_act_mj = [np.zeros(29, np.float32)] * 10
        self.h_quat   = [np.array([1,0,0,0], np.float32)] * 10
        self.init_base_quat   = np.array([1,0,0,0], np.float32)
        self.init_ref_quat    = np.array([1,0,0,0], np.float32)
        self._heading_init    = False
        self.encode_mode      = 0
        self.vr_3point_local_target    = VR_TARGET_DEF.copy()
        self.vr_3point_local_orn_target = VR_ORN_DEF.copy()
        self.smpl_joints_10frame_step1 = SMPL_DEF.copy()
        self.set_zero_reference()

    def update_history(self, q, dq, ang, quat):
        quat = quat / (np.linalg.norm(quat)+1e-8)
        q_mj = _to_mujoco(q); dq_mj = _to_mujoco(dq)
        self.h_q_mj   = [q_mj - DEFAULT_ANGLES_MUJOCO] + self.h_q_mj[:-1]
        self.h_dq_mj  = [dq_mj]       + self.h_dq_mj[:-1]
        self.h_ang    = [ang.copy()]   + self.h_ang[:-1]
        self.h_act_mj = [self.last_action_mj.copy()] + self.h_act_mj[:-1]
        self.h_quat   = [quat.copy()]  + self.h_quat[:-1]
        if not self._heading_init:
            self.init_base_quat = quat.copy(); self._heading_init = True

    def _heading_quat(self, q):
        h = calc_heading(q) / 2.0
        return np.array([np.cos(h), 0, 0, np.sin(h)], np.float32)

    def _heading_quat_inv(self, q):
        h = calc_heading(q) / 2.0
        return np.array([np.cos(-h), 0, 0, np.sin(-h)], np.float32)

    def _anchor_6d(self, base_quat, ref_quat=None):
        if ref_quat is None: ref_quat = self.init_ref_quat
        delta = quat_mul(self._heading_quat(self.init_base_quat), self._heading_quat_inv(self.init_ref_quat))
        new_ref = quat_mul(delta, ref_quat)
        return quat_to_6d(quat_mul(quat_conj(base_quat), new_ref))

    def set_zero_reference(self):
        self.motion_joint_positions  = [ENCODER_STANDING_REF.copy()]
        self.motion_joint_velocities = [np.zeros(29, np.float32)]
        self.motion_body_quats       = [np.array([1,0,0,0], np.float32)]
        self.motion_body_z           = [DEFAULT_HEIGHT]
        self.motion_timesteps        = 1
        self.freeze_ref_frame        = 0
        self.init_ref_quat           = self.motion_body_quats[0].copy()

    def build_encoder_obs(self):
        obs = np.zeros(1762, np.float32)
        obs[0] = float(self.encode_mode)
        rf = min(self.freeze_ref_frame, self.motion_timesteps - 1)
        ref_pos, ref_quat = self.motion_joint_positions[rf], self.motion_body_quats[rf]
        if self.encode_mode == 0:
            for f in range(10):
                obs[4+29*f:4+29*(f+1)] = ref_pos
                obs[601+6*f:601+6*(f+1)] = self._anchor_6d(self.h_quat[0], ref_quat)
        elif self.encode_mode == 1:
            ref_lower = ref_pos[LOWER_BODY_IL]
            for f in range(10):
                obs[661+12*f:661+12*(f+1)] = ref_lower
            obs[901:910] = self.vr_3point_local_target
            obs[910:922] = self.vr_3point_local_orn_target
            obs[595:601]  = self._anchor_6d(self.h_quat[0], ref_quat)
        elif self.encode_mode == 2:
            obs[922:1642] = self.smpl_joints_10frame_step1
            for f in range(10):
                obs[1642+6*f:1642+6*(f+1)] = self._anchor_6d(self.h_quat[0], ref_quat)
                obs[1702+6*f:1702+6*(f+1)] = ref_pos[WRIST_IL]
        else:
            raise RuntimeError(f"Unsupported encoder mode: {self.encode_mode}")
        return obs

    def build_decoder_obs(self):
        obs = np.zeros(994, np.float32); off = 0
        obs[off:off+64] = self.token; off += 64
        for h, sz in [(list(reversed(self.h_ang)),3), (list(reversed(self.h_q_mj)),29),
                      (list(reversed(self.h_dq_mj)),29), (list(reversed(self.h_act_mj)),29)]:
            for f in range(10): obs[off:off+sz] = h[f]; off += sz
        for q in reversed(self.h_quat):
            obs[off:off+3] = gravity_dir(q); off += 3
        assert off == 994, f"Decoder obs mismatch: {off}"
        return obs

    def run_encoder(self):
        return self.encoder.run(None, {self.encoder_input: self.build_encoder_obs().reshape(1,-1)})[0].squeeze().astype(np.float32)

    def step(self, robot_obs, update_encoder, debug=False):
        jnames = [m.name for m in G1_29_JointIndex]
        q   = np.array([robot_obs.get(f"{n}.q",  DEFAULT_ANGLES[m.value]) for m,n in zip(G1_29_JointIndex,jnames)], np.float32)
        dq  = np.array([robot_obs.get(f"{n}.dq", 0.0) for n in jnames], np.float32)
        quat = np.array([robot_obs.get("imu.quat.w",1), robot_obs.get("imu.quat.x",0),
                         robot_obs.get("imu.quat.y",0), robot_obs.get("imu.quat.z",0)], np.float32)
        ang  = np.array([robot_obs.get(f"imu.gyro.{a}",0) for a in "xyz"], np.float32)
        self.update_history(q, dq, ang, quat)
        if update_encoder: self.token = self.run_encoder()
        action_mj = self.decoder.run(None, {self.decoder_input: self.build_decoder_obs().reshape(1,-1)})[0].squeeze().astype(np.float32)
        self.last_action_mj = action_mj.copy()
        target = DEFAULT_ANGLES + action_mj[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        if debug:
            delta = target - q
            print(f"token_norm={np.linalg.norm(self.token):.4f} action_norm={np.linalg.norm(action_mj):.4f} "
                  f"delta_max={np.max(np.abs(delta)):.4f} delta_rms={np.sqrt(np.mean(delta**2)):.4f}")
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}

    def print_input_diagnostics(self):
        print("\n[Diag] Standing reference checks")
        names = {0:"g1", 1:"teleop", 2:"smpl"}
        print(f"  encoder mode: {self.encode_mode} ({names.get(self.encode_mode,'unknown')})")
        print(f"  DEFAULT_ANGLES range: [{DEFAULT_ANGLES.min():+.4f}, {DEFAULT_ANGLES.max():+.4f}]")
        print(f"  anchor_6d(identity): {self._anchor_6d(np.array([1,0,0,0],np.float32), np.array([1,0,0,0],np.float32))}")
        print(f"  gravity(identity): {gravity_dir(np.array([1,0,0,0],np.float32))} (expect [0,0,-1])")
        dec0 = self.build_decoder_obs()
        print(f"  decoder q-delta max: {np.max(np.abs(dec0[94:384])):.6f}")
        print(f"  decoder dq max:      {np.max(np.abs(dec0[384:674])):.6f}")

# ── Planner motion buffer ─────────────────────────────────────────────────────

class PlannerMotion:
    def __init__(self, max_frames=1500):
        self.timesteps         = 0
        self.joint_positions   = np.zeros((max_frames, 29), np.float64)
        self.joint_velocities  = np.zeros((max_frames, 29), np.float64)
        self.body_positions    = np.zeros((max_frames, 3),  np.float64)
        self.body_quaternions  = np.zeros((max_frames, 4),  np.float64)
        self.body_quaternions[:, 0] = 1.0

# ── Subprocess planner ────────────────────────────────────────────────────────

def _resample_30_to_50(qpos, n30):
    t50 = int(np.floor(n30 / 30.0 * 50))
    f30 = np.arange(t50) / 50.0 * 30.0
    f0  = np.floor(f30).astype(int)
    f1  = np.minimum(f0+1, n30-1)
    frac, w0 = (f30-f0).astype(np.float64), None
    w0 = 1.0 - frac
    jp = (w0[:,None]*qpos[f0,7:36] + frac[:,None]*qpos[f1,7:36])[:,MUJOCO_TO_ISAACLAB]
    jv = np.zeros_like(jp)
    if t50 >= 2: jv[:t50-1] = (jp[1:] - jp[:-1]) * 50.0; jv[-1] = jv[-2]
    return {
        "timesteps": t50,
        "joint_positions":  jp,
        "joint_velocities": jv,
        "body_positions":   w0[:,None]*qpos[f0,:3]   + frac[:,None]*qpos[f1,:3],
        "body_quaternions": quat_slerp_batch(qpos[f0,3:7], qpos[f1,3:7], frac),
    }

def _build_planner_inputs(ctx, ms_dict, version, seed):
    inp = {
        "context_mujoco_qpos": ctx.astype(np.float32).reshape(1,4,36),
        "target_vel":          np.array([ms_dict["speed"]], np.float32),
        "mode":                np.array([ms_dict["mode"]], np.int64),
        "movement_direction":  np.array(ms_dict["movement_direction"], np.float32).reshape(1,3),
        "facing_direction":    np.array(ms_dict["facing_direction"],   np.float32).reshape(1,3),
        "random_seed":         np.array([seed], np.int64),
    }
    if version >= 1:
        allowed = np.zeros((1,K), np.int64); allowed[0,:6] = 1
        inp.update({
            "height": np.array([ms_dict["height"]], np.float32),
            "has_specific_target":       np.array([[0]], np.int64),
            "specific_target_positions": np.zeros((1,4,3), np.float32),
            "specific_target_headings":  np.zeros((1,4),   np.float32),
            "allowed_pred_num_tokens":   allowed,
        })
    return inp

def _planner_worker(path, req_q, res_q, stop_evt, version, seed):
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    while not stop_evt.is_set():
        try: ctx, gf, ms_dict = req_q.get(timeout=0.05)
        except Exception: continue
        try:
            inp = _build_planner_inputs(ctx, ms_dict, version, seed)
            t0 = time.time()
            qpos_out, num_pred = sess.run(None, inp)
            t_inf = time.time()
            n = int(num_pred.flat[0])
            qpos = qpos_out[0,:n]
            if np.any(np.isnan(qpos)): continue
            motion = _resample_30_to_50(qpos, n)
            motion["gen_frame"] = gf
            print(f"[Planner] inf={1000*(t_inf-t0):.1f}ms total={1000*(time.time()-t0):.1f}ms frames={n}", flush=True)
            while not res_q.empty():
                try: res_q.get_nowait()
                except Exception: break
            res_q.put(motion)
        except Exception as e:
            print(f"[Planner] Error: {e}", flush=True)

# ── SonicPlanner ──────────────────────────────────────────────────────────────

class SonicPlanner:
    def __init__(self, session, planner_path):
        self.session      = session
        self.planner_path = planner_path
        self.gen_frame    = 0
        self.random_seed  = INITIAL_RANDOM_SEED
        self.version      = 1 if len(session.get_inputs()) >= 11 else 0
        self.motion_50hz  = PlannerMotion()
        self._snapshot    = PlannerMotion()
        self._req_q = self._res_q = self._stop_evt = self._proc = None
        self._ctrl = None

    def _build_inputs(self, ctx, ms):
        return _build_planner_inputs(
            ctx,
            {"mode": ms.mode, "speed": ms.speed, "height": ms.height,
             "movement_direction": list(ms.movement_direction),
             "facing_direction":   list(ms.facing_direction)},
            self.version, self.random_seed)

    @staticmethod
    def build_initial_context(joint_positions):
        ctx = np.zeros((4,36), np.float32)
        for n in range(4):
            ctx[n,2] = DEFAULT_HEIGHT; ctx[n,3] = 1.0
            ctx[n,7:36] = joint_positions.astype(np.float32)
        return ctx

    def _context_from_controller(self, current_frame):
        ctrl = self._ctrl
        gen_frame = current_frame + MOTION_LOOK_AHEAD_STEPS
        t_arr = gen_frame/50.0 + np.arange(4)/30.0
        f50 = t_arr * 50.0
        with ctrl.motion_lock:
            ts = ctrl.motion_timesteps
            bp = ctrl.motion_body_pos[:ts].copy()
            bq = ctrl.motion_body_quats[:ts].copy()
            jp = ctrl.motion_joint_positions[:ts].copy()
        f0 = np.minimum(np.floor(f50).astype(int), ts-1)
        f1 = np.minimum(f0+1, ts-1)
        frac, w0 = f50-f0, None; w0 = 1.0-frac
        ctx = np.zeros((4,36), np.float32)
        ctx[:,0:3] = w0[:,None]*bp[f0] + frac[:,None]*bp[f1]
        ctx[:,3:7] = quat_slerp_batch(bq[f0], bq[f1], frac)
        ij = w0[:,None]*jp[f0] + frac[:,None]*jp[f1]
        ctx[:,7:36] = ij[:,ISAACLAB_TO_MUJOCO]
        self.gen_frame = gen_frame
        return ctx

    def _load_motion_in_place(self, qpos, n30, target=None):
        if target is None: target = self.motion_50hz
        r = _resample_30_to_50(qpos, n30)
        n = r["timesteps"]; target.timesteps = n
        target.joint_positions[:n]  = r["joint_positions"]
        target.joint_velocities[:n] = r["joint_velocities"]
        target.body_positions[:n]   = r["body_positions"]
        target.body_quaternions[:n] = r["body_quaternions"]
        return target

    def initialize(self, joint_positions, ms):
        ctx = self.build_initial_context(joint_positions)
        qpos_out, num_pred = self.session.run(None, self._build_inputs(ctx, ms))
        n = int(num_pred.flat[0]); qpos = qpos_out[0,:n]
        if np.any(np.isnan(qpos)): raise RuntimeError("Planner initial output contains NaN")
        print(f"[Planner] Init: {n} frames @ 30 Hz")
        self._load_motion_in_place(qpos, n)
        print(f"[Planner] Resampled to {self.motion_50hz.timesteps} frames @ 50 Hz")
        return self.motion_50hz

    def request_replan(self, cursor, ms):
        if self._req_q is None: return
        ctx = self._context_from_controller(cursor)
        ms_dict = {"mode": ms.mode, "speed": ms.speed, "height": ms.height,
                   "movement_direction": list(ms.movement_direction),
                   "facing_direction":   list(ms.facing_direction)}
        while not self._req_q.empty():
            try: self._req_q.get_nowait()
            except Exception: break
        self._req_q.put((ctx, self.gen_frame, ms_dict))

    def try_get_new_motion(self):
        if self._res_q is None: return None
        result = None
        while not self._res_q.empty():
            try: result = self._res_q.get_nowait()
            except Exception: break
        if result is None: return None
        n, gf = result["timesteps"], result["gen_frame"]
        s = self._snapshot; s.timesteps = n
        s.joint_positions[:n]  = result["joint_positions"]
        s.joint_velocities[:n] = result["joint_velocities"]
        s.body_positions[:n]   = result["body_positions"]
        s.body_quaternions[:n] = result["body_quaternions"]
        return s, gf

    def start_subprocess(self, controller):
        self._ctrl = controller
        self._req_q, self._res_q, self._stop_evt = mp.Queue(), mp.Queue(), mp.Event()
        self._proc = mp.Process(
            target=_planner_worker,
            args=(self.planner_path, self._req_q, self._res_q,
                  self._stop_evt, self.version, self.random_seed),
            daemon=True)
        self._proc.start()
        print(f"[Planner] Background process started (PID={self._proc.pid})")

    def stop_subprocess(self):
        if self._stop_evt: self._stop_evt.set()
        if self._proc:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive(): self._proc.terminate()
            print("[Planner] Background process stopped")
        for q in (self._req_q, self._res_q):
            if q: q.close()

# ── PlannerController ─────────────────────────────────────────────────────────

class PlannerController(StandingEncoderDecoder):
    def __init__(self, planner, encoder, decoder):
        super().__init__(encoder, decoder)
        self.planner = planner
        self.ref_cursor       = 0
        self.motion_timesteps = 0
        self.motion_joint_positions  = np.zeros((1500,29), np.float64)
        self.motion_joint_velocities = np.zeros((1500,29), np.float64)
        self.motion_body_quats       = np.zeros((1500,4),  np.float64); self.motion_body_quats[:,0] = 1.0
        self.motion_body_pos         = np.zeros((1500,3),  np.float64)
        self.init_ref_quat        = np.array([1,0,0,0], np.float64)
        self.heading_init_base_quat = np.array([1,0,0,0], np.float64)
        self.delta_heading = 0.0
        self.reinit_heading = False
        self.playing = self.first_motion = False
        self.motion_lock = threading.Lock()

    def load_initial_motion(self, motion):
        with self.motion_lock:
            n = motion.timesteps
            self.motion_timesteps = n
            self.motion_joint_positions[:n]  = motion.joint_positions[:n]
            self.motion_joint_velocities[:n] = motion.joint_velocities[:n]
            self.motion_body_quats[:n]       = motion.body_quaternions[:n]
            self.motion_body_pos[:n]         = motion.body_positions[:n]
            self.init_ref_quat = motion.body_quaternions[0].copy()
            self.ref_cursor = 0; self.first_motion = True
            self.playing = True; self.delta_heading = 0.0

    def blend_new_motion(self, new_motion, gen_frame):
        with self.motion_lock:
            cur = self.ref_cursor
            new_len = gen_frame - cur + new_motion.timesteps
            if new_len <= 0: return
            f_arr = np.arange(new_len)
            f_old = np.minimum(f_arr + cur, self.motion_timesteps - 1)
            f_new = np.clip(f_arr + cur - gen_frame, 0, new_motion.timesteps - 1)
            blend_start = max(0, gen_frame - cur)
            w_new = np.clip((f_arr - blend_start) / BLEND_FRAMES if BLEND_FRAMES > 0
                            else np.ones(new_len), 0.0, 1.0)
            w_old = 1.0 - w_new
            self.motion_joint_positions[:new_len]  = w_old[:,None]*self.motion_joint_positions[f_old]  + w_new[:,None]*new_motion.joint_positions[f_new]
            self.motion_joint_velocities[:new_len] = w_old[:,None]*self.motion_joint_velocities[f_old] + w_new[:,None]*new_motion.joint_velocities[f_new]
            self.motion_body_pos[:new_len]         = w_old[:,None]*self.motion_body_pos[f_old]         + w_new[:,None]*new_motion.body_positions[f_new]
            self.motion_body_quats[:new_len]       = quat_slerp_batch(self.motion_body_quats[f_old], new_motion.body_quaternions[f_new], w_new)
            self.motion_timesteps = new_len; self.first_motion = False; self.ref_cursor = 0
            self.init_ref_quat = self.motion_body_quats[0].copy()

    def _heading_apply_delta(self):
        delta = quat_mul(heading_quat(self.heading_init_base_quat).astype(np.float32),
                         heading_quat_inv(self.init_ref_quat).astype(np.float32))
        if self.delta_heading:
            h = self.delta_heading / 2.0
            delta = quat_mul(np.array([np.cos(h),0,0,np.sin(h)], np.float32), delta)
        return delta

    def _anchor_6d(self, base_quat, ref_quat=None):
        if ref_quat is None: ref_quat = self.init_ref_quat
        new_ref = quat_mul(self._heading_apply_delta(), ref_quat.astype(np.float32))
        return quat_to_6d(quat_mul(quat_conj(base_quat.astype(np.float32)), new_ref))

    def build_encoder_obs(self):
        obs = np.zeros(1762, np.float32); obs[0] = float(self.encode_mode)
        with self.motion_lock:
            for f in range(10):
                tf = min(self.ref_cursor + f*5 if self.playing else self.ref_cursor,
                         self.motion_timesteps - 1)
                obs[4+29*f:4+29*(f+1)] = self.motion_joint_positions[tf].astype(np.float32)
                if self.playing:
                    obs[294+29*f:294+29*(f+1)] = self.motion_joint_velocities[tf].astype(np.float32)
                obs[601+6*f:601+6*(f+1)] = self._anchor_6d(
                    self.h_quat[0], self.motion_body_quats[tf].astype(np.float32))
        return obs

    def step(self, robot_obs, update_encoder, debug=False):
        if robot_obs and (self.first_motion or self.reinit_heading):
            q = robot_obs.get("imu.quaternion")
            if q is not None:
                self.heading_init_base_quat = np.array(q, np.float64)
                with self.motion_lock:
                    rf = min(self.ref_cursor, self.motion_timesteps - 1)
                    self.init_ref_quat = self.motion_body_quats[rf].copy()
                self.delta_heading = 0.0
                self.first_motion = False
                self.reinit_heading = False
                print(f"[Heading] init quat: {self.heading_init_base_quat}")
        return super().step(robot_obs, update_encoder=update_encoder, debug=debug)

    def advance_cursor(self, wall_dt):
        if not self.playing: return
        frames = max(1, round(wall_dt / CONTROL_DT))
        with self.motion_lock:
            self.ref_cursor = min(self.ref_cursor + frames, self.motion_timesteps - 1)

# ── Keyboard ──────────────────────────────────────────────────────────────────

class RawKeyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
    def __enter__(self): tty.setcbreak(self.fd); return self
    def __exit__(self, *_): termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def get_key(self):
        return sys.stdin.read(1) if select.select([sys.stdin],[],[],0)[0] else None

def process_keyboard(key, ms, controller=None):
    if key is None: return False
    if key == '\x1b': return True
    if key == ' ':
        ms.mode = LM.IDLE; ms.speed = ms.height = -1.0
        ms.has_movement = False; ms.needs_replan = True
        if controller: controller.playing = False; controller.reinit_heading = True
        print("\n  >> EMERGENCY STOP -> IDLE"); return False
    if key in ('r','R'):
        ms.needs_replan = True; print("\n  >> Manual replan"); return False
    if key in ('n','N','p','P'):
        ms.motion_set_idx = (ms.motion_set_idx + (1 if key in ('n','N') else -1)) % len(MOTION_SETS)
        name, modes = MOTION_SETS[ms.motion_set_idx]
        print(f"\n  >> Motion set: {name}")
        [print(f"       {i+1}: {m.name}") for i,m in enumerate(modes)]
        return False
    if key.isdigit() and key not in ('9','0'):
        idx = int(key) - 1; modes = MOTION_SETS[ms.motion_set_idx][1]
        if 0 <= idx < len(modes):
            ms.mode = modes[idx]; ms.needs_replan = True
            if controller: controller.playing = True; controller.reinit_heading = True
            print(f"\n  >> Mode: {LM(ms.mode).name} ({ms.mode}) [replanning...]")
        return False
    if key == '9':
        ms.speed = max(0.0, (ms.speed if ms.speed>=0 else 1.0) - 0.1)
        print(f"\n  >> Speed: {ms.speed:.1f}"); return False
    if key == '0':
        ms.speed = min(5.0, (ms.speed if ms.speed>=0 else 1.0) + 0.1)
        print(f"\n  >> Speed: {ms.speed:.1f}"); return False
    if key == '-':
        ms.height = max(0.2, (ms.height if ms.height>=0 else DEFAULT_HEIGHT) - 0.02)
        print(f"\n  >> Height: {ms.height:.2f}"); return False
    if key == '=':
        ms.height = min(1.0, (ms.height if ms.height>=0 else DEFAULT_HEIGHT) + 0.02)
        print(f"\n  >> Height: {ms.height:.2f}"); return False
    if key.lower() == 'w': ms.movement_angle = ms.facing_angle
    elif key.lower() == 's': ms.movement_angle = ms.facing_angle + math.pi
    elif key.lower() == 'a': ms.movement_angle = ms.facing_angle + math.pi/2
    elif key.lower() == 'd': ms.movement_angle = ms.facing_angle - math.pi/2
    if key.lower() in ('w','s','a','d'):
        ms.has_movement = ms.needs_replan = True
    elif key.lower() == 'q':
        ms.facing_angle += 0.1
        if controller: controller.delta_heading += 0.1
        print(f"\n  >> Facing: {math.degrees(ms.facing_angle):.0f}°")
    elif key.lower() == 'e':
        ms.facing_angle -= 0.1
        if controller: controller.delta_heading -= 0.1
        print(f"\n  >> Facing: {math.degrees(ms.facing_angle):.0f}°")
    return False

_joy_prev_active = False


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


def process_joystick(obs, ms, controller=None):
    """Joystick mirrors keyboard: left stick=WASD, right stick X=Q/E, right stick Y=height."""
    global _joy_prev_active
    wr = obs.get("wireless_remote")
    if wr is None:
        return
    parsed = _parse_wireless(wr)
    if parsed is None:
        return
    lx, ly, rx, ry = parsed

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
        if not _joy_prev_active:
            ms.needs_replan = True
        _joy_prev_active = True
    elif _joy_prev_active and not (abs(rx) > 0 or abs(ry) > 0):
        _joy_prev_active = False
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SONIC planner with keyboard + gamepad control")
    parser.add_argument("--ip", type=str, default=None,
                        help="Robot IP for real hardware (e.g. 192.168.123.164). "
                             "Omit for simulation.")
    args = parser.parse_args()

    print("=" * 60)
    print("SONIC planner - full mode control")
    print("  N/P  cycle sets | 1-8 select mode | WASD move")
    print("  Q/E  rotate     | 9/0 speed       | -/= height")
    print("  R    replan     | Space IDLE       | Esc quit")
    if args.ip:
        print(f"  Robot IP: {args.ip}")
    else:
        print("  Mode: simulation")
    print("=" * 60 + "\n")

    planner_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="planner_sonic.onnx")
    encoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_encoder.onnx")
    decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

    providers = ort.get_available_providers()
    use_gpu   = "CUDAExecutionProvider" in providers
    gpu_ep    = (["CUDAExecutionProvider","CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"])
    so = ort.SessionOptions(); so.log_severity_level = 3

    print(f"[ONNX] enc/dec={'GPU' if use_gpu else 'CPU'}, planner=CPU")
    planner_sess = ort.InferenceSession(planner_path, sess_options=so, providers=["CPUExecutionProvider"])
    encoder_sess = ort.InferenceSession(encoder_path, sess_options=so, providers=gpu_ep)
    decoder_sess = ort.InferenceSession(decoder_path, sess_options=so, providers=gpu_ep)
    print(f"[Planner] version={'v1+' if len(planner_sess.get_inputs())>=11 else 'v0'}")

    cfg = UnitreeG1Config()
    if args.ip:
        cfg.is_simulation = False
        cfg.robot_ip = args.ip
    robot = UnitreeG1(cfg); robot.connect()
    kp, kd = _kp_kd(); robot.kp = kp.copy(); robot.kd = kd.copy()

    ms         = MovementState()
    planner    = SonicPlanner(planner_sess, planner_path)
    controller = PlannerController(planner, encoder_sess, decoder_sess)

    motion = planner.initialize(DEFAULT_ANGLES, ms)
    controller.load_initial_motion(motion)
    controller.print_input_diagnostics()
    planner.start_subprocess(controller)

    print(f"\nStarting: {MOTION_SETS[0][0]}")
    [print(f"  {i+1}: {m.name}") for i,m in enumerate(MOTION_SETS[0][1])]

    with RawKeyboard() as kb:
        try:
            gc.disable(); gc_timer = 0.0
            robot.reset(CONTROL_DT, DEFAULT_ANGLES); time.sleep(1.0)

            step = 0; last_status = replan_timer = 0.0
            loop_t = enc_t = dec_t = obs_t = act_t = []
            slow_n = blend_n = 0; stall_src = ""; did_blend = False
            prev_end = time.time(); t_start = time.time()

            log_path = "/tmp/sonic_pose_log.csv"
            jnames   = [m.name for m in G1_29_JointIndex]
            with open(log_path, "w") as log_f:
                log_f.write("t,step,cursor,ts,blend,mode," +
                             ",".join(f"q{i}" for i in range(29)) + "," +
                             ",".join(f"ref{i}" for i in range(29)) + "," +
                             ",".join(f"act{i}" for i in range(29)) +
                             ",delta_max,action_norm,token_norm\n")

                while not robot._shutdown_event.is_set():
                    t0 = time.time()
                    if process_keyboard(kb.get_key(), ms, controller): break

                    obs = robot.get_observation(); t_obs = time.time()
                    obs_t.append(1000*(t_obs - t0))
                    if not obs:
                        step += 1; prev_end = time.time()
                        time.sleep(max(0.0, CONTROL_DT-(time.time()-t0))); continue

                    process_joystick(obs, ms, controller)
                    clamp_mode_params(ms)

                    is_static = LM(ms.mode) in STATIC_MODES
                    do_req = ms.needs_replan and step > 0
                    if do_req: ms.needs_replan = False; replan_timer = 0.0
                    elif not is_static and step > 0 and ms.speed != 0:
                        replan_timer += CONTROL_DT
                        if replan_timer >= replan_interval(ms.mode):
                            do_req = True; replan_timer = 0.0
                    if do_req: planner.request_replan(controller.ref_cursor, ms)

                    do_enc = (step % ENCODER_UPDATE_EVERY == 0)
                    t_step = time.time()
                    action = controller.step(obs, update_encoder=do_enc, debug=(step % DEBUG_PRINT_EVERY == 0))
                    step_ms = 1000*(time.time()-t_step)
                    (enc_t if do_enc else dec_t).append(step_ms)

                    t_act = time.time()
                    robot.send_action(action)
                    act_t.append(1000*(time.time()-t_act))

                    result = planner.try_get_new_motion()
                    t_blend = time.time()
                    if result:
                        controller.blend_new_motion(*result)
                        blend_ms = 1000*(time.time()-t_blend)
                        blend_n += 1; did_blend = True
                    else:
                        blend_ms = 0.0

                    if step % 5 == 0:
                        t_rel = time.time() - t_start
                        q_r  = np.array([obs.get(f"{n}.q", 0) for n in jnames])
                        a_v  = np.array([action.get(f"{n}.q", 0) for n in jnames])
                        cur, ts = controller.ref_cursor, controller.motion_timesteps
                        q_ref = controller.motion_joint_positions[min(cur,ts-1)] if ts > 0 else np.zeros(29)
                        log_f.write(f"{t_rel:.4f},{step},{cur},{ts},{int(did_blend)},{ms.mode}," +
                                    ",".join(f"{v:.6f}" for v in q_r) + "," +
                                    ",".join(f"{v:.6f}" for v in q_ref) + "," +
                                    ",".join(f"{v:.6f}" for v in a_v) + "," +
                                    f"{np.max(np.abs(a_v-q_r)):.6f},"
                                    f"{np.linalg.norm(a_v):.6f},"
                                    f"{np.linalg.norm(controller.token):.6f}\n")
                        did_blend = False

                    now = time.time(); loop_ms = 1000*(now-t0)
                    wall_dt = now - prev_end; loop_t.append(loop_ms)
                    if loop_ms > 50:
                        stall_src = (f"[STALL] {loop_ms:.0f}ms: "
                                     f"obs={obs_t[-1]:.0f} blend={blend_ms:.0f} step={step_ms:.0f} act={act_t[-1]:.0f}")
                    if loop_ms > CONTROL_DT*1500: slow_n += 1

                    controller.advance_cursor(wall_dt)

                    if now - last_status > 2.0:
                        def _avg(l): return sum(l)/len(l) if l else 0
                        hz = 1000/_avg(loop_t) if _avg(loop_t) else 0
                        print(f"\r  {ms.status_line()}  step={step} ref={controller.ref_cursor}/{controller.motion_timesteps} "
                              f"loop={_avg(loop_t):.1f}ms(max={max(loop_t,default=0):.1f}) hz={hz:.0f} "
                              f"enc={_avg(enc_t):.1f} dec={_avg(dec_t):.1f} obs={_avg(obs_t):.1f} "
                              f"slow={slow_n} blends={blend_n}", end="", flush=True)
                        if stall_src: print(f"\n  {stall_src}"); stall_src = ""
                        last_status = now
                        loop_t=enc_t=dec_t=obs_t=act_t=[]; slow_n=blend_n=0

                    prev_end = time.time()
                    gc_timer += CONTROL_DT
                    if gc_timer >= 10.0: gc.collect(); gc_timer = 0.0
                    step += 1
                    time.sleep(max(0.0, CONTROL_DT-(time.time()-t0)))

        except KeyboardInterrupt:
            pass
        finally:
            gc.enable()
            print(f"\n[Log] Saved to {log_path}")
            planner.stop_subprocess()
            print("\nStopping...")
            if robot.is_connected: robot.disconnect()
            print("Done.")

if __name__ == "__main__":
    main()