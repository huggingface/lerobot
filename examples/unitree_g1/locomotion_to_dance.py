#!/usr/bin/env python3
"""
Locomotion ↔ Dance Toggle for Unitree G1

Press Enter to instantly switch between locomotion and dance modes.
- Starts in LOCOMOTION mode (joystick control)
- Press Enter → DANCE mode (resets to frame 0)
- Press Enter → LOCOMOTION mode
- Repeat...

Auto-recovery feature:
- If robot tilts beyond threshold during dance, auto-switches to locomotion
- When robot recovers (tilt below recovery threshold), resumes dance from where it left off

Usage:
    python examples/unitree_g1/locomotion_to_dance.py
    python examples/unitree_g1/locomotion_to_dance.py --tilt-threshold 25 --recovery-threshold 10
"""

import argparse
import json
import logging
import select
import sys
import threading
import time
from xml.etree import ElementTree

import numpy as np
import onnx
import onnxruntime as ort
import pinocchio as pin
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_DOFS = 29
CONTROL_DT = 0.02  # 50Hz

# Locomotion config
DEFAULT_HOLOSOMA_REPO_ID = "nepyope/holosoma_locomotion"
LOCOMOTION_ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
GAIT_PERIOD = 1.0

# Dance config
DANCE_ONNX_PATH = "examples/unitree_g1/fastsac_g1_29dof_dancing.onnx"
FROZEN_JOINTS = [13, 14, 20, 21, 27, 28]
FROZEN_KP = 500.0
FROZEN_KD = 5.0

# fmt: off
# 29-DOF defaults (holosoma training)
DEFAULT_29DOF_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    0.0, 0.0, 0.0,                          # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,    # right arm
], dtype=np.float32)

DEFAULT_29DOF_KP = np.array([
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 28.501, 28.501,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
], dtype=np.float32)

DEFAULT_29DOF_KD = np.array([
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 1.814, 1.814,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
], dtype=np.float32)

# 23-DOF config (no waist_roll/pitch, no wrist_pitch/yaw)
DEFAULT_23DOF_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    0.0,                                    # waist_yaw only
    0.2, 0.2, 0.0, 0.6, 0.0,               # left arm (5 joints)
    0.2, -0.2, 0.0, 0.6, 0.0,              # right arm (5 joints)
], dtype=np.float32)

DEFAULT_23DOF_KP = np.array([
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179,
    14.251, 14.251, 14.251, 14.251, 14.251,
    14.251, 14.251, 14.251, 14.251, 14.251,
], dtype=np.float32)

DEFAULT_23DOF_KD = np.array([
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558,
    0.907, 0.907, 0.907, 0.907, 0.907,
    0.907, 0.907, 0.907, 0.907, 0.907,
], dtype=np.float32)

# 23-DOF policy index → 29-DOF motor index
DOF_23_TO_MOTOR = [
    0, 1, 2, 3, 4, 5,       # left leg
    6, 7, 8, 9, 10, 11,     # right leg
    12,                      # waist_yaw
    15, 16, 17, 18, 19,     # left arm (skip wrist_pitch/yaw)
    22, 23, 24, 25, 26,     # right arm (skip wrist_pitch/yaw)
]
MISSING_23DOF_MOTORS = [13, 14, 20, 21, 27, 28]
# fmt: on

# =============================================================================
# QUATERNION UTILITIES
# =============================================================================

def quat_inverse(q):
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=1)

def quat_mul(a, b):
    a, b = a.reshape(-1, 4), b.reshape(-1, 4)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z]).T.reshape(a.shape)

def subtract_frame_transforms(q01, q02):
    return quat_mul(quat_inverse(q01), q02)

def matrix_from_quat(q):
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    o = np.stack((
        1 - two_s * (j*j + k*k), two_s * (i*j - k*r), two_s * (i*k + j*r),
        two_s * (i*j + k*r), 1 - two_s * (i*i + k*k), two_s * (j*k - i*r),
        two_s * (i*k - j*r), two_s * (j*k + i*r), 1 - two_s * (i*i + j*j),
    ), -1)
    return o.reshape(q.shape[:-1] + (3, 3))

def xyzw_to_wxyz(xyzw):
    return np.concatenate([xyzw[:, -1:], xyzw[:, :3]], axis=1)

def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw

def rpy_to_quat(rpy):
    roll, pitch, yaw = rpy
    cy, sy = np.cos(yaw*0.5), np.sin(yaw*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cr, sr = np.cos(roll*0.5), np.sin(roll*0.5)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

# =============================================================================
# PINOCCHIO FK
# =============================================================================

DOF_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class PinocchioFK:
    def __init__(self, urdf_text: str):
        root = ElementTree.fromstring(urdf_text)
        for parent in root.iter():
            for child in list(parent):
                if child.tag.split("}")[-1] in {"visual", "collision"}:
                    parent.remove(child)
        xml_text = '<?xml version="1.0"?>\n' + ElementTree.tostring(root, encoding="unicode")
        self.model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        pin_names = [n for n in self.model.names if n not in ["universe", "root_joint"]]
        self.idx_map = np.array([DOF_NAMES.index(n) for n in pin_names])
        self.ref_frame_id = self.model.getFrameId("torso_link")

    def get_torso_quat(self, pos, quat_wxyz, dof_pos):
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        config = np.concatenate([pos, quat_xyzw, dof_pos[self.idx_map]])
        pin.framesForwardKinematics(self.model, self.data, config)
        coeffs = pin.Quaternion(self.data.oMf[self.ref_frame_id].rotation).coeffs()
        return np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]]).reshape(1, 4)

    def get_torso_tilt(self, pos, quat_wxyz, dof_pos):
        """Get torso tilt angle from upright (degrees). Uses roll and pitch."""
        torso_q = self.get_torso_quat(pos, quat_wxyz, dof_pos)
        roll, pitch, _ = quat_to_rpy(torso_q.flatten())
        # Tilt is the angle from vertical - combine roll and pitch
        tilt_rad = np.sqrt(roll**2 + pitch**2)
        return np.degrees(tilt_rad), np.degrees(roll), np.degrees(pitch)


# =============================================================================
# LOCOMOTION CONTROLLER
# =============================================================================

class LocomotionController:
    """Holosoma whole-body locomotion (23-DOF or 29-DOF)."""

    def __init__(self, policy, robot, obs_dim: int):
        self.policy = policy
        self.robot = robot
        self.obs_dim = obs_dim

        # Detect DOF mode
        self.is_23dof = (obs_dim == 82)
        self.num_dof = 23 if self.is_23dof else 29

        if self.is_23dof:
            self.default_angles = DEFAULT_23DOF_ANGLES
            self.kp = DEFAULT_23DOF_KP
            self.kd = DEFAULT_23DOF_KD
            self.motor_map = DOF_23_TO_MOTOR
            logger.info("Locomotion: 23-DOF (82D obs)")
        else:
            self.default_angles = DEFAULT_29DOF_ANGLES
            self.kp = DEFAULT_29DOF_KP
            self.kd = DEFAULT_29DOF_KD
            self.motor_map = list(range(29))
            logger.info("Locomotion: 29-DOF (100D obs)")

        self.cmd = np.zeros(3, dtype=np.float32)
        self.qj = np.zeros(self.num_dof, dtype=np.float32)
        self.dqj = np.zeros(self.num_dof, dtype=np.float32)
        self.obs = np.zeros(obs_dim, dtype=np.float32)
        self.last_action = np.zeros(self.num_dof, dtype=np.float32)

        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.phase_dt = 2 * np.pi / (50.0 * GAIT_PERIOD)
        self.is_standing = True

    def run_step(self):
        """Single locomotion step."""
        state = self.robot.lowstate_buffer.get_data()
        if state is None:
            return

        # Joystick
        if state.wireless_remote is not None:
            self.robot.remote_controller.set(state.wireless_remote)

        ly = self.robot.remote_controller.ly if abs(self.robot.remote_controller.ly) > 0.1 else 0.0
        lx = self.robot.remote_controller.lx if abs(self.robot.remote_controller.lx) > 0.1 else 0.0
        rx = self.robot.remote_controller.rx if abs(self.robot.remote_controller.rx) > 0.1 else 0.0
        self.cmd[0], self.cmd[1], self.cmd[2] = ly, -lx, -rx

        # Read joints via motor map
        for i in range(self.num_dof):
            self.qj[i] = state.motor_state[self.motor_map[i]].q
            self.dqj[i] = state.motor_state[self.motor_map[i]].dq

        # IMU
        quat = state.imu_state.quaternion
        ang_vel = np.array(state.imu_state.gyroscope, dtype=np.float32)
        gravity = self.robot.get_gravity_orientation(quat)

        # Scale
        qj_obs = (self.qj - self.default_angles) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_s = ang_vel * ANG_VEL_SCALE

        # Phase
        cmd_mag = np.linalg.norm(self.cmd[:2])
        ang_mag = abs(self.cmd[2])
        if cmd_mag < 0.01 and ang_mag < 0.01:
            self.phase[0, :] = np.pi
            self.is_standing = True
        elif self.is_standing:
            self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
            self.is_standing = False
        else:
            self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2*np.pi) - np.pi

        sin_ph, cos_ph = np.sin(self.phase[0]), np.cos(self.phase[0])

        # Build obs
        if self.is_23dof:
            self.obs[0:23] = self.last_action
            self.obs[23:26] = ang_vel_s
            self.obs[26] = self.cmd[2]
            self.obs[27:29] = self.cmd[:2]
            self.obs[29:31] = cos_ph
            self.obs[31:54] = qj_obs
            self.obs[54:77] = dqj_obs
            self.obs[77:80] = gravity
            self.obs[80:82] = sin_ph
        else:
            self.obs[0:29] = self.last_action
            self.obs[29:32] = ang_vel_s
            self.obs[32] = self.cmd[2]
            self.obs[33:35] = self.cmd[:2]
            self.obs[35:37] = cos_ph
            self.obs[37:66] = qj_obs
            self.obs[66:95] = dqj_obs
            self.obs[95:98] = gravity
            self.obs[98:100] = sin_ph

        # Inference
        obs_in = self.obs.reshape(1, -1).astype(np.float32)
        ort_in = {self.policy.get_inputs()[0].name: obs_in}
        raw_action = self.policy.run(None, ort_in)[0].squeeze()
        clipped = np.clip(raw_action, -100.0, 100.0)
        self.last_action = clipped.copy()
        scaled = clipped * LOCOMOTION_ACTION_SCALE
        target = self.default_angles + scaled

        # Send commands
        for i in range(self.num_dof):
            motor_idx = self.motor_map[i]
            self.robot.msg.motor_cmd[motor_idx].q = float(target[i])
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = self.kp[i]
            self.robot.msg.motor_cmd[motor_idx].kd = self.kd[i]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # Zero missing joints for 23-DOF
        if self.is_23dof:
            for idx in MISSING_23DOF_MOTORS:
                self.robot.msg.motor_cmd[idx].q = 0.0
                self.robot.msg.motor_cmd[idx].qd = 0
                self.robot.msg.motor_cmd[idx].kp = 40.0
                self.robot.msg.motor_cmd[idx].kd = 2.0
                self.robot.msg.motor_cmd[idx].tau = 0

        self.robot.send_action(self.robot.msg)

    def reset(self):
        """Reset state for fresh start."""
        self.last_action.fill(0)
        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.is_standing = True


# =============================================================================
# DANCE CONTROLLER
# =============================================================================

class DanceController:
    """WBT dance policy with FK for torso tracking."""

    def __init__(self, policy, robot, pinocchio_fk, motor_kp, motor_kd, action_scale):
        self.policy = policy
        self.robot = robot
        self.pinocchio_fk = pinocchio_fk
        self.motor_kp = motor_kp
        self.motor_kd = motor_kd
        self.action_scale = action_scale

        self.obs_dim = policy.get_inputs()[0].shape[1]
        self.last_action = np.zeros((1, NUM_DOFS), dtype=np.float32)
        self.motion_command = None
        self.ref_quat_xyzw = None
        self.timestep = 0
        self.yaw_offset = 0.0

        logger.info(f"Dance: obs_dim={self.obs_dim}, action_scale={action_scale}")

    def initialize(self, reset_to_frame_0: bool = True):
        """Initialize dance. If reset_to_frame_0=True, starts from frame 0. Otherwise resumes."""
        if reset_to_frame_0:
            self.timestep = 0
            self.last_action.fill(0)

            # Get initial motion data at frame 0
            dummy = np.zeros((1, self.obs_dim), dtype=np.float32)
            outs = self.policy.run(["joint_pos", "joint_vel", "ref_quat_xyzw"],
                                  {"obs": dummy, "time_step": np.array([[0]], dtype=np.float32)})
            self.motion_command = np.concatenate(outs[0:2], axis=1)
            self.ref_quat_xyzw = outs[2]
            logger.info("Dance: reset to frame 0")
        else:
            # Resume from current timestep - just update motion command for current frame
            dummy = np.zeros((1, self.obs_dim), dtype=np.float32)
            outs = self.policy.run(["joint_pos", "joint_vel", "ref_quat_xyzw"],
                                  {"obs": dummy, "time_step": np.array([[self.timestep]], dtype=np.float32)})
            self.motion_command = np.concatenate(outs[0:2], axis=1)
            self.ref_quat_xyzw = outs[2]
            logger.info(f"Dance: resuming from frame {self.timestep}")

        # Capture yaw offset
        state = self.robot.lowstate_buffer.get_data()
        if state and self.pinocchio_fk:
            quat = np.array(state.imu_state.quaternion, dtype=np.float32)
            dof = np.array([state.motor_state[i].q for i in range(NUM_DOFS)], dtype=np.float32)
            torso_q = self.pinocchio_fk.get_torso_quat(np.zeros(3), quat, dof)
            _, _, self.yaw_offset = quat_to_rpy(torso_q.flatten())
            logger.info(f"Dance yaw offset: {np.degrees(self.yaw_offset):.1f}°")

    def _remove_yaw_offset(self, quat_wxyz):
        if abs(self.yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_q = rpy_to_quat((0, 0, -self.yaw_offset)).reshape(1, 4)
        return quat_mul(yaw_q, quat_wxyz)

    def run_step(self):
        """Single dance step."""
        state = self.robot.lowstate_buffer.get_data()
        if state is None:
            return

        quat = np.array(state.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(state.imu_state.gyroscope, dtype=np.float32)
        dof_pos = np.array([state.motor_state[i].q for i in range(NUM_DOFS)], dtype=np.float32)
        dof_vel = np.array([state.motor_state[i].dq for i in range(NUM_DOFS)], dtype=np.float32)

        # FK for torso orientation
        if self.pinocchio_fk:
            torso_q = self.pinocchio_fk.get_torso_quat(np.zeros(3), quat, dof_pos)
            torso_q = self._remove_yaw_offset(torso_q)
            motion_ori = xyzw_to_wxyz(self.ref_quat_xyzw)
            rel_quat = subtract_frame_transforms(torso_q, motion_ori)
            ori_b = matrix_from_quat(rel_quat)[..., :2].reshape(1, -1)
        else:
            ori_b = np.zeros((1, 6), dtype=np.float32)

        dof_rel = (dof_pos - DEFAULT_29DOF_ANGLES).reshape(1, -1)

        # Build obs (alphabetical)
        obs_dict = {
            "actions": self.last_action,
            "base_ang_vel": ang_vel.reshape(1, 3),
            "dof_pos": dof_rel,
            "dof_vel": dof_vel.reshape(1, -1),
            "motion_command": self.motion_command,
            "motion_ref_ori_b": ori_b,
        }
        obs = np.concatenate([obs_dict[k].astype(np.float32) for k in sorted(obs_dict.keys())], axis=1)
        obs = np.clip(obs, -100, 100)

        # Inference
        outs = self.policy.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"],
                              {"obs": obs, "time_step": np.array([[self.timestep]], dtype=np.float32)})
        action = np.clip(outs[0], -100, 100)
        self.motion_command = np.concatenate(outs[1:3], axis=1)
        self.ref_quat_xyzw = outs[3]
        self.last_action = action.copy()

        target = DEFAULT_29DOF_ANGLES + action.flatten() * self.action_scale

        # Send commands
        for i in range(NUM_DOFS):
            if i in FROZEN_JOINTS:
                self.robot.msg.motor_cmd[i].q = 0.0
                self.robot.msg.motor_cmd[i].kp = FROZEN_KP
                self.robot.msg.motor_cmd[i].kd = FROZEN_KD
            else:
                self.robot.msg.motor_cmd[i].q = float(target[i])
                self.robot.msg.motor_cmd[i].kp = self.motor_kp[i]
                self.robot.msg.motor_cmd[i].kd = self.motor_kd[i]
            self.robot.msg.motor_cmd[i].qd = 0
            self.robot.msg.motor_cmd[i].tau = 0

        self.robot.send_action(self.robot.msg)
        self.timestep += 1


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Locomotion ↔ Dance Toggle")
    parser.add_argument("--loco-repo", type=str, default=DEFAULT_HOLOSOMA_REPO_ID)
    parser.add_argument("--dance-onnx", type=str, default=DANCE_ONNX_PATH)
    args = parser.parse_args()

    print("=" * 70)
    print("🚶 LOCOMOTION ↔ 💃 DANCE")
    print("=" * 70)
    print("Press ENTER to toggle between modes")
    print("=" * 70)

    # Load locomotion policy
    logger.info("Loading locomotion policy...")
    loco_path = hf_hub_download(repo_id=args.loco_repo, filename="fastsac_g1_29dof.onnx")
    loco_policy = ort.InferenceSession(loco_path)
    loco_obs_dim = loco_policy.get_inputs()[0].shape[1]
    logger.info(f"Locomotion: {loco_obs_dim}D obs")

    # Load dance policy
    logger.info("Loading dance policy...")
    dance_policy = ort.InferenceSession(args.dance_onnx)
    dance_model = onnx.load(args.dance_onnx)
    dance_meta = {p.key: json.loads(p.value) for p in dance_model.metadata_props}
    dance_kp = np.array(dance_meta.get("kp", DEFAULT_29DOF_KP), dtype=np.float32)
    dance_kd = np.array(dance_meta.get("kd", DEFAULT_29DOF_KD), dtype=np.float32)
    dance_action_scale = float(dance_meta.get("action_scale", 1.0))
    logger.info(f"Dance: {dance_policy.get_inputs()[0].shape[1]}D obs, scale={dance_action_scale}")

    # Build Pinocchio FK
    pinocchio_fk = None
    if "robot_urdf" in dance_meta:
        logger.info("Building Pinocchio FK...")
        pinocchio_fk = PinocchioFK(dance_meta["robot_urdf"])

    # Initialize robot
    logger.info("Initializing robot...")
    config = UnitreeG1Config()
    robot = UnitreeG1(config)
    logger.info("Robot connected!")

    # Create controllers
    loco_ctrl = LocomotionController(loco_policy, robot, loco_obs_dim)
    dance_ctrl = DanceController(dance_policy, robot, pinocchio_fk, dance_kp, dance_kd, dance_action_scale)

    # State
    mode = "locomotion"
    toggle_event = threading.Event()
    shutdown = threading.Event()

    # Input thread
    def input_loop():
        while not shutdown.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                toggle_event.set()

    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()

    print("\n🚶 LOCOMOTION MODE - Use joystick to walk")
    print("   Press ENTER to switch to DANCE")
    print("-" * 70)

    step = 0
    try:
        while not shutdown.is_set():
            t0 = time.time()

            # Check toggle
            if toggle_event.is_set():
                toggle_event.clear()
                if mode == "locomotion":
                    mode = "dance"
                    dance_ctrl.initialize()
                    print("\n" + "=" * 70)
                    print("💃 DANCE MODE (frame 0)")
                    print("   Press ENTER to switch to LOCOMOTION")
                    print("=" * 70)
                else:
                    mode = "locomotion"
                    loco_ctrl.reset()
                    print("\n" + "=" * 70)
                    print("🚶 LOCOMOTION MODE")
                    print("   Press ENTER to switch to DANCE")
                    print("=" * 70)

            # Run controller
            if mode == "locomotion":
                loco_ctrl.run_step()
            else:
                dance_ctrl.run_step()

            # Log
            if step % 100 == 0:
                if mode == "locomotion":
                    print(f"[LOCO ] step={step:5d} cmd=[{loco_ctrl.cmd[0]:.2f},{loco_ctrl.cmd[1]:.2f},{loco_ctrl.cmd[2]:.2f}]")
                else:
                    print(f"[DANCE] step={step:5d} timestep={dance_ctrl.timestep}")

            step += 1
            elapsed = time.time() - t0
            if elapsed < CONTROL_DT:
                time.sleep(CONTROL_DT - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        shutdown.set()
        robot.disconnect()

    print("Done!")


if __name__ == "__main__":
    main()
