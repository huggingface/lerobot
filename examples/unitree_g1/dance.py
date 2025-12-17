#!/usr/bin/env python3
"""
WBT (Whole Body Tracking) Dance Policy for Unitree G1

Uses ONNX model with motion data baked in.
Pattern matches gr00t_locomotion.py - uses UnitreeG1 robot class.

Usage:
    python examples/unitree_g1/dance.py
"""

import argparse
import json
import logging
import threading
import time
from xml.etree import ElementTree

import numpy as np
import onnx
import onnxruntime as ort
import pinocchio as pin

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DANCE_ONNX_PATH = "examples/unitree_g1/fastsac_g1_29dof_dancing.onnx"
CONTROL_DT = 0.02  # 50 Hz
NUM_DOFS = 29

# Default joint positions (holosoma training defaults)
DEFAULT_DOF_POS = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # Left leg (6)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # Right leg (6)
    0.0, 0.0, 0.0,                           # Waist (3)
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # Left arm (7)
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # Right arm (7)
], dtype=np.float32)

# Stiff hold KP/KD (for initialization)
STIFF_KP = np.array([
    150, 150, 200, 200, 40, 40,
    150, 150, 200, 200, 40, 40,
    200, 200, 100,
    100, 100, 100, 100, 50, 50, 50,
    100, 100, 100, 100, 50, 50, 50,
], dtype=np.float32)

STIFF_KD = np.array([
    2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
    2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
    5.0, 5.0, 5.0,
    2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
    2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
], dtype=np.float32)

# Joints to freeze at 0 with high KP
FROZEN_JOINTS = [13, 14, 20, 21, 27, 28]
FROZEN_KP = 500.0
FROZEN_KD = 5.0

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
    """Pinocchio forward kinematics for torso_link orientation."""

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
        logger.info(f"Pinocchio FK: {len(pin_names)} joints, torso_link frame={self.ref_frame_id}")

    def get_torso_quat(self, pos, quat_wxyz, dof_pos):
        """Get torso_link orientation in world frame."""
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        config = np.concatenate([pos, quat_xyzw, dof_pos[self.idx_map]])
        pin.framesForwardKinematics(self.model, self.data, config)
        coeffs = pin.Quaternion(self.data.oMf[self.ref_frame_id].rotation).coeffs()
        return np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]]).reshape(1, 4)


# =============================================================================
# DANCE CONTROLLER
# =============================================================================

class DanceController:
    """
    Handles WBT dance policy for the Unitree G1 robot.

    This controller manages:
    - 29-joint observation processing
    - Pinocchio FK for torso orientation
    - Policy inference with motion data from ONNX
    """

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

        # Get initial motion data from ONNX
        dummy = np.zeros((1, self.obs_dim), dtype=np.float32)
        outs = self.policy.run(["joint_pos", "joint_vel", "ref_quat_xyzw"],
                              {"obs": dummy, "time_step": np.array([[0]], dtype=np.float32)})
        self.motion_command = np.concatenate(outs[0:2], axis=1)
        self.ref_quat_xyzw = outs[2]
        self.motion_start_pose = outs[0].flatten()

        # Thread management
        self.dance_running = False
        self.dance_thread = None

        logger.info(f"DanceController: obs_dim={self.obs_dim}, action_scale={action_scale}")

    def capture_yaw_offset(self):
        """Capture robot's current yaw for relative tracking."""
        robot_state = self.robot.lowstate_buffer.get_data()
        if robot_state and self.pinocchio_fk:
            quat = np.array(robot_state.imu_state.quaternion, dtype=np.float32)
            dof = np.array([robot_state.motor_state[i].q for i in range(NUM_DOFS)], dtype=np.float32)
            torso_q = self.pinocchio_fk.get_torso_quat(np.zeros(3), quat, dof)
            _, _, self.yaw_offset = quat_to_rpy(torso_q.flatten())
            logger.info(f"Captured yaw offset: {np.degrees(self.yaw_offset):.1f}°")

    def _remove_yaw_offset(self, quat_wxyz):
        """Remove stored yaw offset from orientation."""
        if abs(self.yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_q = rpy_to_quat((0, 0, -self.yaw_offset)).reshape(1, 4)
        return quat_mul(yaw_q, quat_wxyz)

    def run_step(self):
        """Single dance step - reads state, runs policy, sends commands."""
        robot_state = self.robot.lowstate_buffer.get_data()
        if robot_state is None:
            return

        # Read robot state
        quat = np.array(robot_state.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(robot_state.imu_state.gyroscope, dtype=np.float32)
        dof_pos = np.array([robot_state.motor_state[i].q for i in range(NUM_DOFS)], dtype=np.float32)
        dof_vel = np.array([robot_state.motor_state[i].dq for i in range(NUM_DOFS)], dtype=np.float32)

        # Compute motion_ref_ori_b using FK
        if self.pinocchio_fk:
            torso_q = self.pinocchio_fk.get_torso_quat(np.zeros(3), quat, dof_pos)
            torso_q = self._remove_yaw_offset(torso_q)
            motion_ori = xyzw_to_wxyz(self.ref_quat_xyzw)
            rel_quat = subtract_frame_transforms(torso_q, motion_ori)
            ori_b = matrix_from_quat(rel_quat)[..., :2].reshape(1, -1)
        else:
            ori_b = np.zeros((1, 6), dtype=np.float32)

        dof_rel = (dof_pos - DEFAULT_DOF_POS).reshape(1, -1)

        # Build observation (alphabetical order)
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

        # Run policy
        outs = self.policy.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"],
                              {"obs": obs, "time_step": np.array([[self.timestep]], dtype=np.float32)})

        action = np.clip(outs[0], -100, 100)
        self.motion_command = np.concatenate(outs[1:3], axis=1)
        self.ref_quat_xyzw = outs[3]
        self.last_action = action.copy()

        # Compute target positions
        target_pos = DEFAULT_DOF_POS + action.flatten() * self.action_scale

        # Send commands
        for i in range(NUM_DOFS):
            if i in FROZEN_JOINTS:
                self.robot.msg.motor_cmd[i].q = 0.0
                self.robot.msg.motor_cmd[i].kp = FROZEN_KP
                self.robot.msg.motor_cmd[i].kd = FROZEN_KD
            else:
                self.robot.msg.motor_cmd[i].q = float(target_pos[i])
                self.robot.msg.motor_cmd[i].kp = self.motor_kp[i]
                self.robot.msg.motor_cmd[i].kd = self.motor_kd[i]
            self.robot.msg.motor_cmd[i].qd = 0
            self.robot.msg.motor_cmd[i].tau = 0

        self.robot.send_action(self.robot.msg)
        self.timestep += 1

    def _dance_thread_loop(self):
        """Background thread that runs the dance policy."""
        logger.info("Dance thread started")
        while self.dance_running:
            start_time = time.time()
            try:
                self.run_step()
            except Exception as e:
                logger.error(f"Error in dance loop: {e}")
                import traceback
                traceback.print_exc()

            elapsed = time.time() - start_time
            sleep_time = max(0, CONTROL_DT - elapsed)
            time.sleep(sleep_time)
        logger.info("Dance thread stopped")

    def start_dance_thread(self):
        """Start the dance control thread."""
        if self.dance_running:
            logger.warning("Dance thread already running")
            return

        # Reset state for fresh start
        self.timestep = 0
        self.last_action.fill(0)

        # Re-get initial motion data
        dummy = np.zeros((1, self.obs_dim), dtype=np.float32)
        outs = self.policy.run(["joint_pos", "joint_vel", "ref_quat_xyzw"],
                              {"obs": dummy, "time_step": np.array([[0]], dtype=np.float32)})
        self.motion_command = np.concatenate(outs[0:2], axis=1)
        self.ref_quat_xyzw = outs[2]

        self.capture_yaw_offset()

        logger.info("Starting dance control thread...")
        self.dance_running = True
        self.dance_thread = threading.Thread(target=self._dance_thread_loop, daemon=True)
        self.dance_thread.start()

    def stop_dance_thread(self):
        """Stop the dance control thread."""
        if not self.dance_running:
            return

        logger.info("Stopping dance control thread...")
        self.dance_running = False
        if self.dance_thread:
            self.dance_thread.join(timeout=2.0)
        logger.info("Dance control thread stopped")

    def reset_to_motion_pose(self, duration: float = 3.0):
        """Move robot to initial motion pose over given duration."""
        logger.info(f"Moving to dance start pose ({duration}s)...")

        robot_state = self.robot.lowstate_buffer.get_data()
        init_pos = np.array([robot_state.motor_state[i].q for i in range(NUM_DOFS)], dtype=np.float32)
        target_pos = self.motion_start_pose

        num_steps = int(duration / CONTROL_DT)
        for step in range(num_steps):
            alpha = step / num_steps
            interp = init_pos * (1 - alpha) + target_pos * alpha

            for i in range(NUM_DOFS):
                if i in FROZEN_JOINTS:
                    self.robot.msg.motor_cmd[i].q = 0.0
                    self.robot.msg.motor_cmd[i].kp = FROZEN_KP
                    self.robot.msg.motor_cmd[i].kd = FROZEN_KD
                else:
                    self.robot.msg.motor_cmd[i].q = float(interp[i])
                    self.robot.msg.motor_cmd[i].kp = STIFF_KP[i]
                    self.robot.msg.motor_cmd[i].kd = STIFF_KD[i]
                self.robot.msg.motor_cmd[i].qd = 0
                self.robot.msg.motor_cmd[i].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(CONTROL_DT)

        logger.info("At dance start pose!")


# =============================================================================
# MAIN
# =============================================================================

def load_dance_policy(onnx_path: str):
    """Load dance policy and extract metadata."""
    logger.info(f"Loading dance policy: {onnx_path}")

    policy = ort.InferenceSession(onnx_path)
    model = onnx.load(onnx_path)
    metadata = {p.key: json.loads(p.value) for p in model.metadata_props}

    motor_kp = np.array(metadata.get("kp", STIFF_KP), dtype=np.float32)
    motor_kd = np.array(metadata.get("kd", STIFF_KD), dtype=np.float32)
    action_scale = float(metadata.get("action_scale", 1.0))
    urdf_text = metadata.get("robot_urdf", None)

    logger.info(f"  Obs dim: {policy.get_inputs()[0].shape[1]}")
    logger.info(f"  Action scale: {action_scale}")
    logger.info(f"  KP range: [{motor_kp.min():.1f}, {motor_kp.max():.1f}]")

    # Build Pinocchio FK if URDF available
    pinocchio_fk = None
    if urdf_text:
        logger.info("  Building Pinocchio FK from URDF...")
        pinocchio_fk = PinocchioFK(urdf_text)
    else:
        logger.warning("  No URDF in metadata - FK will not work!")

    return policy, pinocchio_fk, motor_kp, motor_kd, action_scale


def main():
    parser = argparse.ArgumentParser(description="WBT Dance Policy for Unitree G1")
    parser.add_argument("--onnx", type=str, default=DANCE_ONNX_PATH, help="Path to dance ONNX model")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()

    print("=" * 70)
    print("💃 WBT DANCE POLICY")
    print("=" * 70)

    # Load policy
    policy, pinocchio_fk, motor_kp, motor_kd, action_scale = load_dance_policy(args.onnx)

    # Initialize robot
    logger.info("Initializing robot...")
    config = UnitreeG1Config()
    robot = UnitreeG1(config)
    logger.info("Robot connected!")

    # Create controller
    controller = DanceController(policy, robot, pinocchio_fk, motor_kp, motor_kd, action_scale)

    try:
        # Move to start pose
        controller.reset_to_motion_pose(duration=3.0)

        # Start dancing
        controller.start_dance_thread()

        logger.info("Dancing! Press Ctrl+C to stop.")
        print("-" * 70)

        # Log status periodically
        while True:
            time.sleep(2.0)
            logger.info(f"timestep={controller.timestep}")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.stop_dance_thread()
        robot.disconnect()

    print("\nDone!")


if __name__ == "__main__":
    main()
