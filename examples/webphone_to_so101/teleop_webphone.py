#!/usr/bin/env python
"""
Web-based phone teleoperation for SO-101 robot arm.

Uses device orientation sensors (no AR/VR) via a FastAPI web server.
Supports both MuJoCo simulation and real robot hardware.

Usage:
    # MuJoCo simulation
    python examples/webphone_to_so101/teleop_webphone.py --mode mujoco

    # Real robot (placeholder — wire up your motor driver)
    python examples/webphone_to_so101/teleop_webphone.py --mode real
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

# ── CLI args ──────────────────────────────────────────────────────
_MODE = "mujoco"
_HOST = "0.0.0.0"
_PORT = 4443
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        _MODE = sys.argv[i + 1]
    elif arg == "--host" and i + 1 < len(sys.argv):
        _HOST = sys.argv[i + 1]
    elif arg == "--port" and i + 1 < len(sys.argv):
        _PORT = int(sys.argv[i + 1])

USE_MUJOCO = _MODE == "mujoco"
USE_REAL = _MODE == "real"

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # lerobot/
URDF_PATH = PROJECT_ROOT / "SO101" / "so101_new_calib.urdf"

# ── Imports ───────────────────────────────────────────────────────
from web_server import WebPhoneServer

from lerobot.model.kinematics import RobotKinematicsDLS
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep

if USE_MUJOCO:
    from gym_so101.envs.so101_gym_env import (
        ALL_JOINT_NAMES,
        ARM_JOINT_NAMES,
        GRIPPER_JOINT_NAME,
        SO101GymEnv,
    )
    from gym_so101.wrappers.viewer_wrapper import PassiveViewerWrapper

# ── Config ────────────────────────────────────────────────────────
FPS = 30
DEBUG_PRINT_INTERVAL = 15
PHONE_STALE_TIMEOUT_FRAMES = int(FPS * 2.0)  # 2 seconds

ARM_JOINT_NAMES_LIST = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
]
ALL_JOINT_NAMES_LIST = ARM_JOINT_NAMES_LIST + ["gripper"]
if USE_MUJOCO:
    ARM_JOINT_NAMES_LIST = list(ARM_JOINT_NAMES)
    ALL_JOINT_NAMES_LIST = list(ALL_JOINT_NAMES)

# ── Helpers ───────────────────────────────────────────────────────

def robot_obs_from_mujoco(env) -> dict:
    """Extract robot observation from MuJoCo env in lerobot format."""
    state = env.unwrapped.get_robot_state()
    # state layout: [arm_qpos(5), arm_qvel(5), gripper_ctrl(1), gripper_qpos(1), tcp_pos(3)]
    obs = {}
    for i, name in enumerate(ALL_JOINT_NAMES_LIST):
        if name == "gripper":
            obs[f"{name}.pos"] = float(state[11])  # gripper_qpos is at index 11
        else:
            obs[f"{name}.pos"] = float(state[i])
    return obs


def joint_action_to_mujoco_ctrl(joint_action: dict) -> np.ndarray:
    ctrl = np.zeros(len(ALL_JOINT_NAMES_LIST), dtype=np.float32)
    for i, name in enumerate(ALL_JOINT_NAMES_LIST):
        ctrl[i] = float(joint_action.get(f"{name}.pos", 0.0))
    return ctrl


def load_joint_limits_from_mujoco(env, margin_pct: float = 0.25) -> dict:
    limits = {}
    model = env.unwrapped.model
    for name in ALL_JOINT_NAMES_LIST:
        jid = model.joint(name).id
        low = model.jnt_range[jid][0]
        high = model.jnt_range[jid][1]
        margin = (high - low) * margin_pct * 0.5
        limits[name] = (low + margin, high - margin)
    return limits


def deadband_joint_commands(action: dict, threshold: float = 0.1) -> dict:
    result = {}
    for k, v in action.items():
        if isinstance(v, (int, float, np.floating)):
            result[k] = v if abs(float(v)) > threshold else 0.0
        else:
            result[k] = v
    return result


def clip_joint_action_deg(action: dict, limits: dict) -> dict:
    result = {}
    for k, v in action.items():
        if isinstance(k, str) and k.endswith(".pos"):
            name = k.removesuffix(".pos")
            if name in limits:
                lo, hi = limits[name]
                result[k] = max(lo, min(hi, float(v)))
            else:
                result[k] = v
        else:
            result[k] = v
    return result


# ── Main ──────────────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 60}")
    print(f"  WebPhone Teleop — SO-101 Arm Control")
    print(f"  Mode: {'MuJoCo Simulation' if USE_MUJOCO else 'Real Robot'}")
    print(f"{'=' * 60}\n")

    # ── Environment ──
    if USE_MUJOCO:
        env = SO101GymEnv(render_mode="human", control_dt=1.0 / FPS, physics_dt=0.002)
        env = PassiveViewerWrapper(env)
    else:
        env = None  # Real robot: no MuJoCo env

    # ── Web server ──
    server = WebPhoneServer(host=_HOST, port=_PORT)
    server.connect()

    # ── Pipeline ──
    kinematics_solver = RobotKinematicsDLS(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES_LIST,
    )

    processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=PhoneOS.ANDROID),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=ARM_JOINT_NAMES_LIST,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.50,
            ),
            GripperVelocityToJoint(speed_factor=1.5, clip_min=-1.0, clip_max=1.0),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=ARM_JOINT_NAMES_LIST,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ── Limits ──
    if USE_MUJOCO:
        joint_limits = load_joint_limits_from_mujoco(env, 0.25)
    else:
        joint_limits = {
            "shoulder_pan": (-150, 150), "shoulder_lift": (-120, 120),
            "elbow_flex": (-120, 120), "wrist_flex": (-120, 120),
            "wrist_roll": (-180, 180), "gripper": (0, 1),
        }

    if USE_MUJOCO:
        env.reset()

    # Wait for client
    print("Waiting for phone/tablet to connect...")
    print(f"Open https://<IP>:{_PORT} or scan QR code\n")
    while not server.is_connected:
        time.sleep(0.5)
    print("Client connected! Calibrate by enabling the toggle.\n")

    # ── Main loop ──
    stale_counter = 0
    frame = 0
    t_last_report = time.perf_counter()
    _last_valid_ctrl = np.zeros(len(ARM_JOINT_NAMES_LIST) + 1, dtype=np.float32)
    ctrl = _last_valid_ctrl.copy()
    _prev_enabled = False
    joint_action = {}

    try:
        while True:
            t0 = time.perf_counter()

            # ── Observation ──
            if USE_MUJOCO:
                robot_obs = robot_obs_from_mujoco(env)
            else:
                robot_obs = {
                    "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0,
                    "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0,
                    "wrist_roll.pos": 0.0, "gripper.pos": 0.0,
                }

            # ── Teleop action ──
            phone_obs = server.get_action()
            phone_ok = (
                phone_obs is not None
                and len(phone_obs) > 0
                and phone_obs.get("phone.rot") is not None
            )

            # ── Process ──
            if not phone_ok:
                stale_counter += 1
                if stale_counter >= PHONE_STALE_TIMEOUT_FRAMES:
                    print("  *** SENSOR STALE — SENDING ZERO ***")
                    stale_counter = 0
                    if USE_MUJOCO:
                        env.step(np.zeros(len(ALL_JOINT_NAMES_LIST), dtype=np.float32))
            else:
                stale_counter = 0
                enabled = phone_obs.get("phone.enabled", False)

                if enabled != _prev_enabled:
                    print(f"  [TOGGLE] {'ON' if enabled else 'OFF'}")
                    _prev_enabled = enabled

                joint_action = processor((phone_obs, robot_obs))
                joint_action = deadband_joint_commands(joint_action)
                joint_action = clip_joint_action_deg(joint_action, joint_limits)

                if USE_MUJOCO:
                    ctrl = joint_action_to_mujoco_ctrl(joint_action)
                    env.step(ctrl)
                else:
                    # Real robot: send to motors (placeholder)
                    ctrl = np.array([
                        float(joint_action.get(f"{n}.pos", 0))
                        for n in ALL_JOINT_NAMES_LIST
                    ])

            # ── Debug print ──
            frame += 1
            if frame % DEBUG_PRINT_INTERVAL == 0:
                now = time.perf_counter()
                elapsed = now - t_last_report
                t_last_report = now

                enabled = phone_obs.get("phone.enabled", False) if phone_obs else False
                status = "ON " if enabled else "OFF"
                if not phone_ok:
                    status = "NO DATA"

                if phone_obs and phone_obs.get("phone.rot") is not None:
                    rv = phone_obs["phone.rot"].as_rotvec() * 180 / math.pi
                else:
                    rv = [0, 0, 0]

                ja = joint_action if isinstance(joint_action, dict) else {}
                sp = ja.get("shoulder_pan.pos", 0)
                sl = ja.get("shoulder_lift.pos", 0)
                ef = ja.get("elbow_flex.pos", 0)
                wf = ja.get("wrist_flex.pos", 0)
                wr = ja.get("wrist_roll.pos", 0)
                gr = ja.get("gripper.pos", 0)

                print(
                    f"\n{'═' * 60}\n"
                    f"  Frame {frame:>5d}  │  {DEBUG_PRINT_INTERVAL / elapsed:4.0f} FPS  │  STATUS: {status}\n"
                    f"{'─' * 60}\n"
                    f"  Orientation (°): [α,β,γ] rotvec = [{rv[0]:+7.2f}, {rv[1]:+7.2f}, {rv[2]:+7.2f}]\n"
                    f"  Joints (°):       pan={sp:+8.2f}  lift={sl:+8.2f}  elbow={ef:+8.2f}  "
                    f"wflex={wf:+8.2f}  wroll={wr:+8.2f}  grip={gr:+.2f}\n"
                    f"{'═' * 60}"
                )

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if USE_MUJOCO:
            env.close()
        server.disconnect()


if __name__ == "__main__":
    main()
