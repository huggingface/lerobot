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
# See the License for the specif

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from gym_so101.envs.so101_gym_env import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    SO101GymEnv,
)
from gym_so101.wrappers.viewer_wrapper import PassiveViewerWrapper
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
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
RERUN_EVERY_N_FRAMES = 5
DEBUG_PRINT_INTERVAL = 15
SAFETY_MARGIN_PCT = 0.25

CALIB_FPATH = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json"
USE_CALIB_FILE = CALIB_FPATH.is_file()

JOINT_MAX_DELTA_DEG = {
    "shoulder_pan": 999.0,
    "shoulder_lift": 999.0,
    "elbow_flex": 999.0,
    "wrist_flex": 999.0,
    "wrist_roll": 999.0,
    "gripper": 999.0,
}
JOINT_DEADBAND_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
PHONE_STALE_TIMEOUT_FRAMES = 90

_prev_joint_action: dict | None = None
_phone_stale_counter: int = 0


def _deg_from_calib(raw_val: int, rmin: int, rmax: int) -> float:
    mid = (rmin + rmax) / 2.0
    return (raw_val - mid) * 360.0 / 4095.0


def load_joint_limits_from_calib(fpath: Path, margin_pct: float) -> dict:
    with open(fpath) as f:
        calib = json.load(f)

    limits_deg = {}
    print(f"\n  Joint limits from calibration ({fpath}) — {100*(1-2*margin_pct):.0f}% center range:")
    print(f"  {'Motor':<16} {'min_deg':>8} {'max_deg':>8} {'effective_min':>8} {'effective_max':>8}")
    print(f"  {'─' * 16} {'─' * 8} {'─' * 8} {'─' * 14} {'─' * 14}")

    for name in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]:
        c = calib[name]
        rmin, rmax = c["range_min"], c["range_max"]
        if name == "gripper":
            min_val = 0.0
            max_val = 100.0
        else:
            min_val = _deg_from_calib(rmin, rmin, rmax)
            max_val = _deg_from_calib(rmax, rmin, rmax)
        margin = (max_val - min_val) * margin_pct
        if name == "gripper":
            unit = "%"
        else:
            unit = "°"
        limits_deg[name] = (min_val + margin, max_val - margin)
        print(f"  {name:<16} {min_val:>8.2f} {max_val:>8.2f}  {limits_deg[name][0]:>8.2f}{unit}  {limits_deg[name][1]:>8.2f}{unit}")

    print()
    return limits_deg


def load_joint_limits_from_mujoco(env: SO101GymEnv, margin_pct: float) -> dict:
    limits_deg = {}
    print(f"\n  Joint limits from MuJoCo ctrlrange — {100*(1-2*margin_pct):.0f}% center range:")
    print(f"  {'Motor':<16} {'ctrl_low':>10} {'ctrl_high':>10} {'effective_min':>8} {'effective_max':>8}")
    print(f"  {'─' * 16} {'─' * 10} {'─' * 10} {'─' * 14} {'─' * 14}")

    base = env.unwrapped
    for i, name in enumerate(ALL_JOINT_NAMES):
        lo = float(base._ctrl_low[i])
        hi = float(base._ctrl_high[i])
        min_deg = math.degrees(lo)
        max_deg = math.degrees(hi)
        margin = (max_deg - min_deg) * margin_pct
        limits_deg[name] = (min_deg + margin, max_deg - margin)
        print(f"  {name:<16} {lo:>10.4f} {hi:>10.4f}  {limits_deg[name][0]:>8.2f}°  {limits_deg[name][1]:>8.2f}°")

    print()
    return limits_deg


def clip_joint_action_deg(joint_action: dict, limits_deg: dict) -> dict:
    for name, (lo, hi) in limits_deg.items():
        key = f"{name}.pos"
        if key in joint_action:
            joint_action[key] = max(lo, min(hi, joint_action[key]))
    return joint_action


def deadband_joint_commands(joint_action: dict) -> dict:
    global _prev_joint_action
    filtered = {}
    for name in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]:
        key = f"{name}.pos"
        val = joint_action.get(key, 0.0)
        if _prev_joint_action is None:
            filtered[key] = val
            continue
        prev = _prev_joint_action.get(key, 0.0)
        delta = val - prev
        deadband = JOINT_DEADBAND_DEG.get(name, 0.2)
        max_delta = JOINT_MAX_DELTA_DEG.get(name, 5.0)
        if abs(delta) < deadband:
            filtered[key] = prev
        else:
            delta = max(-max_delta, min(max_delta, delta))
            filtered[key] = prev + delta
    _prev_joint_action = filtered.copy()
    return filtered


def joint_action_to_mujoco_ctrl(joint_action: dict) -> np.ndarray:
    ctrl = np.zeros(len(ALL_JOINT_NAMES), dtype=np.float32)
    for i, name in enumerate(ALL_JOINT_NAMES):
        ctrl[i] = math.radians(joint_action.get(f"{name}.pos", 0.0))
    return ctrl


def robot_obs_from_mujoco(env: SO101GymEnv) -> RobotObservation:
    base = env.unwrapped
    qpos = base.data.qpos
    joint_ids = np.array([base.model.joint(n).id for n in ALL_JOINT_NAMES], dtype=np.int32)
    obs = {}
    for i, name in enumerate(ALL_JOINT_NAMES):
        obs[f"{name}.pos"] = math.degrees(float(qpos[joint_ids[i]]))
    return obs


def main():
    LOG_FPATH = Path(__file__).resolve().parent / "teleop_phone_mujoco.log"
    log_f = open(LOG_FPATH, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f
    print(f"=== Teleop Mujoco log started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    env = SO101GymEnv(
        render_mode="human",
        control_dt=1.0 / FPS,
        physics_dt=0.002,
    )
    env = PassiveViewerWrapper(env)

    teleop_config = PhoneConfig(
        phone_os=PhoneOS.ANDROID,
    )
    teleop_device = Phone(teleop_config)

    kinematics_solver = RobotKinematicsDLS(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(ARM_JOINT_NAMES),
    )


    kinematics_solver = RobotKinematicsDLS(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(ARM_JOINT_NAMES),
    )
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 2.0, "y": 2.0, "z": 2.0},
                motor_names=list(ARM_JOINT_NAMES),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.50,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(ARM_JOINT_NAMES),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    if USE_CALIB_FILE:
        joint_limits_deg = load_joint_limits_from_calib(CALIB_FPATH, SAFETY_MARGIN_PCT)
    else:
        joint_limits_deg = load_joint_limits_from_mujoco(env, SAFETY_MARGIN_PCT)

    env.reset()
    teleop_device.connect()
    init_rerun(session_name="phone_so101_mujoco")

    if not teleop_device.is_connected:
        raise ValueError("Teleop (phone) is not connected!")

    print("Starting teleop loop (MuJoCo simulation). Move your phone to teleoperate the robot...")
    print("=" * 60)
    print("  Android WebXR 控制方式:")
    print("  1. 手机/平板浏览器打开: https://<本机IP>:4443")
    print("  2. 接受 HTTPS 证书警告（自签名证书）")
    print("  3. 授权 WebXR / 运动传感器权限")
    print("  4. 手指在屏幕上触摸并拖动 = 使能 (ENABLE)")
    print("  5. 按住虚拟按钮A = 夹爪闭合, 按住虚拟按钮B = 夹爪张开")
    print("  STATUS: 'TOGGLE=ON' = 已使能, 'TOGGLE=OFF' = 暂停")
    print("=" * 60 + "\n")
    global _phone_stale_counter, _prev_joint_action
    frame = 0
    t_last_report = time.perf_counter()
    _last_valid_ctrl = np.zeros(len(ALL_JOINT_NAMES), dtype=np.float32)
    joint_action: dict = {}
    ctrl = _last_valid_ctrl.copy()
    _last_b1_state: bool | None = None

    try:
        while True:
            t0 = time.perf_counter()

            t_obs_start = time.perf_counter()
            robot_obs = robot_obs_from_mujoco(env)
            t_obs = (time.perf_counter() - t_obs_start) * 1000

            t_teleop_start = time.perf_counter()
            phone_obs = teleop_device.get_action()
            t_teleop = (time.perf_counter() - t_teleop_start) * 1000

            phone_ok = (
                phone_obs is not None
                and len(phone_obs) > 0
                and phone_obs.get("phone.pos") is not None
            )

            if phone_ok:
                raw = phone_obs.get("phone.raw_inputs", {})
                enabled_now = phone_obs.get("phone.enabled", False)
                if _last_b1_state is None:
                    _last_b1_state = enabled_now
                    print(f"  [DIAG] Initial raw_inputs: {dict(sorted(raw.items()))}")
                    print(f"  [DIAG] Initial enabled={enabled_now}")
                if enabled_now != _last_b1_state:
                    print(f"  [EVENT] ENABLED: {_last_b1_state} → {enabled_now} (move={raw.get('move', False)})")
                    _last_b1_state = enabled_now

            if frame == 60 and phone_ok:
                raw = phone_obs.get("phone.raw_inputs", {})
                print(f"  [DIAG] Frame 60 raw_inputs: {dict(sorted(raw.items()))}")

            if not phone_ok:
                _phone_stale_counter += 1
            else:
                _phone_stale_counter = 0

            if _phone_stale_counter >= PHONE_STALE_TIMEOUT_FRAMES:
                ctrl = np.zeros(len(ALL_JOINT_NAMES), dtype=np.float32)
                env.step(ctrl)
                print(f"\n  *** PHONE STALE ({_phone_stale_counter} frames) — SENDING ZERO COMMAND ***")
                _prev_joint_action = None
                _phone_stale_counter = 0
                _last_valid_ctrl = ctrl.copy()
                joint_action = {}
                t_proc = 0.0
                t_send = 0.0
            elif phone_ok:
                t_proc_start = time.perf_counter()
                joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))
                t_proc = (time.perf_counter() - t_proc_start) * 1000

                t_send_start = time.perf_counter()
                joint_action = deadband_joint_commands(joint_action)
                joint_action = clip_joint_action_deg(joint_action, joint_limits_deg)
                ctrl = joint_action_to_mujoco_ctrl(joint_action)
                env.step(ctrl)
                _last_valid_ctrl = ctrl.copy()
                t_send = (time.perf_counter() - t_send_start) * 1000
            else:
                env.step(_last_valid_ctrl)
                joint_action = {}
                t_proc = 0.0
                t_send = 0.0

            t_vis = 0.0
            if frame % RERUN_EVERY_N_FRAMES == 0:
                t_vis_start = time.perf_counter()
                log_rerun_data(observation=phone_obs, action=joint_action)
                t_vis = (time.perf_counter() - t_vis_start) * 1000

            frame += 1

            if frame % DEBUG_PRINT_INTERVAL == 0:
                now = time.perf_counter()
                elapsed = now - t_last_report
                t_last_report = now

                if not phone_ok:
                    status = f"NO DATA ({_phone_stale_counter})"
                elif not phone_obs.get("phone.enabled", False):
                    status = "TOGGLE=OFF"
                else:
                    status = "TOGGLE=ON "

                phone_pos = phone_obs.get("phone.pos", None) if phone_obs else None
                phone_rot = phone_obs.get("phone.rot", None) if phone_obs else None
                phone_enabled = phone_obs.get("phone.enabled", False) if phone_obs else False
                raw_inputs = phone_obs.get("phone.raw_inputs", {}) if phone_obs else {}

                pos_x = phone_pos[0] * 1000 if phone_pos is not None else 0.0
                pos_y = phone_pos[1] * 1000 if phone_pos is not None else 0.0
                pos_z = phone_pos[2] * 1000 if phone_pos is not None else 0.0
                rotvec = phone_rot.as_rotvec() * 180.0 / math.pi if phone_rot is not None else [0, 0, 0]
                wx, wy, wz = rotvec[0], rotvec[1], rotvec[2]
                gripper_btnA = raw_inputs.get("reservedButtonA", False)
                gripper_btnB = raw_inputs.get("reservedButtonB", False)
                move_touch = raw_inputs.get("move", False)

                sp = joint_action.get("shoulder_pan.pos", 0.0)
                sl = joint_action.get("shoulder_lift.pos", 0.0)
                ef = joint_action.get("elbow_flex.pos", 0.0)
                wf = joint_action.get("wrist_flex.pos", 0.0)
                wr = joint_action.get("wrist_roll.pos", 0.0)
                gr = joint_action.get("gripper.pos", 0.0)

                print(
                    f"\n{'═' * 78}\n"
                    f"  {frame:>6d} frames  │  {DEBUG_PRINT_INTERVAL / elapsed:5.0f} FPS  │ STATUS: {status:<18}"
                    f"│ obs={t_obs:3.0f}ms teleop={t_teleop:3.0f}ms proc={t_proc:3.0f}ms send={t_send:3.0f}ms vis={t_vis:3.0f}ms\n"
                    f"{'─' * 78}\n"
                    f"  PHONE (calibrated)                │  JOINTS (deg, MuJoCo)\n"
                    f"{'─' * 38}┼{'─' * 39}\n"
                    f"  pos.x (mm): {pos_x:+8.2f}            │  shoulder_pan:  {sp:+8.2f}°\n"
                    f"  pos.y (mm): {pos_y:+8.2f}            │  shoulder_lift: {sl:+8.2f}°\n"
                    f"  pos.z (mm): {pos_z:+8.2f}            │  elbow_flex:    {ef:+8.2f}°\n"
                    f"  rot.wx (°): {wx:+8.2f}            │  wrist_flex:    {wf:+8.2f}°\n"
                    f"  rot.wy (°): {wy:+8.2f}            │  wrist_roll:    {wr:+8.2f}°\n"
                    f"  rot.wz (°): {wz:+8.2f}            │  gripper:       {gr:+8.2f}°\n"
                    f"  enabled: {str(phone_enabled):>5}  move: {str(move_touch):>5}  btnA: {str(gripper_btnA):>5}  btnB: {str(gripper_btnB):>5}   │\n"
                    f"{'═' * 78}"
                )

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        env.close()
        teleop_device.disconnect()
        print(f"\n=== Teleop Mujoco log ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_f.close()
        print(f"Log saved to: {LOG_FPATH.resolve()}")


if __name__ == "__main__":
    main()
