#!/usr/bin/env python

"""
Find joint limits, joint velocity ranges, TCP bounds, and TCP wrench ranges for the UR10e
robot via gamepad teleoperation

Usage:
    python -m lerobot.robots.ur10.ur10_find_limits \
        --ip 192.168.0.100 \
        --rtde_frequency 500 \
        --gripper_port /dev/ttyUSB0 \
        --teleop_time_s 60 \
        --warmup_time_s 5 \
        --fps 10 \
        --ee_step_x 0.001 --ee_step_y 0.001 --ee_step_z 0.001

The orientation is pinned to the *current* TCP at the end of warmup, so pre-pose the wrist
during warmup before recording begins.

Yaw discovery (optional):
    Add `--use_yaw` to allow Right Stick X to rotate the wrist about its tool Z axis
    during the recording phase. The script tracks the min/max yaw OFFSET reached
    (relative to the warmup-end orientation) and prints a `yaw_bounds` line for the
    JSON config:
        processor.inverse_kinematics.end_effector_bounds.min/max  # 4th element = yaw offset
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(description="Find UR10e workspace limits via gamepad teleop")
    parser.add_argument("--ip", type=str, default="192.168.0.100")
    parser.add_argument("--rtde_frequency", type=int, default=500)
    parser.add_argument("--tcp_offset", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        help="[x y z rx ry rz] tool offset from flange (m, axis-angle rad)")
    parser.add_argument("--payload_mass", type=float, default=0.0)
    parser.add_argument("--payload_cog", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--gripper_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--gripper_baudrate", type=int, default=115200)
    parser.add_argument("--teleop_time_s", type=float, default=120.0)
    parser.add_argument("--warmup_time_s", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--ee_step_x", type=float, default=0.002)
    parser.add_argument("--ee_step_y", type=float, default=0.002)
    parser.add_argument("--ee_step_z", type=float, default=0.002)
    parser.add_argument("--servo_lookahead_time", type=float, default=0.15)
    parser.add_argument("--servo_gain", type=float, default=100.0)
    # UR10e base frame is rotated relative to the operator at our workstation; flip x/y so
    # the gamepad sticks map intuitively to forward/back and left/right.
    parser.add_argument("--invert_delta_x", action="store_true", default=True)
    parser.add_argument("--invert_delta_y", action="store_true", default=True)
    parser.add_argument("--invert_delta_z", action="store_true", default=False)
    # Optional yaw DOF — when enabled, Right Stick X rotates the wrist about tool Z
    # during the recording phase, and the printed results include a `yaw_bounds` line.
    parser.add_argument("--use_yaw", action="store_true", default=False)
    parser.add_argument("--invert_delta_yaw", action="store_true", default=False)
    parser.add_argument("--yaw_step", type=float, default=0.01,
                        help="Yaw step size in radians per unit action (default 0.01)")
    args = parser.parse_args()

    # Imports that need hardware
    from lerobot.cameras.configs import CameraConfig  # noqa: F401  (kept for future cam tests)
    from lerobot.robots.ur10 import UR10Robot, UR10RobotConfig
    from lerobot.teleoperators.gamepad import GamepadTeleop
    from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig

    cfg = UR10RobotConfig(
        ip=args.ip,
        rtde_frequency=args.rtde_frequency,
        tcp_offset=list(args.tcp_offset),
        payload_mass=args.payload_mass,
        payload_cog=list(args.payload_cog),
        gripper_port=args.gripper_port,
        gripper_baudrate=args.gripper_baudrate,
        cameras={},
    )
    robot = UR10Robot(cfg)

    print(f"Connecting to UR10e at {args.ip} ...")
    robot.connect()

    # Start the background streaming thread so the URScript stays alive and the controller
    # sees a continuous 200 Hz reference. The teleop loop below just updates the shared
    # target via robot.set_target_pose(...) — no per-frame servoL call.
    robot.start_streaming(
        dt=1.0 / 200,
        lookahead_time=args.servo_lookahead_time,
        gain=args.servo_gain,
    )

    print("Connecting gamepad ...")
    teleop = GamepadTeleop(
        GamepadTeleopConfig(
            use_gripper=True,
            use_yaw=args.use_yaw,
            invert_delta_x=args.invert_delta_x,
            invert_delta_y=args.invert_delta_y,
            invert_delta_z=args.invert_delta_z,
            invert_delta_yaw=args.invert_delta_yaw,
        )
    )
    teleop.connect()

    ee_step = np.array([args.ee_step_x, args.ee_step_y, args.ee_step_z], dtype=np.float32)
    yaw_step = float(args.yaw_step)
    dt = 1.0 / args.fps

    # Latched commanded xyz target — deltas apply to the previous command, not to the live
    # TCP. Without this latching the robot's tracking lag turns each step into a
    # receding-carrot setpoint, causing jerkiness while held and a small reverse on release.
    target_xyz = np.array(robot.get_current_tcp()[:3], dtype=np.float32)

    min_joint_pos = None
    max_joint_pos = None
    min_joint_vel = None
    max_joint_vel = None
    # Absolute TCP — used to derive `ee_bounds` (workspace-clip bounds), which the env
    # consumes in absolute base-frame coordinates.
    min_tcp = None
    max_tcp = None
    # RELATIVE TCP xyz — used to derive `dataset_stats."observation.state"`'s xyz dims,
    # which the policy consumes relative to the per-episode anchor (HIL-SERL paper).
    # Anchored to the warmup-end pose, mirroring the env's reset-time anchor.
    min_tcp_xyz_rel = None
    max_tcp_xyz_rel = None

    # Captured at end of warmup so the user can pre-pose the wrist.
    fixed_rx = fixed_ry = fixed_rz = None
    # Captured at end of warmup as the position-only relative-observation baseline.
    initial_tcp_xyz: np.ndarray | None = None
    # Yaw tracking — only populated when --use_yaw. `R_home_warm` is built at warmup
    # end to mirror the env's `_R_home`; `target_yaw_offset` accumulates per step and is
    # tracked into `min_yaw_offset` / `max_yaw_offset` for the printed `yaw_bounds`.
    R_home_warm: Rotation | None = None
    target_yaw_offset: float = 0.0
    min_yaw_offset: float = 0.0
    max_yaw_offset: float = 0.0

    start_t = time.perf_counter()
    warmup_done = False

    print()
    print("=" * 50)
    print(f"  WARMUP PHASE ({args.warmup_time_s}s)")
    print("  Move freely and pre-pose the wrist. Data is NOT recorded yet.")
    print("=" * 50)
    print()

    try:
        while True:
            t0 = time.perf_counter()

            # print(robot.get_tcp_force())

            teleop_action = teleop.get_action()
            dx = teleop_action.get("delta_x", 0.0)
            dy = teleop_action.get("delta_y", 0.0)
            dz = teleop_action.get("delta_z", 0.0)
            dyaw = float(teleop_action.get("delta_yaw", 0.0)) if args.use_yaw else 0.0
            grip = teleop_action.get("gripper", 1)  # 0=close, 1=stay, 2=open

            tcp = robot.get_current_tcp()
            delta_xyz = np.array([dx, dy, dz], dtype=np.float32) * ee_step
            # Apply delta to the latched commanded target, not the live TCP.
            target_xyz = target_xyz + delta_xyz

            if fixed_rx is None:
                # Use the live wrist orientation while warming up. Yaw input (if any) is
                # ignored here — pre-posing happens via manual physical manipulation, and
                # accumulating dyaw before R_home is captured would just bias the baseline.
                rx, ry, rz = float(tcp[3]), float(tcp[4]), float(tcp[5])
            elif args.use_yaw and R_home_warm is not None:
                # Recording phase with yaw enabled: accumulate yaw OFFSET, clip to a wide
                # ±π window (the printed yaw_bounds capture the *actual* range used), and
                # compose against the warmup-end orientation. Matches UR10RobotEnv.step.
                target_yaw_offset = float(np.clip(
                    target_yaw_offset + dyaw * yaw_step, -np.pi, np.pi,
                ))
                min_yaw_offset = min(min_yaw_offset, target_yaw_offset)
                max_yaw_offset = max(max_yaw_offset, target_yaw_offset)
                R_target = R_home_warm * Rotation.from_euler("z", target_yaw_offset)
                rx, ry, rz = (float(v) for v in R_target.as_rotvec())
            else:
                rx, ry, rz = fixed_rx, fixed_ry, fixed_rz

            target_pose = [
                float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), rx, ry, rz,
            ]
            # Non-blocking: hand the new target to the streaming thread.
            robot.set_target_pose(target_pose)

            if grip == 0:
                robot.send_gripper(0)
            elif grip == 2:
                robot.send_gripper(2)

            joint_pos = robot.get_joint_positions()
            joint_vel = robot.get_joint_velocities()
            tcp_full = robot.get_current_tcp()

            elapsed = time.perf_counter() - start_t

            if elapsed < args.warmup_time_s:
                pass  # warmup
            else:
                if not warmup_done:
                    print()
                    print("=" * 50)
                    print("  RECORDING STARTED")
                    print("  Drive to ALL workspace extremes and exercise the gripper.")
                    print("  Press Ctrl+C to stop and see results.")
                    print("=" * 50)
                    print()

                    # Pin the wrist orientation to whatever the user posed during warmup.
                    fixed_rx = float(tcp_full[3])
                    fixed_ry = float(tcp_full[4])
                    fixed_rz = float(tcp_full[5])
                    # Cache the home rotation so per-step yaw composition mirrors the
                    # env's `_R_home` exactly. Only consulted when --use_yaw.
                    R_home_warm = Rotation.from_rotvec([fixed_rx, fixed_ry, fixed_rz])

                    # Anchor the position-only relative-observation baseline at warmup
                    # end. This is the analog of `UR10Robot.capture_baselines()` in the
                    # env's reset path. Force is left absolute — see env's module
                    # docstring for the rationale.
                    initial_tcp_xyz = tcp_full[:3].copy()

                    min_joint_pos = joint_pos.copy()
                    max_joint_pos = joint_pos.copy()
                    min_joint_vel = joint_vel.copy()
                    max_joint_vel = joint_vel.copy()
                    min_tcp = tcp_full.copy()
                    max_tcp = tcp_full.copy()
                    min_tcp_xyz_rel = (tcp_full[:3] - initial_tcp_xyz).copy()
                    max_tcp_xyz_rel = (tcp_full[:3] - initial_tcp_xyz).copy()
                    warmup_done = True

                tcp_xyz_rel = tcp_full[:3] - initial_tcp_xyz

                min_joint_pos = np.minimum(min_joint_pos, joint_pos)
                max_joint_pos = np.maximum(max_joint_pos, joint_pos)
                min_joint_vel = np.minimum(min_joint_vel, joint_vel)
                max_joint_vel = np.maximum(max_joint_vel, joint_vel)
                min_tcp = np.minimum(min_tcp, tcp_full)
                max_tcp = np.maximum(max_tcp, tcp_full)
                min_tcp_xyz_rel = np.minimum(min_tcp_xyz_rel, tcp_xyz_rel)
                max_tcp_xyz_rel = np.maximum(max_tcp_xyz_rel, tcp_xyz_rel)

                recording_time = elapsed - args.warmup_time_s
                remaining = args.teleop_time_s - recording_time
                print(f"  Recording ... {remaining:.1f}s remaining", end="\r")

                if recording_time >= args.teleop_time_s:
                    print("\nTime limit reached.")
                    break

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nInterrupted. Stopping safely ...")

    finally:
        robot.disconnect()
        teleop.disconnect()

    # -- print results -------------------------------------------------------
    if min_joint_pos is None:
        print("No data recorded (exited during warmup).")
        return

    r = 4

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n# Joint Position Limits (radians) — 6 values [absolute]")
    print(f"min_joint_pos = {np.round(min_joint_pos, r).tolist()}")
    print(f"max_joint_pos = {np.round(max_joint_pos, r).tolist()}")

    print("\n# Joint Velocity Limits (rad/s) — 6 values [absolute]")
    print(f"min_joint_vel = {np.round(min_joint_vel, r).tolist()}")
    print(f"max_joint_vel = {np.round(max_joint_vel, r).tolist()}")

    print("\n# TCP Limits [x, y, z, rx, ry, rz]  [ABSOLUTE — for env workspace bounds]")
    print(f"min_tcp = {np.round(min_tcp, r).tolist()}")
    print(f"max_tcp = {np.round(max_tcp, r).tolist()}")

    print("\n# Wrist orientation (axis-angle) captured at end of warmup")
    print(f"fixed_rx, fixed_ry, fixed_rz = {fixed_rx:.6f}, {fixed_ry:.6f}, {fixed_rz:.6f}")

    print("\n# Per-episode position baseline (informational; the env captures a fresh one")
    print("# each reset — this is just what this find-limits run anchored to)")
    print(f"initial_tcp_xyz = {np.round(initial_tcp_xyz, r).tolist()}")

    # Compose the full 6-DoF home pose [x, y, z, rx, ry, rz] from the warmup-end anchor.
    # The lerobot config schema's `processor.reset.fixed_reset_joint_positions` key is a
    # misnomer — it actually carries the home TCP pose (RC10 inherited this naming, UR10
    # follows). Pasting this list under that key makes env.reset() drive the arm to the
    # exact pose used as the relative-observation baseline, so policy and recorded demos
    # stay frame-consistent.
    home_tcp_pose = [
        float(initial_tcp_xyz[0]),
        float(initial_tcp_xyz[1]),
        float(initial_tcp_xyz[2]),
        float(fixed_rx),
        float(fixed_ry),
        float(fixed_rz),
    ]
    print("\n# Home TCP pose [x, y, z, rx, ry, rz] — paste into ur10_env_*.json AND ur10_train_*.json")
    print("# under processor.reset.fixed_reset_joint_positions")
    print("# (the key is a misnomer — it carries a 6-DoF TCP pose, not joint positions)")
    print(f"fixed_reset_joint_positions = {[round(v, r) for v in home_tcp_pose]}")

    print("\n# End-Effector XYZ Bounds — paste into ur10_env_*.json [ABSOLUTE base-frame]")
    print("# under processor.inverse_kinematics.end_effector_bounds")
    print(f"ee_bounds_min = {np.round(min_tcp[:3], r).tolist()}")
    print(f"ee_bounds_max = {np.round(max_tcp[:3], r).tolist()}")

    if args.use_yaw:
        # Yaw OFFSET range observed during recording (radians, relative to fixed_rz).
        # Paste as the 4th element of end_effector_bounds.min/max in the JSON config —
        # the env reads bounds[3] when `use_yaw=true` (see UR10RobotEnv.__init__).
        print("\n# Yaw OFFSET bounds (radians, relative to fixed_rz) — paste as the 4th element")
        print("# of processor.inverse_kinematics.end_effector_bounds.min and .max")
        print("# Also set processor.inverse_kinematics.use_yaw = true and add 'yaw' to")
        print("# processor.inverse_kinematics.end_effector_step_sizes (e.g. 0.01).")
        print(f"yaw_offset_min = {round(min_yaw_offset, r)}")
        print(f"yaw_offset_max = {round(max_yaw_offset, r)}")
        print("# Example merged bounds row:")
        bounds_min_with_yaw = np.round(min_tcp[:3], r).tolist() + [round(min_yaw_offset, r)]
        bounds_max_with_yaw = np.round(max_tcp[:3], r).tolist() + [round(max_yaw_offset, r)]
        print(f'#   "min": {bounds_min_with_yaw}')
        print(f'#   "max": {bounds_max_with_yaw}')

    # Dataset_stats for observation.state. 16-D in no-yaw mode, 17-D in yaw mode:
    #   no-yaw: [joint_pos(6), joint_vel(6), tcp_xyz(3) RELATIVE, gripper(1)]
    #   yaw   : [joint_pos(6), joint_vel(6), tcp_xyz(3) RELATIVE, yaw_offset(1), gripper(1)]
    # Matches `UR10RobotEnv._augment_observation()` — gripper is always at index -1.
    gripper_min = 0.0
    gripper_max = 1.0
    if args.use_yaw:
        obs_min = np.concatenate([
            min_joint_pos, min_joint_vel, min_tcp_xyz_rel,
            [min_yaw_offset], [gripper_min],
        ])
        obs_max = np.concatenate([
            max_joint_pos, max_joint_vel, max_tcp_xyz_rel,
            [max_yaw_offset], [gripper_max],
        ])
        banner = "17D  [tcp_xyz RELATIVE, yaw_offset RELATIVE]"
        order = "[6 joint_pos, 6 joint_vel, 3 tcp_xyz_rel, 1 yaw_offset, 1 gripper]"
    else:
        obs_min = np.concatenate([
            min_joint_pos, min_joint_vel, min_tcp_xyz_rel, [gripper_min],
        ])
        obs_max = np.concatenate([
            max_joint_pos, max_joint_vel, max_tcp_xyz_rel, [gripper_max],
        ])
        banner = "16D  [tcp_xyz RELATIVE]"
        order = "[6 joint_pos, 6 joint_vel, 3 tcp_xyz_rel, 1 gripper]"

    print(f"\n# dataset_stats for observation.state ({banner})")
    print(f"# Order: {order}")
    print(f'"min": {np.round(obs_min, r).tolist()}')
    print(f'"max": {np.round(obs_max, r).tolist()}')

    print()
    print("=== Paste guide ===")
    print("  processor.reset.fixed_reset_joint_positions  ← `fixed_reset_joint_positions` above")
    print("  processor.inverse_kinematics.end_effector_bounds  ← `ee_bounds_min/max` above")
    print("  policy.dataset_stats.\"observation.state\"  ← the `min`/`max` block above")
    print("Update both ur10_env_*.json AND ur10_train_*.json so they agree.")


if __name__ == "__main__":
    main()
