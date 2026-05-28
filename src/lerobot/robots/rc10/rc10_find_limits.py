#!/usr/bin/env python

"""
Find joint limits, joint velocity ranges, and TCP bounds for the RC10 robot
via gamepad teleoperation

Usage:
    python -m lerobot.robots.rc10.rc10_find_limits \
        --ip 10.10.10.10 \
        --gripper_port /dev/ttyUSB0 \
        --teleop_time_s 60 \
        --warmup_time_s 5 \
        --fps 10 \
        --ee_step_x 0.01 \
        --ee_step_y 0.01 \
        --ee_step_z 0.01
"""

import argparse
import time

import numpy as np

from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(description="Find RC10 joint/TCP limits via gamepad teleop")
    parser.add_argument("--ip", type=str, default="10.10.10.10")
    parser.add_argument("--rate_hz", type=int, default=100)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--acceleration", type=float, default=1.0)
    parser.add_argument("--threshold_position", type=float, default=0.001)
    parser.add_argument("--threshold_angle", type=float, default=1.0)
    parser.add_argument("--gripper_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--gripper_baudrate", type=int, default=115200)
    parser.add_argument("--teleop_time_s", type=float, default=60.0)
    parser.add_argument("--warmup_time_s", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--ee_step_x", type=float, default=0.01)
    parser.add_argument("--ee_step_y", type=float, default=0.01)
    parser.add_argument("--ee_step_z", type=float, default=0.01)
    args = parser.parse_args()

    # -- imports that need hardware ------------------------------------------
    from rc10_api.controller import TaskSpaceJogController
    from rc10_api.gripper import Gripper

    from lerobot.teleoperators.gamepad import GamepadTeleop
    from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig

    # -- connect robot -------------------------------------------------------
    print(f"Connecting to RC10 at {args.ip} ...")
    controller = TaskSpaceJogController(
        ip=args.ip,
        rate_hz=args.rate_hz,
        velocity=args.velocity,
        acceleration=args.acceleration,
        treshold_position=args.threshold_position,
        treshold_angel=args.threshold_angle,
    )
    controller.start()

    gripper = Gripper(device=args.gripper_port, baudrate=args.gripper_baudrate)

    # -- connect gamepad -----------------------------------------------------
    print("Connecting gamepad ...")
    teleop = GamepadTeleop(GamepadTeleopConfig(use_gripper=True))
    teleop.connect()

    # -- tracking arrays -----------------------------------------------------
    ee_step = np.array([args.ee_step_x, args.ee_step_y, args.ee_step_z])
    dt = 1.0 / args.fps

    min_joint_pos = None
    max_joint_pos = None
    min_joint_vel = None
    max_joint_vel = None
    min_tcp = None
    max_tcp = None

    start_t = time.perf_counter()
    warmup_done = False

    print()
    print("=" * 50)
    print(f"  WARMUP PHASE ({args.warmup_time_s}s)")
    print("  Move the robot freely. Data is NOT recorded yet.")
    print("=" * 50)
    print()

    try:
        while True:
            t0 = time.perf_counter()

            # -- teleop control ----------------------------------------------
            teleop_action = teleop.get_action()
            dx = teleop_action.get("delta_x", 0.0)
            dy = teleop_action.get("delta_y", 0.0)
            dz = teleop_action.get("delta_z", 0.0)
            grip = teleop_action.get("gripper", 1)  # 0=close, 1=stay, 2=open

            tcp = controller.get_current_tcp()  # [x, y, z, roll, pitch, yaw]
            current_xyz = tcp[:3]
            delta_xyz = np.array([dx, dy, dz]) * ee_step
            new_xyz = current_xyz + delta_xyz

            controller.set_target(
                float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]),
                float(tcp[3]), float(tcp[4]), float(tcp[5]),
            )

            if grip == 0:
                gripper.send(-1)
            elif grip == 2:
                gripper.send(1)

            # -- read state --------------------------------------------------
            joint_pos = controller.get_current_joint()       # (6,)
            joint_vel = controller.get_current_joint_vel()    # (6,)
            joint_torques = controller.get_current_torque()     # (6,)
            tcp_full = controller.get_current_tcp()           # (6,)

            # -- phase logic -------------------------------------------------
            elapsed = time.perf_counter() - start_t

            if elapsed < args.warmup_time_s:
                pass  # warmup
            else:
                if not warmup_done:
                    print()
                    print("=" * 50)
                    print("  RECORDING STARTED")
                    print("  Move the robot to ALL workspace extremes.")
                    print("  Open and close the gripper.")
                    print("  Press Ctrl+C to stop and see results.")
                    print("=" * 50)
                    print()

                    min_joint_pos = joint_pos.copy()
                    max_joint_pos = joint_pos.copy()
                    min_joint_vel = joint_vel.copy()
                    max_joint_vel = joint_vel.copy()
                    min_joint_torques = joint_torques.copy()
                    max_joint_torques = joint_torques.copy()
                    min_tcp = tcp_full.copy()
                    max_tcp = tcp_full.copy()
                    warmup_done = True

                # Update limits
                min_joint_pos = np.minimum(min_joint_pos, joint_pos)
                max_joint_pos = np.maximum(max_joint_pos, joint_pos)
                min_joint_vel = np.minimum(min_joint_vel, joint_vel)
                max_joint_vel = np.maximum(max_joint_vel, joint_vel)
                min_joint_torques = np.minimum(min_joint_torques, joint_torques)
                max_joint_torques = np.maximum(max_joint_torques, joint_torques)
                min_tcp = np.minimum(min_tcp, tcp_full)
                max_tcp = np.maximum(max_tcp, tcp_full)

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
        controller.stop()
        gripper.close()
        teleop.disconnect()

    # -- print results -------------------------------------------------------
    if min_joint_pos is None:
        print("No data recorded (exited during warmup).")
        return

    r = 4  # rounding precision

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n# Joint Position Limits (radians) — 6 values")
    print(f"min_joint_pos = {np.round(min_joint_pos, r).tolist()}")
    print(f"max_joint_pos = {np.round(max_joint_pos, r).tolist()}")

    print("\n# Joint Velocity Limits (rad/s) — 6 values")
    print(f"min_joint_vel = {np.round(min_joint_vel, r).tolist()}")
    print(f"max_joint_vel = {np.round(max_joint_vel, r).tolist()}")

    print("\n# Joint Torque Limits (N-m) — 6 values")
    print(f"min_joint_torques = {np.round(min_joint_torques, r).tolist()}")
    print(f"max_joint_torques = {np.round(max_joint_torques, r).tolist()}")

    print("\n# TCP Limits [x, y, z, roll, pitch, yaw]")
    print(f"min_tcp = {np.round(min_tcp, r).tolist()}")
    print(f"max_tcp = {np.round(max_tcp, r).tolist()}")

    print("\n# End-Effector XYZ Bounds (for ee_bounds in config)")
    print(f"ee_bounds_min = {np.round(min_tcp[:3], r).tolist()}")
    print(f"ee_bounds_max = {np.round(max_tcp[:3], r).tolist()}")

    # -- dataset_stats format ------------------------------------------------
    gripper_min = 0.0
    gripper_max = 1.0

    obs_min = np.concatenate([min_joint_pos, min_joint_vel, min_joint_torques, [gripper_min], min_tcp[:3]])
    obs_max = np.concatenate([max_joint_pos, max_joint_vel, max_joint_torques, [gripper_max], max_tcp[:3]])

    print("\n# dataset_stats for observation.state (22D)")
    print("# Order: [6 joint_pos, 6 joint_vel, 6 joint_torques, 1 gripper, 3 tcp_xyz]")
    print(f'"min": {np.round(obs_min, r).tolist()}')
    print(f'"max": {np.round(obs_max, r).tolist()}')

    print()
    print("Paste the ee_bounds into your rc10_env.json / rc10_train.json")
    print("under processor.inverse_kinematics.end_effector_bounds.")
    print("Paste the dataset_stats into rc10_train.json under")
    print("policy.dataset_stats.\"observation.state\".")


if __name__ == "__main__":
    main()
