"""
OpenArms Teleoperation with Gravity Compensation

Leader RIGHT arm: Gravity compensation (weightless, easy to move)
Follower RIGHT arm: Mirrors leader movements
Both LEFT arms: Free to move (disabled)

The urdf file tested with this script is found in:
https://github.com/michel-aractingi/openarm_description/blob/main/openarm_bimanual_pybullet.urdf
"""

import time
import os

import numpy as np
import pinocchio as pin

from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader

# Path to the URDF file
URDF_PATH = "/home/croissant/Documents/openarm_description/openarm_bimanual_pybullet.urdf"

def compute_gravity_torques(leader, positions_rad):
    """
    Compute gravity torques for all joints in the robot.

    Args:
        leader: OpenArmsLeader instance with pin_robot set
        positions_rad: Dictionary mapping motor names (with arm prefix) to positions in radians

    Returns:
        Dictionary mapping motor names to gravity torques in N·m
    """
    if not hasattr(leader, "pin_robot") or leader.pin_robot is None:
        raise RuntimeError("URDF model not loaded on leader")

    # Build position vector in the order of motors (right arm, then left arm)
    q = np.zeros(leader.pin_robot.model.nq)
    idx = 0

    # Right arm motors
    for motor_name in leader.bus_right.motors:
        full_name = f"right_{motor_name}"
        q[idx] = positions_rad.get(full_name, 0.0)
        idx += 1

    # Left arm motors
    for motor_name in leader.bus_left.motors:
        full_name = f"left_{motor_name}"
        q[idx] = positions_rad.get(full_name, 0.0)
        idx += 1

    # Compute generalized gravity vector
    g = pin.computeGeneralizedGravity(leader.pin_robot.model, leader.pin_robot.data, q)

    # Map back to motor names
    result = {}
    idx = 0
    for motor_name in leader.bus_right.motors:
        result[f"right_{motor_name}"] = float(g[idx])
        idx += 1
    for motor_name in leader.bus_left.motors:
        result[f"left_{motor_name}"] = float(g[idx])
        idx += 1

    return result


def main():
    """Main teleoperation loop with gravity compensation"""

    print("=" * 70)
    print("OpenArms Teleoperation with Gravity Compensation")
    print("=" * 70)

    # Configuration
    follower_config = OpenArmsFollowerConfig(
        port_right="can0",
        port_left="can1",
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=10.0,
    )

    leader_config = OpenArmsLeaderConfig(
        port_right="can2",
        port_left="can3",
        can_interface="socketcan",
        id="openarms_leader",
        manual_control=False,  # Enable torque control for gravity compensation
    )

    # Initialize and connect
    print("\nInitializing devices...")
    follower = OpenArmsFollower(follower_config)
    leader = OpenArmsLeader(leader_config)

    follower.connect(calibrate=True)
    leader.connect(calibrate=True)

    # Load URDF for gravity compensation
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF file not found at {URDF_PATH}")
    pin_robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, os.path.dirname(URDF_PATH))
    pin_robot.data = pin_robot.model.createData()
    leader.pin_robot = pin_robot

    print("\nLeader RIGHT: G-comp | Follower RIGHT: Teleop")
    print("Press ENTER to start...")
    input()

    # Enable motors
    leader.bus_right.enable_torque()
    leader.bus_left.enable_torque()
    time.sleep(0.1)

    print("Press Ctrl+C to stop\n")

    # Main control loop
    loop_times = []
    last_print_time = time.perf_counter()

    # Right arm joints only
    right_joints = [
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_joint_6",
        "right_joint_7",
        "right_gripper",
    ]

    try:
        while True:
            loop_start = time.perf_counter()

            # Get leader state
            leader_action = leader.get_action()

            leader_positions_deg = {}
            for motor in leader.bus_right.motors:
                key = f"right_{motor}.pos"
                if key in leader_action:
                    leader_positions_deg[f"right_{motor}"] = leader_action[key]

            for motor in leader.bus_left.motors:
                key = f"left_{motor}.pos"
                if key in leader_action:
                    leader_positions_deg[f"left_{motor}"] = leader_action[key]

            # Calculate gravity torques for leader
            leader_positions_rad = {k: np.deg2rad(v) for k, v in leader_positions_deg.items()}
            leader_torques_nm = compute_gravity_torques(leader, leader_positions_rad)

            # Apply gravity compensation to leader right arm
            for motor in leader.bus_right.motors:
                full_name = f"right_{motor}"
                position = leader_positions_deg.get(full_name, 0.0)
                torque = leader_torques_nm.get(full_name, 0.0)

                leader.bus_right._mit_control(
                    motor=motor,
                    kp=0.0,
                    kd=0.0,
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=torque,
                )

            # Keep leader left arm free
            for motor in leader.bus_left.motors:
                full_name = f"left_{motor}"
                position = leader_positions_deg.get(full_name, 0.0)

                leader.bus_left._mit_control(
                    motor=motor,
                    kp=0.0,
                    kd=0.0,
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=0.0,
                )

            # Send leader positions to follower right arm
            follower_action = {}
            for joint in right_joints:
                pos_key = f"{joint}.pos"
                if pos_key in leader_action:
                    follower_action[pos_key] = leader_action[pos_key]

            if follower_action:
                follower.send_action(follower_action)

            # Performance monitoring
            loop_end = time.perf_counter()
            loop_time = loop_end - loop_start
            loop_times.append(loop_time)

            if loop_end - last_print_time >= 2.0:
                if loop_times:
                    avg_time = sum(loop_times) / len(loop_times)
                    current_hz = 1.0 / avg_time if avg_time > 0 else 0
                    sample_pos = leader_positions_deg.get("right_joint_2", 0.0)
                    sample_torque = leader_torques_nm.get("right_joint_2", 0.0)

                    print(f"[{current_hz:.1f} Hz] J2: {sample_pos:5.1f}° | Torque: {sample_torque:5.2f} N·m")

                    loop_times = []
                    last_print_time = loop_end

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        try:
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
            leader.disconnect()
            follower.disconnect()
            print("✓ Shutdown complete")
        except Exception as e:
            print(f"Shutdown error: {e}")


if __name__ == "__main__":
    main()
