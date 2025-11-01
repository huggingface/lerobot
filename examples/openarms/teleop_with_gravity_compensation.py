"""
OpenArms Teleoperation with Gravity Compensation

Leader arms (both LEFT and RIGHT): Gravity compensation (weightless, easy to move)
Follower arms (both LEFT and RIGHT): Mirror leader movements

Uses the URDF file from the lerobot repository.
"""

import time

import numpy as np

from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader


def main():
    """Main teleoperation loop with gravity compensation"""

    print("=" * 70)
    print("OpenArms Teleoperation with Gravity Compensation")
    print("=" * 70)

    # Configuration
    follower_config = OpenArmsFollowerConfig(
        port_left="can0",
        port_right="can1",
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=10.0,
    )

    leader_config = OpenArmsLeaderConfig(
        port_left="can2",
        port_right="can3",
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

    # URDF is automatically loaded in the leader constructor
    if leader.pin_robot is None:
        raise RuntimeError("URDF model not loaded on leader. Gravity compensation not available.")

    print("\nLeader BOTH arms: G-comp | Follower BOTH arms: Teleop")
    print("Press ENTER to start...")
    input()

    # Enable motors on both leader arms for gravity compensation
    leader.bus_right.enable_torque()
    leader.bus_left.enable_torque()
    time.sleep(0.1)

    print("Press Ctrl+C to stop\n")

    # Main control loop
    loop_times = []
    last_print_time = time.perf_counter()

    # All joints (both arms)
    all_joints = []
    for motor in leader.bus_right.motors:
        all_joints.append(f"right_{motor}")
    for motor in leader.bus_left.motors:
        all_joints.append(f"left_{motor}")

    try:
        while True:
            loop_start = time.perf_counter()

            # Get leader state
            leader_action = leader.get_action()

            # Extract positions in degrees
            leader_positions_deg = {}
            for motor in leader.bus_right.motors:
                key = f"right_{motor}.pos"
                if key in leader_action:
                    leader_positions_deg[f"right_{motor}"] = leader_action[key]

            for motor in leader.bus_left.motors:
                key = f"left_{motor}.pos"
                if key in leader_action:
                    leader_positions_deg[f"left_{motor}"] = leader_action[key]

            # Calculate gravity torques for leader using built-in method
            leader_positions_rad = {k: np.deg2rad(v) for k, v in leader_positions_deg.items()}
            leader_torques_nm = leader._gravity_from_q(leader_positions_rad)

            # Apply gravity compensation to leader RIGHT arm (all joints except gripper)
            for motor in leader.bus_right.motors:
                if motor == "gripper":
                    # Skip gripper - keep it free
                    full_name = f"right_{motor}"
                    position = leader_positions_deg.get(full_name, 0.0)
                    leader.bus_right._mit_control(
                        motor=motor,
                        kp=0.0,
                        kd=0.0,
                        position_degrees=position,
                        velocity_deg_per_sec=0.0,
                        torque=0.0,
                    )
                    continue
                
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

            # Apply gravity compensation to leader LEFT arm (all joints except gripper)
            for motor in leader.bus_left.motors:
                if motor == "gripper":
                    # Skip gripper - keep it free
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
                    continue
                
                full_name = f"left_{motor}"
                position = leader_positions_deg.get(full_name, 0.0)
                torque = leader_torques_nm.get(full_name, 0.0)

                leader.bus_left._mit_control(
                    motor=motor,
                    kp=0.0,
                    kd=0.0,
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=torque,
                )

            # Send leader positions to follower (both arms)
            follower_action = {}
            for joint in all_joints:
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

                    print(f"{current_hz:.1f} Hz ({avg_time*1000:.1f} ms)")

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
