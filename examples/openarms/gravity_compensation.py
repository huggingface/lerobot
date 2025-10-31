"""
OpenArms Gravity Compensation Demo

This script demonstrates gravity compensation on a real OpenArms follower robot.
It uses Pinocchio to calculate gravity torques and applies them via MIT control mode.

Starting with one joint (joint_2 - shoulder lift) to test the implementation.

Controls:
- Press ENTER to start gravity compensation
- Press Ctrl+C to stop
"""

import time
import numpy as np
import pinocchio as pin
from os.path import join, dirname, exists

from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig


def main() -> None:
    """Main entry point for gravity compensation demo"""
    
    # ===== Configuration =====
    print("=" * 70)
    print("OpenArms Gravity Compensation Demo")
    print("=" * 70)
    
    # Configure follower (right arm only for now)
    config = OpenArmsFollowerConfig(
        port_right="can0",
        port_left="can1",
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        # Safety: No max_relative_target since we're doing torque control
        max_relative_target=None,
    )
    
    
    print("Initializing robot...")
    follower = OpenArmsFollower(config)
    follower.connect(calibrate=True)
    urdf_path = "/Users/michel_aractingi/code/openarm_description/openarm_bimanual_pybullet.urdf",

    pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
    pin_robot.data = pin_robot.model.createData()
    print(f"✓ Loaded Pinocchio model with {pin_robot.nq} DoFs")
    
    # Set the Pinocchio model on the follower
    follower.pin_robot = pin_robot
    
    # Start with joint_2 (shoulder lift)
    test_joint = "joint_2"
    test_arm = "right"
    test_joint_full = f"{test_arm}_{test_joint}"
    
    print(f"Testing gravity compensation on: {test_joint_full.upper()}")
    print("\nThis joint will have gravity compensation applied.")
    print("All other joints will be disabled (free to move).")
    print("\nIMPORTANT:")
    print("  1. Support the arm before starting")
    print("  2. The arm will be free to move when you release it")
    print("  3. Gravity compensation should keep the joint stable")
    print("\nPress ENTER when ready to start...")
    input()
    
    # ===== Enable all motors (we'll control which ones get torque commands) =====
    print(f"\nEnabling motors...")
    
    # Enable all motors on both arms
    follower.bus_right.enable_torque()
    follower.bus_left.enable_torque()
    time.sleep(0.1)
    
    print(f"✓ Motors enabled")
    print(f"   Only {test_joint_full} will receive torque commands")
    print(f"   Other joints will have zero torque (kp=0, kd=0, torque=0)")
    
    print("\nStarting gravity compensation loop...")
    print("Press Ctrl+C to stop\n")
    
    # ===== Main Control Loop =====
    loop_times = []
    last_print_time = time.perf_counter()
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Get current joint positions from robot
            obs = follower.get_observation()
            
            # Extract positions in degrees
            positions_deg = {}
            for motor in follower.bus_right.motors:
                key = f"right_{motor}.pos"
                if key in obs:
                    positions_deg[f"right_{motor}"] = obs[key]
            
            for motor in follower.bus_left.motors:
                key = f"left_{motor}.pos"
                if key in obs:
                    positions_deg[f"left_{motor}"] = obs[key]
            
            # Convert to radians and calculate gravity torques
            # Use the built-in method from OpenArmsFollower
            positions_rad = {k: np.deg2rad(v) for k, v in positions_deg.items()}
            torques_nm = follower._gravity_from_q(positions_rad)
            
            # Apply gravity compensation torque to test joint ONLY
            # Send zero torque to all other joints to keep them free-moving
            
            for motor in follower.bus_right.motors:
                full_name = f"right_{motor}"
                position = positions_deg.get(full_name, 0.0)
                
                if motor == test_joint:
                    # Apply gravity compensation to test joint
                    torque = torques_nm.get(test_joint_full, 0.0) * 0.0
                else:
                    # Zero torque for all other joints (free to move)
                    torque = 0.0
                
                # Send MIT control command
                follower.bus_right._mit_control(
                    motor=motor,
                    kp=0.0,  # No position control
                    kd=0.0,  # No velocity damping
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=torque
                )
            
            # Also ensure left arm joints have zero torque
            for motor in follower.bus_left.motors:
                full_name = f"left_{motor}"
                position = positions_deg.get(full_name, 0.0)
                
                follower.bus_left._mit_control(
                    motor=motor,
                    kp=0.0,
                    kd=0.0,
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=0.0
                )
            
            # Measure loop time
            loop_end = time.perf_counter()
            loop_time = loop_end - loop_start
            loop_times.append(loop_time)
            
            # Print status every 2 seconds
            if loop_end - last_print_time >= 2.0:
                if loop_times:
                    avg_time = sum(loop_times) / len(loop_times)
                    current_hz = 1.0 / avg_time if avg_time > 0 else 0
                    
                    # Get test joint values for display
                    test_position = positions_deg.get(test_joint_full, 0.0)
                    test_torque = torques_nm.get(test_joint_full, 0.0)
                    
                    print(f"[{test_joint_full}] "
                          f"Pos: {test_position:6.1f}° | "
                          f"Torque: {test_torque:6.3f} N·m | "
                          f"Loop: {current_hz:.1f} Hz ({avg_time*1000:.1f} ms)")
                    
                    # Reset for next measurement window
                    loop_times = []
                    last_print_time = loop_end
            
            # Small sleep to avoid overwhelming the CAN bus
            time.sleep(0.01)  # 100 Hz
            
    except KeyboardInterrupt:
        print("\n\nStopping gravity compensation...")
    
    finally:
        print("\nDisabling all motors and disconnecting...")
        follower.bus_right.disable_torque()
        follower.bus_left.disable_torque()
        time.sleep(0.1)
        follower.disconnect()
        print("✓ Safe shutdown complete")


if __name__ == "__main__":
    main()

