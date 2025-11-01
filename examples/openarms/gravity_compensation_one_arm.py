import time
import numpy as np
import pinocchio as pin
from os.path import join, dirname, exists, expanduser

from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig


def main() -> None:
    config = OpenArmsFollowerConfig(
        port_right="can0",
        port_left="can1",
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=5.0,
    )
    
    
    print("Initializing robot...")
    follower = OpenArmsFollower(config)
    follower.connect(calibrate=True)
    
    # Load URDF for Pinocchio dynamics
    urdf_path = "/home/croissant/Documents/openarm_description/openarm_bimanual_pybullet.urdf"
    
    pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, dirname(urdf_path))
    pin_robot.data = pin_robot.model.createData()
    print(f"✓ Loaded Pinocchio model with {pin_robot.nq} DoFs")
    
    follower.pin_robot = pin_robot
    
    control_arm = "right"
    
    print(f"Applying gravity compensation to: {control_arm.upper()} ARM (all joints)")
    print("\nAll joints on the right arm will have gravity compensation applied.")
    print("Left arm joints will be disabled (free to move).")
    print("\nIMPORTANT:")
    print("  1. Support the arm before starting")
    print("  2. The arm will be held in place by gravity compensation")
    print("  3. You should be able to move it with gentle force")
    print("\nPress ENTER when ready to start...")
    input()
    
    follower.bus_left.enable_torque()
    time.sleep(0.1)
    
    print(f"✓ Motors enabled")
    print(f"   {control_arm.upper()} arm: Gravity compensation active")
    print(f"   LEFT arm: Zero torque (free to move)")
    
    print("\nStarting gravity compensation loop...")
    print("Press Ctrl+C to stop\n")
    
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
            
            # Apply gravity compensation torques to all joints on the right arm
            for motor in follower.bus_right.motors:
                full_name = f"right_{motor}"
                position = positions_deg.get(full_name, 0.0)
                
                # Apply gravity compensation to this joint
                torque = torques_nm.get(full_name, 0.0)
                
                # Send MIT control command with gravity compensation torque
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
                    
                    # Display status for all joints on the controlled arm
                    print(f"\n[RIGHT ARM] Loop: {current_hz:.1f} Hz ({avg_time*1000:.1f} ms)")
                    
                    # Show each joint's position and torque
                    for motor in follower.bus_right.motors:
                        full_name = f"right_{motor}"
                        pos = positions_deg.get(full_name, 0.0)
                        torque = torques_nm.get(full_name, 0.0)
                        print(f"  {motor:8s}: Pos={pos:7.1f}° | Torque={torque:7.3f} N·m")
                    
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

