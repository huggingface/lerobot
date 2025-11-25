import time
import numpy as np
import pinocchio as pin
from os.path import join, dirname, exists, expanduser

from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig


def main() -> None:
    config = OpenArmsFollowerConfig(
        port_left="can0",
        port_right="can1",
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
    
    print(f"Applying gravity compensation")
    print("  1. Support the arm before starting")
    print("  2. The arm will be held in place by gravity compensation")
    print("  3. You should be able to move it with gentle force")
    print("\nPress ENTER when ready to start...")
    input()
    
    print(f"✓ Motors enabled")
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
            
            # Apply gravity compensation to right arm (all joints except gripper)
            for motor in follower.bus_right.motors:
                if motor == "gripper":
                    continue  # Skip gripper
                    
                full_name = f"right_{motor}"
                position = positions_deg.get(full_name, 0.0)
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

            # Apply gravity compensation to left arm (all joints except gripper)
            for motor in follower.bus_left.motors:
                if motor == "gripper":
                    continue  # Skip gripper
                    
                full_name = f"left_{motor}"
                position = positions_deg.get(full_name, 0.0)
                torque = torques_nm.get(full_name, 0.0)
                
                # Send MIT control command with gravity compensation torque
                follower.bus_left._mit_control(
                    motor=motor,
                    kp=0.0,  # No position control
                    kd=0.0,  # No velocity damping
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=torque
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
                    
                    print(f"{current_hz:.1f} Hz ({avg_time*1000:.1f} ms)")

                    loop_times = []
                    last_print_time = loop_end
            
            time.sleep(0.005)
            
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

