import time
import numpy as np

from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig


# Friction model parameters from OpenArms config/follower.yaml
# τ_fric(ω) = Fo + Fv·ω + Fc·tanh(k·ω)
# For 8 motors: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
FRICTION_PARAMS = {
    "Fc": [0.306, 0.306, 0.40, 0.166, 0.050, 0.093, 0.172, 0.0512],  # Coulomb friction [Nm]
    "k": [28.417, 28.417, 29.065, 130.038, 151.771, 242.287, 7.888, 4.000],  # tanh steepness
    "Fv": [0.063, 0.0630, 0.604, 0.813, 0.029, 0.072, 0.084, 0.084],  # Viscous friction [Nm·s/rad]
    "Fo": [0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059, -0.050],  # Offset torque [Nm]
}

# Constants from OpenArms C++ implementation
AMP_TMP = 1.0
COEF_TMP = 0.1

FRICTION_SCALE = 1.0  # OpenArms C++ uses 0.3 factor in unilateral mode
DAMPING_KD = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]  # Damping gains for stability

def compute_friction_torque(velocity_rad_per_sec: float, motor_index: int) -> float:
    """
    Compute friction torque for a single motor using the tanh friction model.
    
    Args:
        velocity_rad_per_sec: Angular velocity in rad/s
        motor_index: Index of the motor (0-7)
        
    Returns:
        Friction torque in N·m (scaled for stability)
    """
    
    Fc = FRICTION_PARAMS["Fc"][motor_index]
    k = FRICTION_PARAMS["k"][motor_index]
    Fv = FRICTION_PARAMS["Fv"][motor_index]
    Fo = FRICTION_PARAMS["Fo"][motor_index]
    
    # Friction model: τ_fric = amp * Fc * tanh(coef * k * ω) + Fv * ω + Fo
    friction_torque = (
        AMP_TMP * Fc * np.tanh(COEF_TMP * k * velocity_rad_per_sec) +
        Fv * velocity_rad_per_sec +
        Fo
    )
    
    # Scale down friction compensation for stability at lower control rates
    # (OpenArms C++ uses 0.3 factor in unilateral mode)!!
    friction_torque *= FRICTION_SCALE
    
    return friction_torque


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
    
    print(f"Applying friction compensation")
    print("  1. Support the arm before starting")
    print("  2. The arm will be held in place by friction compensation")
    print("  3. You should be able to move it with gentle force")
    print("\nPress ENTER when ready to start...")
    input()
    
    print(f"✓ Motors enabled")
    print("\nStarting friction compensation loop...")
    print("Press Ctrl+C to stop\n")
    
    loop_times = []
    last_print_time = time.perf_counter()
    
    # Motor name to index mapping
    motor_name_to_index = {
        "joint_1": 0,
        "joint_2": 1,
        "joint_3": 2,
        "joint_4": 3,
        "joint_5": 4,
        "joint_6": 5,
        "joint_7": 6,
        "gripper": 7,
    }
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Get current joint positions and velocities from robot
            obs = follower.get_observation()
            
            # Extract velocities in degrees per second
            velocities_deg_per_sec = {}
            positions_deg = {}
            
            for motor in follower.bus_right.motors:
                vel_key = f"right_{motor}.vel"
                pos_key = f"right_{motor}.pos"
                if vel_key in obs:
                    velocities_deg_per_sec[f"right_{motor}"] = obs[vel_key]
                if pos_key in obs:
                    positions_deg[f"right_{motor}"] = obs[pos_key]
            
            for motor in follower.bus_left.motors:
                vel_key = f"left_{motor}.vel"
                pos_key = f"left_{motor}.pos"
                if vel_key in obs:
                    velocities_deg_per_sec[f"left_{motor}"] = obs[vel_key]
                if pos_key in obs:
                    positions_deg[f"left_{motor}"] = obs[pos_key]
            
            # Convert velocities to rad/s and compute friction torques
            friction_torques_nm = {}
            for motor_full_name, velocity_deg_per_sec in velocities_deg_per_sec.items():
                # Extract motor name without arm prefix
                if motor_full_name.startswith("right_"):
                    motor_name = motor_full_name.removeprefix("right_")
                elif motor_full_name.startswith("left_"):
                    motor_name = motor_full_name.removeprefix("left_")
                else:
                    continue
                
                # Get motor index for friction parameters
                motor_index = motor_name_to_index.get(motor_name, 0)
                
                # Convert velocity to rad/s
                velocity_rad_per_sec = np.deg2rad(velocity_deg_per_sec)
                
                # Compute friction torque
                friction_torque = compute_friction_torque(velocity_rad_per_sec, motor_index)
                friction_torques_nm[motor_full_name] = friction_torque
            
            # Apply friction compensation to right arm (all joints INCLUDING gripper)
            for motor in follower.bus_right.motors:
                full_name = f"right_{motor}"
                position = positions_deg.get(full_name, 0.0)
                torque = friction_torques_nm.get(full_name, 0.0)
                
                # Get motor index for damping gain
                motor_index = motor_name_to_index.get(motor, 0)
                kd = DAMPING_KD[motor_index]
                
                # Send MIT control command with friction compensation + damping
                follower.bus_right._mit_control(
                    motor=motor,
                    kp=0.0,  # No position control
                    kd=kd,   # Add damping for stability
                    position_degrees=position,
                    velocity_deg_per_sec=0.0,
                    torque=torque
                )

            # Apply friction compensation to left arm (all joints INCLUDING gripper)
            for motor in follower.bus_left.motors:
                full_name = f"left_{motor}"
                position = positions_deg.get(full_name, 0.0)
                torque = friction_torques_nm.get(full_name, 0.0)
                
                # Get motor index for damping gain
                motor_index = motor_name_to_index.get(motor, 0)
                kd = DAMPING_KD[motor_index]
                
                # Send MIT control command with friction compensation + damping
                follower.bus_left._mit_control(
                    motor=motor,
                    kp=0.0,  # No position control
                    kd=kd,   # Add damping for stability
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
                    
                    print(f"{current_hz:.1f} Hz")

                    loop_times = []
                    last_print_time = loop_end
            
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\nStopping friction compensation...")
    
    finally:
        print("\nDisabling all motors and disconnecting...")
        follower.bus_right.disable_torque()
        follower.bus_left.disable_torque()
        time.sleep(0.1)
        follower.disconnect()
        print("✓ Safe shutdown complete")


if __name__ == "__main__":
    main()

