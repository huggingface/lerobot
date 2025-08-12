#!/usr/bin/env python3
"""
Minimalist PID Gains Manager for SO101 Follower Motors

Configure PID gains below and run the script to apply them to all motors.
"""

# =============================================================================
# CONFIGURATION - Edit these values to set PID gains
# =============================================================================
P_GAIN = 255  # Proportional gain (0-255) - Higher = faster response, more oscillation
I_GAIN = 0   # Integral gain (0-255) - Usually 0 to avoid windup issues  
D_GAIN = 200  # Derivative gain (0-255) - Higher = more damping, less overshoot

PORT = "/dev/ttyACM0"  # Serial port for SO101 Follower
# =============================================================================

from lerobot.common.motors.feetech import FeetechMotorsBus
from lerobot.common.motors.motors_bus import Motor, MotorNormMode

def main():
    print("=== SO101 FOLLOWER PID GAINS MANAGER ===")
    print(f"Target PID gains: P={P_GAIN}, I={I_GAIN}, D={D_GAIN}")
    
    # SO101 Follower motor configuration
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    
    bus = FeetechMotorsBus(port=PORT, motors=motors)
    
    print(f"Connecting to {PORT}...")
    bus.connect()
    print("✓ Connected")
    
    # Read current gains
    print("\nCurrent PID gains:")
    for motor_name in motors.keys():
        motor_id = motors[motor_name].id
        p = bus.read("P_Coefficient", motor_name, normalize=False)
        i = bus.read("I_Coefficient", motor_name, normalize=False)
        d = bus.read("D_Coefficient", motor_name, normalize=False)
        print(f"  Motor {motor_id}: P={p}, I={i}, D={d}")
    
    # Set new gains
    print(f"\nSetting PID gains to P={P_GAIN}, I={I_GAIN}, D={D_GAIN}...")
    for motor_name in motors.keys():
        motor_id = motors[motor_name].id
        bus.write("P_Coefficient", motor_name, P_GAIN, normalize=False)
        bus.write("I_Coefficient", motor_name, I_GAIN, normalize=False)
        bus.write("D_Coefficient", motor_name, D_GAIN, normalize=False)
        print(f"  Motor {motor_id}: ✓")
    
    # Verify changes
    print(f"\nVerifying changes...")
    for motor_name in motors.keys():
        motor_id = motors[motor_name].id
        p = bus.read("P_Coefficient", motor_name, normalize=False)
        i = bus.read("I_Coefficient", motor_name, normalize=False)
        d = bus.read("D_Coefficient", motor_name, normalize=False)
        
        correct = (p == P_GAIN and i == I_GAIN and d == D_GAIN)
        status = "✓" if correct else "❌"
        print(f"  Motor {motor_id}: P={p}, I={i}, D={d} {status}")
    
    print("\n✓ All motors updated!")
    
    bus.disconnect()
    print("✓ Disconnected")

if __name__ == "__main__":
    main() 