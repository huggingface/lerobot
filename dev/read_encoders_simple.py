#!/usr/bin/env python3
"""
Simple Joint Encoder Reader for SO101 Follower Motors
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
PORT = "/dev/ttyACM0"  # Serial port for SO101 Follower
UPDATE_RATE = 5        # Updates per second
# =============================================================================

import time
from lerobot.common.motors.feetech import FeetechMotorsBus
from lerobot.common.motors.motors_bus import Motor, MotorNormMode

def main():
    print("=== JOINT ENCODER READINGS ===")
    print("Press Ctrl+C to stop\n")
    
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    
    bus = FeetechMotorsBus(port=PORT, motors=motors)
    bus.connect()
    
    sleep_time = 1.0 / UPDATE_RATE
    
    while True:
        positions = bus.sync_read("Present_Position", normalize=False)
        
        print(f"M1:{positions['shoulder_pan']:4d} M2:{positions['shoulder_lift']:4d} M3:{positions['elbow_flex']:4d} M4:{positions['wrist_flex']:4d} M5:{positions['wrist_roll']:4d} M6:{positions['gripper']:4d}")
        
        time.sleep(sleep_time)

if __name__ == "__main__":
    main() 