#!/usr/bin/env python3
"""
Minimalist Velocity Controller for SO101 First Joint
Hold 'a'/'d' to move left/right, 'q' to quit
"""

import sys
import termios
import tty
import signal
import select
import time
from lerobot.common.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.common.motors.motors_bus import Motor, MotorNormMode

PORT = "/dev/ttyACM0"
TARGET_VELOCITY = 200
KEY_TIMEOUT = 0.15

# Setup motor
motors = {"shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)}
bus = FeetechMotorsBus(port=PORT, motors=motors)

def cleanup_and_exit(signum=None, frame=None):
    bus.write("Goal_Velocity", "shoulder_pan", 0, normalize=False)
    bus.disable_torque()
    bus.disconnect()
    sys.exit(0)

def main():
    print("Minimalist Velocity Controller - Hold 'a'/'d' to move, 'q' to quit")
    
    signal.signal(signal.SIGINT, cleanup_and_exit)
    
    # Connect and configure
    bus.connect()
    bus.configure_motors()
    bus.write("Operating_Mode", "shoulder_pan", OperatingMode.VELOCITY.value)
    bus.enable_torque()
    print("Ready!")
    
    # Terminal setup
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setraw(sys.stdin.fileno())
        
        last_key = None
        last_key_time = 0
        current_velocity = 0
        
        while True:
            loop_start = time.time()
            
            # Check for key press
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1).lower()
                if char == 'q' or char == '\x03':
                    break
                elif char in ['a', 'd']:
                    last_key = char
                    last_key_time = time.time()
            
            # Determine velocity
            if last_key and (time.time() - last_key_time) < KEY_TIMEOUT:
                velocity = -TARGET_VELOCITY if last_key == 'a' else TARGET_VELOCITY
            else:
                velocity = 0
                last_key = None
            
            # Send command at 100Hz
            bus.write("Goal_Velocity", "shoulder_pan", velocity, normalize=False)
            
            # Status update
            if velocity != current_velocity:
                if velocity == 0:
                    print("Stop")
                else:
                    print(f"{'Left' if velocity < 0 else 'Right'}: {velocity}")
                current_velocity = velocity
            
            # Maintain 100Hz
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.01 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        cleanup_and_exit()

if __name__ == "__main__":
    main()
