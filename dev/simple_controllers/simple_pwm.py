#!/usr/bin/env python3
"""
Fixed PWM Controller for SO101 First Joint
Hold 'a'/'d' to move left/right, 'q' to quit

Fixed issues:
- Use Goal_Time instead of Goal_Velocity for PWM mode
- Proper cleanup to restore motor to position mode
- Better error handling
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
TARGET_PWM = 500  # PWM torque limit (0-1000 range)
RUNNING_TIME = 100  # Goal_Time in milliseconds
KEY_TIMEOUT = 0.15

# Setup motor
motors = {"shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)}
bus = FeetechMotorsBus(port=PORT, motors=motors)

def cleanup_and_exit(signum=None, frame=None):
    """Proper cleanup to restore motor to usable state."""
    try:
        print("\nCleaning up...")
        # Stop PWM output
        bus.write("Torque_Limit", "shoulder_pan", 0, normalize=False)
        bus.write("Goal_Time", "shoulder_pan", 0, normalize=False)
        
        # Restore to position mode
        bus.write("Operating_Mode", "shoulder_pan", OperatingMode.POSITION.value)
        
        # Brief delay to let mode change take effect
        time.sleep(0.1)
        
        # Disable torque and disconnect
        bus.disable_torque()
        bus.disconnect()
        print("✓ Motor restored to position mode")
    except Exception as e:
        print(f"Cleanup error: {e}")
    finally:
        sys.exit(0)

def main():
    print("Fixed PWM Controller - Hold 'a'/'d' to move, 'q' to quit")
    print("Note: Using Goal_Time parameter for PWM mode")
    
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    
    try:
        # Connect and configure
        bus.connect()
        bus.configure_motors()
        
        # Set to PWM mode
        bus.write("Operating_Mode", "shoulder_pan", OperatingMode.PWM.value)
        
        # Enable torque
        bus.enable_torque()
        print("Ready!")
        
        # Terminal setup
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            
            last_key = None
            last_key_time = 0
            current_pwm = 0
            
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
                
                # Determine PWM value and direction
                if last_key and (time.time() - last_key_time) < KEY_TIMEOUT:
                    pwm_value = TARGET_PWM
                    # In PWM mode, Goal_Time bit 11 controls direction
                    # 0 = forward, bit 11 set = reverse
                    direction_bit = 0x0800 if last_key == 'a' else 0x0000
                    goal_time = RUNNING_TIME | direction_bit
                else:
                    pwm_value = 0
                    goal_time = 0
                    last_key = None
                
                # Send PWM commands
                bus.write("Torque_Limit", "shoulder_pan", pwm_value, normalize=False)
                if pwm_value > 0:
                    bus.write("Goal_Time", "shoulder_pan", goal_time, normalize=False)
                
                # Status update
                if pwm_value != current_pwm:
                    if pwm_value == 0:
                        print("Stop")
                    else:
                        print(f"{'Left' if last_key == 'a' else 'Right'}: PWM={pwm_value}, Time={goal_time:04x}")
                    current_pwm = pwm_value
                
                # Maintain 50Hz (less aggressive than 100Hz)
                elapsed = time.time() - loop_start
                sleep_time = max(0, 0.02 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
