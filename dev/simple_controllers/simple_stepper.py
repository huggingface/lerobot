#!/usr/bin/env python3
"""
Simple Stepper Mode Controller for SO101 First Joint (Shoulder Pan)

Controls:
- 'a' key: Turn left
- 'd' key: Turn right  
- 'q' key: Quit
"""

import sys
import termios
import tty
import signal
from lerobot.common.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.common.motors.motors_bus import Motor, MotorNormMode

# Configuration
PORT = "/dev/ttyACM0"  # Change to your follower port
STEP_SIZE = 10  # Step size in motor units (adjust as needed)

class StepperController:
    def __init__(self, port):
        self.port = port
        self.motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        }
        self.bus = FeetechMotorsBus(port=port, motors=self.motors)
        self.connected = False
        
    def connect(self):
        """Connect to the motor and configure for stepper mode."""
        print(f"Connecting to motor on {self.port}...")
        self.bus.connect()
        self.bus.configure_motors()
        
        # Configure motor for stepper mode
        print("Configuring motor for stepper mode...")
        motor_name = "shoulder_pan"
        
        # Set to position mode first
        self.bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)
        
        # Enable torque
        self.bus.enable_torque()
        
        # Get current position
        current_pos = self.bus.read("Present_Position", motor_name, normalize=False)
        print(f"Current position: {current_pos}")
        
        self.connected = True
        print("✓ Motor connected and ready for stepper control!")
        
    def disconnect(self):
        """Disconnect from motor."""
        if self.connected:
            print("\nDisabling torque and disconnecting...")
            self.bus.disable_torque()
            self.bus.disconnect()
            self.connected = False
            print("✓ Motor disconnected")
    
    def step_left(self):
        """Move motor one step to the left."""
        if not self.connected:
            return
            
        motor_name = "shoulder_pan"
        current_pos = self.bus.read("Present_Position", motor_name, normalize=False)
        new_pos = current_pos - STEP_SIZE
        
        # Apply position limits (typical STS3215 range: 0-4095)
        new_pos = max(0, min(4095, new_pos))
        
        print(f"Moving left: {current_pos} → {new_pos}")
        self.bus.write("Goal_Position", motor_name, new_pos, normalize=False)
    
    def step_right(self):
        """Move motor one step to the right."""
        if not self.connected:
            return
            
        motor_name = "shoulder_pan"
        current_pos = self.bus.read("Present_Position", motor_name, normalize=False)
        new_pos = current_pos + STEP_SIZE
        
        # Apply position limits (typical STS3215 range: 0-4095)
        new_pos = max(0, min(4095, new_pos))
        
        print(f"Moving right: {current_pos} → {new_pos}")
        self.bus.write("Goal_Position", motor_name, new_pos, normalize=False)

def get_char():
    """Get a single character from stdin without pressing Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def cleanup_and_exit(controller):
    """Clean shutdown."""
    controller.disconnect()
    sys.exit(0)

def main():
    print("=== SO101 STEPPER MODE CONTROLLER ===")
    print("First joint (shoulder_pan) stepper control")
    print(f"Step size: {STEP_SIZE} motor units")
    print()
    
    # Create controller
    controller = StepperController(PORT)
    
    # Setup signal handler for Ctrl+C
    def signal_handler(signum, frame):
        cleanup_and_exit(controller)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Connect to motor
        controller.connect()
        
        print("\nControls:")
        print("  'a' - Step left")
        print("  'd' - Step right")
        print("  'q' - Quit")
        print("\nPress keys to control the motor...")
        
        # Main control loop
        while True:
            try:
                char = get_char().lower()
                
                if char == 'a':
                    controller.step_left()
                elif char == 'd':
                    controller.step_right()
                elif char == 'q':
                    print("\nQuitting...")
                    break
                elif char == '\x03':  # Ctrl+C
                    break
                else:
                    print(f"Unknown key: '{char}' (use 'a', 'd', or 'q')")
                    
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_and_exit(controller)

if __name__ == "__main__":
    main()
