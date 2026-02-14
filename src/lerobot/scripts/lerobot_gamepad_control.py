#!/usr/bin/env python3
"""
SO-101 Xbox Controller Control Script
Controls SO-101 robotic arm using an Xbox controller (or any compatible gamepad)

Button Mapping:
- Left Stick: Control joints 1-2 (base rotation and shoulder)
- Right Stick: Control joints 3-4 (elbow and wrist)
- D-Pad Up/Down: Control joint 5 (wrist rotation)
- LT (Left Trigger): Close gripper
- RT (Right Trigger): Open gripper
- RB (Right Bumper): Enable/disable control (safety)
- Start button: Exit program
- B button: Reset to home position

Safety Features:
- Must hold RB to enable movement (dead-man switch)
- Max movement speed limit
- Automatic disconnect on exit
"""

import time
import numpy as np
import torch
import pygame
from pathlib import Path

# Import LeRobot components
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


class SO101GamepadController:
    """Xbox controller interface for SO-101 robot arm"""
    
    def __init__(
        self,
        robot_port="/dev/ttyACM0",
        robot_id="so101_follower",
        max_speed=2.0,  # Maximum change per control loop (in degrees for joints)
        control_frequency=30,  # Hz
    ):
        self.robot_port = robot_port
        self.robot_id = robot_id
        self.max_speed = max_speed
        self.control_dt = 1.0 / control_frequency
        
        # Initialize pygame for gamepad
        pygame.init()
        pygame.joystick.init()
        
        # Check for gamepad
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected! Please connect an Xbox controller.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"✓ Gamepad connected: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")
        
        # Initialize robot
        print(f"\n✓ Connecting to SO-101 on {robot_port}...")
        config = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
        )
        self.robot = SO101Follower(config)
        self.robot.connect()
        
        # Get initial state
        self.current_position = None
        self.num_joints = None
        
        # Initialize preset positions dictionary (before calling _initialize_presets)
        self.preset_positions = {}
        
        self._initialize_robot_state()
        
        # Initialize preset positions
        self._initialize_presets()
        
        # Control state
        self.enabled = False
        self.running = True
        self.gripper_state = 0.0  # -1.0 = closed, 1.0 = open
        
        # Button/axis indices (standard Xbox controller layout)
        self.AXIS_LEFT_X = 0
        self.AXIS_LEFT_Y = 1
        self.AXIS_RIGHT_X = 2
        self.AXIS_RIGHT_Y = 3
        self.AXIS_LT = 4  # Left trigger
        self.AXIS_RT = 5  # Right trigger
        
        self.BTN_A = 0
        self.BTN_B = 1
        self.BTN_X = 2
        self.BTN_Y = 3
        self.BTN_LB = 4
        self.BTN_RB = 5  # Right bumper - enable control
        self.BTN_START = 7
        
        # Dead zone for analog sticks
        self.DEAD_ZONE = 0.15
        
    def _initialize_robot_state(self):
        """Get initial robot state"""
        obs = self.robot.get_observation()
        
        # SO-101 returns individual keys for each joint position
        joint_keys = [
            'shoulder_pan.pos',
            'shoulder_lift.pos', 
            'elbow_flex.pos',
            'wrist_flex.pos',
            'wrist_roll.pos',
            'gripper.pos'
        ]
        
        # Extract positions into numpy array
        positions = []
        for key in joint_keys:
            if key in obs:
                positions.append(obs[key])
            else:
                print(f"⚠️  Warning: Key '{key}' not found in observation")
        
        self.current_position = np.array(positions, dtype=np.float32)
        self.num_joints = len(self.current_position)
        self.joint_keys = joint_keys
        
        print(f"✓ Robot initialized with {self.num_joints} joints")
        print(f"  Current position (degrees): {np.round(self.current_position, 2)}")
        
        # Load calibrated joint limits
        self._load_joint_limits()
    
    def _load_joint_limits(self):
        """Load joint limits from calibration file"""
        import platform
        import json
        
        # Find calibration file
        if platform.system() == "Windows":
            calib_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / f"{self.robot_id}.json"
        else:
            calib_path = Path.home() / ".cache" / "calibration" / "so101" / f"{self.robot_id}.json"
        
        # Default limits (conservative ±180°)
        self.joint_limits_lower = np.array([-180.0] * self.num_joints, dtype=np.float32)
        self.joint_limits_upper = np.array([180.0] * self.num_joints, dtype=np.float32)
        
        # Try to load calibration
        if calib_path.exists():
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
                
                print(f"  📄 Calibration file loaded from: {calib_path.name}")
                
                print(f"  ✓ Reading range_min/range_max from calibration motors")
                
                # STS3215 servo conversion: servo units to degrees
                # Servo range: 0-4095 maps to 0-360 degrees (approx)
                # But calibration uses a centered system where 2048 ≈ 0°
                SERVO_CENTER = 2048
                SERVO_UNITS_PER_DEGREE = 4096 / 360.0  # ~11.38 units per degree
                
                limits_found = False
                for i, key in enumerate(self.joint_keys):
                    joint_name = key.removesuffix('.pos')
                    
                    if joint_name in calib_data:
                        motor_config = calib_data[joint_name]
                        
                        if 'range_min' in motor_config and 'range_max' in motor_config:
                            range_min = motor_config['range_min']
                            range_max = motor_config['range_max']
                            
                            # Convert servo units to degrees (centered at 2048)
                            # Degrees = (servo_units - 2048) / (4096/360)
                            min_degrees = (range_min - SERVO_CENTER) / SERVO_UNITS_PER_DEGREE
                            max_degrees = (range_max - SERVO_CENTER) / SERVO_UNITS_PER_DEGREE
                            
                            self.joint_limits_lower[i] = min_degrees
                            self.joint_limits_upper[i] = max_degrees
                            limits_found = True
                
                if limits_found:
                    print(f"  ✓ Converted servo ranges to degrees:")
                    for i, key in enumerate(self.joint_keys):
                        print(f"    {key}: [{self.joint_limits_lower[i]:.1f}°, {self.joint_limits_upper[i]:.1f}°]")
                else:
                    print(f"  ⚠️  No range_min/range_max found in motor configs")
                    print(f"  Using default ±180° limits")
                
            except Exception as e:
                print(f"  ⚠️  Could not load calibration limits: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Using default ±180° limits")
        else:
            print(f"  ⚠️  Calibration file not found at {calib_path}")
            print(f"  Using default ±180° limits")
    
    def _initialize_presets(self):
        """Initialize preset positions"""
        # Ready position: all joints at 0°
        self.preset_positions['ready'] = np.zeros(self.num_joints, dtype=np.float32)
        
        # Home position:
        self.preset_positions['home'] = np.array([
            0.0,    # shoulder_pan: centered
            -90.0,  # shoulder_lift: back
            90.0,   # elbow_flex: 90° bend
            -90.0,  # wrist_flex: back
            0.0,    # wrist_roll: centered
            0.0     # gripper: neutral
        ], dtype=np.float32)
        
        # Vertical reach position: arm reaching upward
        self.preset_positions['vertical'] = np.array([
            0.0,    # shoulder_pan: centered
            0.0,    # shoulder_lift: pointing up
            -90.0,  # elbow_flex: 90° bend
            0.0,    # wrist_flex: straight
            0.0,    # wrist_roll: centered
            0.0     # gripper: neutral
        ], dtype=np.float32)
        
        print(f"  ✓ Initialized preset positions:")
        print(f"    A button → Home (all zeros)")
        print(f"    X button → Ready (forward reach)")
        print(f"    Y button → Vertical (upward reach)")
        
    def apply_deadzone(self, value):
        """Apply dead zone to analog stick values"""
        if abs(value) < self.DEAD_ZONE:
            return 0.0
        # Re-scale to smooth transition
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEAD_ZONE) / (1.0 - self.DEAD_ZONE)
    
    def get_gamepad_input(self):
        """Read gamepad state and return action vector"""
        pygame.event.pump()
        
        # Check enable button (RB - dead man switch)
        rb_pressed = self.joystick.get_button(self.BTN_RB)
        
        # Check exit button
        if self.joystick.get_button(self.BTN_START):
            self.running = False
            return None
        
        # Check preset position buttons (A, X, Y)
        if self.joystick.get_button(self.BTN_A):
            print("→ Moving to HOME position")
            return "preset_home"
        
        if self.joystick.get_button(self.BTN_X):
            print("→ Moving to READY position")
            return "preset_ready"
        
        if self.joystick.get_button(self.BTN_Y):
            print("→ Moving to VERTICAL position")
            return "preset_vertical"
            
        # Check reset button (B)
        if self.joystick.get_button(self.BTN_B):
            print("Resetting to home position...")
            return "reset"
        
        # If RB not pressed, don't move
        if not rb_pressed:
            if self.enabled:
                print("Control disabled - release RB")
                self.enabled = False
            return np.zeros(self.num_joints)
        
        if not self.enabled:
            print("Control enabled - hold RB to move")
            self.enabled = True
        
        # Read analog sticks (apply deadzone, no inversion)
        left_x = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
        left_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))
        right_x = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_X))
        right_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))
        
        # Read triggers (normalize from -1.0~1.0 to 0.0~1.0 range)
        # Xbox triggers default to -1.0 when not pressed, +1.0 when fully pressed
        lt = (self.joystick.get_axis(self.AXIS_LT) + 1.0) / 2.0
        rt = (self.joystick.get_axis(self.AXIS_RT) + 1.0) / 2.0
        
        # Read D-pad (hat) for wrist roll control
        # Hat returns (x, y) where x is -1 (left), 0 (center), or 1 (right)
        hat_x = 0
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)
            hat_x = -hat[0]  # -1 for left, 1 for right
        
        # Map to joint deltas
        action = np.zeros(self.num_joints)
        
        # Assuming SO-101 has 6 joints: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        if self.num_joints >= 6:
            action[0] = left_x * self.max_speed       # Shoulder pan (base rotation)
            action[1] = left_y * self.max_speed       # Shoulder lift
            action[2] = right_y * self.max_speed      # Elbow flex
            action[3] = right_x * self.max_speed      # Wrist flex (pitch)
            action[4] = hat_x * self.max_speed        # Wrist roll (D-pad left/right)
            
            # Gripper control (joint 5)
            if rt > 0.1:  # Right trigger - open gripper
                action[5] = rt * self.max_speed
            elif lt > 0.1:  # Left trigger - close gripper
                action[5] = -lt * self.max_speed
        
        return action
    
    def run(self):
        """Main control loop"""
        print("\n" + "="*60)
        print("SO-101 GAMEPAD CONTROL")
        print("="*60)
        print("\nControls:")
        print("  Left Stick:         Shoulder pan & lift (joints 0-1)")
        print("  Right Stick:        Elbow flex & wrist flex (joints 2-3)")
        print("  D-Pad Left/Right:   Wrist roll (joint 4)")
        print("  LT (Left Trigger):  Close gripper")
        print("  RT (Right Trigger): Open gripper")
        print("  RB (Hold):          Enable movement (SAFETY)")
        print("\nPreset Positions:")
        print("  A Button:           Move to HOME (all zeros)")
        print("  X Button:           Move to READY (forward reach)")
        print("  Y Button:           Move to VERTICAL (upward reach)")
        print("  B Button:           Reset to home (legacy)")
        print("\nOther:")
        print("  Start Button:       Exit")
        print("\n" + "="*60)
        print("\n⚠️  SAFETY: Hold RB button to enable manual movement!")
        print("⚠️  Preset buttons (A/X/Y) work without holding RB")
        print("Starting in 3 seconds...\n")
        time.sleep(3)
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get gamepad input
                action = self.get_gamepad_input()
                
                if action is None:
                    break
                
                # Handle preset positions
                if isinstance(action, str):
                    if action == "reset":
                        # Reset to home position (all zeros in degrees)
                        target_position = self.preset_positions['home'].copy()
                    elif action == "preset_home":
                        target_position = self.preset_positions['home'].copy()
                    elif action == "preset_ready":
                        target_position = self.preset_positions['ready'].copy()
                    elif action == "preset_vertical":
                        target_position = self.preset_positions['vertical'].copy()
                    else:
                        continue
                    
                    # Convert to dictionary format for SO-101
                    action_dict = {key: float(target_position[i]) for i, key in enumerate(self.joint_keys)}
                    
                    self.robot.send_action(action_dict)
                    self.current_position = target_position
                    time.sleep(0.5)  # Brief pause after preset movement
                    continue
                
                # Calculate new target position (relative control)
                target_position = self.current_position + action
                
                # Clip to calibrated safe ranges
                # This prevents sending positions outside the servo's physical limits
                # Uses the actual calibrated min/max from your robot's calibration file
                target_position = np.clip(target_position, self.joint_limits_lower, self.joint_limits_upper)
                
                # Convert to dictionary format for SO-101
                action_dict = {key: float(target_position[i]) for i, key in enumerate(self.joint_keys)}
                
                # Send action to robot
                self.robot.send_action(action_dict)
                
                # Update current position
                self.current_position = target_position
                
                # Display status (every 30 frames = ~1 second)
                if int(time.time() * 30) % 30 == 0:
                    active = np.any(np.abs(action) > 0.001)
                    if active or self.enabled:
                        status = "ACTIVE" if active else "READY"
                        print(f"[{status}] Position: {np.round(self.current_position, 2)}")
                
                # Maintain control frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.control_dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        
        finally:
            print("\nShutting down...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Disconnecting robot...")
        self.robot.disconnect()
        pygame.quit()
        print("✓ Cleanup complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Control SO-101 robot arm with Xbox controller"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO-101 (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="so101_follower",
        help="Robot ID (default: so101_follower)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=2.0,
        help="Maximum joint speed in degrees/step (default: 2.0)",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="Control frequency in Hz (default: 30)",
    )
    
    args = parser.parse_args()
    
    # Check if calibration exists (Windows and Linux paths)
    import platform
    if platform.system() == "Windows":
        calib_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / f"{args.robot_id}.json"
    else:
        calib_path = Path.home() / ".cache" / "calibration" / "so101" / f"{args.robot_id}.json"
    
    if not calib_path.exists():
        print(f"\n⚠️  WARNING: Calibration file not found at {calib_path}")
        print("Please calibrate your robot first with:")
        print(f"  lerobot-calibrate --robot.type=so101_follower --robot.port={args.port} --robot.id={args.robot_id}")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Create controller and run
    controller = SO101GamepadController(
        robot_port=args.port,
        robot_id=args.robot_id,
        max_speed=args.max_speed,
        control_frequency=args.frequency,
    )
    
    controller.run()


if __name__ == "__main__":
    main()
