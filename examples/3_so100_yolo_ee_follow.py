#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import time
import logging
import traceback
import math
import cv2
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scale factor
    ['shoulder_lift', 2.0, 0.97],     # Joint2: zero position offset, scale factor
    ['elbow_flex', 0.0, 1.05],        # Joint3: zero position offset, scale factor
    ['wrist_flex', 0.0, 0.94],        # Joint4: zero position offset, scale factor
    ['wrist_roll', 0.0, 0.5],        # Joint5: zero position offset, scale factor
    ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scale factor
]

def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients
    
    Args:
        joint_name: Joint name
        raw_position: Raw position value
    
    Returns:
        calibrated_position: Calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # Zero position offset
            scale = joint_cal[2]   # Scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # If no calibration coefficient found, return raw value

def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets
    
    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)
        
    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0
    
    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance
    
    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max
    
    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min
    
    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)
    
    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    
    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    
    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    
    # Convert from radians to degrees
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90-joint2_deg
    joint3_deg = joint3_deg-90
    
    return joint2_deg, joint3_deg

def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position
    
    Args:
        robot: Robot instance
        duration: Time required to move to zero position (seconds)
        kp: Proportional gain
    """
    print("Using P control to slowly move robot to zero position...")
    
    # Get current robot state
    current_obs = robot.get_observation()
    
    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value
    
    # Zero position target
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # Calculate control steps
    control_freq = 50  # 60Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will move to zero position in {duration} seconds using P control, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value
        
        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Show progress
        if step % (control_freq // 2) == 0:  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("Robot has moved to zero position")

def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position
    
    Args:
        robot: Robot instance
        start_positions: Start joint positions dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    print("Returning to start position...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds
    
    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # Don't apply calibration coefficients
        
        # P control calculation
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)
                
                # P control: output = Kp * error
                control_output = kp * error
                
                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        
        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)
        
        # Check if start position is reached
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break
        
        time.sleep(control_period)
    
    print("Return to start position completed")

# Mapping coefficients for vision control
K_pan = -0.006  # radians per pixel (tune as needed)
K_y = 0.00004   # meters per pixel (tune as needed)

# Vision control update function
def vision_control_update(target_positions, current_x, current_y, model, cap, K_pan, K_y, target_objects=["mouse"]):
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not available")
        return current_x, current_y  # No update

    results = model(frame)
    if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
        print("No objects detected")
        annotated_frame = frame
    else:
        # Find target objects in detections
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]
            if label in target_objects:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                h, w = frame.shape[:2]
                dx = cx - w // 2
                dy = cy - h // 2
                # Map dx, dy to robot control
                d_current_x = -K_pan * dx
                if abs(d_current_x) < 0.1:
                    d_current_x = 0

                target_positions['shoulder_pan'] += d_current_x
                d_current_y = -K_y * dy 
                # Clamp d_current_y to [-0.004, 0.004] and zero if within ±0.0005
                d_current_y = max(min(d_current_y, 0.005), -0.005)
                if abs(d_current_y) < 0.001:
                    d_current_y = 0
                current_y += d_current_y   # Negative sign: up in image = increase y
                # Update joint targets using inverse kinematics
                joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                target_positions['shoulder_lift'] = joint2_target
                target_positions['elbow_flex'] = joint3_target
                print(f"{label.capitalize()} center offset: dx={dx}, dy={dy} -> dc_y={d_current_y}, pan: {target_positions['shoulder_pan']:.2f}, y: {current_y:.3f}, joint2: {joint2_target:.2f}, joint3: {joint3_target:.2f}")
                break  # Only use first target object found
    # Show annotated frame in a window
    cv2.imshow("YOLO11 Live", annotated_frame)
    # Allow quitting vision mode with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt
    return current_x, current_y

def p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50, model=None, cap=None, vision_mode=False, target_objects=["mouse"]):
    """
    P control loop
    
    Args:
        robot: Robot instance
        keyboard: Keyboard instance
        target_positions: Target joint positions dictionary
        start_positions: Start joint positions dictionary
        current_x: Current x coordinate
        current_y: Current y coordinate
        kp: Proportional gain
        control_freq: Control frequency (Hz)
        model: YOLO model instance (optional)
        cap: Camera instance (optional)
        vision_mode: Whether to enable vision control
    """
    control_period = 1.0 / control_freq
    
    # Initialize pitch control variables
    pitch = 0.0  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size
    
    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    while True:
        try:
            # Vision-based control (if enabled)
            if vision_mode and model is not None and cap is not None:
                current_x, current_y = vision_control_update(
                    target_positions, current_x, current_y, model, cap, K_pan, K_y, target_objects
                )
            
            # Keyboard input (always available)
            keyboard_action = keyboard.get_action()
            
            if keyboard_action:
                # Process keyboard input, update target positions
                for key, value in keyboard_action.items():
                        if key == 'x':
                            # Exit program, first return to start position
                            print("Exit command detected, returning to start position...")
                            return_to_start_position(robot, start_positions, 0.2, control_freq)
                            return
                        
                        # Joint control mapping
                        joint_controls = {
                            'q': ('shoulder_pan', -1),    # Joint1 decrease
                            'a': ('shoulder_pan', 1),     # Joint1 increase
                            't': ('wrist_roll', -1),      # Joint5 decrease
                            'g': ('wrist_roll', 1),       # Joint5 increase
                            'y': ('gripper', -1),         # Joint6 decrease
                            'h': ('gripper', 1),          # Joint6 increase
                        }
                        
                        # x,y coordinate control
                        xy_controls = {
                            'w': ('x', -0.004),  # x decrease
                            's': ('x', 0.004),   # x increase
                            'e': ('y', -0.004),  # y decrease
                            'd': ('y', 0.004),   # y increase
                        }
                        
                        # pitch control
                        if key == 'r':
                            pitch += pitch_step
                            print(f"Increase pitch adjustment: {pitch:.3f}")
                        elif key == 'f':
                            pitch -= pitch_step
                            print(f"Decrease pitch adjustment: {pitch:.3f}")
                        
                        if key in joint_controls:
                            joint_name, delta = joint_controls[key]
                            if joint_name in target_positions:
                                current_target = target_positions[joint_name]
                                new_target = int(current_target + delta)
                                target_positions[joint_name] = new_target
                                print(f"Update target position {joint_name}: {current_target} -> {new_target}")
                        
                        elif key in xy_controls:
                            coord, delta = xy_controls[key]
                            if coord == 'x':
                                current_x += delta
                                # Calculate joint2 and joint3 target angles
                                joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                                target_positions['shoulder_lift'] = joint2_target
                                target_positions['elbow_flex'] = joint3_target
                                print(f"Update x coordinate: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                            elif coord == 'y':
                                current_y += delta
                                # Calculate joint2 and joint3 target angles
                                joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                                target_positions['shoulder_lift'] = joint2_target
                                target_positions['elbow_flex'] = joint3_target
                                print(f"Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
            
            # Apply pitch adjustment to wrist_flex
            # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
            if 'shoulder_lift' in target_positions and 'elbow_flex' in target_positions:
                target_positions['wrist_flex'] = - target_positions['shoulder_lift'] - target_positions['elbow_flex'] + pitch
                # Display current pitch value (every 100 steps to avoid spam)
                if hasattr(p_control_loop, 'step_counter'):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0
                
                if p_control_loop.step_counter % 100 == 0:
                    print(f"Current pitch adjustment: {pitch:.3f}, wrist_flex target: {target_positions['wrist_flex']:.3f}")
            
            # Get current robot state
            current_obs = robot.get_observation()
            
            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # Apply calibration coefficients
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value
            
            # P control calculation
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos
                    
                    # P control: output = Kp * error
                    control_output = kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                robot.send_action(robot_action)
            
            time.sleep(control_period)
            
        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break

def main():
    """Main function"""
    print("LeRobot Simplified Keyboard Control Example (P Control)")
    print("="*50)
    
    try:
        # Import necessary modules
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
        
        # Get port
        port = input("Please enter SO100 robot USB port (e.g.: /dev/ttyACM0): ").strip()
        
        # If Enter is pressed directly, use default port
        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")
        
        # Configure robot
        robot_config = SO100FollowerConfig(port=port)
        robot = SO100Follower(robot_config)
        
        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        
        # Connect devices
        robot.connect()
        keyboard.connect()
        
        print("Devices connected successfully!")
        
        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")
        
        # Read starting joint angles
        print("Reading starting joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients
        
        print("Starting joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")
        
        # Move to zero position
        move_to_zero_position(robot, duration=3.0)
        
        # Initialize target positions to current positions (integers)
        target_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
          }
        
        # Initialize x,y coordinate control
        x0, y0 = 0.1629, 0.1131
        current_x, current_y = x0, y0
        print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}")
        
        # Initialize YOLO and camera
        model = YOLO("yolo11x.pt")
        
        # Get detection targets from user input
        print("\n" + "="*60)
        print("YOLO Detection Target Setup")
        print("="*60)
        target_input = input("Enter objects to detect (separate multiple objects with commas, e.g., bottle,cup,mouse): ").strip()
        
        # If Enter is pressed directly, use default targets
        if not target_input:
            target_objects = ["mouse"]
            print(f"Using default targets: {target_objects}")
        else:
            # Parse multiple objects separated by commas
            target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
            print(f"Detection targets: {target_objects}")
        
        # List available cameras and prompt user
        def list_cameras(max_index=5):
            available = []
            for idx in range(max_index):
                cap_test = cv2.VideoCapture(idx)
                if cap_test.isOpened():
                    available.append(idx)
                    cap_test.release()
            return available
        cameras = list_cameras()
        if not cameras:
            print("No cameras found!")
            return
        print(f"Available cameras: {cameras}")
        selected = int(input(f"Select camera index from {cameras}: "))
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            print("Camera not found!")
            return
        
        print("Control instructions:")
        print(f"Vision mode: Robot will automatically track {target_objects} objects")
        print("Keyboard control (can be used simultaneously with vision):")
        print("- Q/A: Joint1 (shoulder_pan) decrease/increase")
        print("- W/S: Control end effector x coordinate (joint2+3)")
        print("- E/D: Control end effector y coordinate (joint2+3)")
        print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
        print("- T/G: Joint5 (wrist_roll) decrease/increase")
        print("- Y/H: Joint6 (gripper) close/open")
        print("- X: Exit program (return to start position first)")
        print("- ESC: Exit program")
        print("- Q (in camera window): Exit vision mode")
        print("="*50)
        print("Note: Vision and keyboard control work together")
        
        # Enable vision control
        vision_mode = True
        p_control_loop(robot, keyboard, target_positions, start_positions, current_x, current_y, kp=0.5, control_freq=50, model=model, cap=cap, vision_mode=vision_mode, target_objects=target_objects)
        
        # Disconnect
        robot.disconnect()
        keyboard.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Is the robot correctly connected")
        print("2. Is the USB port correct")
        print("3. Do you have sufficient permissions to access USB device")
        print("4. Is the robot correctly configured")

if __name__ == "__main__":
    main() 