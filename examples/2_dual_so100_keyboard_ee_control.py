#!/usr/bin/env python3
"""
Dual-arm keyboard control for SO100/SO101 robots
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
Supports simultaneous control of two robot arms: /dev/ttyACM0 and /dev/ttyACM1
Keyboard mapping: First arm (7y8u9i0o-p=[), Second arm (hbjnkml,;.'/)
"""

import time
import logging
import traceback
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scaling_factor]
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scaling factor
    ['shoulder_lift', 2.0, 0.97],     # Joint2: zero position offset, scaling factor
    ['elbow_flex', 0.0, 1.05],        # Joint3: zero position offset, scaling factor
    ['wrist_flex', 0.0, 0.94],        # Joint4: zero position offset, scaling factor
    ['wrist_roll', 0.0, 0.5],        # Joint5: zero position offset, scaling factor
    ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scaling factor
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
            scale = joint_cal[2]   # Scaling factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # If no calibration coefficient found, return original value

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

def move_to_zero_position(robots, duration=3.0, kp=0.5):
    """
    Use P control to slowly move all robots to zero position
    
    Args:
        robots: Robot instance dictionary {'arm1': robot1, 'arm2': robot2}
        duration: Time required to move to zero position (seconds)
        kp: Proportional gain
    """
    print("Using P control to slowly move all robots to zero position...")
    
    # Get current states of all robots
    current_obs = {}
    for arm_name, robot in robots.items():
        current_obs[arm_name] = robot.get_observation()
    
    # Extract current joint positions
    current_positions = {}
    for arm_name, obs in current_obs.items():
        current_positions[arm_name] = {}
        for key, value in obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[arm_name][motor_name] = value
    
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
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq
    
    print(f"Will move to zero position in {duration} seconds using P control, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    for step in range(total_steps):
        # Get current states of all robots
        current_obs = {}
        current_positions = {}
        for arm_name, robot in robots.items():
            current_obs[arm_name] = robot.get_observation()
            current_positions[arm_name] = {}
            for key, value in current_obs[arm_name].items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    # Apply calibration coefficients
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[arm_name][motor_name] = calibrated_value
        
        # Perform P control calculation for each robot arm
        for arm_name, robot in robots.items():
            robot_action = {}
            for joint_name, target_pos in zero_positions.items():
                if joint_name in current_positions[arm_name]:
                    current_pos = current_positions[arm_name][joint_name]
                    error = target_pos - current_pos
                    
                    # P control: output = Kp * error
                    control_output = kp * error
                    
                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position
            
            # Send action to robot
            if robot_action:
                robot.send_action(robot_action)
        
        # Display progress
        if step % (control_freq // 2) == 0:  # Display progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Move to zero position progress: {progress:.1f}%")
        
        time.sleep(step_time)
    
    print("All robots have moved to zero position")

def return_to_start_position(robots, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position
    
    Args:
        robots: Robot instance dictionary {'arm1': robot1, 'arm2': robot2}
        start_positions: Start joint position dictionary {'arm1': {}, 'arm2': {}}
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    print("Returning to start position...")
    
    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds
    
    for step in range(max_steps):
        # Get current states of all robots
        current_obs = {}
        current_positions = {}
        for arm_name, robot in robots.items():
            current_obs[arm_name] = robot.get_observation()
            current_positions[arm_name] = {}
            for key, value in current_obs[arm_name].items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    current_positions[arm_name][motor_name] = value  # Don't apply calibration coefficients
        
        # Perform P control calculation for each robot arm
        total_error = 0
        for arm_name, robot in robots.items():
            robot_action = {}
            for joint_name, target_pos in start_positions[arm_name].items():
                if joint_name in current_positions[arm_name]:
                    current_pos = current_positions[arm_name][joint_name]
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
        if total_error < 4.0:  # If total error is less than 4 degrees (dual arms), consider reached
            print("Returned to start position")
            break
        
        time.sleep(control_period)
    
    print("Return to start position completed")

def p_control_loop(robots, keyboard, target_positions, start_positions, current_positions, kp=0.5, control_freq=50):
    """
    P control loop
    
    Args:
        robots: Robot instance dictionary {'arm1': robot1, 'arm2': robot2}
        keyboard: Keyboard instance
        target_positions: Target joint position dictionary
        start_positions: Start joint position dictionary
        current_positions: Current joint position dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    control_period = 1.0 / control_freq
    
    # Initialize dual-arm pitch control variables
    pitch = {'arm1': 0.0, 'arm2': 0.0}  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size
    
    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")
    
    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()
            
            if keyboard_action:
                # Process keyboard input, update target positions
                for key, value in keyboard_action.items():
                    if key == 'x':
                        # Exit program, return to start position first
                        print("Exit command detected, returning to start position...")
                        return_to_start_position(robots, start_positions, 0.2, control_freq)
                        return
                    
                    # First arm control mapping: 7y8u9i0o-p=[
                    arm1_joint_controls = {
                        '7': ('shoulder_pan', -1),    # Joint1 decrease
                        'y': ('shoulder_pan', 1),     # Joint1 increase
                        '0': ('wrist_roll', -1),      # Joint5 decrease
                        'o': ('wrist_roll', 1),       # Joint5 increase
                        '-': ('gripper', -1),         # Joint6 decrease
                        'p': ('gripper', 1),          # Joint6 increase
                    }
                    
                    # First arm x,y coordinate control
                    arm1_xy_controls = {
                        '8': ('x', -0.004),  # x decrease
                        'u': ('x', 0.004),   # x increase
                        '9': ('y', -0.004),  # y decrease
                        'i': ('y', 0.004),   # y increase
                    }
                    
                    # Second arm control mapping: hbjnkml,;.'/
                    arm2_joint_controls = {
                        'h': ('shoulder_pan', -1),    # Joint1 decrease
                        'b': ('shoulder_pan', 1),     # Joint1 increase
                        ';': ('wrist_roll', -1),      # Joint5 decrease
                        'l': ('wrist_roll', 1),       # Joint5 increase
                        "'": ('gripper', -1),         # Joint6 decrease
                        '/': ('gripper', 1),          # Joint6 increase
                    }
                    
                    # Second arm x,y coordinate control
                    arm2_xy_controls = {
                        'j': ('x', -0.004),  # x decrease
                        'n': ('x', 0.004),   # x increase
                        'k': ('y', -0.004),  # y decrease
                        'm': ('y', 0.004),   # y increase
                    }
                    
                    # First arm pitch control
                    if key == '=':
                        pitch['arm1'] += pitch_step
                        print(f"First arm increase pitch adjustment: {pitch['arm1']:.3f}")
                    elif key == '[':
                        pitch['arm1'] -= pitch_step
                        print(f"First arm decrease pitch adjustment: {pitch['arm1']:.3f}")
                    
                    # Second arm pitch control
                    elif key == ',':
                        pitch['arm2'] += pitch_step
                        print(f"Second arm increase pitch adjustment: {pitch['arm2']:.3f}")
                    elif key == '.':
                        pitch['arm2'] -= pitch_step
                        print(f"Second arm decrease pitch adjustment: {pitch['arm2']:.3f}")
                    
                    # First arm joint control
                    if key in arm1_joint_controls:
                        joint_name, delta = arm1_joint_controls[key]
                        if joint_name in target_positions['arm1']:
                            current_target = target_positions['arm1'][joint_name]
                            new_target = int(current_target + delta)
                            target_positions['arm1'][joint_name] = new_target
                            print(f"First arm update target position {joint_name}: {current_target} -> {new_target}")
                    
                    # Second arm joint control
                    elif key in arm2_joint_controls:
                        joint_name, delta = arm2_joint_controls[key]
                        if joint_name in target_positions['arm2']:
                            current_target = target_positions['arm2'][joint_name]
                            new_target = int(current_target + delta)
                            target_positions['arm2'][joint_name] = new_target
                            print(f"Second arm update target position {joint_name}: {current_target} -> {new_target}")
                    
                    # First arm xy control
                    elif key in arm1_xy_controls:
                        coord, delta = arm1_xy_controls[key]
                        if coord == 'x':
                            current_positions['arm1']['x'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm1']['x'], current_positions['arm1']['y'])
                            target_positions['arm1']['shoulder_lift'] = joint2_target
                            target_positions['arm1']['elbow_flex'] = joint3_target
                            print(f"First arm update x coordinate: {current_positions['arm1']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_positions['arm1']['y'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm1']['x'], current_positions['arm1']['y'])
                            target_positions['arm1']['shoulder_lift'] = joint2_target
                            target_positions['arm1']['elbow_flex'] = joint3_target
                            print(f"First arm update y coordinate: {current_positions['arm1']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                    
                    # Second arm xy control
                    elif key in arm2_xy_controls:
                        coord, delta = arm2_xy_controls[key]
                        if coord == 'x':
                            current_positions['arm2']['x'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm2']['x'], current_positions['arm2']['y'])
                            target_positions['arm2']['shoulder_lift'] = joint2_target
                            target_positions['arm2']['elbow_flex'] = joint3_target
                            print(f"Second arm update x coordinate: {current_positions['arm2']['x']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
                        elif coord == 'y':
                            current_positions['arm2']['y'] += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_positions['arm2']['x'], current_positions['arm2']['y'])
                            target_positions['arm2']['shoulder_lift'] = joint2_target
                            target_positions['arm2']['elbow_flex'] = joint3_target
                            print(f"Second arm update y coordinate: {current_positions['arm2']['y']:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")
            
            # Apply pitch adjustment to wrist_flex for each robot arm
            for arm_name in ['arm1', 'arm2']:
                if 'shoulder_lift' in target_positions[arm_name] and 'elbow_flex' in target_positions[arm_name]:
                    target_positions[arm_name]['wrist_flex'] = - target_positions[arm_name]['shoulder_lift'] - target_positions[arm_name]['elbow_flex'] + pitch[arm_name]
            
            # Display current pitch values (every 100 steps to avoid spam)
            if hasattr(p_control_loop, 'step_counter'):
                p_control_loop.step_counter += 1
            else:
                p_control_loop.step_counter = 0
            
            if p_control_loop.step_counter % 100 == 0:
                print(f"Current pitch adjustment: arm1={pitch['arm1']:.3f}, arm2={pitch['arm2']:.3f}")
            
            # Get current states of all robots
            current_obs = {}
            current_joint_positions = {}
            for arm_name, robot in robots.items():
                current_obs[arm_name] = robot.get_observation()
                current_joint_positions[arm_name] = {}
                for key, value in current_obs[arm_name].items():
                    if key.endswith('.pos'):
                        motor_name = key.removesuffix('.pos')
                        # Apply calibration coefficients
                        calibrated_value = apply_joint_calibration(motor_name, value)
                        current_joint_positions[arm_name][motor_name] = calibrated_value
            
            # Perform P control calculation for each robot arm
            for arm_name, robot in robots.items():
                robot_action = {}
                for joint_name, target_pos in target_positions[arm_name].items():
                    if joint_name in current_joint_positions[arm_name]:
                        current_pos = current_joint_positions[arm_name][joint_name]
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
    print("LeRobot Dual-arm Keyboard Control Example (P Control)")
    print("="*50)
    
    try:
        # Import necessary modules
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
        
        # Configure dual-arm robots
        arm1_port = "/dev/ttyACM0"
        arm2_port = "/dev/ttyACM1"
        
        print(f"Configuring first arm: {arm1_port}")  
        print(f"Configuring second arm: {arm2_port}")
        
        # Create dual-arm robot instances
        arm1_config = SO100FollowerConfig(port=arm1_port)
        arm2_config = SO100FollowerConfig(port=arm2_port)
        
        arm1_robot = SO100Follower(arm1_config)
        arm2_robot = SO100Follower(arm2_config)
        
        robots = {
            'arm1': arm1_robot,
            'arm2': arm2_robot
        }
        
        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)
        
        # Connect devices
        print("Connecting first arm...")
        arm1_robot.connect()
        print("Connecting second arm...")
        arm2_robot.connect()
        print("Connecting keyboard...")
        keyboard.connect()
        
        print("All devices connected successfully!")
        
        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robots? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration of dual arms...")
                for arm_name, robot in robots.items():
                    print(f"Calibrating {arm_name}...")
                    robot.calibrate()
                print("Dual arm calibration completed!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration files")
                break
            else:
                print("Please enter y or n")
        
        # Read starting joint angles
        print("Reading dual arm starting joint angles...")
        start_positions = {}
        for arm_name, robot in robots.items():
            start_obs = robot.get_observation()
            start_positions[arm_name] = {}
            for key, value in start_obs.items():
                if key.endswith('.pos'):
                    motor_name = key.removesuffix('.pos')
                    start_positions[arm_name][motor_name] = int(value)  # Don't apply calibration coefficients
        
        print("Dual arm starting joint angles:")
        for arm_name, positions in start_positions.items():
            print(f"{arm_name}:")
            for joint_name, position in positions.items():
                print(f"  {joint_name}: {position}°")
        
        # Move to zero position
        move_to_zero_position(robots, duration=3.0)
        
        # Initialize dual arm target positions to current position (integer)
        target_positions = {
            'arm1': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            },
            'arm2': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            }
        }
        
        # Initialize dual arm x,y coordinate control
        x0, y0 = 0.1629, 0.1131
        current_positions = {
            'arm1': {'x': x0, 'y': y0},
            'arm2': {'x': x0, 'y': y0}
        }
        print(f"Initialize dual arm end effector positions: arm1=({x0:.4f}, {y0:.4f}), arm2=({x0:.4f}, {y0:.4f})")
        
        
        print("Dual arm keyboard control instructions:")
        print("First arm control (7y8u9i0o-p=[):")
        print("- 7/y: Joint1 (shoulder_pan) decrease/increase")
        print("- 8/u: Control end effector x coordinate (joint2+3)")
        print("- 9/i: Control end effector y coordinate (joint2+3)")
        print("- =/[: Pitch adjustment increase/decrease (affects wrist_flex)")
        print("- 0/o: Joint5 (wrist_roll) decrease/increase")
        print("- -/p: Joint6 (gripper) decrease/increase")
        print("")
        print("Second arm control (hbjnkml,;.'/):")
        print("- h/b: Joint1 (shoulder_pan) decrease/increase")
        print("- j/n: Control end effector x coordinate (joint2+3)")
        print("- k/m: Control end effector y coordinate (joint2+3)")
        print("- ,/.: Pitch adjustment increase/decrease (affects wrist_flex)")
        print("- ;/l: Joint5 (wrist_roll) decrease/increase")
        print("- '/: Joint6 (gripper) decrease/increase")
        print("")
        print("- X: Exit program (return to start position first)")
        print("- ESC: Exit program")
        print("="*50)
        print("Note: Dual arm robots will continuously move to target positions")
        
        # Start P control loop
        p_control_loop(robots, keyboard, target_positions, start_positions, current_positions, kp=0.5, control_freq=50)
        
        # Disconnect
        for arm_name, robot in robots.items():
            print(f"Disconnecting {arm_name}...")
            robot.disconnect()
        keyboard.disconnect()
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Are the robots properly connected")
        print("2. Are the USB ports correct")
        print("3. Do you have sufficient permissions to access USB devices")
        print("4. Are the robots properly configured")

if __name__ == "__main__":
    main() 