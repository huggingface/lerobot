#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import logging
import math
import time
import traceback
from joyconrobotics import JoyconRobotics


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edited
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
JOINT_CALIBRATION = [
    ["shoulder_pan", 6.0, 1.0],  # Joint 1: zero position offset, scale factor
    ["shoulder_lift", 2.0, 0.97],  # Joint 2: zero position offset, scale factor
    ["elbow_flex", 0.0, 1.05],  # Joint 3: zero position offset, scale factor
    ["wrist_flex", 0.0, 0.94],  # Joint 4: zero position offset, scale factor
    ["wrist_roll", 0.0, 0.5],  # Joint 5: zero position offset, scale factor
    ["gripper", 0.0, 1.0],  # Joint 6: zero position offset, scale factor
]

class FixedAxesJoyconRobotics(JoyconRobotics):
    def common_update(self):
        # 修改后的更新逻辑：摇杆只控制固定轴向
        speed_scale = 0.0008
        # pitch = -self.position[4] * 60 + 20
        # print(f"pitch_ctrl: {pitch}")
        # 垂直摇杆：只控制X轴（前后）
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_0 = 1800
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + joycon_stick_v_0:
            self.position[0] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]
        elif joycon_stick_v < joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]
        
        # 水平摇杆：只控制Y轴（左右）  
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_0 = 2000
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + joycon_stick_h_0:
            self.position[1] += speed_scale * (joycon_stick_h - joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale * (joycon_stick_h - joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z轴只通过按钮控制
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        # 其他按钮控制（复制原来的逻辑）
        joycon_button_xup = self.joycon.get_button_x() if self.joycon.is_right() else self.joycon.get_button_up()
        joycon_button_xback = self.joycon.get_button_b() if self.joycon.is_right() else self.joycon.get_button_down()
        if joycon_button_xup == 1:
            self.position[0] += 0.001 * self.dof_speed[0]
        elif joycon_button_xback == 1:
            self.position[0] -= 0.001 * self.dof_speed[0]
        
        # Home按钮重置逻辑（简化版）
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # 夹爪控制逻辑（复制原来的）
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            elif ((self.joycon.is_right() and event_type == 'zr') or (self.joycon.is_left() and event_type == 'zl')) and not self.change_down_to_gripper:
                self.gripper_toggle_button = status
            elif ((self.joycon.is_right() and event_type == 'stick_r_btn') or (self.joycon.is_left() and event_type == 'stick_l_btn')) and self.change_down_to_gripper:
                self.gripper_toggle_button = status
            else: 
                self.reset_button = 0
            
        if self.gripper_toggle_button == 1 :
            if self.gripper_state == self.gripper_open:
                self.gripper_state = self.gripper_close
            else:
                self.gripper_state = self.gripper_open
            self.gripper_toggle_button = 0

        # 按钮控制状态
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control



def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients

    Args:
        joint_name: joint name
        raw_position: raw position value

    Returns:
        calibrated_position: calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # zero position offset
            scale = joint_cal[2]  # scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # if no calibration coefficient found, return original value


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

    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position

    Args:
        robot: robot instance
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    print("Using P control to slowly move robot to zero position...")

    # Get current robot state
    current_obs = robot.get_observation()

    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith(".pos"):
            motor_name = key.removesuffix(".pos")
            current_positions[motor_name] = value

    # Zero position targets
    zero_positions = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }

    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(
        f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}"
    )

    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
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
        robot: robot instance
        start_positions: start joint position dictionary
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
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

        # Check if reached start position
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break

        time.sleep(control_period)

    print("Return to start position completed")


def p_control_loop(
    robot, keyboard, target_positions, start_positions, current_x, current_y, joyconrobotics_right, kp=0.5, control_freq=50
):
    """
    P control loop

    Args:
        robot: robot instance
        keyboard: keyboard instance
        target_positions: target joint position dictionary
        start_positions: start joint position dictionary
        current_x: current x coordinate
        current_y: current y coordinate
        joyconrobotics_right: joycon robotics instance
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    control_period = 1.0 / control_freq

    # Initialize pitch control variables
    pitch = 0.0  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()

            pose, gripper, control_button = joyconrobotics_right.get_control()
            x, y, z, roll_, pitch_, yaw = pose
            pitch = -pitch_ * 60 + 20
            current_x = 0.1629 + x
            current_y = 0.1131 + z
            roll = roll_ * 50 
            print(f"pitch: {pitch}")
            
            # 添加y值控制shoulder_pan关节
            # y值直接映射到shoulder_pan的目标位置，可以调整缩放因子
            y_scale = 300.0  # 缩放因子，可以根据需要调整
            target_positions["shoulder_pan"] = y * y_scale
            
            # Calculate target angles for joint2 and joint3
            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
            target_positions["shoulder_lift"] = joint2_target
            target_positions["elbow_flex"] = joint3_target 
            # target_positions["shoulder_lift"] = joint2_target + pitch
            # target_positions["elbow_flex"] = joint3_target + pitch
            target_positions["wrist_flex"] = -target_positions["shoulder_lift"] - target_positions["elbow_flex"] + pitch
            target_positions["wrist_roll"] = roll

            if gripper == 1:
                target_positions["gripper"] = 60
            else:
                target_positions["gripper"] = 0
            
            # Get current robot state
            current_obs = robot.get_observation()

            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
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
    print("=" * 50)

    try:
        # Import necessary modules
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
        from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        # Get port
        port = input("Please enter the USB port for SO100 robot (e.g., /dev/ttyACM0): ").strip()

        # If directly press Enter, use default port
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
        # 使用修改后的控制类
        joyconrobotics_right = FixedAxesJoyconRobotics(
            "right",
            dof_speed=[2, 2, 2, 1, 1, 1]
        )

        print("固定轴向控制测试:")
        print("垂直摇杆: 只控制X轴（前后）")  
        print("水平摇杆: 只控制Y轴（左右）")
        print("R按钮: Z轴上升")
        print("摇杆按钮: Z轴下降")
        print("Home按钮: 重置位置")
        print("ZR按钮: 切换夹爪")
        print("按Ctrl+C停止")
        print()


        print("Device connection successful!")

        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ["y", "yes"]:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ["n", "no"]:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")

        # Read initial joint angles
        print("Reading initial joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients

        print("Initial joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")

        # Move to zero position
        move_to_zero_position(robot, duration=3.0)

        # Initialize target positions as current positions (integers)
        target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        # Initialize x,y coordinate control
        x0, y0 = 0.1629, 0.1131
        current_x, current_y = x0, y0
        print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}")

        print("Keyboard control instructions:")
        print("- Q/A: Joint 1 (shoulder_pan) decrease/increase")
        print("- W/S: Control end effector x coordinate (joint2+3)")
        print("- E/D: Control end effector y coordinate (joint2+3)")
        print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
        print("- T/G: Joint 5 (wrist_roll) decrease/increase")
        print("- Y/H: Joint 6 (gripper) decrease/increase")
        print("- X: Exit program (return to start position first)")
        print("- ESC: Exit program")
        print("=" * 50)
        print("Note: Robot will continuously move to target positions")

        # Start P control loop
        p_control_loop(
            robot, keyboard, target_positions, start_positions, current_x, current_y, joyconrobotics_right, kp=0.5, control_freq=50
        )

        # Disconnect
        robot.disconnect()
        keyboard.disconnect()
        print("Program ended")

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Whether the robot is properly connected")
        print("2. Whether the USB port is correct")
        print("3. Whether you have sufficient permissions to access USB devices")
        print("4. Whether the robot is properly configured")


if __name__ == "__main__":
    main()
