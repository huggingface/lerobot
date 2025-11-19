# To Run on the host
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host --robot.id=my_xlerobot_2wheels
'''

# To Run the teleop:
'''
PYTHONPATH=src python -m examples.xlerobot_2wheels.teleoperate_joycon
'''

# Base speed control instructions:
# - When holding any base control button (X forward, B backward, Y left turn, A right turn), speed will linearly accelerate to maximum speed
# - After releasing the button, speed will linearly decelerate to 0
# - You can adjust the acceleration and deceleration slopes by modifying the following parameters:
#   * BASE_ACCELERATION_RATE: acceleration slope (speed/second)
#   * BASE_DECELERATION_RATE: deceleration slope (speed/second)
#   * BASE_MAX_SPEED: maximum speed multiplier

import time
import numpy as np
import math

from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig, XLerobot2Wheels
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics
from joyconrobotics import JoyconRobotics

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        
        # Set different center values for left and right Joy-Cons
        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:  # left Joy-Con
            self.joycon_stick_v_0 = 2300
            self.joycon_stick_h_0 = 2000
        
        # Gripper control related variables
        self.gripper_speed = 0.4  # Gripper open/close speed (degrees/frame)
        self.gripper_direction = 1  # 1 means open, -1 means close
        self.gripper_min = 0  # Minimum angle (fully closed)
        self.gripper_max = 90  # Maximum angle (fully open)
        self.last_gripper_button_state = 0  # Record previous frame button state for detecting press events
    
    def common_update(self):
        # Modified update logic: joystick only controls fixed axes
        speed_scale = 0.001
        
        # Get current orientation data to print pitch
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        
        # Vertical joystick: controls X and Z axes (forward/backward)
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        
        # Horizontal joystick: only controls Y axis (left/right)
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z-axis button control
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        # Home button reset logic (simplified version)
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # Gripper control logic (hold for linear increase/decrease mode)
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            else: 
                self.reset_button = 0
        
        # Gripper button state detection and direction control
        gripper_button_pressed = False
        if self.joycon.is_right():
            # Right Joy-Con uses ZR button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            # Left Joy-Con uses ZL button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1
        
        # Detect button press events (from 0 to 1) to change direction
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            # Button just pressed, change direction
            self.gripper_direction *= -1
            print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
        
        # Update button state record
        self.last_gripper_button_state = gripper_button_pressed
        
        # Linear control of gripper open/close when holding gripper button
        if gripper_button_pressed:
            # Check if exceeding limits
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            
            # If exceeding limits, stop moving
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state
            # If exceeding limits, stay at current position, don't change direction

        

        # Button control state
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
    
class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        
        # Initialize smooth controller for arm joints
        self.smooth_controller = SmoothArmController()
        
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Set step size
        self.degree_step = 2
        self.xy_step = 0.005
        
        # P control target positions, set to zero position
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        
        # Reset kinematics variables to initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon_pose, gripper_state):
        """Handle Joy-Con input, update arm control - based on 6_so100_joycon_ee_control.py"""
        x, y, z, roll_, pitch_, yaw = joycon_pose
        
        # Calculate pitch control - consistent with 6_so100_joycon_ee_control.py
        pitch = -pitch_ * 60 + 10
        
        # Set coordinates - consistent with 6_so100_joycon_ee_control.py
        current_x = 0.1629 + x
        current_y = 0.1131 + z
        
        # Calculate roll - consistent with 6_so100_joycon_ee_control.py
        roll = roll_ * 45
        
        print(f"[{self.prefix}] pitch: {pitch}")
        
        # Add y value to control shoulder_pan joint - consistent with 6_so100_joycon_ee_control.py
        y_scale = 250.0  # Scaling factor, can be adjusted as needed
        self.target_positions["shoulder_pan"] = y * y_scale
        
        # Use inverse kinematics to calculate joint angles - consistent with 6_so100_joycon_ee_control.py
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as e:
            print(f"[{self.prefix}] IK failed: {e}")
        
        # Set wrist_flex - consistent with 6_so100_joycon_ee_control.py
        self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + pitch
        
        # Set wrist_roll - consistent with 6_so100_joycon_ee_control.py
        self.target_positions["wrist_roll"] = roll
        
        # Gripper control - now set directly in main loop, no need to handle here
        pass

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        
        # Apply smooth control to the first three joints
        smoothed_positions = self.smooth_controller.update(self.target_positions, current)
        
        action = {}
        for j in self.target_positions:
            if j in ["shoulder_pan", "shoulder_lift", "elbow_flex"]:
                # Use smoothed positions for the first three joints
                error = smoothed_positions[j] - current[j]
            else:
                # Use direct control for other joints (wrist_flex, wrist_roll, gripper)
                error = self.target_positions[j] - current[j]
            
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # Move 2 degrees each time
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        print(f"[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon):
        """Handle left Joy-Con directional pad input to control head motors"""
        # Get left Joy-Con directional pad state
        button_up = joycon.joycon.get_button_up()      # Up: head_motor_1+
        button_down = joycon.joycon.get_button_down()  # Down: head_motor_1-
        button_left = joycon.joycon.get_button_left()  # Left: head_motor_2+
        button_right = joycon.joycon.get_button_right() # Right: head_motor_2-
        
        if button_up == 1:
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_down == 1:
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_left == 1:
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if button_right == 1:
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

def get_joycon_base_action(joycon, robot):
    """
    Get base control commands from Joy-Con for differential drive
    X: left turn, B: right turn, Y: forward, A: backward
    """
    # Get button states
    button_x = joycon.joycon.get_button_x()  # left turn
    button_b = joycon.joycon.get_button_b()  # right turn
    button_y = joycon.joycon.get_button_y()  # forward
    button_a = joycon.joycon.get_button_a()  # backward
    
    # Build key set (simulate keyboard input)
    pressed_keys = set()
    
    if button_x == 1:
        pressed_keys.add(robot.teleop_keys['rotate_left'])  # left turn
        print("[BASE] Left turn")
    if button_b == 1:
        pressed_keys.add(robot.teleop_keys['rotate_right'])  # right turn
        print("[BASE] Right turn")
    if button_y == 1:
        pressed_keys.add(robot.teleop_keys['forward'])  # forward
        print("[BASE] Forward")
    if button_a == 1:
        pressed_keys.add(robot.teleop_keys['backward'])  # backward
        print("[BASE] Backward")
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action

# Base speed control parameters - adjustable slopes
BASE_ACCELERATION_RATE = 10.0  # acceleration slope (speed/second)
BASE_DECELERATION_RATE = 10.0  # deceleration slope (speed/second)
BASE_MAX_SPEED = 5.0          # maximum speed multiplier
MIN_VELOCITY_THRESHOLD = 0.02 # minimum velocity to send to motors during deceleration

# Arm smooth control parameters - adjustable slopes
ARM_ACCELERATION_RATE = 5.0   # acceleration slope (degrees/second)
ARM_DECELERATION_RATE = 8.0   # deceleration slope (degrees/second)
ARM_MAX_SPEED = 2.0           # maximum speed multiplier
ARM_MIN_VELOCITY_THRESHOLD = 0.1 # minimum velocity to send to motors during deceleration

class SmoothBaseController:
    """Simplified smooth base controller with acceleration/deceleration for differential drive"""
    
    def __init__(self):
        self.current_speed = 0.0
        self.last_time = time.time()
        self.last_direction = {"x.vel": 0.0, "theta.vel": 0.0}
        self.is_moving = False
    
    def update(self, pressed_keys, robot):
        """Update smooth control and return base action"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Check if any base keys are pressed
        base_keys = [
            robot.teleop_keys['forward'],
            robot.teleop_keys['backward'], 
            robot.teleop_keys['rotate_left'],
            robot.teleop_keys['rotate_right']
        ]
        any_key_pressed = any(key in pressed_keys for key in base_keys)
        
        # Calculate base action directly (bypass robot's built-in speed control)
        base_action = {"x.vel": 0.0, "theta.vel": 0.0}
        
        if any_key_pressed:
            # Keys pressed - calculate direction and accelerate
            if not self.is_moving:
                self.is_moving = True
                print("[BASE] Starting acceleration")
            
            # Get current speed level from robot
            speed_setting = robot.speed_levels[robot.speed_index]
            linear_speed = speed_setting["linear"]  # e.g. 0.1, 0.2, or 0.3
            angular_speed = speed_setting["angular"]  # e.g. 30, 60, or 90
            
            # Calculate direction based on pressed keys
            if robot.teleop_keys["forward"] in pressed_keys:
                base_action["x.vel"] += linear_speed
            if robot.teleop_keys["backward"] in pressed_keys:
                base_action["x.vel"] -= linear_speed
            if robot.teleop_keys["rotate_left"] in pressed_keys:
                base_action["theta.vel"] += angular_speed
            if robot.teleop_keys["rotate_right"] in pressed_keys:
                base_action["theta.vel"] -= angular_speed
            
            # Store current direction for deceleration
            self.last_direction = base_action.copy()
            
            # Accelerate
            self.current_speed += BASE_ACCELERATION_RATE * dt
            self.current_speed = min(self.current_speed, BASE_MAX_SPEED)
                
        else:
            # No keys pressed - decelerate
            if self.is_moving:
                self.is_moving = False
                print("[BASE] Starting deceleration")
            
            # Use last direction for deceleration
            if self.current_speed > 0.01 and self.last_direction:
                base_action = self.last_direction.copy()
            
            # Decelerate
            self.current_speed -= BASE_DECELERATION_RATE * dt
            self.current_speed = max(self.current_speed, 0.0)
        
        # Apply speed multiplier
        if base_action:
            for key in base_action:
                if 'vel' in key:
                    original_value = base_action[key]
                    base_action[key] *= self.current_speed
                    
                    # Ensure minimum velocity during deceleration to prevent motor cutoff
                    if self.current_speed > 0.01 and abs(base_action[key]) < MIN_VELOCITY_THRESHOLD:
                        # During deceleration, maintain minimum velocity to keep motors moving
                        base_action[key] = MIN_VELOCITY_THRESHOLD if original_value > 0 else -MIN_VELOCITY_THRESHOLD
        
        # Debug output
        if any_key_pressed:
            print(f"[BASE] ACCEL: Speed={self.current_speed:.2f}, Action={base_action}")
        elif self.current_speed > 0.01:
            print(f"[BASE] DECEL: Speed={self.current_speed:.2f}, Action={base_action}")
        elif self.current_speed <= 0.01:
            print(f"[BASE] STOPPED: Speed={self.current_speed:.2f}")
        
        return base_action

class SmoothArmController:
    """Smooth arm controller with acceleration/deceleration for the first three joints"""
    
    def __init__(self):
        self.current_speeds = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0
        }
        self.last_time = time.time()
        self.last_directions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0
        }
        self.is_moving = {
            "shoulder_pan": False,
            "shoulder_lift": False,
            "elbow_flex": False
        }
    
    def update(self, target_positions, current_positions):
        """Update smooth control and return smoothed target positions"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        smoothed_positions = {}
        
        for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex"]:
            target = target_positions.get(joint, 0.0)
            current = current_positions.get(joint, 0.0)
            
            # Calculate direction and magnitude of movement needed
            error = target - current
            abs_error = abs(error)
            
            if abs_error > 0.1:  # Only move if error is significant
                # Determine direction
                direction = 1.0 if error > 0 else -1.0
                
                # Check if we're starting to move
                if not self.is_moving[joint]:
                    self.is_moving[joint] = True
                    print(f"[ARM] Starting {joint} movement")
                
                # Store current direction for deceleration
                self.last_directions[joint] = direction
                
                # Accelerate
                self.current_speeds[joint] += ARM_ACCELERATION_RATE * dt
                self.current_speeds[joint] = min(self.current_speeds[joint], ARM_MAX_SPEED)
                
                # Calculate movement step
                movement_step = self.current_speeds[joint] * dt * direction
                
                # Apply movement
                smoothed_positions[joint] = current + movement_step
                
            else:
                # No significant error - decelerate
                if self.is_moving[joint]:
                    self.is_moving[joint] = False
                    print(f"[ARM] Starting {joint} deceleration")
                
                # Use last direction for deceleration
                if self.current_speeds[joint] > 0.01 and self.last_directions[joint] != 0:
                    direction = self.last_directions[joint]
                    movement_step = self.current_speeds[joint] * dt * direction
                    
                    # Ensure minimum velocity during deceleration
                    if abs(movement_step) < ARM_MIN_VELOCITY_THRESHOLD:
                        movement_step = ARM_MIN_VELOCITY_THRESHOLD if direction > 0 else -ARM_MIN_VELOCITY_THRESHOLD
                    
                    smoothed_positions[joint] = current + movement_step
                else:
                    smoothed_positions[joint] = current
                
                # Decelerate
                self.current_speeds[joint] -= ARM_DECELERATION_RATE * dt
                self.current_speeds[joint] = max(self.current_speeds[joint], 0.0)
        
        return smoothed_positions

# Global smooth controller instances
smooth_controller = SmoothBaseController()

def main():
    FPS = 30
    
    # Try to use saved calibration file to avoid recalibrating each time
    # You can modify robot_id here to match your robot configuration
    robot_config = XLerobot2WheelsConfig(id="my_xlerobot_2wheels_lab")  # Can be modified to your robot ID
    robot = XLerobot2Wheels(robot_config)
    
    try:
        robot.connect()
        print(f"[MAIN] Successfully connected to robot")
        if robot.is_calibrated:
            print(f"[MAIN] Robot is calibrated and ready to use!")
        else:
            print(f"[MAIN] Robot requires calibration")
    except Exception as e:
        print(f"[MAIN] Failed to connect to robot: {e}")
        print(f"[MAIN] Robot config: {robot_config}")
        print(f"[MAIN] Robot: {robot}")
        return

    init_rerun(session_name="xlerobot_2wheels_teleop_joycon")

    # Initialize right Joy-Con controller - based on 6_so100_joycon_ee_control.py
    print("[MAIN] Initializing right Joy-Con controller...")
    joycon_right = FixedAxesJoyconRobotics(
        "right",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    print(f"[MAIN] Right Joy-Con controller connected")
    print("[MAIN] Initializing left Joy-Con controller...")
    joycon_left = FixedAxesJoyconRobotics(
        "left",
        dof_speed=[2, 2, 2, 1, 1, 1]
    )
    print(f"[MAIN] Left Joy-Con controller connected")

    # Init the arm and head instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, prefix="left")
    right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, prefix="right")
    head_control = SimpleHeadControl(obs)

    # Move both arms and head to zero position at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)

    # Print comprehensive keymap information based on robot config
    print("\n" + "="*80)
    print("🤖 XLeRobot 2Wheels Joy-Con Control Instructions")
    print("="*80)
    
    print("\n📱 Base Control (Differential Drive):")
    print(f"    X: Rotate Left")
    print(f"    B: Rotate Right") 
    print(f"    Y: Forward")
    print(f"    A: Backward")
    print("    🚀 Smooth Control: Linear acceleration when holding, linear deceleration when released")
    
    print("\n🦾 Right Arm Control:")
    print("   Joystick Control:")
    print("    - Vertical Joystick: X-axis and Z-axis movement (forward/backward)")
    print("    - Horizontal Joystick: Y-axis movement (left/right)")
    print("    - R Button: Z-axis up")
    print("    - Right Stick Button: Z-axis down")
    print("   Gripper Control:")
    print("    - ZR Button: Hold to toggle open/close direction, continue holding for linear open/close")
    print("    - D-pad: Control head motors")
    print("   🚀 Smooth Control: Linear acceleration/deceleration for shoulder_pan, shoulder_lift, elbow_flex")
    
    print("\n🦾 Left Arm Control:")
    print("   Joystick Control:")
    print("    - Vertical Joystick: X-axis and Z-axis movement (forward/backward)")
    print("    - Horizontal Joystick: Y-axis movement (left/right)")
    print("    - L Button: Z-axis up")
    print("    - Left Stick Button: Z-axis down")
    print("   Gripper Control:")
    print("    - ZL Button: Hold to toggle open/close direction, continue holding for linear open/close")
    print("    - D-pad: Control head motors")
    print("   🚀 Smooth Control: Linear acceleration/deceleration for shoulder_pan, shoulder_lift, elbow_flex")
    
    print("\n👁️ Head Control:")
    print("   Left Joy-Con D-pad:")
    print("    - Up/Down: Head Motor 2 +/-")
    print("    - Left/Right: Head Motor 1 +/-")
    
    print(f"\n⚙️ Robot Configuration:")
    print(f"   Wheel Radius: {robot.config.wheel_radius:.3f}m")
    print(f"   Wheelbase: {robot.config.wheelbase:.3f}m")
    print(f"   Speed Levels: {len(robot.speed_levels)} levels")
    for i, level in enumerate(robot.speed_levels):
        print(f"      Level {i+1}: Linear {level['linear']:.1f}m/s, Angular {level['angular']:.0f}°/s")
    
    print(f"\n🚀 Smooth Control Parameters:")
    print(f"   Base Control:")
    print(f"     Acceleration Rate: {BASE_ACCELERATION_RATE:.1f} speed/second")
    print(f"     Deceleration Rate: {BASE_DECELERATION_RATE:.1f} speed/second")
    print(f"     Max Speed Multiplier: {BASE_MAX_SPEED:.1f}x")
    print(f"   Arm Control (shoulder_pan, shoulder_lift, elbow_flex):")
    print(f"     Acceleration Rate: {ARM_ACCELERATION_RATE:.1f} degrees/second")
    print(f"     Deceleration Rate: {ARM_DECELERATION_RATE:.1f} degrees/second")
    print(f"     Max Speed Multiplier: {ARM_MAX_SPEED:.1f}x")
    
    print("\n" + "="*80)
    print("🎮 Control started! Use Joy-Con to control robot")
    print("="*80 + "\n")

    try:
        while True:
            pose_right, gripper_right, control_button_right = joycon_right.get_control()
            print(f"pose_right: {pose_right}, gripper_right: {gripper_right}, control_button_right: {control_button_right}")
            pose_left, gripper_left, control_button_left = joycon_left.get_control()
            print(f"pose_left: {pose_left}, gripper_left: {gripper_left}, control_button_left: {control_button_left}")

            if control_button_right == 8:  # reset button
                print("[MAIN] Reset to zero position!")
                right_arm.move_to_zero_position(robot)
                left_arm.move_to_zero_position(robot)
                head_control.move_to_zero_position(robot)
                continue

            # Handle gripper control - directly use Joy-Con gripper state
            right_arm.target_positions["gripper"] = gripper_right
            left_arm.target_positions["gripper"] = gripper_left
            
            right_arm.handle_joycon_input(pose_right, gripper_right)
            right_action = right_arm.p_control_action(robot)
            left_arm.handle_joycon_input(pose_left, gripper_left)
            left_action = left_arm.p_control_action(robot)
            head_control.handle_joycon_input(joycon_left) # Pass joycon_left to head_control
            head_action = head_control.p_control_action(robot)

            # Get base action from Joy-Con buttons
            base_action = get_joycon_base_action(joycon_right, robot)
            
            # Apply smooth speed control to base action
            pressed_keys = set()
            if joycon_right.joycon.get_button_x() == 1:
                pressed_keys.add(robot.teleop_keys['rotate_left'])
            if joycon_right.joycon.get_button_b() == 1:
                pressed_keys.add(robot.teleop_keys['rotate_right'])
            if joycon_right.joycon.get_button_y() == 1:
                pressed_keys.add(robot.teleop_keys['forward'])
            if joycon_right.joycon.get_button_a() == 1:
                pressed_keys.add(robot.teleop_keys['backward'])
            
            # Get smooth base action with linear acceleration/deceleration
            smooth_base_action = smooth_controller.update(pressed_keys, robot)

            # Merge all actions
            action = {**left_action, **right_action, **head_action, **smooth_base_action}
            robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(obs, action)
    finally:
        joycon_right.disconnect()
        joycon_left.disconnect()
        robot.disconnect()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
