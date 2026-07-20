#!/usr/bin/env python3
"""
VR control for XLerobot robot
Uses handle_vr_input with delta action control
"""

# Standard library imports
import asyncio
import logging
import math
import sys
import threading
import time
import traceback
import queue

# Third-party imports
import numpy as np

# Local imports
from XLeVR.vr_monitor import VRMonitor
from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint mapping configurations
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

# --- Minimal flags to disable one whole hand and head teleop ---
# Set to False to disable left-hand teleop and head teleop.
# Re-enable by setting to True.
ENABLE_LEFT_HAND = False
ENABLE_HEAD = False

EPISODE_LEN = 450  # Number of steps per episode
RESET_LEN = 150
NR_OF_EPISODES = 110
TASK = "Grab the cup"
MAIN_CAMERA_INDEX = "/dev/ttyACM0" 
#RIGHT_ARM_CAMERA_INDEX = "/dev/video2"
LEFT_ARM_CAMERA_INDEX = "/dev/video2"
DATASET_REPO = "Grigorij/XLeRobot_arms_5"
FPS = 30

class SimpleTeleopArm:
    """
    A class for controlling a robot arm using VR input with delta action control.
    
    This class provides inverse kinematics-based arm control with proportional control
    for smooth movement and gripper operations based on VR controller input.
    """
    
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        
        # Initial joint positions - adapted for XLerobot observation format
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
        
        # Delta control state variables for VR input
        self.last_vr_time = 0.0
        self.vr_deadzone = 0.001  # Minimum movement threshold
        self.max_delta_per_frame = 0.005  # Maximum position change per frame
        
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
        
        # Reset delta control state
        self.last_vr_time = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_vr_input(self, vr_goal, gripper_state):
        """
        Handle VR input with delta action control - incremental position updates.
        
        Args:
            vr_goal: VR controller goal data containing target position and orientations
            gripper_state: Current gripper state (not used in current implementation)
        """
        if vr_goal is None:
            return
        
        # VR goal contains: target_position [x, y, z], wrist_roll_deg, wrist_flex_deg, gripper_closed
        if not hasattr(vr_goal, 'target_position') or vr_goal.target_position is None:
            return
            
        # Extract VR position data
        # Get current VR position
        current_vr_pos = vr_goal.target_position  # [x, y, z] in meters
        
        # Initialize previous VR position if not set
        if not hasattr(self, 'prev_vr_pos'):
            self.prev_vr_pos = current_vr_pos
            return  # Skip first frame to establish baseline
        
        # Calculate relative change (delta) from previous frame
        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * 220 # Scale for the shoulder
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * 70 
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * 70

        # print(f'vr_x: {vr_x}, vr_y: {vr_y}, vr_z: {vr_z}')

        # Update previous position for next frame
        self.prev_vr_pos = current_vr_pos
        
        # Delta control parameters - adjust these for sensitivity
        pos_scale = 0.01  # Position sensitivity scaling
        angle_scale = 4.0  # Angle sensitivity scaling
        delta_limit = 0.01  # Maximum delta per update (meters)
        angle_limit = 8.0  # Maximum angle delta per update (degrees)
        
        delta_x = vr_x * pos_scale
        delta_y = vr_y * pos_scale  
        delta_z = vr_z * pos_scale
        
        # Limit delta values to prevent sudden movements
        delta_x = max(-delta_limit, min(delta_limit, delta_x))
        delta_y = max(-delta_limit, min(delta_limit, delta_y))
        delta_z = max(-delta_limit, min(delta_limit, delta_z))
        
        self.current_x += -delta_z  # yy: VR Z maps to robot x, change the direction
        self.current_y += delta_y  # yy:VR Y maps to robot y

        # Handle wrist angles with delta control - use relative changes
        if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
            # Initialize previous wrist_flex if not set
            if not hasattr(self, 'prev_wrist_flex'):
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
                return
            
            # Calculate relative change from previous frame
            delta_pitch = (vr_goal.wrist_flex_deg - self.prev_wrist_flex) * angle_scale
            delta_pitch = max(-angle_limit, min(angle_limit, delta_pitch))
            self.pitch += delta_pitch
            self.pitch = max(-90, min(90, self.pitch))  # Limit pitch range
            
            # Update previous value for next frame
            self.prev_wrist_flex = vr_goal.wrist_flex_deg
        
        if hasattr(vr_goal, 'wrist_roll_deg') and vr_goal.wrist_roll_deg is not None:
            # Initialize previous wrist_roll if not set
            if not hasattr(self, 'prev_wrist_roll'):
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
                return
            
            delta_roll = (vr_goal.wrist_roll_deg - self.prev_wrist_roll) * angle_scale
            delta_roll = max(-angle_limit, min(angle_limit, delta_roll))
            
            current_roll = self.target_positions.get("wrist_roll", 0.0)
            new_roll = current_roll + delta_roll
            new_roll = max(-90, min(90, new_roll))  # Limit roll range
            self.target_positions["wrist_roll"] = new_roll
            
            # Update previous value for next frame
            self.prev_wrist_roll = vr_goal.wrist_roll_deg
        
        # VR Z axis controls shoulder_pan joint (delta control)
        if abs(delta_x) > 0.001:  # Only update if significant movement
            x_scale = 200.0  # Reduced scaling factor for delta control
            delta_pan = delta_x * x_scale
            delta_pan = max(-angle_limit, min(angle_limit, delta_pan))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            new_pan = current_pan + delta_pan
            new_pan = max(-180, min(180, new_pan))  # Limit pan range
            self.target_positions["shoulder_pan"] = new_pan
        
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            # Smooth transition to new joint positions,  Smoothing factor 0-1, lower = smoother
            alpha = 0.1
            self.target_positions["shoulder_lift"] = (1-alpha) * self.target_positions.get("shoulder_lift", 0.0) + alpha * joint2_target
            self.target_positions["elbow_flex"] = (1-alpha) * self.target_positions.get("elbow_flex", 0.0) + alpha * joint3_target
        except Exception as e:
            print(f"[{self.prefix}] VR IK failed: {e}")
        
        # Calculate wrist_flex to maintain end-effector orientation
        self.target_positions["wrist_flex"] = (-self.target_positions["shoulder_lift"] - 
                                               self.target_positions["elbow_flex"] + self.pitch)
   
        # Handle gripper state directly
        if vr_goal.metadata.get('trigger', 0) > 0.5:
            self.target_positions["gripper"] = 45
        else:
            self.target_positions["gripper"] = 0.0

    def p_control_action(self, robot):
        """
        Generate proportional control action based on target positions.
        
        Args:
            robot: Robot instance to get current observations
            
        Returns:
            dict: Action dictionary with position commands for each joint
        """
        if self.prefix=="left":
            obs_raw = robot.bus_1.sync_read("Present_Position", robot.left_arm_motors)
        else:
            obs_raw = robot.bus_2.sync_read("Present_Position", robot.right_arm_motors)

        obs_pos_suffix = {f"{v}.pos": obs_raw[v] for v in self.joint_map.values()}
        current = {k: obs_raw[v] for k, v in self.joint_map.items()}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action, obs_pos_suffix


class SimpleHeadControl:
    """
    A class for controlling robot head motors using VR thumbstick input.
    
    Provides simple head movement control with proportional control for smooth operation.
    """
    
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # Move 2 degrees each time
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def handle_vr_input(self, vr_goal):
        # Map VR input to head motor targets
        thumb = vr_goal.metadata.get('thumbstick', {})
        if thumb:
            thumb_x = thumb.get('x', 0)
            thumb_y = thumb.get('y', 0)
            if abs(thumb_x) > 0.1:
                if thumb_x > 0:
                    self.target_positions["head_motor_1"] += self.degree_step
                else:
                    self.target_positions["head_motor_1"] -= self.degree_step
            if abs(thumb_y) > 0.1:
                if thumb_y > 0:
                    self.target_positions["head_motor_2"] += self.degree_step
                else:
                    self.target_positions["head_motor_2"] -= self.degree_step
                    
    def move_to_zero_position(self, robot):
        print(f"[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def p_control_action(self, robot):
        """
        Generate proportional control action for head motors.
        
        Args:
            robot: Robot instance to get current observations
            
        Returns:
            dict: Action dictionary with position commands for head motors
        """
        obs_raw = robot.bus_1.sync_read("Present_Position", robot.head_motors)
        action = {}
        for motor in self.target_positions:
            current = obs_raw.get(HEAD_MOTOR_MAP[motor], 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action


def get_vr_base_action(vr_goal, robot):
    """
    Get base control commands from VR input.
    
    Args:
        vr_goal: VR controller goal data containing metadata
        robot: Robot instance for action conversion
        
    Returns:
        dict: Base movement actions based on VR thumbstick input
    """
    if vr_goal is None or not hasattr(vr_goal, 'metadata'):
        return {}
    
    # Build key set based on VR input (you can customize this mapping)
    pressed_keys = set()
    
    # Example VR to base movement mapping - adjust according to your VR system
    # You may need to customize these mappings based on your VR controller buttons
    thumb = vr_goal.metadata.get('thumbstick', {})
    if thumb:
        thumb_x = thumb.get('x', 0)
        thumb_y = thumb.get('y', 0)
        if abs(thumb_x) > 0.2:
            if thumb_x > 0:
                pressed_keys.add('o')  # Move backward
            else:
                pressed_keys.add('u')  # Move forward
        if abs(thumb_y) > 0.2:
            if thumb_y > 0:
                pressed_keys.add('k')  # Move right
            else:
                pressed_keys.add('i')  # Move backward
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action


# Base speed control parameters - adjustable slopes
BASE_ACCELERATION_RATE = 2.0/2  # acceleration slope (speed/second)
BASE_DECELERATION_RATE = 2.5/2    # deceleration slope (speed/second)
BASE_MAX_SPEED = 3.0/2          # maximum speed multiplier


def get_vr_speed_control(vr_goal):
    """
    Get speed control from VR input with linear acceleration and deceleration.
    
    Linearly accelerates to maximum speed when holding any base control input,
    and linearly decelerates to 0 when released.
    
    Args:
        vr_goal: VR controller goal data
        
    Returns:
        float: Current speed multiplier (0.0 to BASE_MAX_SPEED)
    """
    global current_base_speed, last_update_time, is_accelerating
    
    # Initialize global variables
    if 'current_base_speed' not in globals():
        current_base_speed = 0.0
        last_update_time = time.time()
        is_accelerating = False
    
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time
    
    # Check if any base control input is active from VR
    any_base_input_active = False
    if vr_goal and hasattr(vr_goal, 'metadata'):
        thumb = vr_goal.metadata.get('thumbstick', {})
        if thumb:
            thumb_x = thumb.get('x', 0)
            thumb_y = thumb.get('y', 0)
            # Check if thumbstick is being used for base movement
            any_base_input_active = abs(thumb_x) > 0.2 or abs(thumb_y) > 0.2
    
    if any_base_input_active:
        # VR input active - accelerate
        if not is_accelerating:
            is_accelerating = True
            print("[BASE] Starting acceleration")
        
        # Linear acceleration
        current_base_speed += BASE_ACCELERATION_RATE * dt
        current_base_speed = min(current_base_speed, BASE_MAX_SPEED)
        
    else:
        # No VR input - decelerate
        if is_accelerating:
            is_accelerating = False
            print("[BASE] Starting deceleration")
        
        # Linear deceleration
        current_base_speed -= BASE_DECELERATION_RATE * dt
        current_base_speed = max(current_base_speed, 0.0)
    
    # Print current speed (optional, for debugging)
    if abs(current_base_speed) > 0.01:  # Only print when speed is not 0
        print(f"[BASE] Current speed: {current_base_speed:.2f}")
    
    return current_base_speed


def init_dataset(robot):
    if ENABLE_LEFT_HAND:
        features = {
            "action": {
                "dtype": "float32",
                "shape": (12,),
                "names": [
                    "left_arm_shoulder_pan.pos", "left_arm_shoulder_lift.pos", "left_arm_elbow_flex.pos", 
                    "left_arm_wrist_flex.pos", "left_arm_wrist_roll.pos", "left_arm_gripper.pos",
                    "right_arm_shoulder_pan.pos", "right_arm_shoulder_lift.pos", "right_arm_elbow_flex.pos", 
                    "right_arm_wrist_flex.pos", "right_arm_wrist_roll.pos", "right_arm_gripper.pos",]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (12,),
                "names": [
                    "left_arm_shoulder_pan.pos", "left_arm_shoulder_lift.pos", "left_arm_elbow_flex.pos", 
                    "left_arm_wrist_flex.pos", "left_arm_wrist_roll.pos", "left_arm_gripper.pos",
                    "right_arm_shoulder_pan.pos", "right_arm_shoulder_lift.pos", "right_arm_elbow_flex.pos", 
                    "right_arm_wrist_flex.pos", "right_arm_wrist_roll.pos", "right_arm_gripper.pos",]
            },
        }
    else:
        features = {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "right_arm_shoulder_pan.pos", "right_arm_shoulder_lift.pos", "right_arm_elbow_flex.pos", 
                    "right_arm_wrist_flex.pos", "right_arm_wrist_roll.pos", "right_arm_gripper.pos",]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "right_arm_shoulder_pan.pos", "right_arm_shoulder_lift.pos", "right_arm_elbow_flex.pos", 
                    "right_arm_wrist_flex.pos", "right_arm_wrist_roll.pos", "right_arm_gripper.pos",]
            },
        }
    camera_features = hw_to_dataset_features(robot._cameras_ft, OBS_STR)
    features = {**features, **camera_features}
    import time
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO,
        root=f"my_dataset_{time.time()}",
        features=features,
        fps=FPS,
        image_writer_processes=10,
        image_writer_threads=5,
    )
    return dataset



def saving_dataset_worker(dataset, frame_queue, shutdown_event, saving_in_progress_event):
    """
    Worker function to save dataset frames in the background.
    
    Continuously checks for new dataset frames and saves them to disk.
    """
    try:
        # save videos into separate files to avoid concatenation problems
        dataset.meta.update_chunk_settings(video_files_size_in_mb=0.001)
        recording_dataset = True
        frame_nr = 0
        episode = 0
        while not shutdown_event.is_set():
            try:
                lerobot_frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
    
            if recording_dataset:
                dataset.add_frame(lerobot_frame)
                print(f"step {frame_nr} added to dataset")

            frame_nr += 1

            if frame_nr == EPISODE_LEN:
                print(f"Finishing episode {episode}, reset...")
                saving_in_progress_event.set()
                dataset.save_episode()
                dataset.image_writer.wait_until_done()
                #recording_dataset = False
                saving_in_progress_event.clear()
                #recording_dataset = True
                frame_nr = 0
                episode += 1
                if episode >= NR_OF_EPISODES:
                    print("Reached maximum number of episodes. Exiting...")
                    shutdown_event.set()
                    break
                print(f"Starting episode {episode} recording...")
   
    except Exception as e:
        logger.error(f"Error in dataset saving worker: {e}", exc_info=True)
    finally:
        if dataset:
            dataset.image_writer.wait_until_done()
            dataset.save_episode()
            dataset.push_to_hub()

def main():
    """
    Main function for VR teleoperation of XLerobot.
    
    Initializes the robot connection, VR monitoring, and runs the main control loop
    for dual-arm robot control with VR input.
    """
    print("XLerobot VR Control Example")
    print("="*50)

    shutdown_event = threading.Event()
    saving_in_progress_event = threading.Event()

    robot, vr_monitor, dataset_saving_thread = None, None, None

    try:
        # Try to use saved calibration file to avoid recalibrating each time
        # You can modify robot_id here to match your robot configuration
        robot_config = XLerobotConfig()  # Can be modified to your robot ID
        robot = XLerobot(robot_config)
        
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
        
        # Initialize VR monitor
        print("🔧 Initializing VR monitor...")
        vr_monitor = VRMonitor()
        if not vr_monitor.initialize():
            print("❌ VR monitor initialization failed")
            return
        print("🚀 Starting VR monitoring...")
        vr_thread = threading.Thread(target=lambda: asyncio.run(vr_monitor.start_monitoring()), daemon=True)
        vr_thread.start()
        print("✅ VR system ready")

        # Init the arm and head instances
        obs = robot.get_observation()
        kin_left = SO101Kinematics()
        kin_right = SO101Kinematics()
        left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, prefix="left") if ENABLE_LEFT_HAND else None
        right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, prefix="right")
        head_control = SimpleHeadControl(obs) if ENABLE_HEAD else None

        # Move both arms and head to zero position at start
        if ENABLE_LEFT_HAND and left_arm:
            left_arm.move_to_zero_position(robot)
        right_arm.move_to_zero_position(robot)
        if ENABLE_HEAD and head_control:
            head_control.move_to_zero_position(robot)
        
        # Main VR control loop
        print("Starting VR control loop.")
        frame_queue = queue.Queue()
        dataset = init_dataset(robot)
        thread_args = (dataset, frame_queue, shutdown_event, saving_in_progress_event)
        dataset_saving_thread = threading.Thread(target=saving_dataset_worker, args=thread_args, daemon=False)
        dataset_saving_thread.start()

        while not shutdown_event.is_set():
            if saving_in_progress_event.is_set():
                print("[MAIN] Waiting for dataset saving to complete...  ", end='\r')
                time.sleep(0.1)
                continue
            
            start_loop_t = time.perf_counter()

            # Get VR controller data
            dual_goals = vr_monitor.get_latest_goal_nowait()
            left_goal = dual_goals.get("left") if dual_goals else None
            right_goal = dual_goals.get("right") if dual_goals else None
            headset_goal = dual_goals.get("headset") if dual_goals else None

            # Wait for VR connection before proceeding
            if dual_goals is None:
                time.sleep(0.01)  # Wait 10ms for VR connection
                continue

            # Handle VR input for both arms
            if ENABLE_LEFT_HAND and left_arm:
                left_arm.handle_vr_input(left_goal, gripper_state=None)
            right_arm.handle_vr_input(right_goal, gripper_state=None)
            
            # Get actions from both arms and head
            left_action, left_obs = left_arm.p_control_action(robot) if (ENABLE_LEFT_HAND and left_arm is not None) else ({}, {})
            right_action, right_obs = right_arm.p_control_action(robot)
            head_action = head_control.p_control_action(robot) if (ENABLE_HEAD and head_control) else {}

            # Get base control from VR
            base_action = get_vr_base_action(right_goal, robot)
            # speed_multiplier = get_vr_speed_control(right_goal)
            
            # if base_action:
            #     for key in base_action:
            #         if 'vel' in key or 'velocity' in key:  
                        # base_action[key] *= speed_multiplier 

            # Merge all actions
            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)
            
            # Get camera frames through async_read
            camera_obs = robot.get_camera_observation()

            # Merge action and observation features for new dataset frame
            if (ENABLE_LEFT_HAND and left_arm):
                action_features = {**left_action, **right_action}
                obs_features = {**left_obs, **right_obs, **camera_obs}
            else:
                action_features = right_action
                obs_features = { **right_obs, **camera_obs}
            action_frame = build_dataset_frame(dataset.features, action_features, prefix=ACTION)
            observation_frame = build_dataset_frame(dataset.features, obs_features, prefix=OBS_STR)
            lerobot_frame = {**observation_frame, **action_frame, "task": TASK}
            frame_queue.put(lerobot_frame)

            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(1 / FPS - dt_s)
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup
        shutdown_event.set()

        if dataset_saving_thread and dataset_saving_thread.is_alive():
            print("[MAIN] Waiting for dataset saving thread to finish...")
            dataset_saving_thread.join()
   
        if robot:
            robot.disconnect()

if __name__ == "__main__":
    main()
