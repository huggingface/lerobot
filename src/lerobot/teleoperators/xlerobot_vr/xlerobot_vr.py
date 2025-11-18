#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XLerobot VR Teleoperator
Refactored based on VR control logic from 8_xlerobot_VR_teleop.py, following teleop_keyboard format
"""
import math

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from queue import Queue
from typing import Any, Dict, Optional

import numpy as np

# from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.model.SO101Robot import SO101Kinematics

from ..teleoperator import Teleoperator
from .configuration_xlerobot_vr import XLerobotVRTeleopConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check VR Monitor availability
VR_AVAILABLE = True
try:
    # Dynamically import VR Monitor 
    from .vr_monitor import VRMonitor
except ImportError as e:
    VR_AVAILABLE = False
    VRMonitor = None
    logging.warning(f"VR Monitor not available: {e}")
except Exception as e:
    VR_AVAILABLE = False
    VRMonitor = None
    logging.warning(f"Could not import VR Monitor: {e}")


# Joint mapping configurations (copied from 8_xlerobot_VR_teleop.py)
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

# Joint calibration coefficients (copied from 8_xlerobot_VR_teleop.py)
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      
    ['shoulder_lift', 2.0, 0.97],     
    ['elbow_flex', 0.0, 1.05],        
    ['wrist_flex', 0.0, 0.94],        
    ['wrist_roll', 0.0, 0.5],        
    ['gripper', 0.0, 1.0],           
]

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
        return action

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
        
        # print(current_vr_pos)
        
        # Calculate relative change (delta) from previous frame
        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * 170  # Scale for the shoulder
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * 80
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * 80

        # print(f'vr_x: {vr_x}, vr_y: {vr_y}, vr_z: {vr_z}')

        # Update previous position for next frame
        self.prev_vr_pos = current_vr_pos
        
        # Delta control parameters - adjust these for sensitivity
        pos_scale = 0.015  # Position sensitivity scaling
        angle_scale = 3.0  # Angle sensitivity scaling (for wrist flex/pitch)
        wrist_roll_scale = 1.0  # Separate, slower scaling for wrist roll (reduced from 3.0)
        delta_limit = 0.02  # Maximum delta per update (meters)
        angle_limit = 6.0  # Maximum angle delta per update (degrees)
        wrist_roll_limit = 3.0  # Maximum wrist roll delta per update (degrees, reduced for precision)
        
        delta_x = vr_x * pos_scale
        delta_y = vr_y * pos_scale  
        delta_z = vr_z * pos_scale

        # Dead zone
        threshold = 0.001
        if delta_x < threshold and delta_x > -threshold:
            delta_x = 0.0
        if delta_y < threshold and delta_y > -threshold:
            delta_y = 0.0
        if delta_z < threshold and delta_z > -threshold:
            delta_z = 0.0

        # Limit delta values to prevent sudden movements
        delta_x = max(-delta_limit, min(delta_limit, delta_x))
        delta_y = max(-delta_limit, min(delta_limit, delta_y))
        delta_z = max(-delta_limit, min(delta_limit, delta_z))
        
        self.current_x += -delta_z  # VR Z maps to robot x, change the direction
        self.current_y += delta_y  # VR Y maps to robot y

        # Handle wrist angles with delta control - use relative changes
        if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
            # Initialize previous wrist_flex if not set
            if not hasattr(self, 'prev_wrist_flex'):
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
                return
            
            # Calculate relative change from previous frame
            delta_pitch = (vr_goal.wrist_flex_deg - self.prev_wrist_flex) * angle_scale
            if delta_pitch < 1 and delta_pitch > -1:
                delta_pitch = 0.0
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
            
            # Use separate, slower scaling for wrist roll
            delta_roll = (vr_goal.wrist_roll_deg - self.prev_wrist_roll) * wrist_roll_scale
            delta_roll = max(-wrist_roll_limit, min(wrist_roll_limit, delta_roll))

            # Smaller dead zone for wrist roll to allow fine control
            if abs(delta_roll) < 0.5:
                delta_roll = 0.0
            
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
            # Validate workspace before IK solving
            r = math.sqrt(self.current_x**2 + self.current_y**2)
            r_max = self.kinematics.l1 + self.kinematics.l2
            r_min = abs(self.kinematics.l1 - self.kinematics.l2)
            
            # Clamp to workspace if needed
            if r > r_max:
                scale = r_max / r
                self.current_x *= scale
                self.current_y *= scale
            elif r < r_min and r > 0:
                scale = r_min / r
                self.current_x *= scale
                self.current_y *= scale
            
            # Solve IK with improved precision
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            
            # Use lower alpha for smoother, more precise IK tracking
            # Lower values = more precise but slower response
            # Higher values = faster but less precise
            alpha = 0.15  # Reduced from 0.2 for better precision (was 0.1, then 0.2)
            
            # Apply exponential smoothing for precise tracking
            current_shoulder = self.target_positions.get("shoulder_lift", 0.0)
            current_elbow = self.target_positions.get("elbow_flex", 0.0)
            
            self.target_positions["shoulder_lift"] = (1-alpha) * current_shoulder + alpha * joint2_target
            self.target_positions["elbow_flex"] = (1-alpha) * current_elbow + alpha * joint3_target
            
        except Exception as e:
            print(f"[{self.prefix}] VR IK failed: {e}")
            # On IK failure, maintain current positions to prevent jumps
        
        # Calculate wrist_flex to maintain end-effector orientation
        self.target_positions["wrist_flex"] = (-self.target_positions["shoulder_lift"] - 
                                               self.target_positions["elbow_flex"] + self.pitch)
   
        # Handle gripper state directly
        if vr_goal.metadata.get('trigger', 0) > 0.5:
            self.target_positions["gripper"] = 45
        else:
            self.target_positions["gripper"] = 0.0

    def p_control_action(self, robot_obs):
        """
        Generate proportional control action based on target positions.
        
        Args:
            robot: Robot instance to get current observations
            
        Returns:
            dict: Action dictionary with position commands for each joint
        """
        obs = robot_obs
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
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
    pressed_keys = set()
    if vr_goal is not None and hasattr(vr_goal, 'metadata'):
    
    # Build key set based on VR input (you can customize this mapping)
    
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

class XLerobotVRTeleop(Teleoperator):
    """
    XLerobot VR Teleoperator class
    Following the format of teleop_keyboard, integrating VR control logic from 8_xlerobot_VR_teleop.py
    """

    config_class = XLerobotVRTeleopConfig
    name = "xlerobot_vr"

    def __init__(self, config: XLerobotVRTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # VR system related
        self.vr_monitor = None
        self.vr_thread = None
        self.vr_data_queue = Queue()
        self.latest_vr_data = None
        
        # New: VR event handler
        self.vr_event_handler = None
                    
        # Kinematics instances
        self.kin_left = SO101Kinematics()
        self.kin_right = SO101Kinematics()
        
        # Arm controllers (initialized during calibrate, guarded elsewhere)
        self.left_arm = None
        self.right_arm = None
        
        # Base speed control
        self.current_base_speed = 0.0
        self.last_update_time = time.time()
        self.last_event_update_time = 0.0
        self.is_accelerating = False
        
        # Status flags
        self._connected = False
        self._calibrated = False
        
        # Store robot reference (set during connect)
        self.robot = None
        
        # Cache for observations to avoid double reads
        self._cached_obs = None
        self._obs_cache_time = 0.0
        self._obs_cache_duration = 0.01  # Cache for 10ms (faster than camera refresh)
        
        self.logs = {}

    @property
    def action_features(self) -> dict:
        """Define action feature structure"""
        # Define based on XLerobot's action space
        # Including dual arm joints, head motors, base movement
        features = {}
        
        # Left arm joints
        for joint_name in LEFT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
        
        # Right arm joints
        for joint_name in RIGHT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
            
        # Base control (according to XLerobot's base control method)
        features["base_action"] = "dict"
        
        return features

    @property
    def feedback_features(self) -> dict:
        """Define feedback feature structure"""
        return {}  # VR controllers usually don't need feedback

    @property
    def is_connected(self) -> bool:
        """Check connection status"""
        return (
            self._connected and 
            VR_AVAILABLE and 
            self.vr_monitor is not None and
            (self.vr_thread is not None and self.vr_thread.is_alive())
        )

    @property
    def is_calibrated(self) -> bool:
        """Check calibration status"""
        return self._calibrated

    def connect(self, calibrate: bool = True, robot=None) -> None:
        """Establish VR connection - optimized version"""
        if self.is_connected:
            raise RuntimeError(
                "XLerobot VR is already connected. Do not run `connect()` twice."
            )

        if not VR_AVAILABLE:
            raise RuntimeError(
                "VR Monitor is not available. Please check VR system installation."
            )

        try:
            logger.info("🔧 Initializing VR monitor...")
            self.vr_monitor = VRMonitor()
            
            # Use timeout mechanism to avoid infinite waiting
            init_success = False
            start_time = time.time()
            timeout = 10.0  # 10 second timeout
            
            while time.time() - start_time < timeout:
                if self.vr_monitor.initialize():
                    init_success = True
                    break
                time.sleep(0.1)
            
            if not init_success:
                raise Exception("VR monitor initialization timeout")
                
            logger.info("🚀 Starting VR monitoring...")
            self.vr_thread = threading.Thread(
                target=lambda: asyncio.run(self.vr_monitor.start_monitoring()), 
                daemon=True
            )
            self.vr_thread.start()
            
            # Wait for thread to start
            time.sleep(0.5)
            
            if not self.vr_thread.is_alive():
                raise Exception("VR monitoring thread failed to start")
                
            logger.info("✅ VR system ready")
            self._connected = True
            
            # Initialize VR event handler
            self.vr_event_handler = VREventHandler(self.vr_monitor)
            logger.info("🎮 VR event handler initialized")
            
            # Store robot reference for use in get_action
            self.robot = robot
            
            if calibrate and robot is not None:
                robot_obs = robot.get_observation()
                self.calibrate(robot_obs)
                
        except Exception as e:
            logger.error(f"[VR] Connection failed: {e}")
            self._connected = False
            raise RuntimeError(f"Failed to connect to VR: {e}")

    def calibrate(self, robot_obs: Optional[Dict] = None) -> None:
        """Calibrate VR controllers - optimized version"""
        if robot_obs is None:
            logger.warning("[VR] No robot observation provided for calibration")
            return
            
        try:
            # Initialize arm controllers
            self.left_arm = SimpleTeleopArm(
                LEFT_JOINT_MAP, robot_obs, self.kin_left, 
                prefix="left", kp=self.config.kp
            )
            self.right_arm = SimpleTeleopArm(
                RIGHT_JOINT_MAP, robot_obs, self.kin_right, 
                prefix="right", kp=self.config.kp
            )
            
            logger.info("[VR] Controllers initialized successfully")
            self._calibrated = True
            
        except Exception as e:
            logger.error(f"[VR] Calibration failed: {e}")
            self._calibrated = False
            raise


    def update_observation_cache(self, obs: dict[str, Any]) -> None:
        """
        Update the observation cache with externally-read observation.
        Called by lerobot_teleoperate.py to avoid double reads.
        """
        self._cached_obs = obs
        self._obs_cache_time = time.perf_counter()
    
    def get_action(self) -> dict[str, Any]:
        """Get VR control action with detailed profiling"""
        total_start = time.perf_counter()
        
        action = {}
        
        # Quick check VR monitoring status and robot reference
        if not self.vr_monitor or self.robot is None:
            self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
            return action
        
        # Get VR data once to avoid repeated calls
        vr_start = time.perf_counter()
        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
            if dual_goals is None:
                self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
                return action
                
            left_goal = dual_goals.get("left")
            right_goal = dual_goals.get("right")
            
        except Exception as e:
            logger.warning(f"VR data acquisition failed: {e}")
            self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
            return action
        vr_dt_ms = (time.perf_counter() - vr_start) * 1e3
        logger.info(f"🎮 VR data fetch: {vr_dt_ms:.1f}ms")
        
        # Get current robot observation with caching to avoid double reads
        # lerobot_teleoperate.py calls robot.get_observation() then teleop.get_action()
        # This cache prevents reading twice in the same loop iteration
        obs_start = time.perf_counter()
        current_time = time.perf_counter()
        
        # Use cached observation if recent (within 10ms)
        if (self._cached_obs is not None and 
            (current_time - self._obs_cache_time) < self._obs_cache_duration):
            robot_obs = self._cached_obs
            obs_dt_ms = (time.perf_counter() - obs_start) * 1e3
            logger.info(f"🤖 Robot observation (cached): {obs_dt_ms:.2f}ms")
        else:
            # Read fresh observation and cache it
            try:
                robot_obs = self.robot.get_observation()
                self._cached_obs = robot_obs
                self._obs_cache_time = current_time
            except Exception as e:
                logger.warning(f"Failed to get robot observation: {e}")
                self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
                return action
            obs_dt_ms = (time.perf_counter() - obs_start) * 1e3
            logger.info(f"🤖 Robot observation (fresh): {obs_dt_ms:.1f}ms")
        
        # IK and control computation
        ik_start = time.perf_counter()
        try:
            current_time = time.perf_counter()
            
            # Robot control - high frequency execution
            if left_goal is not None and self.left_arm is not None:
                self.left_arm.handle_vr_input(left_goal, None)
                
            if right_goal is not None and self.right_arm is not None:
                self.right_arm.handle_vr_input(right_goal, None)
            
            # Event processing - optimized frequency (10Hz)
            if (current_time - self.last_event_update_time) >= 0.1:
                if left_goal is not None:
                    self._update_events_inline(left_goal)
                self.last_event_update_time = current_time
            
            # Generate action dictionary
            left_action = self.left_arm.p_control_action(robot_obs) if self.left_arm is not None else {}
            right_action = self.right_arm.p_control_action(robot_obs) if self.right_arm is not None else {}
            base_action = get_vr_base_action(right_goal, self.robot)
            
            # Merge actions
            action.update(left_action)
            action.update(right_action)
            action.update(base_action)
            
        except Exception as e:
            logger.error(f"Action generation failed: {e}")
        
        ik_dt_ms = (time.perf_counter() - ik_start) * 1e3
        logger.info(f"🧮 IK + control: {ik_dt_ms:.1f}ms")
        
        total_dt_ms = (time.perf_counter() - total_start) * 1e3
        logger.info(f"⏱️  TOTAL get_action: {total_dt_ms:.1f}ms")
        logger.info(f"=" * 60)
        
        self.logs["read_pos_dt_s"] = time.perf_counter() - total_start
        return action
    
    def _update_events_inline(self, left_goal):
        """
        Low frequency event update - 10Hz frequency, reuse already acquired left_goal data
        Only execute when event interval time is reached, greatly reducing processing overhead
        """
        if not self.vr_event_handler or not left_goal or not hasattr(left_goal, 'metadata'):
            return
            
        # Directly use already acquired data, no need to call VR interface again
        try:
            self.vr_event_handler._process_left_controller(left_goal.metadata)
        except Exception as e:
            logger.debug(f"Low frequency event update failed: {e}")  # Downgrade to debug to avoid disrupting main flow

    def send_feedback(self) -> None:
        """Send feedback - optimized version, reduce blocking wait"""
        if not self.vr_monitor:
            logger.warning("VR monitor not available for feedback")
            return

        max_attempts = 200  # Maximum 200 attempts
        attempt = 0
        
        while attempt < max_attempts:
            try:
                dual_goals = self.vr_monitor.get_latest_goal_nowait()
                if dual_goals and sum(dual_goals.get('right').metadata['vr_position']):
                    logger.info("VR controller data received")
                    return
                    
            except Exception as e:
                logger.warning(f"Error getting VR data: {e}")
            
            attempt += 1
            logger.info(f'Waiting for VR controller data (attempt {attempt}/{max_attempts})')
            time.sleep(0.5)  # Reduce wait time from 8 seconds to 0.5 seconds
        
        logger.warning("Timeout waiting for VR controller data")

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        """Disconnect VR connection"""
        if not self.is_connected:
            raise RuntimeError(
                "XLerobot VR is not connected."
            )
        
        try:
            if self.vr_monitor:
                # VR Monitor usually runs in a thread, stop the thread
                pass
            
            self._connected = False
            self._calibrated = False
            print("[VR] Disconnected")
            
        except Exception as e:
            print(f"[VR] Error during disconnect: {e}")

    def move_to_zero_position(self, robot):
        """Move all controllers to zero position"""
        robot_obs = robot.get_observation()
        action = {}
        left_action = self.left_arm.move_to_zero_position(robot_obs)
        right_action = self.right_arm.move_to_zero_position(robot_obs)
        base_action = get_vr_base_action(None, robot)
        action.update(left_action)
        action.update(right_action)
        action.update(base_action)

        return action
    
    def get_vr_events(self):
        """Get VR event status (high-performance version - use cache to avoid repeated VR data acquisition)"""
        if self.vr_event_handler:
            # Get current event status
            events = self.vr_event_handler.get_events()
            
            # Automatically reset one-time events to prevent infinite loops
            # Only reset when event is True to avoid affecting normal state
            if events.get("exit_early", False) or events.get("rerecord_episode", False):
                self.vr_event_handler.reset_events()
            
            return events
        else:
            # Return default event status
            return {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
                "reset_position": False,
                "back_position": False,
            }
    
    def reset_vr_events(self):
        """Reset VR event status"""
        if self.vr_event_handler:
            self.vr_event_handler.reset_events()
    
    def print_vr_control_guide(self):
        """Print VR control guide"""
        if self.vr_event_handler:
            self.vr_event_handler.print_control_guide()
        else:
            logger.info("VR event handler not initialized")


def init_vr_listener(teleop_vr):
    """
    Initialize VR listener, providing the same interface as init_keyboard_listener
    Used to replace keyboard event listening, used in record.py
    
    Args:
        teleop_vr: XLerobotVRTeleop instance
        
    Returns:
        tuple: (listener, events) - same return format as init_keyboard_listener
    """
    if not isinstance(teleop_vr, XLerobotVRTeleop):
        logger.error("teleop_vr must be an XLerobotVRTeleop instance")
        return None, {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "reset_position": False,
            "back_position": False,
        }
    
    # Print control guide
    teleop_vr.print_vr_control_guide()
    
    # Create virtual listener object (compatible with keyboard listener)
    class VRListener:
        def __init__(self, teleop_vr):
            self.teleop_vr = teleop_vr
            self.is_alive = True
            
        def stop(self):
            self.is_alive = False
            logger.info("VR listener stopped")
    
    vr_listener = VRListener(teleop_vr)
    
    # Get initial event status
    events = teleop_vr.get_vr_events()
    
    return vr_listener, events

class VREventHandler:
    """
    VR event handler, specifically handles recording control events
    Use left VR controller to replace keyboard control
    """
    
    def __init__(self, vr_monitor):
        self.vr_monitor = vr_monitor
        self.events = {
            "exit_early": False,      # Left controller right: Exit loop early (original right arrow key)
            "rerecord_episode": False, # Left controller left: Re-record episode (original left arrow key)
            "stop_recording": False,   # Left controller up: Stop recording (original ESC key)
            "reset_position": False,   # Left controller down: Reset robot (new feature)
            "back_position": False,    # In the bucket (new feature)
        }
        self.prev_states = {
            'thumbstick_x': 0,
            'thumbstick_y': 0,
            'trigger': False,
            'button_a': False,
            'button_b': False,
        }
        self.threshold = 0.7  # Thumbstick trigger threshold
        
    def update_events(self):
        """Update VR event status"""
        if not self.vr_monitor:
            return self.events
            
        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
            if not dual_goals:
                return self.events
                
            left_goal = dual_goals.get("left")
            if not left_goal or not hasattr(left_goal, 'metadata'):
                return self.events
                
            self._process_left_controller(left_goal.metadata)
            
        except Exception as e:
            logger.error(f"VR事件更新失败: {e}")
            
        return self.events
    
    def _process_left_controller(self, metadata):
        """处理左手柄输入"""
        # 获取摇杆输入
        thumb = metadata.get('thumbstick', {})
        thumb_x = thumb.get('x', 0)
        thumb_y = thumb.get('y', 0)

        
        # Detect thumbstick direction events (only trigger when crossing threshold)
        if thumb_x > self.threshold and self.prev_states['thumbstick_x'] <= self.threshold:
            logger.info("🎮 VR left controller right -> Exit loop early")
            self.events["exit_early"] = True
            
        elif thumb_x < -self.threshold or self.events['rerecord_episode'] == True:
            logger.info("🎮 VR left controller left -> Re-record episode")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
            
        if thumb_y > self.threshold and self.prev_states['thumbstick_y'] <= self.threshold:
            logger.info("🎮 VR left controller up -> Stop recording")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
            # self.events["back_position"] = True

        elif thumb_y < -self.threshold and self.prev_states['thumbstick_y'] >= -self.threshold:
            logger.info("🎮 VR left controller down -> Reset robot")
            self.events["reset_position"] = True
        else:
            self.events["reset_position"] = False  # Reset event is instantaneous
            self.events["back_position"] = False
        
        # Detect trigger key events
        trigger = metadata.get('trigger', 0) > 0.5
        
        # Update status
        self.prev_states.update({
            'thumbstick_x': thumb_x,
            'thumbstick_y': thumb_y,
            'trigger': trigger,
        })
    
    def reset_events(self):
        """Reset all event status"""
        for key in self.events:
            self.events[key] = False
    
    def get_events(self):
        """Get current event status"""
        return self.events.copy()
    
    def print_control_guide(self):
        """Print VR control guide"""
        guide = """
        🎮 VR Left Controller Guide:
        ├── 👈 Push thumbstick left: Re-record current episode
        ├── 👉 Push thumbstick right: Exit current loop early
        ├── 👆 Push thumbstick up: Stop recording
        ├── 👇 Push thumbstick down: Reset robot position
        """
        logger.info(guide)