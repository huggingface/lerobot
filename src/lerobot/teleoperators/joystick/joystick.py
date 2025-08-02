#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from pathlib import Path
from typing import Any

import pygame

from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..teleoperator import Teleoperator
from .config_joystick import JoystickTeleopConfig

logger = logging.getLogger(__name__)


class JoystickTeleop(Teleoperator):
    """
    Joystick-based teleoperator for robot control.
    
    Supports any joystick device compatible with pygame, with calibration 
    for consistent behavior and relative position control.
    """

    config_class = JoystickTeleopConfig
    name = "joystick"

    def __init__(self, config: JoystickTeleopConfig):
        super().__init__(config)
        self.config = config
        self.joystick = None
        self.axis_ranges = {}
        self.is_initialized = False
        self.center_positions = {}
        self.robot_positions = {}
        self.position_initialized = False
        
        if self.calibration_fpath.is_file():
            self._load_calibration()

    @property
    def action_features(self) -> dict[str, type]:
        """Return the action features for this teleoperator."""
        features = {}
        for joint_name in self.config.axis_mapping.values():
            features[f"{joint_name}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Return the feedback features for this teleoperator."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if joystick is connected and initialized."""
        return self.is_initialized and self.joystick is not None

    def connect(self, calibrate: bool = True) -> None:
        """Initialize pygame and connect to joystick."""
        try:
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No joystick detected. Please connect a joystick and try again.")
            
            self.joystick = pygame.joystick.Joystick(self.config.device_index)
            self.joystick.init()
            
            logger.info(f"Connected to joystick: {self.joystick.get_name()}")
            logger.info(f"Number of axes: {self.joystick.get_numaxes()}")
            logger.info(f"Number of buttons: {self.joystick.get_numbuttons()}")
            
            self.is_initialized = True
            
            if calibrate:
                self.calibrate()
                
        except Exception as e:
            logger.error(f"Failed to connect to joystick: {e}")
            raise

    @property
    def is_calibrated(self) -> bool:
        """Check if joystick has been calibrated."""
        return len(self.axis_ranges) > 0

    def calibrate(self) -> None:
        """Calibrate joystick by recording axis ranges."""
        if not self.is_connected:
            raise RuntimeError("Joystick not connected")
        
        if self.calibration_fpath.is_file():
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Loading calibration file associated with the id {self.id}")
                self._load_calibration()
                return
        
        logger.info(f"\nRunning calibration of {self}")
        
        axis_count = self.joystick.get_numaxes()
        mins = {i: 0.0 for i in range(axis_count)}
        maxes = {i: 0.0 for i in range(axis_count)}
        
        # Get initial positions
        for i in range(axis_count):
            pygame.event.pump()
            value = self.joystick.get_axis(i)
            mins[i] = value
            maxes[i] = value
        
        print(
            "Move all joystick axes sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        
        # Record ranges of motion with live display
        user_pressed_enter = False
        while not user_pressed_enter:
            positions = {}
            for i in range(axis_count):
                pygame.event.pump()
                value = self.joystick.get_axis(i)
                positions[i] = value
                mins[i] = min(mins[i], value)
                maxes[i] = max(maxes[i], value)
            
            # Live display
            print("\n-------------------------------------------")
            print(f"{'AXIS':<6} | {'MIN':>8} | {'POS':>8} | {'MAX':>8}")
            for i in range(axis_count):
                print(f"{i:<6} | {mins[i]:>8.3f} | {positions[i]:>8.3f} | {maxes[i]:>8.3f}")
            
            if enter_pressed():
                user_pressed_enter = True
            
            if not user_pressed_enter:
                move_cursor_up(axis_count + 3)
        
        # Validate calibration
        same_min_max = [i for i in range(axis_count) if abs(maxes[i] - mins[i]) < 0.1]
        if same_min_max:
            raise ValueError(f"Some axes have the same min and max values:\n{same_min_max}")
        
        # Store calibration data
        self.axis_ranges = {}
        for i in range(axis_count):
            self.axis_ranges[i] = {
                'min': mins[i],
                'max': maxes[i]
            }
            logger.info(f"Axis {i}: min={mins[i]:.3f}, max={maxes[i]:.3f}")
        
        self.calibration = self.axis_ranges
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")
        logger.info("Joystick calibration complete!")

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """Load joystick calibration data from file."""
        import json
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, 'r') as f:
            self.calibration = json.load(f)
            self.axis_ranges = self.calibration

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """Save joystick calibration data to file."""
        import json
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, 'w') as f:
            json.dump(self.calibration, f, indent=4)

    def configure(self) -> None:
        """Configure the joystick (no special configuration needed)."""
        pass
    
    def initialize_robot_positions(self, robot_observation: dict[str, Any]) -> None:
        """Initialize robot position tracking from current robot state."""
        for joint_name in self.config.axis_mapping.values():
            joint_key = f"{joint_name}.pos"
            if joint_key in robot_observation:
                self.robot_positions[joint_name] = robot_observation[joint_key]
                logger.info(f"Initialized {joint_name} position: {robot_observation[joint_key]:.1f}")

    def get_action(self) -> dict[str, Any]:
        """Get current joystick action using relative control from center positions."""
        if not self.is_connected:
            raise RuntimeError("Joystick not connected")
        
        if not self.axis_ranges and self.calibration:
            self.axis_ranges = self.calibration
        
        pygame.event.pump()
        
        # Initialize center positions and robot positions on first call
        if not self.position_initialized:
            # Record current joystick positions as center
            for axis_idx in self.config.axis_mapping.keys():
                if axis_idx < self.joystick.get_numaxes():
                    self.center_positions[axis_idx] = self.joystick.get_axis(axis_idx)
            
            # Initialize robot positions to current values (will be set by teleoperate.py)
            # For now, use middle of range as fallback
            for joint_name in self.config.axis_mapping.values():
                if joint_name == "gripper":
                    self.robot_positions[joint_name] = 50.0  # Middle of 0-100 range
                else:
                    self.robot_positions[joint_name] = 0.0   # Middle of -100 to +100 range
            
            self.position_initialized = True
            logger.info("Initialized joystick center positions for relative control")
        
        # Calculate deltas from center positions and apply to robot positions
        for axis_idx, joint_name in self.config.axis_mapping.items():
            if axis_idx < self.joystick.get_numaxes():
                current_value = self.joystick.get_axis(axis_idx)
                center_value = self.center_positions.get(axis_idx, 0.0)
                
                # Calculate delta from center
                delta_raw = current_value - center_value
                
                # Apply deadzone
                if abs(delta_raw) < self.config.deadzone:
                    delta_raw = 0.0
                
                # Convert delta to motor space with step size
                if axis_idx in self.axis_ranges:
                    ranges = self.axis_ranges[axis_idx]
                    min_val = ranges['min']
                    max_val = ranges['max']
                    
                    # Normalize delta to range scale
                    if max_val > min_val:
                        range_scale = max_val - min_val
                        normalized_delta = delta_raw / range_scale
                    else:
                        normalized_delta = delta_raw
                else:
                    # Raw joystick range is typically -1 to +1, so delta range is -2 to +2
                    normalized_delta = delta_raw / 2.0
                
                # Apply step size and convert to appropriate motor range
                step_size = self.config.step_size
                if joint_name == "gripper":
                    # Gripper uses RANGE_0_100
                    delta_motor = normalized_delta * step_size * 50.0
                else:
                    # Body joints use RANGE_M100_100  
                    delta_motor = normalized_delta * step_size * 100.0
                
                # Integrate delta into robot position
                self.robot_positions[joint_name] += delta_motor
                
                # Clamp to valid ranges
                if joint_name == "gripper":
                    self.robot_positions[joint_name] = max(0.0, min(100.0, self.robot_positions[joint_name]))
                else:
                    self.robot_positions[joint_name] = max(-100.0, min(100.0, self.robot_positions[joint_name]))
        
        # Return current robot positions as action
        action = {}
        for joint_name in self.config.axis_mapping.values():
            action[f"{joint_name}.pos"] = self.robot_positions[joint_name]
        
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to joystick (not implemented for basic joystick)."""
        pass

    def disconnect(self) -> None:
        """Disconnect and cleanup pygame resources."""
        if self.joystick:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()
        self.is_initialized = False
        logger.info("Joystick disconnected") 