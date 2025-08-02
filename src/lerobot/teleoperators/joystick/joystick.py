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
import pygame
from pathlib import Path
from typing import Any

from ..teleoperator import Teleoperator
from lerobot.utils.utils import enter_pressed, move_cursor_up
from .config_joystick import JoystickTeleopConfig

logger = logging.getLogger(__name__)


class JoystickTeleop(Teleoperator):
    """
    Teleop class to use joystick inputs for control.
    Supports any joystick device with calibration for consistent behavior.
    """

    config_class = JoystickTeleopConfig
    name = "joystick"

    def __init__(self, config: JoystickTeleopConfig):
        super().__init__(config)
        self.config = config
        self.joystick = None
        self.axis_ranges = {}  # Stores min/max values for each axis
        self.is_initialized = False
        
        # Load calibration if available
        if self.calibration_fpath.is_file():
            self._load_calibration()

    @property
    def action_features(self) -> dict[str, type]:
        """Return the action features for this teleoperator."""
        # Return features for all mapped joints
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
        
        # Check if calibration file exists
        if self.calibration_fpath.is_file():
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Loading calibration file associated with the id {self.id}")
                self._load_calibration()
                return
        
        logger.info(f"\nRunning calibration of {self}")
        
        # Initialize calibration data
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
                # Move cursor up to overwrite the previous output
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
        
        # Save calibration
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

    def get_action(self) -> dict[str, Any]:
        """Get current joystick action mapped to robot joints."""
        if not self.is_connected:
            raise RuntimeError("Joystick not connected")
        
        # Load calibration if not loaded
        if not self.axis_ranges and self.calibration:
            self.axis_ranges = self.calibration
        
        pygame.event.pump()
        
        action = {}
        for axis_idx, joint_name in self.config.axis_mapping.items():
            if axis_idx < self.joystick.get_numaxes():
                # Get raw axis value
                raw_value = self.joystick.get_axis(axis_idx)
                
                # Apply deadzone
                if abs(raw_value) < self.config.deadzone:
                    raw_value = 0.0
                
                # Normalize to 0-1 range using calibration data
                if axis_idx in self.axis_ranges:
                    ranges = self.axis_ranges[axis_idx]
                    min_val = ranges['min']
                    max_val = ranges['max']
                    
                    # Normalize to 0-1 range
                    if max_val > min_val:
                        normalized = (raw_value - min_val) / (max_val - min_val)
                        normalized = max(0.0, min(1.0, normalized))  # Clamp to 0-1
                    else:
                        normalized = 0.5  # Default to middle if no range
                else:
                    # If not calibrated, use raw value directly
                    normalized = (raw_value + 1.0) / 2.0  # Convert from -1,1 to 0,1
                
                action[f"{joint_name}.pos"] = normalized
        
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