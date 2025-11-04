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

import importlib
import json
import logging
from pathlib import Path

from lerobot.motors.motors_bus import MotorNormMode

from ..teleoperator import Teleoperator
from .config_custom import CustomConfig

logger = logging.getLogger(__name__)


class Custom(Teleoperator):
    """
    Custom teleoperator that dynamically wraps a base teleoperator class and applies configurable joint mapping.
    The base class is specified in custom_config.json, allowing flexible teleoperator configurations.
    """

    config_class = CustomConfig
    name = "custom"

    def __init__(self, config: CustomConfig):
        # Load custom configuration from JSON file
        if config.config_path is None:
            raise ValueError(
                "config_path must be provided for custom teleoperator. "
                "Example: --teleop.config_path=/path/to/custom_config.json"
            )
        
        config_path = Path(config.config_path)
        
        with open(config_path) as f:
            custom_config = json.load(f)
        
        logger.info(f"Loaded custom config from {config_path}")
        logger.info(f"Found {len(custom_config)} teleoperator(s): {list(custom_config.keys())}")
        
        # Initialize the base Teleoperator class
        super().__init__(config)
        
        # Store multiple base teleoperators and their action mappings
        self.base_teleops = {}
        self.robot_actions_configs = {}
        
        # Instantiate each base teleoperator from the config
        for device_name, device_config in custom_config.items():
            base_class_name = device_config["base_class"]
            
            # Create a config copy for this teleoperator
            from dataclasses import replace
            teleop_config = replace(
                config,
                port=device_config.get("port", config.port),
                id=device_config.get("id", f"{config.id}_{device_name}"),
                baud_rate=device_config.get("baud_rate", config.baud_rate)
            )
            
            logger.info(f"  {device_name}: class={base_class_name}, port={teleop_config.port}, id={teleop_config.id}")
            
            # Dynamically import and instantiate the base teleoperator class
            module_path, class_name_full = base_class_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            base_class = getattr(module, class_name_full)
            
            # Store the teleoperator and its action mapping
            self.base_teleops[device_name] = base_class(teleop_config)
            self.robot_actions_configs[device_name] = device_config["robot_actions"]

    @property
    def action_features(self) -> dict:
        # Aggregate action features from all teleoperators' action mappings
        all_actions = {}
        for device_config in self.robot_actions_configs.values():
            for robot_action in device_config.keys():
                all_actions[robot_action] = float
        return all_actions
    
    @property
    def feedback_features(self) -> dict:
        # Aggregate feedback features from all base teleoperators
        all_feedback = {}
        for teleop in self.base_teleops.values():
            all_feedback.update(teleop.feedback_features)
        return all_feedback
    
    @property
    def is_connected(self) -> bool:
        # All teleoperators must be connected
        return all(teleop.is_connected for teleop in self.base_teleops.values())
    
    @property
    def is_calibrated(self) -> bool:
        # All teleoperators must be calibrated
        return all(teleop.is_calibrated for teleop in self.base_teleops.values())
    
    def connect(self, calibrate: bool = True) -> None:
        # Connect all base teleoperators
        for device_name, teleop in self.base_teleops.items():
            logger.info(f"Connecting {device_name}...")
            teleop.connect(calibrate=calibrate)
    
    def calibrate(self) -> None:
        # Calibrate all base teleoperators
        for device_name, teleop in self.base_teleops.items():
            logger.info(f"Calibrating {device_name}...")
            teleop.calibrate()
    
    def configure(self) -> None:
        # Configure all base teleoperators
        for teleop in self.base_teleops.values():
            teleop.configure()
    
    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Send feedback to all base teleoperators
        for teleop in self.base_teleops.values():
            teleop.send_feedback(feedback)
    
    def disconnect(self) -> None:
        # Disconnect all base teleoperators
        for device_name, teleop in self.base_teleops.items():
            logger.info(f"Disconnecting {device_name}...")
            teleop.disconnect()

    def _normalize_to_unit_range(self, teleop, joint_name: str, value: float) -> float:
        """Convert a joint value from base teleoperator's normalization mode to [0, 1] range.
        
        Args:
            teleop: The base teleoperator instance
            joint_name: Name of the joint (e.g., "shoulder_pitch")
            value: Value in the base teleoperator's normalization mode
            
        Returns:
            Value normalized to [0, 1] range
        """
        norm_mode = teleop.joints[joint_name]
        
        if norm_mode == MotorNormMode.RANGE_M100_100:
            # Convert from [-100, 100] to [0, 1]
            return (value + 100.0) / 200.0
        elif norm_mode == MotorNormMode.RANGE_0_100:
            # Convert from [0, 100] to [0, 1]
            return value / 100.0
        elif norm_mode == MotorNormMode.DEGREES:
            # For degrees, we need calibration to know the range
            # Use calibration min/max to normalize
            if teleop.calibration and joint_name in teleop.calibration:
                min_deg = teleop.calibration[joint_name].range_min
                max_deg = teleop.calibration[joint_name].range_max
                if max_deg != min_deg:
                    return (value - min_deg) / (max_deg - min_deg)
            # Fallback: assume common range like [-180, 180]
            return (value + 180.0) / 360.0
        else:
            raise ValueError(f"Unknown normalization mode: {norm_mode}")

    def get_action(self) -> dict[str, float]:
        # Build action dict by reading from all base teleoperators
        action = {}
        
        # Loop through each teleoperator
        for device_name, teleop in self.base_teleops.items():
            # Read joint positions from this teleoperator
            # These are in the teleoperator's normalization mode (e.g., -100 to 100)
            joint_positions = teleop._read()

            # Get the robot actions config for this teleoperator
            robot_actions_config = self.robot_actions_configs[device_name]
            
            # Process each robot action for this teleoperator
            for robot_action, config in robot_actions_config.items():
                if config["source"] == "neutral":
                    # Use fixed neutral value (already in [0, 1] range)
                    value = config["value"]
                elif config["source"] == "teleop":
                    # Get value from teleop joint
                    teleop_joint = config["joint"]
                    value = joint_positions[teleop_joint]
                    
                    # Convert from base teleoperator's normalization mode to [0, 1] range
                    value = self._normalize_to_unit_range(teleop, teleop_joint, value)
                    
                    # Apply inversion if specified
                    if config.get("invert", False):
                        value = 1.0 - value
                else:
                    raise ValueError(f"Unknown source '{config['source']}' for robot action '{robot_action}'")
                
                action[robot_action] = value
        return action
