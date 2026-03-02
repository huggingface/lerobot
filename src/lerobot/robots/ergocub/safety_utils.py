#!/usr/bin/env python

# Copyright 2024 Istituto Italiano di Tecnologia. All rights reserved.
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
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class HandSafetyChecker:
    """
    Safety checker for hand position validation based on metaControllClient logic.
    """
    
    def __init__(self, position_tolerance: float = 0.1):
        """
        Initialize safety checker.
        
        Args:
            position_tolerance: Position tolerance in meters (default 0.1m = 10cm)
        """
        self.position_tolerance = max(0.01, position_tolerance)  # Minimum 1cm tolerance
        self.is_arm_controlled = {"left": False, "right": False}
        
    def set_position_tolerance(self, tolerance: float) -> None:
        """
        Set the position tolerance for safety checking.
        
        Args:
            tolerance: Position tolerance in meters
        """
        self.position_tolerance = max(0.01, tolerance)
        logger.info("Position tolerance set to %.3fm", self.position_tolerance)
    
    def reset_arm_control(self) -> None:
        """
        Reset the arm control state, requiring position check before movement.
        """
        self.is_arm_controlled = {"left": False, "right": False}
        logger.info("Arm control reset: hands must be repositioned within tolerance before movement")
    
    def is_valid_action(self, action: Dict[str, Any]) -> bool:
        """
        Check if the action contains valid values (not all zeros, no NaN values).
        Based on the isValidPose function from metaControllClient.
        
        Args:
            action: Action to validate
            arms_to_check: List of arm sides to check ("left", "right")
            
        Returns:
            True if action is valid, False otherwise
        """
        # Check for NaN values
        for key, value in action.items():
            if np.any(np.isnan(value)):
                logger.debug("NaN value detected in action key: %s", key)
                return False
        
        # Check if position values are all zeros for each configured arm
        for side in ["left", "right"]:
            # Check arm position values
            arm_position_keys = [f"{side}_hand.position.x", f"{side}_hand.position.y", f"{side}_hand.position.z"]
            arm_position_values = [action[key] for key in arm_position_keys]
            
            # Sum of absolute values (like metaControllClient isValidPose)
            position_sum = sum(abs(v) for v in arm_position_values)
            if position_sum <= 1e-6:
                logger.debug("Invalid action: %s arm position values are all zeros", side)
                return False
            # If all values of rotation are zeros, return false
            arm_rotation_values = [action[f"{side}_hand.orientation.d{i}" ]for i in range(1, 7)]
            if all([x == 0 for x in arm_rotation_values]):
                logger.debug("Invalid action: %s arm rotation values are all zeros", side)
                return False

        
        return True
    
    def check_hand_position_safety(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> bool:
        """
        Check if the target hand positions are close enough to current positions to safely move.
        Based on the safety logic from metaControllClient.
        
        Args:
            action: Target action containing hand positions
            current_state: Current robot state dictionary from bus.read_state()
            arms_to_check: List of arm sides to check ("left", "right")
            
        Returns:
            True if it's safe to move (all hands within tolerance or controlled)
        """
        for side in ["left", "right"]:
            # Check if target position is valid (not all zeros, like metaControllClient)
            target_pos = np.array([
                action[f"{side}_hand.position.x"],
                action[f"{side}_hand.position.y"],
                action[f"{side}_hand.position.z"]
            ])
            
            # Check for invalid poses (all zeros or NaN values, similar to metaControllClient)
            if np.allclose(target_pos, 0.0, atol=1e-6) or np.any(np.isnan(target_pos)):
                logger.debug("Invalid target position for %s arm: all zeros or NaN values", side)
                return False
            
            # Get current position from state dict
            current_pos = np.array([
                current_state[f"{side}_hand.position.x"],
                current_state[f"{side}_hand.position.y"],
                current_state[f"{side}_hand.position.z"]
            ])
            
            # Calculate position error
            position_error = target_pos - current_pos
            max_error = np.max(np.abs(position_error))
            
            if not self.is_arm_controlled[side]:
                # the arm is not yet controlled, check if the target position is within the tolerance of the current position
                if max_error < self.position_tolerance:
                    self.is_arm_controlled[side] = True
                    print(f"{side.capitalize()} arm is now controlled (error: {max_error:.3f}m < {self.position_tolerance:.3f}m)")
                else:
                    print(f"{side.capitalize()} arm not ready: position error {max_error:.3f}m > {self.position_tolerance:.3f}m")
            else:
                # the arm was controlled, disable it if the target gets too far
                if max_error > self.position_tolerance*10:
                    self.is_arm_controlled[side] = False
                    print(f"{side.capitalize()} arm disabled: position error {max_error:.3f}m > {self.position_tolerance:.3f}m")

        # Only move if all configured arms are controlled
        controlled_arms = [side for side in ["left", "right"] if self.is_arm_controlled[side]]
        return len(controlled_arms) == len(["left", "right"])
