# !/usr/bin/env python

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

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import rtz_to_xyz_delta
from .configuration_gamepad import GamepadTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config

        self.gamepad = None
        
        # Track current polar position for coordinate conversion (RTZ mode)
        self.current_r = 0.0
        self.current_theta = 0.0
        self.current_z = 0.0

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3, "wrist_roll": 4, "wrist_flex": 5},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (5,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "wrist_roll": 3, "wrist_flex": 4},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        # if sys.platform == "darwin":
        #     # NOTE: On macOS, pygame doesn't reliably detect input from some controllers so we fall back to hidapi
        #     from .gamepad_utils import GamepadControllerHID as Gamepad
        # else:
        from .gamepad_utils import GamepadController as Gamepad

        # Determine step sizes based on coordinate system
        if self.config.coordinate_system.value == "rtz":
            x_step_size = self.config.deltas.get("r", 1.0)
            y_step_size = self.config.deltas.get("t", 1.0)
            z_step_size = self.config.deltas.get("z", 1.0)
        else:  # xyz
            x_step_size = self.config.deltas.get("x", 1.0)
            y_step_size = self.config.deltas.get("y", 1.0)
            z_step_size = self.config.deltas.get("z", 1.0)

        self.gamepad = Gamepad(
            x_step_size=x_step_size,
            y_step_size=y_step_size,
            z_step_size=z_step_size
        )
        self.gamepad.start()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Apply coordinate transformation if using RTZ
        if self.config.coordinate_system.value == "rtz":
            # In RTZ mode, the gamepad inputs are interpreted as:
            # delta_x (from gamepad) -> delta_r (radial)
            # delta_y (from gamepad) -> delta_theta (angular)
            # delta_z (from gamepad) -> delta_z (vertical)
            delta_x, delta_y, delta_z, self.current_r, self.current_theta, self.current_z = rtz_to_xyz_delta(
                delta_x, delta_y, delta_z, self.current_r, self.current_theta, self.current_z
            )

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Handle gripper control based on mode
        if self.config.use_gripper:
            # Use toggle control for both normal and pick and place modes
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        # Add wrist roll control
        wrist_roll_delta = self.gamepad.get_wrist_roll_delta()
        action_dict["wrist_roll"] = wrist_roll_delta

        # Add wrist flex control
        wrist_flex_delta = self.gamepad.get_wrist_flex_delta()
        action_dict["wrist_flex"] = wrist_flex_delta

        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
