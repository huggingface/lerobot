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

from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.common.teleoperators.spacemouse.configuration_spacemouse import SpacemouseTeleopConfig

import pyspacemouse

class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class SpacemouseTeleop(Teleoperator):
    """
    Teleop class to use spacemouse inputs for control.
    """

    config_class = SpacemouseTeleopConfig

    name = "spacemouse"

    def __init__(self, config: SpacemouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self._connected = False
        # Gripper toggle state: assume starts OPEN
        self._gripper_state: int = GripperAction.OPEN.value
        self._prev_button_state: int = 0

        # Background reader thread vars (used to keep only the latest state)
        self._latest_state = None  # will store the most recent raw state coming from the driver
        self._reader_thread = None
        self._stop_reader = False

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5, "gripper": 6},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        """Connect to the SpaceMouse device (real or mock)."""
        # Map short device aliases to full names for convenience
        device_alias: str | None
        if self.config.device in {"e", "E"}:
            device_alias = "SpacePilot Enterprise"
        elif self.config.device in {"p", "P"}:
            device_alias = "SpaceMouse Pro"
        else:
            # Treat "", "default", or None as auto-detect (i.e., no device argument)
            if not self.config.device or self.config.device.lower() in {"default", "auto"}:
                device_alias = None
            else:
                device_alias = self.config.device

        # Call open with or without explicit device name depending on alias resolution
        if device_alias is None:
            self._connected = bool(pyspacemouse.open())
        else:
            self._connected = bool(pyspacemouse.open(device=device_alias))

        # Start background reader to avoid piling up driver messages (reduces perceived latency)
        if self._connected:
            import threading

            def _reader_loop():
                """Continuously poll the driver so its internal queue stays empty.

                We only keep the most recent state which `get_action` then consumes. This
                prevents buildup when the driver polls faster than the main control loop
                (e.g. teleoperate.py's ≈60 Hz loop vs. SpaceMouse ≈125 Hz updates).
                """

                while not self._stop_reader:
                    try:
                        self._latest_state = pyspacemouse.read()
                    except Exception:
                        # In case device is unplugged mid-run; exit thread gracefully
                        break

            self._stop_reader = False
            self._reader_thread = threading.Thread(target=_reader_loop, daemon=True)
            self._reader_thread.start()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected():
            raise RuntimeError("SpaceMouse is not connected. Call connect() first.")

        # Prefer the state produced by the background reader (most recent),
        # fall back to direct read if thread hasn't produced anything yet.
        state = self._latest_state if self._latest_state is not None else pyspacemouse.read()

        deltas = [
            state.y ** 3,
            -state.x ** 3,
            state.z ** 3,
            state.roll,
            state.pitch,
            -state.yaw,
        ]

        # Clamp, apply deadzone & scaling
        for i in range(6):
            # Clamp to [-1, 1]
            if deltas[i] > 1.0:
                deltas[i] = 1.0
            elif deltas[i] < -1.0:
                deltas[i] = -1.0

            # Deadzone
            if abs(deltas[i]) < self.config.deadzone:
                deltas[i] = 0.0

            # Scale translation vs rotation
            if i < 3:
                deltas[i] *= self.config.translation_scale
            else:
                deltas[i] *= self.config.rotation_scale

        # Additional yaw scaling
        deltas[5] *= self.config.yaw_scale

        spacemouse_action = np.array(deltas, dtype=np.float32)

        action_dict = {
            "delta_x": spacemouse_action[0],
            "delta_y": spacemouse_action[1],
            "delta_z": spacemouse_action[2],
            "delta_roll": spacemouse_action[3],
            "delta_pitch": spacemouse_action[4],
            "delta_yaw": spacemouse_action[5],
        }

        # Simple gripper control: right button (index 1) toggles open/close.
        # Assumption: the physical gripper starts in the OPEN state when the teleop script boots.
        # Each button press switches the command between OPEN and CLOSE accordingly.
        if self.config.use_gripper and hasattr(state, "buttons") and len(state.buttons) >= 2:
            # Rising-edge detection on right button (index 1) to toggle gripper state
            btn = state.buttons[1]
            if btn and not self._prev_button_state:
                # Toggle the stored gripper state
                self._gripper_state = (
                    GripperAction.CLOSE.value
                    if self._gripper_state == GripperAction.OPEN.value
                    else GripperAction.OPEN.value
                )
            self._prev_button_state = btn

            action_dict["gripper"] = self._gripper_state

        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the spacemouse."""
        # pyspacemouse does not expose an explicit close API but we reset connection flag
        self._connected = False

        # Stop reader thread if running
        self._stop_reader = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.1)
            self._reader_thread = None

    def is_connected(self) -> bool:
        """Check if spacemouse is connected."""
        return self._connected

    def calibrate(self) -> None:
        """Calibrate the spacemouse."""
        # No calibration needed for spacemouse
        pass

    def is_calibrated(self) -> bool:
        """Check if spacemouse is calibrated."""
        # Spacemouse doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the spacemouse."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the spacemouse."""
        # Spacemouse doesn't support feedback
        pass


if __name__ == "__main__":
    # Quick test: create a configuration and read a few actions (auto-detect device)
    config = SpacemouseTeleopConfig(use_gripper=True, device="")
    
    # Initialize the SpacemouseTeleop
    teleop = SpacemouseTeleop(config)
    
    # Connect to the spacemouse
    teleop.connect()
    
    # Check if connected
    if teleop.is_connected():
        print("Connected to spacemouse.")
        
        import time
        print("Streaming actions…  Press Ctrl+C to stop.")
        try:
            while True:
                action = teleop.get_action()
                print("Action:", action)
        except KeyboardInterrupt:
            print("Stopping.")
        finally:
            teleop.disconnect()
    else:
        print("Failed to connect to spacemouse.")
