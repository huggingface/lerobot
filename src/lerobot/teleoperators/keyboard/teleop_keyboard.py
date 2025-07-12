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

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import rtz_to_xyz_delta
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
            self._print_controls()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()

    def _print_controls(self):
        """Print the keyboard controls for the user."""
        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  W key: Wrist flex up")
        print("  S key: Wrist flex down")
        print("  1 key: Wrist roll clockwise")
        print("  2 key: Wrist roll counter-clockwise")
        print("  [ key: Close gripper")
        print("  ] key: Open gripper")
        print("  ESC: Exit")


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    Supports both XYZ cartesian and RTZ polar coordinate systems.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        
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
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "wrist_flex": 3},
            }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        
        # Initialize deltas based on coordinate system
        if self.config.coordinate_system.value == "rtz":
            delta_r = 0.0
            delta_theta = 0.0
            delta_z = 0.0
        else:  # xyz
            delta_x = 0.0
            delta_y = 0.0
            delta_z = 0.0
            
        wrist_roll_action = 0.0
        wrist_flex_action = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if self.config.coordinate_system.value == "rtz":
                # RTZ mode: use polar coordinates
                if key == keyboard.Key.up:
                    delta_r = int(val) * self.config.deltas.get("r", 0.2)  # Radial movement (forward/backward)
                elif key == keyboard.Key.down:
                    delta_r = -int(val) * self.config.deltas.get("r", 0.2)  # Radial movement (forward/backward)
                elif key == keyboard.Key.left:
                    delta_theta = int(val) * self.config.deltas.get("t", 0.05)  # Angular movement (rotation around Z-axis)
                elif key == keyboard.Key.right:
                    delta_theta = -int(val) * self.config.deltas.get("t", 0.05)  # Angular movement (rotation around Z-axis)
                elif key == keyboard.Key.shift:
                    delta_z = int(val) * self.config.deltas.get("z", 1.0)  # Vertical movement (up/down)
                elif key == keyboard.Key.shift_r:
                    delta_z = -int(val) * self.config.deltas.get("z", 1.0)  # Vertical movement (up/down)
            else:
                # XYZ mode: use cartesian coordinates
                if key == keyboard.Key.up:
                    delta_x = int(val) * self.config.deltas.get("x", 1.0)  # Forward/backward
                elif key == keyboard.Key.down:
                    delta_x = -int(val) * self.config.deltas.get("x", 1.0)  # Forward/backward
                elif key == keyboard.Key.left:
                    delta_y = int(val) * self.config.deltas.get("y", 1.0)  # Left/right
                elif key == keyboard.Key.right:
                    delta_y = -int(val) * self.config.deltas.get("y", 1.0)  # Left/right
                elif key == keyboard.Key.shift:
                    delta_z = int(val) * self.config.deltas.get("z", 1.0)  # Vertical movement (up/down)
                elif key == keyboard.Key.shift_r:
                    delta_z = -int(val) * self.config.deltas.get("z", 1.0)  # Vertical movement (up/down)
            
            # Common controls for both modes
            if key == "]":
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == "[":
                gripper_action = int(val) - 1
            elif key == "1":
                wrist_roll_action = int(val)
            elif key == "2":
                wrist_roll_action = -int(val)
            elif key == "w":
                wrist_flex_action = -int(val)
            elif key == "s":
                wrist_flex_action = int(val)
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        # Convert coordinates if needed
        if self.config.coordinate_system.value == "rtz":
            # Convert RTZ polar coordinates to XYZ cartesian coordinates
            delta_x, delta_y, delta_z, self.current_r, self.current_theta, self.current_z = rtz_to_xyz_delta(
                delta_r, delta_theta, delta_z, self.current_r, self.current_theta, self.current_z
            )

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        action_dict["wrist_roll"] = wrist_roll_action
        action_dict["wrist_flex"] = wrist_flex_action

        return action_dict
