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
Keyboard teleoperator for SO-101 MuJoCo robot.
Returns raw keyboard state (which keys are pressed) for the robot to interpret.
"""

import logging
import os
import sys
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.keyboard_event_manager import get_keyboard_manager
from lerobot.utils.keyboard_event_manager import PYNPUT_AVAILABLE

from ..teleoperator import Teleoperator
from .configuration_keyboard import SO101KeyboardTeleopConfig

class SO101KeyboardTeleop(Teleoperator):
    """
    Keyboard teleop for SO-101 that returns raw key states.

    Returns a dict with boolean values for each key (WASD controls to avoid arrow key conflicts):
    {
        "w": bool, "a": bool, "s": bool, "d": bool,  # Movement
        "q": bool, "e": bool,  # Wrist roll
        "r": bool, "f": bool,  # Gripper
        "shift": bool, "ctrl": bool,  # Up/down
    }
    """

    name = "so101_keyboard"
    config_class = SO101KeyboardTeleopConfig

    def __init__(self, config: SO101KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.keyboard_manager = None

    @property
    def action_features(self) -> dict:
        # Return empty since we don't directly control action space
        # The robot's _from_keyboard_to_base_action will handle conversion
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        # Connected if the shared keyboard manager is active
        return self.keyboard_manager is not None and self.keyboard_manager.is_active

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "SO101KeyboardTeleop is already connected. Do not run `connect()` twice."
            )
        
        # Always use the shared keyboard manager (singleton)
        self.keyboard_manager = get_keyboard_manager()
        
        if self.keyboard_manager is None:
            logging.warning("Keyboard event manager not available - SO-101 keyboard teleop cannot connect.")
            return
        
        # Import keyboard here to get Key enums (only if manager is available)
        from pynput import keyboard
        
        # Start the manager if not already started
        if not self.keyboard_manager.is_active:
            logging.info("Starting shared keyboard event manager for SO-101 keyboard teleop.")
            self.keyboard_manager.start()
        else:
            logging.info("Using already-active shared keyboard event manager for SO-101 keyboard teleop.")
        
        # Register handlers for all keys we care about
        for char in ["w", "a", "s", "d", "q", "e", "r", "f", "[", "]", "o", "c"]:
            self.keyboard_manager.register_char_press_handler(
                char, lambda c=char: self.event_queue.put((c, True))
            )
            self.keyboard_manager.register_char_release_handler(
                char, lambda c=char: self.event_queue.put((c, False))
            )
        
        # Register shift and ctrl
        self.keyboard_manager.register_key_press_handler(
            keyboard.Key.shift, lambda: self.event_queue.put(("shift", True))
        )
        self.keyboard_manager.register_key_release_handler(
            keyboard.Key.shift, lambda: self.event_queue.put(("shift", False))
        )
        self.keyboard_manager.register_key_press_handler(
            keyboard.Key.shift_l, lambda: self.event_queue.put(("shift", True))
        )
        self.keyboard_manager.register_key_release_handler(
            keyboard.Key.shift_l, lambda: self.event_queue.put(("shift", False))
        )
        self.keyboard_manager.register_key_press_handler(
            keyboard.Key.ctrl, lambda: self.event_queue.put(("ctrl", True))
        )
        self.keyboard_manager.register_key_release_handler(
            keyboard.Key.ctrl, lambda: self.event_queue.put(("ctrl", False))
        )
        self.keyboard_manager.register_key_press_handler(
            keyboard.Key.ctrl_l, lambda: self.event_queue.put(("ctrl", True))
        )
        self.keyboard_manager.register_key_release_handler(
            keyboard.Key.ctrl_l, lambda: self.event_queue.put(("ctrl", False))
        )

    def calibrate(self) -> None:
        pass  # No calibration needed

    def configure(self) -> None:
        pass  # No configuration needed

    def _drain_pressed_keys(self):
        """Process all queued key events to get current state."""
        while not self.event_queue.empty():
            key_name, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_name] = is_pressed

    def get_action(self) -> dict[str, Any]:
        """
        Get current keyboard state.

        Returns:
            Dict with boolean values for each key (WASD controls):
            {
                "w": bool, "a": bool, "s": bool, "d": bool,
                "q": bool, "e": bool, "r": bool, "f": bool,
                "shift": bool, "ctrl": bool,
            }
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SO101KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Return current state of all keys we care about
        return {
            "w": self.current_pressed.get("w", False),
            "a": self.current_pressed.get("a", False),
            "s": self.current_pressed.get("s", False),
            "d": self.current_pressed.get("d", False),
            "q": self.current_pressed.get("q", False),
            "e": self.current_pressed.get("e", False),
            "r": self.current_pressed.get("r", False),
            "f": self.current_pressed.get("f", False),
            "[": self.current_pressed.get("[", False),
            "]": self.current_pressed.get("]", False),
            "o": self.current_pressed.get("o", False),
            "c": self.current_pressed.get("c", False),
            "shift": self.current_pressed.get("shift", False),
            "ctrl": self.current_pressed.get("ctrl", False),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No feedback for keyboard

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SO101KeyboardTeleop is not connected. You need to run `connect()` before `disconnect()`."
            )
        
        # Note: We don't stop the shared manager here because:
        # 1. It might be used by lerobot_record or other components
        # 2. The manager is a singleton, so stopping it would affect all users
        # 3. lerobot_record.py will stop it when recording is complete
        logging.info("Disconnecting from shared keyboard manager (not stopping it - shared resource)")
        self.keyboard_manager = None
