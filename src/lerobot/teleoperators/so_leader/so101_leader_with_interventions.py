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
import os
import sys
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError

from ..utils import TeleopEvents
from .config_so_leader import SO101LeaderConfig
from .so_leader import SO101Leader

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

logger = logging.getLogger(__name__)


class SO101LeaderWithInterventions(SO101Leader):
    """
    SO-101 Leader Arm with keyboard intervention support.

    This class extends SO101Leader to add keyboard-based intervention capability.
    When the user presses the intervention key (default 's'), the leader enters
    teleoperator mode and actions will be reflected in the follower.
    """

    def __init__(self, config: SO101LeaderConfig, intervention_key: str = "o", auto_key: str = "p"):
        """
        Initialize SO101LeaderWithInterventions.

        Args:
            config: SO101LeaderConfig instance
            intervention_key: Key to press for enabling intervention (default: 'o')
        """
        super().__init__(config)
        self.intervention_key = intervention_key.lower()
        self.auto_key = auto_key.lower()
        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self._is_intervention_active = False

    @property
    def is_connected(self) -> bool:
        """Check if both the motor bus and keyboard listener are connected."""
        motor_connected = super().is_connected
        keyboard_connected = (
            PYNPUT_AVAILABLE
            and keyboard is not None
            and isinstance(self.listener, keyboard.Listener)
            and self.listener.is_alive()
        )
        return motor_connected and keyboard_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the SO101 leader and start keyboard listener.

        Args:
            calibrate: Whether to calibrate the motors after connecting
        """
        # Connect the motor bus first
        super().connect(calibrate=calibrate)

        # Then start keyboard listener
        if self.listener is not None and self.listener.is_alive():
            raise DeviceAlreadyConnectedError(
                "Keyboard listener is already running. Do not run `connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logger.info("pynput is available - enabling keyboard listener for interventions.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
            logger.info(
                f"Keyboard intervention enabled. Press '{self.intervention_key} / {self.auto_key}' to enable/disable "
                "teleoperator mode."
            )
        else:
            logger.warning("pynput not available - keyboard interventions will not work.")
            self.listener = None

    def disconnect(self) -> None:
        """Disconnect from the SO101 leader and stop keyboard listener."""
        # Stop keyboard listener first
        if self.listener is not None and self.listener.is_alive():
            self.listener.stop()
            self.listener = None
            logger.info("Keyboard listener stopped.")

        # Then disconnect motor bus
        super().disconnect()

    def _on_press(self, key):
        """Handle key press events."""
        if hasattr(key, "char") and key.char:
            key_char = key.char.lower()
            self.event_queue.put((key_char, True))

            # Toggle intervention when intervention key is pressed
            if key_char == self.intervention_key:
                self._is_intervention_active = True
                logger.info("Intervention activated - SO101 leader is now in teleoperator mode.")

            if key_char == self.auto_key:
                self._is_intervention_active = False
                logger.info("Auto mode activated - SO101 leader is now in autonomous mode.")

    def _on_release(self, key):
        """Handle key release events."""
        if hasattr(key, "char") and key.char:
            key_char = key.char.lower()
            self.event_queue.put((key_char, False))

        if key == keyboard.Key.esc:
            logger.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        """Process all pending key events from the queue."""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening (key pressed)
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not PYNPUT_AVAILABLE or self.listener is None or not self.listener.is_alive():
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Process any pending key events
        self._drain_pressed_keys()

        # Check if intervention key is currently pressed
        is_intervention = self._is_intervention_active or self.current_pressed.get(
            self.intervention_key, False
        )

        # Check for episode control commands
        terminate_episode = self.current_pressed.get("t", False)
        success = self.current_pressed.get("s", False)
        rerecord_episode = self.current_pressed.get("r", False)

        # Process any other control keys if needed (similar to KeyboardEndEffectorTeleop)
        # For now, we only handle intervention, but can be extended

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def keyboard_intervention(self):
        """
        Legacy method name - use get_teleop_events() instead.
        Returns whether intervention is currently active.
        """
        events = self.get_teleop_events()
        return events.get(TeleopEvents.IS_INTERVENTION, False)
