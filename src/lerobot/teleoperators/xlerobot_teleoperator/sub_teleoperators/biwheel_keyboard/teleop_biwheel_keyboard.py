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
import time
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ....teleoperator import Teleoperator
from ....utils import TeleopEvents
from .configuration_biwheel_keyboard import BiwheelKeyboardTeleopConfig

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


class SmoothBaseController:
    """
    Simplified smooth base controller with acceleration/deceleration.

    This controller implements linear acceleration when keys are pressed and
    linear deceleration when keys are released, providing smooth motion control
    for differential drive mobile bases.
    """

    def __init__(self, config: BiwheelKeyboardTeleopConfig):
        self.config = config
        self.current_speed = 0.0
        self.last_time = time.time()
        self.last_direction = {"x.vel": 0.0, "theta.vel": 0.0}
        self.is_moving = False
        self.speed_index = config.initial_speed_index

        # Validate speed_index
        if not (0 <= self.speed_index < len(config.speed_levels)):
            logging.warning(
                f"Initial speed index {self.speed_index} out of range. "
                f"Setting to {len(config.speed_levels) // 2}"
            )
            self.speed_index = len(config.speed_levels) // 2

    def change_speed_level(self, delta: int) -> None:
        """Change the speed level by delta."""
        old_index = self.speed_index
        self.speed_index = max(0, min(len(self.config.speed_levels) - 1, self.speed_index + delta))

        if old_index != self.speed_index and self.config.debug:
            level = self.config.speed_levels[self.speed_index]
            logging.info(
                f"[BASE] Speed level changed: {old_index + 1} -> {self.speed_index + 1} "
                f"(Linear: {level['linear']:.2f}m/s, Angular: {level['angular']:.0f}°/s)"
            )

    def update(self, pressed_keys: set) -> dict[str, float]:
        """
        Update smooth control and return base action.

        Args:
            pressed_keys: Set of currently pressed key characters

        Returns:
            Dictionary with 'x.vel' (linear velocity) and 'theta.vel' (angular velocity)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Check if any base movement keys are pressed
        base_keys = [
            self.config.key_forward,
            self.config.key_backward,
            self.config.key_rotate_left,
            self.config.key_rotate_right,
        ]
        any_key_pressed = any(key in pressed_keys for key in base_keys)

        # Handle speed level changes
        if self.config.key_speed_up in pressed_keys:
            self.change_speed_level(1)
        if self.config.key_speed_down in pressed_keys:
            self.change_speed_level(-1)

        # Calculate base action
        base_action = {"x.vel": 0.0, "theta.vel": 0.0}

        if any_key_pressed:
            # Keys pressed - calculate direction and accelerate
            if not self.is_moving:
                self.is_moving = True
                if self.config.debug:
                    logging.info("[BASE] Starting acceleration")

            # Get current speed level settings
            speed_setting = self.config.speed_levels[self.speed_index]
            linear_speed = speed_setting["linear"]
            angular_speed = speed_setting["angular"]

            # Calculate direction based on pressed keys
            if self.config.key_forward in pressed_keys:
                base_action["x.vel"] += linear_speed
            if self.config.key_backward in pressed_keys:
                base_action["x.vel"] -= linear_speed
            if self.config.key_rotate_left in pressed_keys:
                base_action["theta.vel"] += angular_speed
            if self.config.key_rotate_right in pressed_keys:
                base_action["theta.vel"] -= angular_speed

            # Store current direction for deceleration
            self.last_direction = base_action.copy()

            # Accelerate
            self.current_speed += self.config.acceleration_rate * dt
            self.current_speed = min(self.current_speed, self.config.max_speed_multiplier)

        else:
            # No keys pressed - decelerate
            if self.is_moving:
                self.is_moving = False
                if self.config.debug:
                    logging.info("[BASE] Starting deceleration")

            # Use last direction for deceleration
            if self.current_speed > 0.01 and self.last_direction:
                base_action = self.last_direction.copy()

            # Decelerate
            self.current_speed -= self.config.deceleration_rate * dt
            self.current_speed = max(self.current_speed, 0.0)

        # Apply speed multiplier
        if base_action:
            for key in base_action:
                if "vel" in key:
                    original_value = base_action[key]
                    base_action[key] *= self.current_speed

                    # Ensure minimum velocity during deceleration
                    if (
                        self.current_speed > 0.01
                        and abs(base_action[key]) < self.config.min_velocity_threshold
                    ):
                        base_action[key] = (
                            self.config.min_velocity_threshold
                            if original_value > 0
                            else -self.config.min_velocity_threshold
                        )

        # Debug output
        if self.config.debug:
            if any_key_pressed:
                logging.debug(f"[BASE] ACCEL: Speed={self.current_speed:.2f}, Action={base_action}")
            elif self.current_speed > 0.01:
                logging.debug(f"[BASE] DECEL: Speed={self.current_speed:.2f}, Action={base_action}")
            elif self.is_moving:
                logging.debug(f"[BASE] STOPPED: Speed={self.current_speed:.2f}")

        return base_action


class BiwheelKeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for bidirectional wheel (differential drive) control.

    This teleoperator provides smooth acceleration/deceleration control for mobile bases
    with differential drive kinematics. It supports configurable key mappings, multiple
    speed levels, and smooth transitions between motion states.
    """

    config_class = BiwheelKeyboardTeleopConfig
    name = "biwheel_keyboard"

    def __init__(self, config: BiwheelKeyboardTeleopConfig):
        self.config = config
        super().__init__(config)

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

        # Initialize smooth controller
        self.smooth_controller = SmoothBaseController(config)

        # Queue for misc keys (for teleop events)
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        """Define the action space for differential drive control."""
        return {
            "dtype": "float32",
            "shape": (2,),
            "names": {"x.vel": 0, "theta.vel": 1},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Biwheel Keyboard is already connected. Do not run `connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener for biwheel control.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
            self._print_keymap()
        else:
            logging.warning("pynput not available - keyboard teleop will not work.")
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
                "BiwheelKeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        # Update current pressed keys
        self._drain_pressed_keys()

        # Get set of currently pressed keys
        pressed_keys = {key for key, val in self.current_pressed.items() if val}

        # Check for misc keys and add to queue
        control_keys = {
            self.config.key_forward,
            self.config.key_backward,
            self.config.key_rotate_left,
            self.config.key_rotate_right,
            self.config.key_speed_up,
            self.config.key_speed_down,
            self.config.key_quit,
        }

        for key in pressed_keys:
            if key not in control_keys:
                self.misc_keys_queue.put(key)

        # Get smooth base action with acceleration/deceleration
        action = self.smooth_controller.update(pressed_keys)

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "BiwheelKeyboardTeleop is not connected. You need to run `connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings (beyond movement keys):
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key or config.key_quit = quit episode

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            self.config.key_forward,
            self.config.key_backward,
            self.config.key_rotate_left,
            self.config.key_rotate_right,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
                terminate_episode = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == self.config.key_quit:
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def _print_keymap(self):
        """Print the keymap information to console."""

        print("\n📱 Base Control (Differential Drive):")
        print(f"    {self.config.key_forward}: Forward")
        print(f"    {self.config.key_backward}: Backward")
        print(f"    {self.config.key_rotate_left}: Rotate Left")
        print(f"    {self.config.key_rotate_right}: Rotate Right")
        print(f"    {self.config.key_speed_up}: Speed Up")
        print(f"    {self.config.key_speed_down}: Speed Down")
        print(f"    {self.config.key_quit}: Quit")

        print("\n Speed Configuration:")
        print(f"   Current Level: {self.smooth_controller.speed_index + 1}/{len(self.config.speed_levels)}")
        print("   Speed Levels:")
        for i, level in enumerate(self.config.speed_levels):
            marker = "→" if i == self.smooth_controller.speed_index else " "
            print(
                f"      {marker} Level {i + 1}: Linear {level['linear']:.2f}m/s, Angular {level['angular']:.0f}°/s"
            )
