#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ....teleoperator import Teleoperator
from .config_panthera_keyboard_ee import PantheraKeyboardEETeleopConfig

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logger.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logger.info("Could not import pynput: %s", e)


class PantheraKeyboardEETeleop(Teleoperator):
    """Keyboard teleop for Panthera end-effector Cartesian + orientation control."""

    config_class = PantheraKeyboardEETeleopConfig
    name = "panthera_keyboard_ee"

    def __init__(self, config: PantheraKeyboardEETeleopConfig):
        super().__init__(config)
        self.config = config
        self.listener = None
        self.event_queue: Queue[tuple[str, bool]] = Queue()
        self.current_pressed: dict[str, bool] = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_roll": float,
            "delta_pitch": float,
            "delta_yaw": float,
            "gripper": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not available - Panthera keyboard EE teleop disabled.")
            self.listener = None
            return

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        logger.info("Panthera keyboard EE teleop connected.")

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _on_press(self, key: Any) -> None:
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char.lower(), True))

    def _on_release(self, key: Any) -> None:
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char.lower(), False))
        if keyboard is not None and key == keyboard.Key.esc:
            self.disconnect()

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        active = {key for key, pressed in self.current_pressed.items() if pressed}

        delta_x = float((self.config.key_x_pos in active) - (self.config.key_x_neg in active))
        delta_y = float((self.config.key_y_pos in active) - (self.config.key_y_neg in active))
        delta_z = float((self.config.key_z_pos in active) - (self.config.key_z_neg in active))
        delta_roll = float((self.config.key_roll_pos in active) - (self.config.key_roll_neg in active))
        delta_pitch = float((self.config.key_pitch_pos in active) - (self.config.key_pitch_neg in active))
        delta_yaw = float((self.config.key_yaw_pos in active) - (self.config.key_yaw_neg in active))

        gripper = 1.0
        if self.config.key_gripper_open in active:
            gripper = 2.0
        elif self.config.key_gripper_close in active:
            gripper = 0.0

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "delta_roll": delta_roll,
            "delta_pitch": delta_pitch,
            "delta_yaw": delta_yaw,
            "gripper": gripper,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
