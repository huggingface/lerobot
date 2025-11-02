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

import numpy as np
from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config import XLeRobotMountGamepadTeleopConfig


class XLeRobotMountGamepadTeleop(Teleoperator):
    
    config_class = XLeRobotMountGamepadTeleopConfig
    name = "xlerobot_mount_gamepad"

    def __init__(self, config: XLeRobotMountGamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self._pygame = None
        self._joystick = None
        self._clock = None

        # Track current position for incremental control (initialized lazily from observation)
        self._current_pan: float | None = None
        self._current_tilt: float | None = None
        self._pan_bias = 0.0
        self._tilt_bias = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "mount_pan.pos": float,
            "mount_tilt.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # not used for gamepad teleop

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for XLeRobotMountGamepadTeleop but is not installed. "
                "Install it with `pip install pygame`."
            ) from exc

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= self.config.joystick_index:
            raise RuntimeError(
                f"No joystick detected at index {self.config.joystick_index}. "
                f"Found {pygame.joystick.get_count()} joystick(s)."
            )

        self._pygame = pygame
        self._joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self._joystick.init()
        self._clock = pygame.time.Clock()

        # Reset current readings; they will be synced from observations if available
        self._current_pan = None
        self._current_tilt = None
        # Record neutral stick offsets so we can compensate for non-zero resting values
        self._pan_bias = self._joystick.get_axis(self.config.pan_axis)
        self._tilt_bias = self._joystick.get_axis(self.config.tilt_axis)

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def calibrate(self) -> None:
        return None

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        return None

    def on_observation(self, robot_obs: dict[str, Any]) -> None:
        if not isinstance(robot_obs, dict):
            return

        pan_val = robot_obs.get("mount_pan.pos")
        tilt_val = robot_obs.get("mount_tilt.pos")

        if isinstance(pan_val, (int, float)):
            self._current_pan = float(pan_val)

        if isinstance(tilt_val, (int, float)):
            self._current_tilt = float(tilt_val)

    def get_action(self) -> dict[str, Any]:
        
        if not self.is_connected or self._pygame is None:
            raise DeviceNotConnectedError("Mount gamepad teleoperator is not connected.")

        pygame = self._pygame

        # Handle gamepad disconnect events
        joy_removed = getattr(pygame, "JOYDEVICEREMOVED", None)
        for event in pygame.event.get():
            if joy_removed is not None and event.type == joy_removed:
                self.disconnect()
                raise DeviceNotConnectedError("Gamepad disconnected.")

        # Read right stick axes
        raw_pan = self._joystick.get_axis(self.config.pan_axis)
        raw_tilt = self._joystick.get_axis(self.config.tilt_axis)

        # Compensate for resting offsets (e.g., triggers default to -1)
        pan_input = raw_pan - self._pan_bias
        tilt_input = raw_tilt - self._tilt_bias

        # Apply axis inversions
        if self.config.invert_pan:
            pan_input = -pan_input
        if self.config.invert_tilt:
            tilt_input = -tilt_input

        # Apply deadzone
        pan_input = self._apply_deadzone(pan_input)
        tilt_input = self._apply_deadzone(tilt_input)

        # Convert to velocity and integrate (incremental control)
        if self._current_pan is None:
            self._current_pan = 0.0
        if self._current_tilt is None:
            self._current_tilt = 0.0

        dt = 1.0 / self.config.polling_fps
        self._current_pan += pan_input * self.config.max_pan_speed_dps * dt
        self._current_tilt += tilt_input * self.config.max_tilt_speed_dps * dt

        # Clamp to safe ranges
        self._current_pan = np.clip(
            self._current_pan,
            self.config.pan_range[0],
            self.config.pan_range[1]
        )
        self._current_tilt = np.clip(
            self._current_tilt,
            self.config.tilt_range[0],
            self.config.tilt_range[1]
        )

        action = {
            "mount_pan.pos": self._current_pan,
            "mount_tilt.pos": self._current_tilt,
        }

        # Rate limit control loop
        if self._clock and self.config.polling_fps > 0:
            self._clock.tick(self.config.polling_fps)

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:

        del feedback

    def disconnect(self) -> None:
        if self._joystick is not None:
            # Not all pygame versions expose quit() on Joystick, guard accordingly
            if hasattr(self._joystick, "quit"):
                try:
                    self._joystick.quit()
                except Exception:
                    pass
            self._joystick = None

        if self._pygame is not None:
            try:
                if hasattr(self._pygame, "joystick"):
                    self._pygame.joystick.quit()
            except Exception:
                pass
            try:
                if hasattr(self._pygame, "quit"):
                    self._pygame.quit()
            except Exception:
                pass
            self._pygame = None

        self._clock = None

    def _apply_deadzone(self, value: float) -> float:
        
        if abs(value) < self.config.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))
