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

"""Xbox gamepad teleoperation for XLeRobot mount (pan/tilt control)."""

import numpy as np
from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config import XLeRobotMountGamepadTeleopConfig


class XLeRobotMountGamepadTeleop(Teleoperator):
    """Teleoperator that maps Xbox controller right stick to mount pan/tilt positions.

    The right stick controls the mount with velocity integration:
    - Right stick horizontal: Controls pan (left/right rotation)
    - Right stick vertical: Controls tilt (up/down rotation)

    Holding the stick moves the mount continuously, while releasing it stops
    the motion at the current position. Safety limits prevent exceeding
    the configured pan/tilt ranges.

    Example:
        ```python
        from lerobot.teleoperators.xlerobot_mount_gamepad import (
            XLeRobotMountGamepadTeleop,
            XLeRobotMountGamepadTeleopConfig,
        )

        config = XLeRobotMountGamepadTeleopConfig()
        teleop = XLeRobotMountGamepadTeleop(config)
        teleop.connect()

        # Get action from controller
        action = teleop.get_action()
        # action = {"mount_pan.pos": 30.0, "mount_tilt.pos": 15.0}

        teleop.disconnect()
        ```
    """

    config_class = XLeRobotMountGamepadTeleopConfig
    name = "xlerobot_mount_gamepad"

    def __init__(self, config: XLeRobotMountGamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self._pygame = None
        self._joystick = None
        self._clock = None

        # Track current position for incremental control
        self._current_pan = 0.0
        self._current_tilt = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        """Define action features: pan and tilt positions."""
        return {
            "mount_pan.pos": float,
            "mount_tilt.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        """No feedback features for this teleoperator."""
        return {}

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the Xbox gamepad.

        Args:
            calibrate: Unused parameter (kept for interface compatibility).

        Raises:
            RuntimeError: If pygame is not installed or no joystick detected.
        """
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

        # Initialize position to center
        self._current_pan = 0.0
        self._current_tilt = 0.0

    @property
    def is_connected(self) -> bool:
        """Check if the gamepad is connected."""
        return self._joystick is not None

    def calibrate(self) -> None:
        """No calibration needed for gamepad."""
        return None

    @property
    def is_calibrated(self) -> bool:
        """Gamepad is always considered calibrated."""
        return True

    def configure(self) -> None:
        """No configuration needed for gamepad."""
        return None

    def get_action(self) -> dict[str, Any]:
        """Read Xbox right stick and compute pan/tilt position commands.

        The method:
        1. Reads right stick axes (horizontal for pan, vertical for tilt)
        2. Applies deadzone filtering
        3. Converts stick deflection to velocity
        4. Integrates velocity to update position
        5. Clamps position to safety limits

        Returns:
            Dictionary with target pan/tilt positions in degrees:
            {
                "mount_pan.pos": float,
                "mount_tilt.pos": float,
            }

        Raises:
            DeviceNotConnectedError: If gamepad is not connected.
        """
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
        pan_input = self._joystick.get_axis(self.config.pan_axis)
        tilt_input = self._joystick.get_axis(self.config.tilt_axis)

        # Apply axis inversions
        if self.config.invert_pan:
            pan_input = -pan_input
        if self.config.invert_tilt:
            tilt_input = -tilt_input

        # Apply deadzone
        pan_input = self._apply_deadzone(pan_input)
        tilt_input = self._apply_deadzone(tilt_input)

        # Convert to velocity and integrate (incremental control)
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
        """No feedback mechanism for this teleoperator.

        Args:
            feedback: Ignored feedback dictionary.
        """
        del feedback

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self._joystick is not None:
            # Not all pygame versions expose quit() on Joystick, guard accordingly
            if hasattr(self._joystick, "quit"):
                self._joystick.quit()
            self._joystick = None

        if self._pygame is not None:
            self._pygame.joystick.quit()
            self._pygame = None

        self._clock = None

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone filtering to joystick axis value.

        Args:
            value: Raw joystick axis value (-1.0 to 1.0).

        Returns:
            Filtered value: 0.0 if within deadzone, otherwise clamped to [-1.0, 1.0].
        """
        if abs(value) < self.config.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))
