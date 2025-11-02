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

import math
from typing import Any

from lerobot.utils.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_lekiwi_base_gamepad import LeKiwiBaseTeleopConfig


class LeKiwiBaseTeleop(Teleoperator):
    """Teleoperator that maps an Xbox-style gamepad to LeKiwi base velocity commands."""

    config_class = LeKiwiBaseTeleopConfig
    name = "lekiwi_base_gamepad"

    def __init__(self, config: LeKiwiBaseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._pygame = None
        self._joystick = None
        self._clock = None
        self._last_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    @property
    def action_features(self) -> dict[str, type]:
        return {"x.vel": float, "y.vel": float, "theta.vel": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # not used for gamepad teleop
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for LeKiwiBaseTeleop but is not installed. "
                "Install it with `pip install pygame`."
            ) from exc

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= self.config.joystick_index:
            raise RuntimeError("No joystick detected at the configured index.")

        self._pygame = pygame
        self._joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self._joystick.init()
        self._clock = pygame.time.Clock()
        self._last_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

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

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected or self._pygame is None:
            raise DeviceNotConnectedError("LeKiwi base teleoperator is not connected.")

        pygame = self._pygame
        joy_removed = getattr(pygame, "JOYDEVICEREMOVED", None)
        for event in pygame.event.get():
            if joy_removed is not None and event.type == joy_removed:
                self.disconnect()
                raise DeviceNotConnectedError("Gamepad disconnected.")

        axis_x = self._joystick.get_axis(self.config.axis_x)
        axis_y = self._joystick.get_axis(self.config.axis_y)
        if self.config.invert_x:
            axis_x = -axis_x
        if self.config.invert_y:
            axis_y = -axis_y

        norm_x = self._apply_deadzone(axis_x)
        norm_y = self._apply_deadzone(axis_y)

        if self.config.normalize_diagonal:
            magnitude = math.hypot(norm_x, norm_y)
            if magnitude > 1.0:
                scale = 1.0 / magnitude
                norm_x *= scale
                norm_y *= scale

        x_vel, y_vel = self._vector_to_velocities(norm_x, norm_y)
        theta_vel = self._read_yaw_velocity()

        action = {"x.vel": x_vel, "y.vel": y_vel, "theta.vel": theta_vel}
        self._last_action = action

        if self._clock and self.config.polling_fps > 0:
            self._clock.tick(self.config.polling_fps)

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self._joystick is not None:
            # Not all pygame versions expose quit() on Joystick, guard accordingly.
            if hasattr(self._joystick, "quit"):
                self._joystick.quit()
            self._joystick = None
        if self._pygame is not None:
            self._pygame.joystick.quit()
            self._pygame = None
        self._clock = None

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.config.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))

    def _vector_to_velocities(self, x_axis: float, y_axis: float) -> tuple[float, float]:
        x_vel = y_axis * self.config.max_speed_mps
        y_vel = x_axis * self.config.max_speed_mps
        return -x_vel, y_vel

    def _read_yaw_velocity(self) -> float:
        if not self.is_connected:
            return 0.0

        joystick = self._joystick
        if joystick.get_numhats() > self.config.hat_index:
            hat = joystick.get_hat(self.config.hat_index)
            hx = hat[0]
        else:
            left = joystick.get_button(self.config.dpad_left_button)
            right = joystick.get_button(self.config.dpad_right_button)
            hx = -1 if left else 0
            hx += 1 if right else 0

        return -self.config.yaw_speed_deg * hx
