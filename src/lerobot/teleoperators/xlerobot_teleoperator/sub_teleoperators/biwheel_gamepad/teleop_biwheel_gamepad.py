#!/usr/bin/env python

# Example teleoperation command for the biwheel base using this gamepad teleop:
#
# lerobot-teleoperate \
#     --robot.type=biwheel \
#     --robot.port=/dev/ttyACM4 \
#     --robot.id=biwheel_base \
#     --teleop.type=biwheel_gamepad \
#     --teleop.joystick_index=0 \
#     --teleop.id=gamepad \
#     --display_data=true

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

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus

from lerobot.utils.errors import DeviceNotConnectedError

from ....teleoperator import Teleoperator
from .config_biwheel_gamepad import BiwheelGamepadTeleopConfig


@dataclass
class GamepadCalibration:
    axis_forward: int
    axis_yaw: int
    invert_forward: bool
    invert_yaw: bool


class BiwheelGamepadTeleop(Teleoperator):
    """Teleoperator that maps an Xbox-style gamepad to differential base commands."""

    config_class = BiwheelGamepadTeleopConfig
    name = "biwheel_gamepad"

    def __init__(self, config: BiwheelGamepadTeleopConfig):
        # _load_calibration which in turn calls _apply_calibration_to_config that needs self.config
        self.config = config
        super().__init__(config)
        self._pygame = None
        self._joystick = None
        self._clock = None
        self._last_action = {"x.vel": 0.0, "theta.vel": 0.0}
        calibration = self.calibration if isinstance(self.calibration, GamepadCalibration) else None
        self.calibration: GamepadCalibration | None = calibration
        self._calibrated = calibration is not None
        if calibration is not None:
            self._apply_calibration_to_config()

    @property
    def action_features(self) -> dict[str, type]:
        return {"x.vel": float, "theta.vel": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for BiwheelGamepadTeleop but is not installed. "
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
        self._last_action = {"x.vel": 0.0, "theta.vel": 0.0}

        if calibrate and not self.is_calibrated:
            self.calibrate()
        elif self.calibration is not None:
            self._apply_calibration_to_config()
            self._calibrated = True

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def calibrate(self) -> None:
        if not self.is_connected or self._pygame is None:
            raise DeviceNotConnectedError("Biwheel gamepad teleoperator is not connected.")

        axis_threshold = 0.4
        release_threshold = 0.2
        poll_interval = 0.05

        if self.calibration is not None:
            user_input = input(
                "Press ENTER to use existing calibration, or type 'c' then ENTER to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                print("Using saved calibration.")
                self._apply_calibration_to_config()
                self._calibrated = True
                return

        print("Calibration started. Follow the prompts.")
        forward_axis, invert_forward = self._detect_axis(
            prompt="Push the stick forward (away from you) to identify the forward axis.",
            exclude_axes=set(),
            axis_threshold=axis_threshold,
            release_threshold=release_threshold,
            poll_interval=poll_interval,
        )
        yaw_axis, invert_yaw = self._detect_axis(
            prompt="Push the stick to the right to identify the yaw axis.",
            exclude_axes={forward_axis},
            axis_threshold=axis_threshold,
            release_threshold=release_threshold,
            poll_interval=poll_interval,
        )

        self.config.axis_forward = forward_axis
        self.config.axis_yaw = yaw_axis
        self.config.invert_forward = invert_forward
        self.config.invert_yaw = invert_yaw

        self.calibration = GamepadCalibration(
            axis_forward=forward_axis,
            axis_yaw=yaw_axis,
            invert_forward=invert_forward,
            invert_yaw=invert_yaw,
        )
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")
        self._calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected or self._pygame is None:
            raise DeviceNotConnectedError("Biwheel gamepad teleoperator is not connected.")

        pygame = self._pygame
        joy_removed = getattr(pygame, "JOYDEVICEREMOVED", None)
        for event in pygame.event.get():
            if joy_removed is not None and event.type == joy_removed:
                self.disconnect()
                raise DeviceNotConnectedError("Gamepad disconnected.")

        axis_forward = self._joystick.get_axis(self.config.axis_forward)
        axis_yaw = self._joystick.get_axis(self.config.axis_yaw)

        if self.config.invert_forward:
            axis_forward = -axis_forward
        if self.config.invert_yaw:
            axis_yaw = -axis_yaw

        forward_cmd = self._apply_deadzone(axis_forward) * self.config.max_speed_mps
        yaw_cmd = self._apply_deadzone(axis_yaw) * self.config.yaw_speed_deg

        action = {"x.vel": forward_cmd, "theta.vel": yaw_cmd}
        self._last_action = action

        if self._clock and self.config.polling_fps > 0:
            self._clock.tick(self.config.polling_fps)

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self._joystick is not None:
            if hasattr(self._joystick, "quit"):
                with suppress(Exception):
                    self._joystick.quit()
            self._joystick = None

        if self._pygame is not None:
            with suppress(Exception):
                if hasattr(self._pygame, "joystick"):
                    self._pygame.joystick.quit()
            self._pygame = None

        self._clock = None

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.config.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))

    def _detect_axis(
        self,
        prompt: str,
        exclude_axes: set[int],
        axis_threshold: float,
        release_threshold: float,
        poll_interval: float,
    ) -> tuple[int, bool]:
        assert self._pygame is not None and self._joystick is not None
        pygame = self._pygame
        joystick = self._joystick

        print(prompt)
        pygame.event.pump()
        baseline = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]

        while True:
            pygame.event.pump()
            num_axes = joystick.get_numaxes()
            best_axis = None
            best_delta = 0.0
            for idx in range(num_axes):
                if idx in exclude_axes:
                    continue
                delta = joystick.get_axis(idx) - baseline[idx]
                if abs(delta) > abs(best_delta):
                    best_delta = delta
                    best_axis = idx
            if best_axis is not None and abs(best_delta) >= axis_threshold:
                invert = best_delta < 0
                direction = "negative" if invert else "positive"
                print(f"Detected axis {best_axis} ({direction})")
                print("Release the stick to continue.")
                while abs(joystick.get_axis(best_axis) - baseline[best_axis]) > release_threshold:
                    pygame.event.pump()
                    time.sleep(poll_interval)
                return best_axis, invert
            time.sleep(poll_interval)

    def _apply_calibration_to_config(self) -> None:
        if self.calibration is None:
            return
        self.config.axis_forward = self.calibration.axis_forward
        self.config.axis_yaw = self.calibration.axis_yaw
        self.config.invert_forward = self.calibration.invert_forward
        self.config.invert_yaw = self.calibration.invert_yaw

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            calibration = draccus.load(GamepadCalibration, f)
        self.calibration = calibration
        self._apply_calibration_to_config()
        self._calibrated = True

    def _save_calibration(self, fpath: Path | None = None) -> None:
        if self.calibration is None:
            return
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)
