#!/usr/bin/env python

# Example teleoperation command for the LeKiwi mobile base using this gamepad teleop:
#
# lerobot-teleoperate \
#     --robot.type=lekiwi_base \
#     --robot.port=/dev/ttyACM2 \
#     --robot.id=lekiwi_base \
#     --teleop.type=lekiwi_base_gamepad \
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

import math
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus

from lerobot.utils.errors import DeviceNotConnectedError

from ....teleoperator import Teleoperator
from .config_lekiwi_base_gamepad import LeKiwiBaseTeleopConfig


@dataclass
class GamepadCalibration:
    axis_x: int
    axis_y: int
    invert_x: bool
    invert_y: bool
    hat_index: int
    dpad_left_button: int
    dpad_right_button: int


class LeKiwiBaseTeleop(Teleoperator):
    """Teleoperator that maps an Xbox-style gamepad to LeKiwi base velocity commands."""

    config_class = LeKiwiBaseTeleopConfig
    name = "lekiwi_base_gamepad"

    def __init__(self, config: LeKiwiBaseTeleopConfig):
        self.config = config
        super().__init__(config)
        self._pygame = None
        self._joystick = None
        self._clock = None
        self._last_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        calibration = self.calibration if isinstance(self.calibration, GamepadCalibration) else None
        self.calibration: GamepadCalibration | None = calibration
        self._calibrated = calibration is not None
        if calibration is not None:
            self._apply_calibration_to_config()

    @property
    def action_features(self) -> dict[str, type]:
        return {"x.vel": float, "y.vel": float, "theta.vel": float}

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, calibrate: bool = True) -> None:
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

        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def calibrate(self) -> None:
        if not self.is_connected or self._pygame is None:
            raise DeviceNotConnectedError("LeKiwi base teleoperator is not connected.")

        pygame = self._pygame
        joystick = self._joystick

        axis_threshold = 0.4
        release_threshold = 0.2
        poll_interval = 0.05

        if self.calibration is not None:
            user_input = input(
                "Press ENTER to use existing calibration file associated with this teleoperator, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                print("Using saved calibration.")
                self._apply_calibration_to_config()
                self._calibrated = True
                return

        print("\nStarting gamepad calibration.")
        print("Follow the prompts. Move/press the indicated control and hold until it is detected.")

        def read_axes() -> list[float]:
            pygame.event.pump()
            return [joystick.get_axis(i) for i in range(joystick.get_numaxes())]

        def wait_for_axis(prompt: str) -> tuple[int, float]:
            print(f"- {prompt}")
            baseline = read_axes()
            while True:
                time.sleep(poll_interval)
                current = read_axes()
                if not current:
                    continue
                diffs = [current[i] - baseline[i] for i in range(len(current))]
                axis_index = max(range(len(diffs)), key=lambda idx: abs(diffs[idx]))
                change = diffs[axis_index]
                if abs(change) >= axis_threshold:
                    detected_value = current[axis_index]
                    print(f"  detected axis {axis_index} with value {detected_value:+.2f}")
                    while True:
                        time.sleep(poll_interval)
                        pygame.event.pump()
                        value = joystick.get_axis(axis_index)
                        if abs(value - baseline[axis_index]) <= release_threshold:
                            break
                    return axis_index, detected_value

        def read_hats() -> list[tuple[int, int]]:
            pygame.event.pump()
            return [joystick.get_hat(i) for i in range(joystick.get_numhats())]

        def read_buttons() -> list[int]:
            pygame.event.pump()
            return [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        def wait_for_dpad(prompt: str, expected_x: int) -> tuple[str, int]:
            print(f"- {prompt}")
            baseline_buttons = read_buttons()
            while True:
                time.sleep(poll_interval)
                hats = read_hats()
                for idx, value in enumerate(hats):
                    if value[0] == expected_x:
                        print(f"  detected hat {idx}")
                        while True:
                            time.sleep(poll_interval)
                            pygame.event.pump()
                            if joystick.get_hat(idx)[0] == 0:
                                break
                        return "hat", idx
                buttons = read_buttons()
                for idx, pressed in enumerate(buttons):
                    if pressed and (idx >= len(baseline_buttons) or not baseline_buttons[idx]):
                        print(f"  detected button {idx}")
                        while joystick.get_button(idx):
                            time.sleep(poll_interval)
                            pygame.event.pump()
                        return "button", idx

        axis_y, axis_y_value = wait_for_axis("Move the left stick UP and hold it")
        axis_x, axis_x_value = wait_for_axis("Move the left stick RIGHT and hold it")
        dpad_right_source = wait_for_dpad("Press the D-pad RIGHT", expected_x=1)
        dpad_left_source = wait_for_dpad("Press the D-pad LEFT", expected_x=-1)

        self.config.axis_y = axis_y
        self.config.invert_y = axis_y_value < 0
        self.config.axis_x = axis_x
        self.config.invert_x = axis_x_value < 0

        use_hat = False
        hat_index = None
        if dpad_right_source[0] == "hat":
            use_hat = True
            hat_index = dpad_right_source[1]
        if dpad_left_source[0] == "hat":
            use_hat = True
            hat_index = dpad_left_source[1]

        if use_hat and hat_index is not None:
            self.config.hat_index = hat_index
        else:
            self.config.hat_index = -1

        if dpad_right_source[0] == "button":
            self.config.dpad_right_button = dpad_right_source[1]
        if dpad_left_source[0] == "button":
            self.config.dpad_left_button = dpad_left_source[1]

        print("\nCalibration results:")
        print(f"  axis_y -> axis {self.config.axis_y} | invert_y={self.config.invert_y}")
        print(f"  axis_x -> axis {self.config.axis_x} | invert_x={self.config.invert_x}")
        if self.config.hat_index >= 0:
            print(f"  using hat {self.config.hat_index} for yaw commands")
        if self.config.hat_index < 0 or dpad_right_source[0] == "button" or dpad_left_source[0] == "button":
            print(
                "  button mapping: "
                f"right={self.config.dpad_right_button}, left={self.config.dpad_left_button}"
            )

        self.calibration = GamepadCalibration(
            axis_x=self.config.axis_x,
            axis_y=self.config.axis_y,
            invert_x=self.config.invert_x,
            invert_y=self.config.invert_y,
            hat_index=self.config.hat_index,
            dpad_left_button=self.config.dpad_left_button,
            dpad_right_button=self.config.dpad_right_button,
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
                with suppress(Exception):
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
        if self.config.hat_index >= 0 and joystick.get_numhats() > self.config.hat_index:
            hat = joystick.get_hat(self.config.hat_index)
            hx = hat[0]
        else:
            left = joystick.get_button(self.config.dpad_left_button)
            right = joystick.get_button(self.config.dpad_right_button)
            hx = -1 if left else 0
            hx += 1 if right else 0

        return -self.config.yaw_speed_deg * hx

    def _apply_calibration_to_config(self) -> None:
        if self.calibration is None:
            return
        self.config.axis_x = self.calibration.axis_x
        self.config.axis_y = self.calibration.axis_y
        self.config.invert_x = self.calibration.invert_x
        self.config.invert_y = self.calibration.invert_y
        self.config.hat_index = self.calibration.hat_index
        self.config.dpad_left_button = self.calibration.dpad_left_button
        self.config.dpad_right_button = self.calibration.dpad_right_button

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
