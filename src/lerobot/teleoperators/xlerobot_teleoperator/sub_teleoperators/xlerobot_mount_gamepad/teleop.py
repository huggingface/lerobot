#!/usr/bin/env python

# Example teleoperation command for XLeRobot mount with this gamepad teleop:
#
# lerobot-teleoperate \
#     --robot.type=xlerobot_mount \
#     --robot.port=/dev/ttyACM5 \
#     --robot.id=mount \
#     --teleop.type=xlerobot_mount_gamepad \
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

import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.utils.errors import DeviceNotConnectedError

from ....teleoperator import Teleoperator
from .config import XLeRobotMountGamepadTeleopConfig


@dataclass
class MountGamepadCalibration:
    pan_axis: int
    tilt_axis: int
    invert_pan: bool
    invert_tilt: bool


class XLeRobotMountGamepadTeleop(Teleoperator):
    config_class = XLeRobotMountGamepadTeleopConfig
    name = "xlerobot_mount_gamepad"

    def __init__(self, config: XLeRobotMountGamepadTeleopConfig):
        self.config = config
        super().__init__(config)
        self._pygame = None
        self._joystick = None
        self._clock = None

        # Track current position for incremental control (initialized lazily from observation)
        self._current_pan: float | None = None
        self._current_tilt: float | None = None
        self._pan_bias = 0.0
        self._tilt_bias = 0.0
        calibration = self.calibration if isinstance(self.calibration, MountGamepadCalibration) else None
        self.calibration: MountGamepadCalibration | None = calibration
        self._calibrated = calibration is not None
        if calibration is not None:
            self._apply_calibration_to_config()

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
        self._pan_bias = 0.0
        self._tilt_bias = 0.0

        if calibrate and not self.is_calibrated:
            self.calibrate()
        elif self.calibration is not None:
            self._apply_calibration_to_config()
            self._calibrated = True

        self._sync_biases()

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def calibrate(self) -> None:
        if not self.is_connected or self._pygame is None:
            raise DeviceNotConnectedError("Mount gamepad teleoperator is not connected.")

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
                self._sync_biases()
                return

        print("\nStarting gamepad calibration for XLeRobot mount.")
        print("Follow the prompts. Move the indicated control and hold until it is detected.")

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

        pan_axis, pan_value = wait_for_axis("Move the right stick to the RIGHT and hold it")

        while True:
            tilt_axis, tilt_value = wait_for_axis("Move the right stick UP and hold it")
            if tilt_axis == pan_axis:
                print("  detected the same axis as pan; release the stick and try moving straight up.")
                continue
            break

        self.config.pan_axis = pan_axis
        self.config.invert_pan = pan_value < 0
        self.config.tilt_axis = tilt_axis
        self.config.invert_tilt = tilt_value < 0

        print("\nCalibration results:")
        print(f"  pan_axis -> axis {self.config.pan_axis} | invert_pan={self.config.invert_pan}")
        print(f"  tilt_axis -> axis {self.config.tilt_axis} | invert_tilt={self.config.invert_tilt}")

        self.calibration = MountGamepadCalibration(
            pan_axis=self.config.pan_axis,
            tilt_axis=self.config.tilt_axis,
            invert_pan=self.config.invert_pan,
            invert_tilt=self.config.invert_tilt,
        )
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

        self._calibrated = True
        self._sync_biases()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

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
        self._current_pan = np.clip(self._current_pan, self.config.pan_range[0], self.config.pan_range[1])
        self._current_tilt = np.clip(self._current_tilt, self.config.tilt_range[0], self.config.tilt_range[1])

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
                with suppress(Exception):
                    self._joystick.quit()
            self._joystick = None

        if self._pygame is not None:
            with suppress(Exception):
                if hasattr(self._pygame, "joystick"):
                    self._pygame.joystick.quit()
            with suppress(Exception):
                if hasattr(self._pygame, "quit"):
                    self._pygame.quit()
            self._pygame = None

        self._clock = None

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.config.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))

    def _sync_biases(self) -> None:
        if not self.is_connected or self._joystick is None:
            self._pan_bias = 0.0
            self._tilt_bias = 0.0
            return

        num_axes = self._joystick.get_numaxes()
        if self.config.pan_axis >= num_axes or self.config.tilt_axis >= num_axes:
            raise RuntimeError(
                "Calibrated axis index exceeds available joystick axes. "
                "Re-run calibration for the gamepad teleoperator."
            )

        self._pan_bias = self._joystick.get_axis(self.config.pan_axis)
        self._tilt_bias = self._joystick.get_axis(self.config.tilt_axis)

    def _apply_calibration_to_config(self) -> None:
        if self.calibration is None:
            return
        self.config.pan_axis = self.calibration.pan_axis
        self.config.tilt_axis = self.calibration.tilt_axis
        self.config.invert_pan = self.calibration.invert_pan
        self.config.invert_tilt = self.calibration.invert_tilt

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            calibration = draccus.load(MountGamepadCalibration, f)
        self.calibration = calibration
        self._apply_calibration_to_config()
        self._calibrated = True

    def _save_calibration(self, fpath: Path | None = None) -> None:
        if self.calibration is None:
            return
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)
