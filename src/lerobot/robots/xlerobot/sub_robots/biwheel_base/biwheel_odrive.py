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
import time
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .biwheel_base import BiwheelBase
from .config_biwheel_base import BiwheelODriveConfig

logger = logging.getLogger(__name__)


class BiwheelODrive(BiwheelBase):
    """Biwheel robot class driven by ODrive controllers."""

    config_class = BiwheelODriveConfig
    name = "biwheel_odrive"
    supports_shared_bus = False

    def __init__(self, config: BiwheelODriveConfig):
        super().__init__(config)
        self.config = config
        self._odrv = None
        self._left = None
        self._right = None
        self._wheel_circumference = 2.0 * np.pi * self.config.wheel_radius

    @property
    def is_connected(self) -> bool:
        return self._odrv is not None and self._left is not None and self._right is not None

    def connect(self, calibrate: bool = True, handshake: bool | None = None) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        _ = calibrate
        _ = handshake

        odrive, enums = self._load_odrive()
        serial = self.config.odrive_serial
        timeout = self.config.odrive_timeout_s
        self._odrv = (
            odrive.find_any(serial_number=serial, timeout=timeout)
            if serial
            else odrive.find_any(timeout=timeout)
        )

        self._left = getattr(self._odrv, f"axis{self.config.axis_left}")
        self._right = getattr(self._odrv, f"axis{self.config.axis_right}")

        self._left.clear_errors()
        self._right.clear_errors()
        time.sleep(0.1)

        self._left.controller.config.control_mode = enums.CONTROL_MODE_VELOCITY_CONTROL
        self._right.controller.config.control_mode = enums.CONTROL_MODE_VELOCITY_CONTROL
        self._left.controller.config.input_mode = enums.INPUT_MODE_PASSTHROUGH
        self._right.controller.config.input_mode = enums.INPUT_MODE_PASSTHROUGH

        if self.config.disable_watchdog:
            self._left.config.enable_watchdog = False
            self._right.config.enable_watchdog = False

        if self.config.request_closed_loop:
            self._left.requested_state = enums.AXIS_STATE_CLOSED_LOOP_CONTROL
            self._right.requested_state = enums.AXIS_STATE_CLOSED_LOOP_CONTROL
            time.sleep(0.5)

        if self._left.error or self._right.error:
            logger.warning(
                "ODrive reported errors after connect (left=%s, right=%s).",
                hex(self._left.error),
                hex(self._right.error),
            )

        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_rps = self._left.encoder.vel_estimate
        right_rps = self._right.encoder.vel_estimate
        left_linear = left_rps * self._wheel_circumference
        right_linear = right_rps * self._wheel_circumference
        left_linear, right_linear = self._remove_inversion(left_linear, right_linear)
        return self._wheel_linear_to_body(left_linear, right_linear)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        base_goal_vel = {
            "x.vel": float(action.get("x.vel", 0.0)),
            "theta.vel": float(action.get("theta.vel", 0.0)),
        }

        left_linear, right_linear = self._body_to_wheel_linear(
            base_goal_vel["x.vel"],
            base_goal_vel["theta.vel"],
        )
        left_linear, right_linear = self._apply_inversion(left_linear, right_linear)

        left_rps = left_linear / self._wheel_circumference
        right_rps = right_linear / self._wheel_circumference

        self._left.controller.input_vel = left_rps
        self._right.controller.input_vel = right_rps

        return base_goal_vel

    def stop_base(self) -> None:
        if not self.is_connected:
            return
        self._left.controller.input_vel = 0.0
        self._right.controller.input_vel = 0.0

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        _, enums = self._load_odrive()
        self.stop_base()
        self._left.requested_state = enums.AXIS_STATE_IDLE
        self._right.requested_state = enums.AXIS_STATE_IDLE
        self._odrv = None
        self._left = None
        self._right = None
        logger.info("%s disconnected.", self)

    @staticmethod
    def _load_odrive():
        import odrive
        import odrive.enums as enums

        return odrive, enums
