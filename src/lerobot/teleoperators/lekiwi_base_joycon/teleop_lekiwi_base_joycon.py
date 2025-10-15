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

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_lekiwi_base_joycon import LeKiwiBaseJoyconConfig

try:  # pragma: no cover - optional runtime dependency
    from pyjoycon import GyroTrackingJoyCon, get_L_id, get_R_id
except ImportError as exc:  # pragma: no cover - handled at runtime
    GyroTrackingJoyCon = None  # type: ignore[assignment]
    _JOYCON_IMPORT_ERROR = exc
else:  # pragma: no cover - import-time constant
    _JOYCON_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _ensure_joycon_available() -> None:
    if GyroTrackingJoyCon is None:
        raise ImportError(
            "pyjoycon is required for the JoyCon teleoperator. Install it with `pip install pyjoycon PyGLM hidapi`."
        ) from _JOYCON_IMPORT_ERROR


class LeKiwiBaseJoycon(Teleoperator):
    """Drive the LeKiwi base using a single Nintendo JoyCon controller."""

    config_class = LeKiwiBaseJoyconConfig
    name = "lekiwi_base_joycon"

    _STICK_ATTR = {"left": "stick_l", "right": "stick_r"}

    def __init__(self, config: LeKiwiBaseJoyconConfig):
        super().__init__(config)
        self.config = config
        self._joycon: GyroTrackingJoyCon | None = None
        self._stick_center = np.zeros(2, dtype=np.float32)
        self._last_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {"x.vel": float, "y.vel": float, "theta.vel": float}

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._joycon is not None

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 JoyCon auto-calibrated by driver
        _ensure_joycon_available()
        which = self.config.which.lower()
        if which not in self._STICK_ATTR:
            raise ValueError("LeKiwiBaseJoyconConfig.which must be 'left' or 'right'")

        joycon_id = self._get_joycon_id(which)
        if joycon_id[0] is None:
            raise RuntimeError(
                f"Could not find {which} JoyCon. Ensure it is paired, connected, and not claimed by another process."
            )

        joycon = GyroTrackingJoyCon(*joycon_id)
        joycon.calibrate(seconds=0.5)
        self._joycon = joycon
        self._capture_stick_center()
        logger.info("Connected %s JoyCon for LeKiwi base teleoperation.", which)

    def calibrate(self) -> None:
        return None

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected or self._joycon is None:
            raise DeviceNotConnectedError("LeKiwiBaseJoycon is not connected.")

        horizontal, vertical = self._read_stick()

        if self.config.normalize_diagonal:
            magnitude = float(np.hypot(horizontal, vertical))
            if magnitude > 1.0:
                horizontal /= magnitude
                vertical /= magnitude

        x_axis = -horizontal if self.config.invert_x else horizontal
        y_axis = -vertical if self.config.invert_y else vertical

        # Map joystick axes to body-frame velocities (same convention as gamepad teleop).
        x_vel = y_axis * self.config.max_speed_mps
        y_vel = x_axis * self.config.max_speed_mps

        yaw_sign = self._read_yaw_direction()
        theta_vel = self.config.yaw_speed_deg * yaw_sign

        self._last_action = {"x.vel": -x_vel, "y.vel": y_vel, "theta.vel": theta_vel}
        return dict(self._last_action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        # if self._joycon is not None:
        #     try:
        #         self._joycon.disconnect()
        #     except Exception:  # pragma: no cover - hardware cleanup
        #         logger.debug("Failed to disconnect JoyCon cleanly.", exc_info=True)
        #     self._joycon = None
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_joycon_id(self, which: str) -> tuple[int | None, int | None, str | None]:
        getter = get_L_id if which == "left" else get_R_id
        poll_interval = max(0.1, self.config.discovery_poll_interval)
        deadline = time.perf_counter() + max(0.0, self.config.discovery_timeout)
        while True:
            joycon_id = getter()
            if joycon_id and joycon_id[0] is not None:
                return joycon_id
            if self.config.serial_hint:
                serial = self.config.serial_hint.lower()
                joycon_id = self._find_by_serial(serial, getter)
                if joycon_id[0] is not None:
                    return joycon_id
            if time.perf_counter() >= deadline:
                return (None, None, None)
            time.sleep(poll_interval)

    def _find_by_serial(
        self,
        serial_hint: str,
        fallback_getter,
    ) -> tuple[int | None, int | None, str | None]:
        joycon_id = fallback_getter()
        if joycon_id[2] and joycon_id[2].lower() == serial_hint:
            return joycon_id
        return (None, None, None)

    def _capture_stick_center(self) -> None:
        joycon = self._joycon
        if joycon is None:
            return
        attr = self._STICK_ATTR[self.config.which.lower()]
        axes = getattr(joycon, attr, None)
        if axes is None:
            self._stick_center = np.zeros(2, dtype=np.float32)
            return
        values = np.asarray(axes, dtype=np.float32)
        if values.shape[0] < 2:
            self._stick_center = np.zeros(2, dtype=np.float32)
        else:
            self._stick_center = values[:2]

    def _read_stick(self) -> tuple[float, float]:
        joycon = self._joycon
        if joycon is None:
            return 0.0, 0.0
        attr = self._STICK_ATTR[self.config.which.lower()]
        axes = getattr(joycon, attr, None)
        if axes is None:
            return 0.0, 0.0
        values = np.asarray(axes, dtype=np.float32)
        if values.shape[0] < 2:
            return 0.0, 0.0
        horizontal = self._apply_deadzone(self._normalize_axis(values[0], self._stick_center[0]))
        vertical = self._apply_deadzone(self._normalize_axis(values[1], self._stick_center[1]))
        return float(horizontal), float(vertical)

    def _read_yaw_direction(self) -> float:
        joycon = self._joycon
        if joycon is None:
            return 0.0
        which = self.config.which.lower()
        if which == "left":
            pos_attr, neg_attr = self.config.yaw_buttons_left
        else:
            pos_attr, neg_attr = self.config.yaw_buttons_right
        return self._button_value(joycon, pos_attr) - self._button_value(joycon, neg_attr)

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.config.deadzone else float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _normalize_axis(value: float, center: float) -> float:
        norm = (value - center) / 2048.0
        return float(np.clip(norm, -1.0, 1.0))

    @staticmethod
    def _button_value(joycon: GyroTrackingJoyCon, attr: str) -> float:
        return 1.0 if bool(getattr(joycon, attr, False)) else 0.0
