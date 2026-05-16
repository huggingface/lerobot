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

import contextlib
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from lerobot.motors import Motor, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus


@dataclass
class SharedComponentAttachment:
    """Represents a component that should attach to a shared bus."""

    name: str
    component: object
    motor_names: tuple[str, ...]
    motor_id_offset: int = 0

    @property
    def calibration(self) -> dict[str, MotorCalibration]:
        return getattr(self.component, "calibration", {}) or {}

    @property
    def bus(self) -> FeetechMotorsBus:
        return self.component.bus


class SharedFeetechBusGroup:
    """Owns a real Feetech bus and provides reference-counted access to it."""

    def __init__(
        self,
        name: str,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration],
        *,
        handshake_on_connect: bool = True,
    ):
        self.name = name
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
        self.handshake_on_connect = handshake_on_connect
        self._active_clients = 0

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def acquire(self, *, handshake: bool | None = None) -> None:
        should_connect = not self.bus.is_connected
        if should_connect:
            use_handshake = self.handshake_on_connect if handshake is None else handshake
            self.bus.connect(handshake=use_handshake)
        self._active_clients += 1

    def release(self, *, disable_torque: bool = True) -> None:
        self._active_clients = max(0, self._active_clients - 1)
        if self._active_clients == 0 and self.bus.is_connected:
            self.bus.disconnect(disable_torque=disable_torque)


class SharedFeetechBusView:
    """Component-scoped view onto a shared Feetech bus."""

    def __init__(self, group: SharedFeetechBusGroup, motor_names: Iterable[str]):
        self._group = group
        self._motor_names = tuple(motor_names)
        self._local_connections = 0
        self.port = group.bus.port
        self.motors = {name: deepcopy(group.bus.motors[name]) for name in self._motor_names}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_subset(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self._motor_names)
        if isinstance(motors, str):
            motors = [motors]
        invalid = [m for m in motors if m not in self._motor_names]
        if invalid:
            raise KeyError(f"Motors {invalid} are not part of this shared-bus view.")
        return motors

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._local_connections > 0

    @property
    def is_calibrated(self) -> bool:
        if not self._group.bus.calibration:
            return False
        current = self._group.bus.read_calibration()
        bus_cal = self._group.bus.calibration
        for motor in self._motor_names:
            if motor not in current or motor not in bus_cal:
                return False
            cal_expected = bus_cal[motor]
            cal_current = current[motor]
            if (
                cal_expected.homing_offset != cal_current.homing_offset
                or cal_expected.range_min != cal_current.range_min
                or cal_expected.range_max != cal_current.range_max
            ):
                return False
        return True

    def connect(self, handshake: bool | None = None) -> None:
        self._group.acquire(handshake=handshake)
        self._local_connections += 1

    def disconnect(self, disable_torque: bool = True) -> None:
        if self._local_connections == 0:
            return
        if disable_torque and self._group.bus.is_connected and self._motor_names:
            self._group.bus.disable_torque(list(self._motor_names))
        self._local_connections = max(0, self._local_connections - 1)
        self._group.release(disable_torque=disable_torque)

    # ------------------------------------------------------------------
    # Bus operations (subset-aware)
    # ------------------------------------------------------------------
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self._group.bus.disable_torque(self._ensure_subset(motors), num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self._group.bus.enable_torque(self._ensure_subset(motors), num_retry=num_retry)

    @contextlib.contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        names = self._ensure_subset(motors)
        with self._group.bus.torque_disabled(names):
            yield

    def configure_motors(self, *args, **kwargs) -> None:
        # Configuring the full bus is acceptable; settings are idempotent.
        self._group.bus.configure_motors(*args, **kwargs)

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        subset = {name: calibration_dict[name] for name in self._motor_names if name in calibration_dict}
        if not subset:
            return
        self._group.bus.write_calibration(subset, cache=cache)

    def write(
        self,
        data_name: str,
        motor: str,
        value,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        if motor not in self._motor_names:
            raise KeyError(f"Motor '{motor}' is not owned by this shared-bus view.")
        self._group.bus.write(
            data_name,
            motor,
            value,
            normalize=normalize,
            num_retry=num_retry,
        )

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, float]:
        names = self._ensure_subset(motors)
        return self._group.bus.sync_read(
            data_name,
            names,
            normalize=normalize,
            num_retry=num_retry,
        )

    def sync_write(
        self,
        data_name: str,
        values,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        if not isinstance(values, dict):
            values = dict.fromkeys(self._motor_names, values)
        else:
            missing = [name for name in values if name not in self._motor_names]
            if missing:
                raise KeyError(f"Motors {missing} are not part of this shared-bus view.")
        self._group.bus.sync_write(
            data_name,
            values,
            normalize=normalize,
            num_retry=num_retry,
        )

    def set_half_turn_homings(self, motors: str | list[str] | None = None):
        names = self._ensure_subset(motors)
        return self._group.bus.set_half_turn_homings(names)

    def record_ranges_of_motion(self, motors: str | list[str] | None = None, display_values: bool = True):
        names = self._ensure_subset(motors)
        return self._group.bus.record_ranges_of_motion(names, display_values=display_values)

    def setup_motor(self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None):
        if motor not in self._motor_names:
            raise KeyError(f"Motor '{motor}' is not part of this shared-bus view.")
        return self._group.bus.setup_motor(motor, initial_baudrate=initial_baudrate, initial_id=initial_id)


def build_shared_bus_group(
    name: str,
    port: str,
    attachments: list[SharedComponentAttachment],
    *,
    handshake_on_connect: bool = True,
) -> tuple[SharedFeetechBusGroup, dict[str, SharedFeetechBusView]]:
    """Create a shared bus group and corresponding component views."""

    if not attachments:
        raise ValueError(f"Shared bus '{name}' has no attached components.")

    motors: dict[str, Motor] = {}
    calibration: dict[str, MotorCalibration] = {}
    motor_lookup: dict[str, list[str]] = {}

    for attachment in attachments:
        for motor_name in attachment.motor_names:
            motor = attachment.bus.motors[motor_name]
            if motor_name in motors:
                raise ValueError(
                    f"Motor name '{motor_name}' is duplicated across components on shared bus '{name}'."
                )
            shifted_motor = deepcopy(motor)
            shifted_motor.id += attachment.motor_id_offset
            motors[motor_name] = shifted_motor
            motor_lookup.setdefault(attachment.name, []).append(motor_name)

        for cal_name, cal in attachment.calibration.items():
            shifted_cal = deepcopy(cal)
            shifted_cal.id += attachment.motor_id_offset
            calibration[cal_name] = shifted_cal

    group = SharedFeetechBusGroup(
        name=name,
        port=port,
        motors=motors,
        calibration=calibration,
        handshake_on_connect=handshake_on_connect,
    )

    views: dict[str, SharedFeetechBusView] = {}
    for attachment in attachments:
        motor_names = motor_lookup.get(attachment.name, [])
        view = SharedFeetechBusView(group, motor_names)
        attachment.component.bus = view
        views[attachment.name] = view

    return group, views
