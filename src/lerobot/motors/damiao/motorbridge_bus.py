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

"""Damiao motor bus backed by the ``motorbridge`` CAN controller.

This is a drop-in replacement for :class:`DamiaoMotorsBus` that routes motor
communication through the external ``motorbridge`` package (as used by the
reBot B601 follower) instead of ``python-can``. It keeps the same
degree-based, MIT-control interface consumed by the OpenArm follower and
leader, converting to/from the radian-based ``motorbridge`` API internally.
"""

import logging
import math
import platform
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import (
        Controller as MotorBridgeController,
        Mode as MotorBridgeMode,
    )
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)


def _is_macos() -> bool:
    return platform.system() == "Darwin"


class MotorBridgeDamiaoBus(MotorsBusBase):
    """MotorBridge-backed Damiao bus exposing the ``DamiaoMotorsBus`` interface.

    Positions are always handled in degrees at the public interface and
    converted to radians for ``motorbridge`` (which is radian-based). Motor
    model strings follow the LeRobot convention (e.g. ``"dm4310"``) and are
    mapped to the ``motorbridge`` naming (e.g. ``"4310"``) by dropping the
    ``dm`` prefix.
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        can_interface: str = "socketcan",
        use_can_fd: bool = True,
        bitrate: int = 1000000,
        data_bitrate: int | None = 5000000,
    ):
        """Initialize the MotorBridge Damiao bus.

        Args:
            port: CAN channel name (e.g. ``"can0"``).
            motors: Mapping of motor name to :class:`Motor` (uses ``id``,
                ``recv_id`` and ``motor_type_str``).
            calibration: Optional in-memory calibration (Damiao motors do not
                persist calibration on-device).
            can_interface: CAN interface type. Only ``"socketcan"`` is
                supported by this backend.
            use_can_fd: Use CAN FD (selects ``Controller.from_socketcanfd``).
            bitrate: Nominal bitrate. Informational only: with MotorBridge the
                CAN interface bitrate is configured at the OS level (e.g. via
                ``lerobot-setup-can``).
            data_bitrate: CAN FD data bitrate. Informational only (see above).
        """
        require_package("motorbridge", extra="openarms")
        super().__init__(port, motors, calibration)
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate

        self._controller: MotorBridgeController | None = None
        self._handles: dict[str, Any] = {}

    @staticmethod
    def _model(motor: Motor) -> str:
        model = motor.motor_type_str or motor.model
        return model.lower().removeprefix("dm")

    @property
    def is_connected(self) -> bool:
        return self._controller is not None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        use_can_fd = self.use_can_fd
        channel = self.port
        if _is_macos():
            # The libusb PCAN backend (motorbridge feat/pcan-usb-fd-native) reaches
            # both adapter channels but is classic-CAN only. Select it via the
            # ``pcanfd:`` channel prefix and force classic CAN.
            use_can_fd = False
            if not channel.startswith("pcanfd:"):
                channel = f"pcanfd:{channel}"

        logger.info(f"Connecting Damiao motors on {channel} (can_fd={use_can_fd})...")
        if use_can_fd:
            self._controller = MotorBridgeController.from_socketcanfd(channel)
        else:
            self._controller = MotorBridgeController(channel=channel)

        for name, motor in self.motors.items():
            self._handles[name] = self._controller.add_damiao_motor(
                motor.id, motor.recv_id, self._model(motor)
            )
        logger.debug(f"{self.__class__.__name__} connected.")

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        for handle in self._handles.values():
            try:
                if disable_torque:
                    handle.disable()
                handle.clear_error()
            except Exception:
                logger.exception("Failed to disable/clear a Damiao motor during disconnect.")
            handle.close()

        if self._controller is not None:
            self._controller.close()
        self._controller = None
        self._handles = {}
        logger.debug(f"{self.__class__.__name__} disconnected.")

    @check_if_not_connected
    def configure_motors(self) -> None:
        """Ensure every motor is in MIT control mode."""
        for handle in self._handles.values():
            handle.ensure_mode(MotorBridgeMode.MIT)

    @check_if_not_connected
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self._controller.enable_all()

    @check_if_not_connected
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        self._controller.disable_all()

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """Context manager that guarantees torque is re-enabled."""
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    @check_if_not_connected
    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        for name in self._get_motors_list(motors):
            self._handles[name].set_zero_position()

    def _read_states(self, motors: list[str]) -> dict[str, dict[str, float]]:
        """Refresh and decode motor feedback in one CAN cycle (degrees)."""
        for name in motors:
            self._handles[name].request_feedback()
        self._controller.poll_feedback_once()

        states: dict[str, dict[str, float]] = {}
        for name in motors:
            state = self._handles[name].get_state()
            if state is None:
                raise ConnectionError(f"No feedback available for motor '{name}'.")
            states[name] = {
                "position": math.degrees(state.pos),
                "velocity": math.degrees(state.vel),
                "torque": state.torq,
                "temp_mos": state.t_mos,
                "temp_rotor": state.t_rotor,
            }
        return states

    @check_if_not_connected
    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Read pos/vel/torque for all motors in one refresh cycle (degrees)."""
        return self._read_states(self._get_motors_list(motors))

    @check_if_not_connected
    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
    ) -> dict[str, Value]:
        target_motors = self._get_motors_list(motors)
        states = self._read_states(target_motors)
        key = self._state_key(data_name)
        return {name: states[name][key] for name in target_motors}

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        return self._read_states([motor])[motor][self._state_key(data_name)]

    @check_if_not_connected
    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> None:
        """Send MIT commands to multiple motors.

        Args:
            commands: Mapping of motor name to
                ``(kp, kd, position_deg, velocity_deg_per_sec, torque)``.
        """
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            self._handles[motor].send_mit(
                math.radians(position_degrees),
                math.radians(velocity_deg_per_sec),
                kp,
                kd,
                torque,
            )

    def write(self, data_name: str, motor: str, value: Value) -> None:
        raise NotImplementedError("MotorBridgeDamiaoBus only supports MIT control via _mit_control_batch.")

    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        raise NotImplementedError("MotorBridgeDamiaoBus only supports MIT control via _mit_control_batch.")

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Damiao motors do not persist calibration on-device."""
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Cache calibration in memory (Damiao motors do not store it)."""
        if cache:
            self.calibration = calibration_dict

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors.keys())
        if isinstance(motors, str):
            return [motors]
        return list(motors)

    @staticmethod
    def _state_key(data_name: str) -> str:
        mapping = {
            "Present_Position": "position",
            "Present_Velocity": "velocity",
            "Present_Torque": "torque",
            "Temperature_MOS": "temp_mos",
            "Temperature_Rotor": "temp_rotor",
        }
        if data_name not in mapping:
            raise ValueError(f"Unsupported data_name for MotorBridgeDamiaoBus: {data_name}")
        return mapping[data_name]
