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

# TODO(Virgile) : Robustify mode control , only the MIT protocole is implemented for now

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property

import can
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase
from .tables import (
    AVAILABLE_BAUDRATES,
    CAN_CMD_CLEAR_FAULT,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_SET_ZERO,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT_MS,
    MODEL_RESOLUTION,
    MOTOR_LIMIT_PARAMS,
    NORMALIZED_DATA,
    PARAM_TIMEOUT,
    RUNNING_TIMEOUT,
    STATE_CACHE_TTL_S,
    ControlMode,
    MotorType,
)

logger = logging.getLogger(__name__)

NameOrID = str | int
Value = int | float


class RobstrideMotorsBus(MotorsBusBase):
    """
    The Robstride implementation for a MotorsBus using CAN bus communication.

    This class uses python-can for CAN bus communication with Robstride motors.
    For more info, see:
    - python-can documentation: https://python-can.readthedocs.io/en/stable/
    - Seedstudio documentation: https://wiki.seeedstudio.com/Robstride_series/
    - DM_Control_Python repo: https://github.com/cmjang/DM_Control_Python
    """

    # CAN-specific settings
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

    # Motor configuration
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        can_interface: str = "auto",
        use_can_fd: bool = True,
        bitrate: int = 1000000,
        data_bitrate: int | None = 5000000,
    ):
        """
        Initialize the Robstride motors bus.

        Args:
            port: CAN interface name (e.g., "can0" for Linux, "/dev/cu.usbmodem*" for macOS)
            motors: Dictionary mapping motor names to Motor objects
            calibration: Optional calibration data
            can_interface: CAN interface type - "auto" (default), "socketcan" (Linux), or "slcan" (macOS/serial)
            use_can_fd: Whether to use CAN FD mode (default: True for OpenArms)
            bitrate: Nominal bitrate in bps (default: 1000000 = 1 Mbps)
            data_bitrate: Data bitrate for CAN FD in bps (default: 5000000 = 5 Mbps), ignored if use_can_fd is False
        """
        super().__init__(port, motors, calibration)
        self.port = port
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate
        self.canbus = None
        self._is_connected = False

        # Map motor names to CAN IDs
        self._motor_can_ids = {}
        self._recv_id_to_motor: dict[int, str] = {}

        # Store motor types and recv IDs
        self._motor_types: dict[str, MotorType] = {}
        self._motor_kps: dict[str, float] = {}
        self._motor_kds: dict[str, float] = {}
        for name, motor in self.motors.items():
            if motor.motor_type_str is not None:
                self._motor_types[name] = getattr(MotorType, motor.motor_type_str.upper())
            else:
                # Default to O0if not specified
                self._motor_types[name] = MotorType.O0

            if hasattr(motor, "kp"):
                self._motor_kps[name] = motor.kp
            else:
                # Default to 15 if not specified
                self._motor_kps[name] = 15

            if hasattr(motor, "kd"):
                self._motor_kds[name] = motor.kd
            else:
                # Default to O.1 if not specified
                self._motor_kds[name] = 0.1

            # Map recv_id to motor name for filtering responses
            if motor.recv_id is not None:
                self._recv_id_to_motor[motor.recv_id] = name
        # Motor Mode
        self.enabled: dict[str, bool] = {}
        self.operation_mode: dict[str, ControlMode] = {}
        self.currentPosition: dict[str, float | None] = {}
        self.currentVelocity: dict[str, float | None] = {}
        self.currentTorque: dict[str, float | None] = {}
        self.currentTemperature: dict[str, float | None] = {}
        self.last_feedback_time: dict[str, float | None] = {}
        self._id_to_name: dict[int, str] = {}
        for name in self.motors:
            self.enabled[name] = False
            self.operation_mode[name] = ControlMode.MIT  # default mode
            self.currentPosition[name] = None
            self.currentVelocity[name] = None
            self.currentTorque[name] = None
            self.currentTemperature[name] = None
            self.last_feedback_time[name] = None

        for name, motor in self.motors.items():
            key = motor.recv_id if motor.recv_id is not None else motor.id
            self._id_to_name[key] = name

    @property
    def is_connected(self) -> bool:
        """Check if the CAN bus is connected."""
        return self._is_connected and self.canbus is not None

    def connect(self, handshake: bool = True) -> None:
        """
        Open the CAN bus and initialize communication.

        Args:
            handshake: If True, ping all motors to verify they're present
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__}('{self.port}') is already connected."
            )

        try:
            # Auto-detect interface type based on port name
            if self.can_interface == "auto":
                if self.port.startswith("/dev/"):
                    # Serial device (macOS/Windows)
                    self.can_interface = "slcan"
                    logger.info(f"Auto-detected slcan interface for port {self.port}")
                else:
                    # Network interface (Linux)
                    self.can_interface = "socketcan"
                    logger.info(f"Auto-detected socketcan interface for port {self.port}")

            # Connect to CAN bus
            if self.can_interface == "socketcan":
                # Linux SocketCAN with CAN FD support
                if self.use_can_fd and self.data_bitrate is not None:
                    self.canbus = can.interface.Bus(
                        channel=self.port,
                        interface="socketcan",
                        bitrate=self.bitrate,
                        data_bitrate=self.data_bitrate,
                        fd=True,
                    )
                    logger.info(
                        f"Connected to {self.port} with CAN FD (bitrate={self.bitrate}, data_bitrate={self.data_bitrate})"
                    )
                else:
                    self.canbus = can.interface.Bus(
                        channel=self.port, interface="socketcan", bitrate=self.bitrate
                    )
                    logger.info(f"Connected to {self.port} with CAN 2.0 (bitrate={self.bitrate})")
            elif self.can_interface == "slcan":
                # Serial Line CAN (macOS, Windows, or USB adapters)
                # Note: SLCAN typically doesn't support CAN FD
                self.canbus = can.interface.Bus(channel=self.port, interface="slcan", bitrate=self.bitrate)
                logger.info(f"Connected to {self.port} with SLCAN (bitrate={self.bitrate})")
            else:
                # Generic interface (vector, pcan, etc.)
                if self.use_can_fd and self.data_bitrate is not None:
                    self.canbus = can.interface.Bus(
                        channel=self.port,
                        interface=self.can_interface,
                        bitrate=self.bitrate,
                        data_bitrate=self.data_bitrate,
                        fd=True,
                    )
                else:
                    self.canbus = can.interface.Bus(
                        channel=self.port, interface=self.can_interface, bitrate=self.bitrate
                    )

            self._is_connected = True

            if handshake:
                self._handshake()

            logger.debug(f"{self.__class__.__name__} connected via {self.can_interface}.")
        except Exception as e:
            self._is_connected = False
            raise ConnectionError("Failed to connect to CAN bus") from e

    def _query_status_via_clear_fault(self, motor) -> None:
        """Query fault status on one motor and log it if a fault is detected."""
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor_name)
        recv_id = self._get_motor_recv_id(motor_name)
        data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        return self._recv_status_via_clear_fault(expected_recv_id=recv_id)

    def _recv_status_via_clear_fault(
        self, expected_recv_id: int | None = None, timeout: float = RUNNING_TIMEOUT
    ):
        """
        Poll the bus for a response to a fault-clear request.

        Args:
            expected_recv_id: Only accept frames from this CAN ID when provided.
            timeout: Maximum time spent polling the bus in seconds.

        Returns:
            Tuple where the first element is True if a fault frame was received,
            and the second element is the CAN message (or None on timeout).
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            msg = self.canbus.recv(timeout=RUNNING_TIMEOUT / 10)
            if not msg:
                continue

            if expected_recv_id is not None and msg.data[0] != expected_recv_id:
                continue

            # Fault-status frame heuristic (doc-based)
            fault_bits = int.from_bytes(msg.data[1:5], "little")
            if fault_bits != 0 and msg.data[5] == msg.data[6] == msg.data[7] == 0:
                logger.error(
                    f"Motor fault received from CAN ID 0x{msg.arbitration_id:02X}: "
                    f"fault_bits=0x{fault_bits:08X}"
                )
                return True, msg

            # Otherwise: valid normal response
            return False, msg

        return False, None

    def update_motor_state(self, motor) -> bool:
        has_fault, msg = self._query_status_via_clear_fault(motor)
        if msg is None:
            logger.warning(f"No response received from motor '{motor}' during state update.")
            raise ConnectionError(f"No response received from motor '{motor}' during state update.")
        if has_fault:
            logger.error(f"Fault reported by motor '{motor}' during state update. msg={msg.data.hex()}")
            raise RuntimeError(f"Fault reported by motor '{motor}' during state update.")

        self._decode_motor_state(msg.data)  # updates cache
        return True

    def _handshake(self) -> None:
        faults = {}

        for motor_name in self.motors:
            has_fault, msg = self._query_status_via_clear_fault(motor_name)
            if has_fault or msg is None:
                faults[motor_name] = msg
            time.sleep(0.01)

        if faults:
            for motor, msg in faults.items():
                logger.error(f"Motor '{motor}' failed handshake. response={msg.data.hex() if msg else None}")
            raise ConnectionError("One or more motors failed handshake. Check fault logs.")

    def _switch_operation_mode(self, motor, mode: ControlMode) -> None:
        """Switch the operation mode of a motor."""
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor_name)
        recv_id = self._get_motor_recv_id(motor_name)
        data = [0xFF] * 8
        data[6] = mode.value
        data[7] = 0xFC
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        msg = self._recv_motor_response(expected_recv_id=recv_id, timeout=PARAM_TIMEOUT)
        if msg is not None:
            self.operation_mode[motor_name] = mode

    def disconnect(self, disable_torque: bool = True) -> None:
        """
        Close the CAN bus connection.

        Args:
            disable_torque: If True, disable torque on all motors before disconnecting
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.__class__.__name__}('{self.port}') is not connected.")

        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        if self.canbus:
            self.canbus.shutdown()
            self.canbus = None
        self._is_connected = False
        logger.debug(f"{self.__class__.__name__} disconnected.")

    def configure_motors(self) -> None:
        """Configure all motors with default settings."""
        # Robstride motors don't require much configuration in MIT mode
        # Just ensure they're enabled
        for motor in self.motors:
            self._enable_motor(motor)
            self._switch_operation_mode(motor, ControlMode.MIT)
            time.sleep(0.01)

    def switch_to_mode(self, mode: ControlMode) -> None:
        """Switch operation mode on selected motors."""
        for motor in self.motors:
            self._switch_operation_mode(motor, mode)
            time.sleep(0.01)

    def _enable_motor(self, motor: NameOrID) -> None:
        """Enable a single motor."""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [CAN_CMD_ENABLE]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        self._recv_motor_response(expected_recv_id=recv_id, timeout=PARAM_TIMEOUT)

    def _disable_motor(self, motor: NameOrID) -> None:
        """Disable a single motor."""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [CAN_CMD_DISABLE]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        self._recv_motor_response(expected_recv_id=recv_id)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        motors = self._get_motors_list(motors)
        for motor in motors:
            for _ in range(num_retry + 1):
                try:
                    self._enable_motor(motor)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(0.01)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        motors = self._get_motors_list(motors)
        for motor in motors:
            for _ in range(num_retry + 1):
                try:
                    self._disable_motor(motor)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(0.01)

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """
        Context manager that guarantees torque is re-enabled.

        This helper is useful to temporarily disable torque when configuring motors.

        Examples:
            >>> with bus.torque_disabled():
            ...     # Safe operations here with torque disabled
            ...     pass
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """Set current position as zero for selected motors."""
        motors = self._get_motors_list(motors)
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            recv_id = self._get_motor_recv_id(motor)
            data = [0xFF] * 7 + [CAN_CMD_SET_ZERO]
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self.canbus.send(msg)
            self._recv_motor_response(expected_recv_id=recv_id)
            time.sleep(0.01)

    def _recv_motor_response(
        self, expected_recv_id: int | None = None, timeout: float = 0.001
    ) -> can.Message | None:
        """
        Receive a response from a motor.

        Args:
            expected_recv_id: If provided, only return messages from this CAN ID
            timeout: Timeout in seconds (default: 1ms for high-speed operation)

        Returns:
            CAN message if received, None otherwise
        """
        try:
            start_time = time.time()
            messages_seen = []
            while time.time() - start_time < timeout:
                msg = self.canbus.recv(timeout=RUNNING_TIMEOUT / 10)  # 100us timeout for fast polling
                if msg:
                    messages_seen.append(f"0x{msg.arbitration_id:02X}")
                    # If no filter specified, return any message
                    if expected_recv_id is None:
                        return msg
                    # Otherwise, only return if it matches the expected recv_id
                    if msg.data[0] == expected_recv_id:
                        return msg
                    else:
                        logger.debug(
                            f"Ignoring message from CAN ID 0x{msg.arbitration_id:02X}, expected 0x{expected_recv_id:02X}"
                        )

            # Only log warnings if we're in debug mode to reduce overhead
            if logger.isEnabledFor(logging.DEBUG):
                if messages_seen:
                    logger.debug(
                        f"Received {len(messages_seen)} message(s) from IDs {set(messages_seen)}, but expected 0x{expected_recv_id:02X}"
                    )
                else:
                    logger.debug(f"No CAN messages received (expected from 0x{expected_recv_id:02X})")
        except Exception as e:
            logger.debug(f"Failed to receive CAN message: {e}")
        return None

    def _recv_all_responses(
        self, expected_recv_ids: list[int], timeout: float = 0.002
    ) -> dict[int, can.Message]:
        """
        Efficiently receive responses from multiple motors at once.
        Uses the OpenArms pattern: collect all available messages within timeout.

        Args:
            expected_recv_ids: List of CAN IDs we expect responses from
            timeout: Total timeout in seconds (default: 2ms)

        Returns:
            Dictionary mapping recv_id to CAN message
        """
        responses = {}
        expected_set = set(expected_recv_ids)
        start_time = time.time()

        try:
            while len(responses) < len(expected_recv_ids) and (time.time() - start_time) < timeout:
                msg = self.canbus.recv(timeout=RUNNING_TIMEOUT / 10)  # 100us poll timeout
                if msg and msg.data[0] in expected_set:
                    responses[msg.data[0]] = msg
                    if len(responses) == len(expected_recv_ids):
                        break  # Got all responses, exit early
        except Exception as e:
            logger.debug(f"Error receiving responses: {e}")

        return responses

    def _speed_control(
        self,
        motor: NameOrID,
        velocity_deg_per_sec: float,
        current_limit_a: float,
    ) -> None:
        """
        Send a Velocity Mode Control Command (Command 11) to a single motor.

        Args:
            motor: Motor name or CAN ID.
            velocity_rad_per_sec: Target speed in rad/s (32-bit float).
            current_limit_a: Current limit in A (32-bit float).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        # Optional: ensure the motor is in velocity control mode

        if self.operation_mode[motor_name] != ControlMode.VEL:
            raise RuntimeError(f"Motor '{motor_name}' is not in velocity control mode.")
        # Convert to rad/s to match protocol specification

        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # Encode float32 little-endian without struct (byte list)
        def _float32_to_le_bytes(x: float) -> list[int]:
            b = np.float32(x).tobytes()  # 4 bytes, little-endian
            return [b[0], b[1], b[2], b[3]]

        speed_bytes = _float32_to_le_bytes(velocity_rad_per_sec)
        limit_bytes = _float32_to_le_bytes(current_limit_a)

        data = speed_bytes + limit_bytes  # 8 octets : [0–3]=speed, [4–7]=current limit

        msg = can.Message(
            arbitration_id=motor_id,
            data=data,
            is_extended_id=False,
        )
        self.canbus.send(msg)

        # Si le proto renvoie une réponse type état, on peut la décoder comme pour MIT
        recv_id = self._get_motor_recv_id(motor)
        if recv_id is not None:
            resp = self._recv_motor_response(expected_recv_id=recv_id)
            if resp:
                self._decode_motor_state(resp.data)

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> None:
        """
        Send MIT control command to a motor.

        Args:
            motor: Motor name or ID
            kp: Position gain
            kd: Velocity gain
            position_degrees: Target position (degrees)
            velocity_deg_per_sec: Target velocity (degrees/s)
            torque: Target torque (N·m)
        """
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        motor_type = self._motor_types.get(motor_name)
        if self.operation_mode[motor_name] != ControlMode.MIT:
            raise RuntimeError(f"Motor '{motor_name}' is not in MIT control mode.")
        # Convert degrees to radians for motor control
        position_rad = np.radians(position_degrees)
        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Encode parameters
        kp_uint = self._float_to_uint(kp, 0, 500, 12)
        kd_uint = self._float_to_uint(kd, 0, 5, 12)
        q_uint = self._float_to_uint(position_rad, -pmax, pmax, 16)
        dq_uint = self._float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
        tau_uint = self._float_to_uint(torque, -tmax, tmax, 12)

        # Pack data
        data = [0] * 8
        data[0] = (q_uint >> 8) & 0xFF
        data[1] = q_uint & 0xFF
        data[2] = dq_uint >> 4
        data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
        data[4] = kp_uint & 0xFF
        data[5] = kd_uint >> 4
        data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
        data[7] = tau_uint & 0xFF

        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        recv_id = self._get_motor_recv_id(motor)
        msg = self._recv_motor_response(expected_recv_id=recv_id)
        if msg:
            self._decode_motor_state(msg.data)  # update cache

    def _float_to_uint(self, x: float, x_min: float, x_max: float, bits: int) -> int:
        """Convert float to unsigned integer for CAN transmission."""
        x = max(x_min, min(x_max, x))  # Clamp to range
        span = x_max - x_min
        data_norm = (x - x_min) / span
        return int(data_norm * ((1 << bits) - 1))

    def _uint_to_float(self, x: int, x_min: float, x_max: float, bits: int) -> float:
        """Convert unsigned integer from CAN to float."""
        span = x_max - x_min
        data_norm = float(x) / ((1 << bits) - 1)
        return data_norm * span + x_min

    def _decode_motor_state(self, data: bytes) -> tuple[float, float, float, float]:
        """
        Decode motor state from CAN data.

        Returns:
            Tuple of (position_degrees, velocity_deg_per_sec, torque, temp_mos)
        """
        if len(data) < 8:
            raise ValueError("Invalid motor state data")

        # Extract encoded values
        motor_id = data[0]
        motor_name = self._id_to_name[motor_id]
        q_uint = (data[1] << 8) | data[2]
        dq_uint = (data[3] << 4) | (data[4] >> 4)
        tau_uint = ((data[4] & 0x0F) << 8) | data[5]
        t_mos = (data[6] << 8) | data[7]

        motor_type = self._motor_types.get(motor_name)
        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Decode to physical values (radians)
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        # Convert to degrees
        position_degrees = np.degrees(position_rad)
        velocity_deg_per_sec = np.degrees(velocity_rad_per_sec)

        # Update cached state
        self.last_feedback_time[motor_name] = time.time()
        self.currentPosition[motor_name] = position_degrees
        self.currentVelocity[motor_name] = velocity_deg_per_sec
        self.currentTorque[motor_name] = torque
        self.currentTemperature[motor_name] = t_mos / 10
        return position_degrees, velocity_deg_per_sec, torque, t_mos / 10

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = False,
        num_retry: int = 0,
    ) -> Value:
        if normalize:
            logger.warning(
                "Normalization parameter is ignored for Robstride motors (positions are always in degrees)."
            )

        """Read a value from a single motor. Positions are always in degrees."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Refresh motor to get latest state
        t_init = time.time()
        if (
            self.last_feedback_time[motor] is None
            or (t_init - self.last_feedback_time[motor]) > STATE_CACHE_TTL_S
        ):
            self.update_motor_state(motor)

        position_degrees = self.currentPosition[motor]
        velocity_deg_per_sec = self.currentVelocity[motor]
        torque = self.currentTorque[motor]
        t_mos = self.currentTemperature[motor]

        # Return requested data (already in degrees for position/velocity)
        if data_name == "Present_Position":
            value = position_degrees
        elif data_name == "Present_Velocity":
            value = velocity_deg_per_sec
        elif data_name == "Present_Torque":
            value = torque
        elif data_name == "Temperature_MOS":
            value = t_mos
        elif data_name == "Temperature_Rotor":
            raise NotImplementedError("Rotor temperature reading not accessible.")
        else:
            raise ValueError(f"Unknown data_name: {data_name}")

        # For Robstride, positions are always in degrees, no normalization needed
        # We keep the normalize parameter for compatibility but don't use it
        return value

    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
        *,
        normalize: bool = False,
        num_retry: int = 0,
    ) -> None:
        if normalize:
            logger.warning(
                "Normalization parameter is ignored for Robstride motors (positions are always in degrees)."
            )

        """Write a value to a single motor. Positions are always in degrees."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Value is expected to be in degrees for positions
        if data_name == "Goal_Position":
            # Use MIT control with position in degrees
            motor_name = self._get_motor_name(motor)
            kp = self._motor_kps.get(motor_name, 15)
            kd = self._motor_kds.get(motor_name, 0.1)
            self._mit_control(motor, kp, kd, value, 0, 0)
        elif data_name == "Goal_Velocity":
            # Use Velocity control mode
            motor_name = self._get_motor_name(motor)
            if self.operation_mode[motor_name] != ControlMode.VEL:
                raise RuntimeError(f"Motor '{motor_name}' is not in velocity control mode.")
            current_limit_a = 5.0  # Example current limit / not specified in doc. This mode is rarely used and primarily intended for diagnostics
            self._speed_control(motor, value, current_limit_a)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = False,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """
        Read the same value from multiple motors simultaneously.
        Uses batched operations: sends all refresh commands, then collects all responses.
        This is MUCH faster than sequential reads (OpenArms pattern).
        """
        if normalize:
            logger.warning(
                "Normalization parameter is ignored for Robstride motors (positions are always in degrees)."
            )

        motors = self._get_motors_list(motors)
        result = {}
        init_time = time.time()
        updated_motor = []
        # Step 1: Send refresh commands to ALL motors first (no waiting)
        for motor in motors:
            if (
                self.last_feedback_time[motor] is not None
                and (init_time - self.last_feedback_time[motor]) < STATE_CACHE_TTL_S
            ):
                # Skip refresh if we got recent feedback (<20ms ago)
                continue
            motor_id = self._get_motor_id(motor)
            data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self.canbus.send(msg)
            updated_motor.append(motor)

        expected_recv_ids = [self._get_motor_recv_id(motor) for motor in updated_motor]
        responses = self._recv_all_responses(expected_recv_ids, timeout=RUNNING_TIMEOUT)
        for response in responses.values():
            self._decode_motor_state(response.data)  # Update cached state

        # Step 2: receive and parse responses
        for motor in motors:
            try:
                # Get latest cached state
                position_degrees = self.currentPosition[motor]
                velocity_deg_per_sec = self.currentVelocity[motor]
                torque = self.currentTorque[motor]
                t_mos = self.currentTemperature[motor]

                if data_name == "Present_Position":
                    value = position_degrees
                elif data_name == "Present_Velocity":
                    value = velocity_deg_per_sec
                elif data_name == "Present_Torque":
                    value = torque
                elif data_name == "Temperature_MOS":
                    value = t_mos
                elif data_name == "Temperature_Rotor":
                    raise NotImplementedError("Rotor temperature reading not accessible.")
                else:
                    raise ValueError(f"Unknown data_name: {data_name}")

                result[motor] = value

            except Exception as e:
                logger.warning(f"Failed to read {data_name} from {motor}: {e}")
                result[motor] = 0.0

        return result

    def sync_write(
        self,
        data_name: str,
        values: dict[str, Value],
        *,
        normalize: bool = False,
        num_retry: int = 0,
    ) -> None:
        """
        Write different values to multiple motors simultaneously. Positions are always in degrees.
        Uses batched operations: sends all commands first, then collects responses when MIT mode is used, otherwise send cmd and wait for response for each motor).
        """
        if normalize:
            logger.warning(
                "Normalization parameter is ignored for Robstride motors (positions are always in degrees)."
            )

        if data_name == "Goal_Position":
            # Step 1: Send all MIT control commands first (no waiting)
            for motor, value_degrees in values.items():
                motor_id = self._get_motor_id(motor)
                motor_name = self._get_motor_name(motor)
                motor_type = self._motor_types.get(motor_name)
                if self.operation_mode[motor_name] != ControlMode.MIT:
                    raise RuntimeError(f"Motor '{motor_name}' is not in MIT control mode.")
                # Convert degrees to radians
                position_rad = np.radians(value_degrees)

                # Default gains for position control
                kp, kd = self._motor_kps.get(motor_name, 15), self._motor_kds.get(motor_name, 0.1)

                # Get motor limits and encode parameters
                pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]
                kp_uint = self._float_to_uint(kp, 0, 500, 12)
                kd_uint = self._float_to_uint(kd, 0, 5, 12)
                q_uint = self._float_to_uint(position_rad, -pmax, pmax, 16)
                dq_uint = self._float_to_uint(0, -vmax, vmax, 12)
                tau_uint = self._float_to_uint(0, -tmax, tmax, 12)

                # Pack data
                data = [0] * 8
                data[0] = (q_uint >> 8) & 0xFF
                data[1] = q_uint & 0xFF
                data[2] = dq_uint >> 4
                data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
                data[4] = kp_uint & 0xFF
                data[5] = kd_uint >> 4
                data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
                data[7] = tau_uint & 0xFF

                msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
                self.canbus.send(msg)

            # Step 2: Collect all responses at once
            expected_recv_ids = [self._get_motor_recv_id(motor) for motor in values]
            responses = self._recv_all_responses(expected_recv_ids, timeout=RUNNING_TIMEOUT)  # 2ms timeout
            for response in responses.values():
                self._decode_motor_state(response.data)  # Update cached state
        else:
            # Fall back to individual writes for other data types
            for motor, value in values.items():
                self.write(data_name, motor, value, normalize=normalize, num_retry=num_retry)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration data from motors."""
        # Robstride motors don't store calibration internally
        # Return existing calibration or empty dict
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration data to motors."""
        # Robstride motors don't store calibration internally
        # Just cache it in memory
        if cache:
            self.calibration = calibration_dict

    def record_ranges_of_motion(
        self, motors: NameOrID | list[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[NameOrID, Value], dict[NameOrID, Value]]:
        """
        Interactively record the min/max values of each motor in degrees.

        Move the joints by hand (with torque disabled) while the method streams live positions.
        Press Enter to finish.
        """
        if motors is None:
            motors = list(self.motors.keys())
        elif isinstance(motors, (str, int)):
            motors = [motors]

        # Disable torque for manual movement
        self.disable_torque(motors)
        time.sleep(0.1)

        # Get initial positions (already in degrees)
        start_positions = self.sync_read("Present_Position", motors, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        print("\nMove joints through their full range of motion. Press ENTER when done.")
        user_pressed_enter = False

        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motors, normalize=False)

            for motor in motors:
                if motor in positions:
                    mins[motor] = int(min(positions[motor], mins.get(motor, positions[motor])))
                    maxes[motor] = int(max(positions[motor], maxes.get(motor, positions[motor])))

            if display_values:
                print("\n" + "=" * 50)
                print(f"{'MOTOR':<20} | {'MIN (deg)':>12} | {'POS (deg)':>12} | {'MAX (deg)':>12}")
                print("-" * 50)
                for motor in motors:
                    if motor in positions:
                        print(
                            f"{motor:<20} | {mins[motor]:>12.1f} | {positions[motor]:>12.1f} | {maxes[motor]:>12.1f}"
                        )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                # Move cursor up to overwrite the previous output
                move_cursor_up(len(motors) + 4)

            time.sleep(0.05)

        # Re-enable torque
        self.enable_torque(motors)

        # Validate ranges
        for motor in motors:
            if motor in mins and motor in maxes and (abs(maxes[motor] - mins[motor]) < 5.0):
                raise ValueError(f"Motor {motor} has insufficient range of motion (< 5 degrees)")

        return mins, maxes

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        """Convert motor specification to list of motor names."""
        if motors is None:
            return list(self.motors.keys())
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors
        else:
            raise TypeError(f"Invalid motors type: {type(motors)}")

    def _get_motor_id(self, motor: NameOrID) -> int:
        """Get CAN ID for a motor."""
        if isinstance(motor, str):
            if motor in self.motors:
                return self.motors[motor].id
            else:
                raise ValueError(f"Unknown motor: {motor}")
        else:
            return motor

    def _get_motor_name(self, motor: NameOrID) -> str:
        """Get motor name from name or ID."""
        if isinstance(motor, str):
            return motor
        else:
            for name, m in self.motors.items():
                if m.id == motor:
                    return name
            raise ValueError(f"Unknown motor ID: {motor}")

    def _get_motor_recv_id(self, motor: NameOrID) -> int:
        """Return the expected ID found in feedback payload byte0 for this motor.

        Robstride MIT feedback frames encode an ID in data[0]. Some setups expose it as
        `motor.recv_id`; otherwise we fall back to the configured `motor.id`.
        """
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors.get(motor_name)

        recv_id = getattr(motor_obj, "recv_id", None)
        if recv_id is None:
            logger.debug(
                "Motor '%s' has no recv_id; falling back to motor.id=%s for feedback demux.",
                motor_name,
                motor_obj.id,
            )
            return motor_obj.id

        return recv_id

    @cached_property
    def is_calibrated(self) -> bool:
        """Check if motors are calibrated."""
        return bool(self.calibration)
