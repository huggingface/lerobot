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

# Portions of this file are derived from DM_Control_Python by cmjang.
# Licensed under the MIT License; see `LICENSE` for the full text:
# https://github.com/cmjang/DM_Control_Python

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypedDict

from lerobot.utils.import_utils import _can_available

if TYPE_CHECKING or _can_available:
    import can
else:
    can.Message = object
    can.interface = None

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value
from .tables import (
    AVAILABLE_BAUDRATES,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_REFRESH,
    CAN_CMD_SET_ZERO,
    CAN_PARAM_ID,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT_MS,
    MIT_KD_RANGE,
    MIT_KP_RANGE,
    MOTOR_LIMIT_PARAMS,
    MotorType,
)

logger = logging.getLogger(__name__)


LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01
SHORT_TIMEOUT_SEC = 0.001
PRECISE_TIMEOUT_SEC = 0.0001


class MotorState(TypedDict):
    position: float
    velocity: float
    torque: float
    temp_mos: float
    temp_rotor: float


class DamiaoMotorsBus(MotorsBusBase):
    """
    The Damiao implementation for a MotorsBus using CAN bus communication.

    This class uses python-can for CAN bus communication with Damiao motors.
    For more info, see:
    - python-can documentation: https://python-can.readthedocs.io/en/stable/
    - Seedstudio documentation: https://wiki.seeedstudio.com/damiao_series/
    - DM_Control_Python repo: https://github.com/cmjang/DM_Control_Python
    """

    # CAN-specific settings
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

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
        Initialize the Damiao motors bus.

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
        self.canbus: can.interface.Bus | None = None
        self._is_connected = False

        # Map motor names to CAN IDs
        self._motor_can_ids: dict[str, int] = {}
        self._recv_id_to_motor: dict[int, str] = {}
        self._motor_types: dict[str, MotorType] = {}

        for name, motor in self.motors.items():
            if motor.motor_type_str is None:
                raise ValueError(f"Motor '{name}' is missing required 'motor_type'")
            self._motor_types[name] = getattr(MotorType, motor.motor_type_str.upper().replace("-", "_"))

            # Map recv_id to motor name for filtering responses
            if motor.recv_id is not None:
                self._recv_id_to_motor[motor.recv_id] = name

        # State cache for handling packet drops safely
        self._last_known_states: dict[str, MotorState] = {
            name: {
                "position": 0.0,
                "velocity": 0.0,
                "torque": 0.0,
                "temp_mos": 0.0,
                "temp_rotor": 0.0,
            }
            for name in self.motors
        }

        # Dynamic gains storage
        # Defaults: Kp=10.0 (Stiffness), Kd=0.5 (Damping)
        self._gains: dict[str, dict[str, float]] = {name: {"kp": 10.0, "kd": 0.5} for name in self.motors}

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
                    self.can_interface = "slcan"
                    logger.info(f"Auto-detected slcan interface for port {self.port}")
                else:
                    self.can_interface = "socketcan"
                    logger.info(f"Auto-detected socketcan interface for port {self.port}")

            # Connect to CAN bus
            kwargs = {
                "channel": self.port,
                "bitrate": self.bitrate,
                "interface": self.can_interface,
            }

            if self.can_interface == "socketcan" and self.use_can_fd and self.data_bitrate is not None:
                kwargs.update({"data_bitrate": self.data_bitrate, "fd": True})
                logger.info(
                    f"Connected to {self.port} with CAN FD (bitrate={self.bitrate}, data_bitrate={self.data_bitrate})"
                )
            else:
                logger.info(f"Connected to {self.port} with {self.can_interface} (bitrate={self.bitrate})")

            self.canbus = can.interface.Bus(**kwargs)
            self._is_connected = True

            if handshake:
                self._handshake()

            logger.debug(f"{self.__class__.__name__} connected via {self.can_interface}.")
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to connect to CAN bus: {e}") from e

    def _handshake(self) -> None:
        """
        Verify all motors are present and populate initial state cache.
        Raises ConnectionError if any motor fails to respond.
        """
        logger.info("Starting handshake with motors...")
        missing_motors = []

        for motor_name in self.motors:
            msg = self._refresh_motor(motor_name)
            if msg is None:
                missing_motors.append(motor_name)
            else:
                self._process_response(motor_name, msg)
            time.sleep(MEDIUM_TIMEOUT_SEC)

        if missing_motors:
            raise ConnectionError(
                f"Handshake failed. The following motors did not respond: {missing_motors}. "
                "Check power (24V) and CAN wiring."
            )
        logger.info("Handshake successful. All motors ready.")

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
        # Damiao motors don't require much configuration in MIT mode
        # Just ensure they're enabled
        for motor in self.motors:
            self._send_simple_command(motor, CAN_CMD_ENABLE)
            time.sleep(MEDIUM_TIMEOUT_SEC)

    def _send_simple_command(self, motor: NameOrID, command_byte: int) -> None:
        """Helper to send simple 8-byte commands (Enable, Disable, Zero)."""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [command_byte]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        if msg := self._recv_motor_response(expected_recv_id=recv_id):
            self._process_response(motor_name, msg)
        else:
            logger.debug(f"No response from {motor_name} after command 0x{command_byte:02X}")

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            for _ in range(num_retry + 1):
                try:
                    self._send_simple_command(motor, CAN_CMD_ENABLE)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            for _ in range(num_retry + 1):
                try:
                    self._send_simple_command(motor, CAN_CMD_DISABLE)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """
        Context manager that guarantees torque is re-enabled.

        This helper is useful to temporarily disable torque when configuring motors.
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """Set current position as zero for selected motors."""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            self._send_simple_command(motor, CAN_CMD_SET_ZERO)
            time.sleep(MEDIUM_TIMEOUT_SEC)

    def _refresh_motor(self, motor: NameOrID) -> can.Message | None:
        """Refresh motor status and return the response."""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
        msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False)
        self.canbus.send(msg)
        return self._recv_motor_response(expected_recv_id=recv_id)

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
                msg = self.canbus.recv(timeout=PRECISE_TIMEOUT_SEC)
                if msg:
                    messages_seen.append(f"0x{msg.arbitration_id:02X}")
                    if expected_recv_id is None or msg.arbitration_id == expected_recv_id:
                        return msg
                    logger.debug(
                        f"Ignoring message from 0x{msg.arbitration_id:02X}, expected 0x{expected_recv_id:02X}"
                    )

            if logger.isEnabledFor(logging.DEBUG):
                if messages_seen:
                    logger.debug(
                        f"Received {len(messages_seen)} msgs from {set(messages_seen)}, expected 0x{expected_recv_id:02X}"
                    )
                else:
                    logger.debug(f"No CAN messages received (expected 0x{expected_recv_id:02X})")
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
                # 100us poll timeout
                msg = self.canbus.recv(timeout=PRECISE_TIMEOUT_SEC)
                if msg and msg.arbitration_id in expected_set:
                    responses[msg.arbitration_id] = msg
                    if len(responses) == len(expected_recv_ids):
                        break
        except Exception as e:
            logger.debug(f"Error receiving responses: {e}")

        return responses

    def _encode_mit_packet(
        self,
        motor_type: MotorType,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> list[int]:
        """Helper to encode control parameters into 8 bytes for MIT mode."""
        # Convert degrees to radians
        position_rad = np.radians(position_degrees)
        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Encode parameters
        kp_uint = self._float_to_uint(kp, *MIT_KP_RANGE, 12)
        kd_uint = self._float_to_uint(kd, *MIT_KD_RANGE, 12)
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
        return data

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> None:
        """Send MIT control command to a motor."""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        motor_type = self._motor_types[motor_name]

        data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)

        recv_id = self._get_motor_recv_id(motor)
        if msg := self._recv_motor_response(expected_recv_id=recv_id):
            self._process_response(motor_name, msg)
        else:
            logger.debug(f"No response from {motor_name} after MIT control command")

    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> None:
        """
        Send MIT control commands to multiple motors in batch.
        Sends all commands first, then collects responses.

        Args:
            commands: Dict mapping motor name/ID to (kp, kd, position_deg, velocity_deg/s, torque)
                     Example: {'joint_1': (10.0, 0.5, 45.0, 0.0, 0.0), ...}
        """
        if not commands:
            return

        recv_id_to_motor: dict[int, str] = {}

        # Step 1: Send all MIT control commands
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            motor_id = self._get_motor_id(motor)
            motor_name = self._get_motor_name(motor)
            motor_type = self._motor_types[motor_name]

            data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self.canbus.send(msg)

            recv_id_to_motor[self._get_motor_recv_id(motor)] = motor_name

        # Step 2: Collect responses and update state cache
        responses = self._recv_all_responses(list(recv_id_to_motor.keys()), timeout=SHORT_TIMEOUT_SEC)
        for recv_id, motor_name in recv_id_to_motor.items():
            if msg := responses.get(recv_id):
                self._process_response(motor_name, msg)

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

    def _decode_motor_state(
        self, data: bytearray | bytes, motor_type: MotorType
    ) -> tuple[float, float, float, int, int]:
        """
        Decode motor state from CAN data.
        Returns: (position_deg, velocity_deg_s, torque, temp_mos, temp_rotor)
        """
        if len(data) < 8:
            raise ValueError("Invalid motor state data")

        # Extract encoded values
        q_uint = (data[1] << 8) | data[2]
        dq_uint = (data[3] << 4) | (data[4] >> 4)
        tau_uint = ((data[4] & 0x0F) << 8) | data[5]
        t_mos = data[6]
        t_rotor = data[7]

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Decode to physical values
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        return np.degrees(position_rad), np.degrees(velocity_rad_per_sec), torque, t_mos, t_rotor

    def _process_response(self, motor: str, msg: can.Message) -> None:
        """Decode a message and update the motor state cache."""
        try:
            motor_type = self._motor_types[motor]
            pos, vel, torque, t_mos, t_rotor = self._decode_motor_state(msg.data, motor_type)

            self._last_known_states[motor] = {
                "position": pos,
                "velocity": vel,
                "torque": torque,
                "temp_mos": float(t_mos),
                "temp_rotor": float(t_rotor),
            }
        except Exception as e:
            logger.warning(f"Failed to decode response from {motor}: {e}")

    def read(self, data_name: str, motor: str) -> Value:
        """Read a value from a single motor. Positions are always in degrees."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Refresh motor to get latest state
        msg = self._refresh_motor(motor)
        if msg is None:
            motor_id = self._get_motor_id(motor)
            recv_id = self._get_motor_recv_id(motor)
            raise ConnectionError(
                f"No response from motor '{motor}' (send ID: 0x{motor_id:02X}, recv ID: 0x{recv_id:02X}). "
                f"Check that: 1) Motor is powered (24V), 2) CAN wiring is correct, "
                f"3) Motor IDs are configured correctly using Damiao Debugging Tools"
            )

        self._process_response(motor, msg)
        return self._get_cached_value(motor, data_name)

    def _get_cached_value(self, motor: str, data_name: str) -> Value:
        """Retrieve a specific value from the cache."""
        state = self._last_known_states[motor]
        mapping: dict[str, Any] = {
            "Present_Position": state["position"],
            "Present_Velocity": state["velocity"],
            "Present_Torque": state["torque"],
            "Temperature_MOS": state["temp_mos"],
            "Temperature_Rotor": state["temp_rotor"],
        }
        if data_name not in mapping:
            raise ValueError(f"Unknown data_name: {data_name}")
        return mapping[data_name]

    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
    ) -> None:
        """
        Write a value to a single motor. Positions are always in degrees.
        Can write 'Goal_Position', 'Kp', or 'Kd'.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if data_name in ("Kp", "Kd"):
            self._gains[motor][data_name.lower()] = float(value)
        elif data_name == "Goal_Position":
            kp = self._gains[motor]["kp"]
            kd = self._gains[motor]["kd"]
            self._mit_control(motor, kp, kd, float(value), 0.0, 0.0)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
    ) -> dict[str, Value]:
        """
        Read the same value from multiple motors simultaneously.
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._get_cached_value(motor, data_name)
        return result

    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, MotorState]:
        """
        Read ALL motor states (position, velocity, torque) from multiple motors in ONE refresh cycle.

        Returns:
            Dictionary mapping motor names to state dicts with keys: 'position', 'velocity', 'torque'
            Example: {'joint_1': {'position': 45.2, 'velocity': 1.3, 'torque': 0.5}, ...}
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._last_known_states[motor].copy()
        return result

    def _batch_refresh(self, motors: list[str]) -> None:
        """Internal helper to refresh a list of motors and update cache."""
        # Send refresh commands
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False)
            self.canbus.send(msg)
            # Small delay to reduce bus congestion if necessary, though removed in sync_read previously
            # precise_sleep(PRECISE_SLEEP_SEC)

        # Collect responses
        expected_recv_ids = [self._get_motor_recv_id(m) for m in motors]
        responses = self._recv_all_responses(expected_recv_ids, timeout=MEDIUM_TIMEOUT_SEC)

        # Update cache
        for motor in motors:
            recv_id = self._get_motor_recv_id(motor)
            msg = responses.get(recv_id)
            if msg:
                self._process_response(motor, msg)
            else:
                logger.warning(f"Packet drop: {motor} (ID: 0x{recv_id:02X}). Using last known state.")

    def sync_write(self, data_name: str, values: Value | dict[str, Value]) -> None:
        """
        Write values to multiple motors simultaneously. Positions are always in degrees.
        """
        if data_name in ("Kp", "Kd"):
            key = data_name.lower()
            for motor, val in values.items():
                self._gains[motor][key] = float(val)

        elif data_name == "Goal_Position":
            # Step 1: Send all MIT control commands
            recv_id_to_motor: dict[int, str] = {}
            for motor, value_degrees in values.items():
                motor_id = self._get_motor_id(motor)
                motor_name = self._get_motor_name(motor)
                motor_type = self._motor_types[motor_name]

                kp = self._gains[motor]["kp"]
                kd = self._gains[motor]["kd"]

                data = self._encode_mit_packet(motor_type, kp, kd, float(value_degrees), 0.0, 0.0)
                msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
                self.canbus.send(msg)
                precise_sleep(PRECISE_TIMEOUT_SEC)

                recv_id_to_motor[self._get_motor_recv_id(motor)] = motor_name

            # Step 2: Collect responses and update state cache
            responses = self._recv_all_responses(list(recv_id_to_motor.keys()), timeout=MEDIUM_TIMEOUT_SEC)
            for recv_id, motor_name in recv_id_to_motor.items():
                if msg := responses.get(recv_id):
                    self._process_response(motor_name, msg)
        else:
            # Fall back to individual writes
            for motor, value in values.items():
                self.write(data_name, motor, value)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration data from motors."""
        # Damiao motors don't store calibration internally
        # Return existing calibration or empty dict
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration data to motors."""
        # Damiao motors don't store calibration internally
        # Just cache it in memory
        if cache:
            self.calibration = calibration_dict

    def record_ranges_of_motion(
        self,
        motors: NameOrID | list[NameOrID] | None = None,
        display_values: bool = True,
    ) -> tuple[dict[NameOrID, Value], dict[NameOrID, Value]]:
        """
        Interactively record the min/max values of each motor in degrees.

        Move the joints by hand (with torque disabled) while the method streams live positions.
        Press Enter to finish.
        """
        target_motors = self._get_motors_list(motors)

        self.disable_torque(target_motors)
        time.sleep(LONG_TIMEOUT_SEC)

        start_positions = self.sync_read("Present_Position", target_motors)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        print("\nMove joints through their full range of motion. Press ENTER when done.")
        user_pressed_enter = False

        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", target_motors)

            for motor in target_motors:
                if motor in positions:
                    mins[motor] = min(positions[motor], mins.get(motor, positions[motor]))
                    maxes[motor] = max(positions[motor], maxes.get(motor, positions[motor]))

            if display_values:
                print("\n" + "=" * 50)
                print(f"{'MOTOR':<20} | {'MIN (deg)':>12} | {'POS (deg)':>12} | {'MAX (deg)':>12}")
                print("-" * 50)
                for motor in target_motors:
                    if motor in positions:
                        print(
                            f"{motor:<20} | {mins[motor]:>12.1f} | {positions[motor]:>12.1f} | {maxes[motor]:>12.1f}"
                        )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(target_motors) + 4)

            time.sleep(LONG_TIMEOUT_SEC)

        self.enable_torque(target_motors)

        for motor in target_motors:
            if (motor in mins) and (motor in maxes) and (int(abs(maxes[motor] - mins[motor])) < 5):
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
        """Get motor recv_id from name or ID."""
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors.get(motor_name)
        if motor_obj and motor_obj.recv_id is not None:
            return motor_obj.recv_id
        else:
            raise ValueError(f"Motor {motor_obj} doesn't have a valid recv_id (None).")

    @cached_property
    def is_calibrated(self) -> bool:
        """Check if motors are calibrated."""
        return bool(self.calibration)
