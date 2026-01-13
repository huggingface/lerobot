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

# TODO(pepijn): add license of: https://github.com/cmjang/DM_Control_Python?tab=MIT-1-ov-file#readme

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property

import can
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
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
    MODEL_RESOLUTION,
    MOTOR_LIMIT_PARAMS,
    NORMALIZED_DATA,
    MotorType,
)

logger = logging.getLogger(__name__)


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
        self.canbus = None
        self._is_connected = False

        # Map motor names to CAN IDs
        self._motor_can_ids = {}
        self._recv_id_to_motor = {}

        # Store motor types and recv IDs
        self._motor_types = {}
        for name, motor in self.motors.items():
            if hasattr(motor, "motor_type"):
                self._motor_types[name] = motor.motor_type
            else:
                # Default to DM4310 if not specified
                self._motor_types[name] = MotorType.DM4310

            # Map recv_id to motor name for filtering responses
            if hasattr(motor, "recv_id"):
                self._recv_id_to_motor[motor.recv_id] = name

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
            raise ConnectionError(f"Failed to connect to CAN bus: {e}") from e

    def _handshake(self) -> None:
        """Verify all motors are present by refreshing their status."""
        for motor_name in self.motors:
            self._refresh_motor(motor_name)
            time.sleep(0.01)  # Small delay between motors

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
            self._enable_motor(motor)
            time.sleep(0.01)

    def _enable_motor(self, motor: NameOrID) -> None:
        """Enable a single motor."""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [CAN_CMD_ENABLE]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        self.canbus.send(msg)
        self._recv_motor_response(expected_recv_id=recv_id)

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
                msg = self.canbus.recv(timeout=0.0001)  # 100us timeout for fast polling
                if msg:
                    messages_seen.append(f"0x{msg.arbitration_id:02X}")
                    # If no filter specified, return any message
                    if expected_recv_id is None:
                        return msg
                    # Otherwise, only return if it matches the expected recv_id
                    if msg.arbitration_id == expected_recv_id:
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
                msg = self.canbus.recv(
                    timeout=0.0002
                )  # 200us poll timeout (increased from 100us for better reliability)
                if msg and msg.arbitration_id in expected_set:
                    responses[msg.arbitration_id] = msg
                    if len(responses) == len(expected_recv_ids):
                        break  # Got all responses, exit early
        except Exception as e:
            logger.debug(f"Error receiving responses: {e}")

        return responses

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
        motor_type = self._motor_types.get(motor_name, MotorType.DM4310)

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
        self._recv_motor_response(expected_recv_id=recv_id)

    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> None:
        """
        Send MIT control commands to multiple motors in batch (optimized).
        Sends all commands first, then collects responses. Much faster than sequential.

        Args:
            commands: Dict mapping motor name/ID to (kp, kd, position_deg, velocity_deg/s, torque)
                     Example: {'joint_1': (10.0, 0.5, 45.0, 0.0, 0.0), ...}
        """
        if not commands:
            return

        expected_recv_ids = []

        # Step 1: Send all MIT control commands (no waiting)
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            motor_id = self._get_motor_id(motor)
            motor_name = self._get_motor_name(motor)
            motor_type = self._motor_types.get(motor_name, MotorType.DM4310)

            # Convert degrees to radians
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

            # Send command
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
            self.canbus.send(msg)

            # Track expected response
            recv_id = self._get_motor_recv_id(motor)
            expected_recv_ids.append(recv_id)

        # Step 2: Collect all responses at once
        self._recv_all_responses(expected_recv_ids, timeout=0.002)

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

    def _decode_motor_state(self, data: bytes, motor_type: MotorType) -> tuple[float, float, float, int, int]:
        """
        Decode motor state from CAN data.

        Returns:
            Tuple of (position_degrees, velocity_deg_per_sec, torque, temp_mos, temp_rotor)
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

        # Decode to physical values (radians)
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        # Convert to degrees
        position_degrees = np.degrees(position_rad)
        velocity_deg_per_sec = np.degrees(velocity_rad_per_sec)

        return position_degrees, velocity_deg_per_sec, torque, t_mos, t_rotor

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
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

        motor_type = self._motor_types.get(motor, MotorType.DM4310)
        position_degrees, velocity_deg_per_sec, torque, t_mos, t_rotor = self._decode_motor_state(
            msg.data, motor_type
        )

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
            value = t_rotor
        else:
            raise ValueError(f"Unknown data_name: {data_name}")

        # For Damiao, positions are always in degrees, no normalization needed
        # We keep the normalize parameter for compatibility but don't use it
        return value

    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write a value to a single motor. Positions are always in degrees."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Value is expected to be in degrees for positions
        if data_name == "Goal_Position":
            # Use MIT control with position in degrees
            self._mit_control(motor, 10.0, 0.5, value, 0, 0)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """
        Read the same value from multiple motors simultaneously.
        Uses batched operations: sends all refresh commands, then collects all responses.
        This is MUCH faster than sequential reads (OpenArms pattern).
        """
        motors = self._get_motors_list(motors)
        result = {}

        # Step 1: Send refresh commands to ALL motors first (no waiting)
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False)
            self.canbus.send(msg)

        # Step 2: Collect all responses at once (batch receive)
        expected_recv_ids = [self._get_motor_recv_id(motor) for motor in motors]
        responses = self._recv_all_responses(expected_recv_ids, timeout=0.01)  # 10ms total timeout

        # Step 3: Parse responses
        for motor in motors:
            try:
                recv_id = self._get_motor_recv_id(motor)
                msg = responses.get(recv_id)

                if msg is None:
                    logger.warning(f"No response from motor '{motor}' (recv ID: 0x{recv_id:02X})")
                    result[motor] = 0.0
                    continue

                motor_type = self._motor_types.get(motor, MotorType.DM4310)
                position_degrees, velocity_deg_per_sec, torque, t_mos, t_rotor = self._decode_motor_state(
                    msg.data, motor_type
                )

                # Return requested data
                if data_name == "Present_Position":
                    value = position_degrees
                elif data_name == "Present_Velocity":
                    value = velocity_deg_per_sec
                elif data_name == "Present_Torque":
                    value = torque
                elif data_name == "Temperature_MOS":
                    value = t_mos
                elif data_name == "Temperature_Rotor":
                    value = t_rotor
                else:
                    raise ValueError(f"Unknown data_name: {data_name}")

                result[motor] = value

            except Exception as e:
                logger.warning(f"Failed to read {data_name} from {motor}: {e}")
                result[motor] = 0.0

        return result

    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, dict[str, Value]]:
        """
        Read ALL motor states (position, velocity, torque) from multiple motors in ONE refresh cycle.
        This is 3x faster than calling sync_read() three times separately.

        Returns:
            Dictionary mapping motor names to state dicts with keys: 'position', 'velocity', 'torque'
            Example: {'joint_1': {'position': 45.2, 'velocity': 1.3, 'torque': 0.5}, ...}
        """
        motors = self._get_motors_list(motors)
        result = {}

        # Step 1: Send refresh commands to ALL motors first (with small delays to reduce bus congestion)
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False)
            self.canbus.send(msg)
            time.sleep(0.0001)  # 100us delay between commands to reduce bus congestion

        # Step 2: Collect all responses at once (batch receive)
        expected_recv_ids = [self._get_motor_recv_id(motor) for motor in motors]
        responses = self._recv_all_responses(
            expected_recv_ids, timeout=0.015
        )  # 15ms timeout (increased for reliability)

        # Step 3: Parse responses and extract ALL state values
        for motor in motors:
            try:
                recv_id = self._get_motor_recv_id(motor)
                msg = responses.get(recv_id)

                if msg is None:
                    logger.warning(f"No response from motor '{motor}' (recv ID: 0x{recv_id:02X})")
                    result[motor] = {"position": 0.0, "velocity": 0.0, "torque": 0.0}
                    continue

                motor_type = self._motor_types.get(motor, MotorType.DM4310)
                position_degrees, velocity_deg_per_sec, torque, t_mos, t_rotor = self._decode_motor_state(
                    msg.data, motor_type
                )

                # Return all state values in one dict
                result[motor] = {
                    "position": position_degrees,
                    "velocity": velocity_deg_per_sec,
                    "torque": torque,
                    "temp_mos": t_mos,
                    "temp_rotor": t_rotor,
                }

            except Exception as e:
                logger.warning(f"Failed to read state from {motor}: {e}")
                result[motor] = {"position": 0.0, "velocity": 0.0, "torque": 0.0}

        return result

    def sync_write(
        self,
        data_name: str,
        values: dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """
        Write different values to multiple motors simultaneously. Positions are always in degrees.
        Uses batched operations: sends all commands first, then collects responses (OpenArms pattern).
        """
        if data_name == "Goal_Position":
            # Step 1: Send all MIT control commands first (no waiting)
            for motor, value_degrees in values.items():
                motor_id = self._get_motor_id(motor)
                motor_name = self._get_motor_name(motor)
                motor_type = self._motor_types.get(motor_name, MotorType.DM4310)

                # Convert degrees to radians
                position_rad = np.radians(value_degrees)

                # Default gains for position control
                kp, kd = 10.0, 0.5

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
                time.sleep(0.0001)  # 100us delay between commands to reduce bus congestion

            # Step 2: Collect all responses at once
            expected_recv_ids = [self._get_motor_recv_id(motor) for motor in values]
            self._recv_all_responses(
                expected_recv_ids, timeout=0.015
            )  # 15ms timeout (increased for reliability)
        else:
            # Fall back to individual writes for other data types
            for motor, value in values.items():
                self.write(data_name, motor, value, normalize=normalize, num_retry=num_retry)

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
                    mins[motor] = min(positions[motor], mins.get(motor, positions[motor]))
                    maxes[motor] = max(positions[motor], maxes.get(motor, positions[motor]))

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
            if (motor in mins) and (motor in maxes) and (abs(maxes[motor] - mins[motor]) < 5.0):
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

    def _get_motor_recv_id(self, motor: NameOrID) -> int | None:
        """Get motor recv_id from name or ID."""
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors.get(motor_name)
        if motor_obj and hasattr(motor_obj, "recv_id"):
            return motor_obj.recv_id
        return None

    @cached_property
    def is_calibrated(self) -> bool:
        """Check if motors are calibrated."""
        return bool(self.calibration)
