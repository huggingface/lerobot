#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# ruff: noqa: N802
# This noqa is for the Protocols classes: PortHandler, PacketHandler GroupSyncRead/Write
# TODO(aliberts): Add block noqa when feature below is available
# https://github.com/astral-sh/ruff/issues/3711

import abc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pprint import pformat
from typing import Protocol, TypeAlias

import serial
from deepdiff import DeepDiff
from tqdm import tqdm

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

NameOrID: TypeAlias = str | int
Value: TypeAlias = int | float

logger = logging.getLogger(__name__)


def get_ctrl_table(model_ctrl_table: dict[str, dict], model: str) -> dict[str, tuple[int, int]]:
    ctrl_table = model_ctrl_table.get(model)
    if ctrl_table is None:
        raise KeyError(f"Control table for {model=} not found.")
    return ctrl_table


def get_address(model_ctrl_table: dict[str, dict], model: str, data_name: str) -> tuple[int, int]:
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    addr_bytes = ctrl_table.get(data_name)
    if addr_bytes is None:
        raise KeyError(f"Address for '{data_name}' not found in {model} control table.")
    return addr_bytes


def assert_same_address(model_ctrl_table: dict[str, dict], motor_models: list[str], data_name: str) -> None:
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = get_address(model_ctrl_table, model, data_name)
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_addr, strict=False))})."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_bytes, strict=False))})."
        )


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode


class PortHandler(Protocol):
    def __init__(self, port_name):
        self.is_open: bool
        self.baudrate: int
        self.packet_start_time: float
        self.packet_timeout: float
        self.tx_time_per_byte: float
        self.is_using: bool
        self.port_name: str
        self.ser: serial.Serial

    def openPort(self): ...
    def closePort(self): ...
    def clearPort(self): ...
    def setPortName(self, port_name): ...
    def getPortName(self): ...
    def setBaudRate(self, baudrate): ...
    def getBaudRate(self): ...
    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class PacketHandler(Protocol):
    def getTxRxResult(self, result): ...
    def getRxPacketError(self, error): ...
    def txPacket(self, port, txpacket): ...
    def rxPacket(self, port): ...
    def txRxPacket(self, port, txpacket): ...
    def ping(self, port, id): ...
    def action(self, port, id): ...
    def readTx(self, port, id, address, length): ...
    def readRx(self, port, id, length): ...
    def readTxRx(self, port, id, address, length): ...
    def read1ByteTx(self, port, id, address): ...
    def read1ByteRx(self, port, id): ...
    def read1ByteTxRx(self, port, id, address): ...
    def read2ByteTx(self, port, id, address): ...
    def read2ByteRx(self, port, id): ...
    def read2ByteTxRx(self, port, id, address): ...
    def read4ByteTx(self, port, id, address): ...
    def read4ByteRx(self, port, id): ...
    def read4ByteTxRx(self, port, id, address): ...
    def writeTxOnly(self, port, id, address, length, data): ...
    def writeTxRx(self, port, id, address, length, data): ...
    def write1ByteTxOnly(self, port, id, address, data): ...
    def write1ByteTxRx(self, port, id, address, data): ...
    def write2ByteTxOnly(self, port, id, address, data): ...
    def write2ByteTxRx(self, port, id, address, data): ...
    def write4ByteTxOnly(self, port, id, address, data): ...
    def write4ByteTxRx(self, port, id, address, data): ...
    def regWriteTxOnly(self, port, id, address, length, data): ...
    def regWriteTxRx(self, port, id, address, length, data): ...
    def syncReadTx(self, port, start_address, data_length, param, param_length): ...
    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length): ...


class GroupSyncRead(Protocol):
    def __init__(self, port, ph, start_address, data_length):
        self.port: str
        self.ph: PortHandler
        self.start_address: int
        self.data_length: int
        self.last_result: bool
        self.is_param_changed: bool
        self.param: list
        self.data_dict: dict

    def makeParam(self): ...
    def addParam(self, id): ...
    def removeParam(self, id): ...
    def clearParam(self): ...
    def txPacket(self): ...
    def rxPacket(self): ...
    def txRxPacket(self): ...
    def isAvailable(self, id, address, data_length): ...
    def getData(self, id, address, data_length): ...


class GroupSyncWrite(Protocol):
    def __init__(self, port, ph, start_address, data_length):
        self.port: str
        self.ph: PortHandler
        self.start_address: int
        self.data_length: int
        self.is_param_changed: bool
        self.param: list
        self.data_dict: dict

    def makeParam(self): ...
    def addParam(self, id, data): ...
    def removeParam(self, id): ...
    def changeParam(self, id, data): ...
    def clearParam(self): ...
    def txPacket(self): ...


class MotorsBus(abc.ABC):
    """
    A MotorsBus allows to efficiently read and write to the attached motors.
    It represents several motors daisy-chained together and connected through a serial port.
    There are currently two implementations of this abstract class:
        - DynamixelMotorsBus
        - FeetechMotorsBus

    Note: This class may evolve in the future should we add support for other types of bus.

    A MotorsBus subclass instance requires a port (e.g. `FeetechMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run our utility script:
    ```bash
    lerobot-find-port.py
    >>> Finding all available ports for the MotorsBus.
    >>> ["/dev/tty.usbmodem575E0032081", "/dev/tty.usbmodem575E0031751"]
    >>> Remove the usb cable from your MotorsBus and press Enter when done.
    >>> The port of this MotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 Feetech sts3215 motor connected to the bus:
    ```python
    bus = FeetechMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={"my_motor": (1, "sts3215")},
    )
    bus.connect()

    position = bus.read("Present_Position", "my_motor", normalize=False)

    # Move from a few motor steps as an example
    few_steps = 30
    bus.write("Goal_Position", "my_motor", position + few_steps, normalize=False)

    # When done, properly disconnect the port using
    bus.disconnect()
    ```
    """

    apply_drive_mode: bool
    available_baudrates: list[int]
    default_baudrate: int
    default_timeout: int
    model_baudrate_table: dict[str, dict]
    model_ctrl_table: dict[str, dict]
    model_encoding_table: dict[str, dict]
    model_number_table: dict[str, int]
    model_resolution_table: dict[str, int]
    normalized_data: list[str]

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

        self.port_handler: PortHandler
        self.packet_handler: PacketHandler
        self.sync_reader: GroupSyncRead
        self.sync_writer: GroupSyncWrite
        self._comm_success: int
        self._no_error: int

        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}
        self._model_nb_to_model_dict = {v: k for k, v in self.model_number_table.items()}

        self._validate_motors()

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")',\n"
        )

    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        if len(self.models) < 2:
            return False

        first_table = self.model_ctrl_table[self.models[0]]
        return any(
            DeepDiff(first_table, get_ctrl_table(self.model_ctrl_table, model)) for model in self.models[1:]
        )

    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _model_nb_to_model(self, motor_nb: int) -> str:
        return self._model_nb_to_model_dict[motor_nb]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motor_model(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors.copy()
        else:
            raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> list[str]:
        if isinstance(values, (int | float)):
            return dict.fromkeys(self.ids, values)
        elif isinstance(values, dict):
            return {self.motors[motor].id: val for motor, val in values.items()}
        else:
            raise TypeError(f"'values' is expected to be a single value or a dict. Got {values}")

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Some motors have the same id!\n{self}")

        # Ensure ctrl table available for all models
        for model in self.models:
            get_ctrl_table(self.model_ctrl_table, model)

    def _is_comm_success(self, comm: int) -> bool:
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        return error != self._no_error

    def _assert_motors_exist(self) -> None:
        expected_models = {m.id: self.model_number_table[m.model] for m in self.motors.values()}

        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found_models[id_])
            for id_ in found_models
            if expected_models.get(id_) != found_models[id_]
        }

        if missing_ids or wrong_models:
            error_lines = [f"{self.__class__.__name__} motor check failed on port '{self.port}':"]

            if missing_ids:
                error_lines.append("\nMissing motor IDs:")
                error_lines.extend(
                    f"  - {id_} (expected model: {expected_models[id_]})" for id_ in missing_ids
                )

            if wrong_models:
                error_lines.append("\nMotors with incorrect model numbers:")
                error_lines.extend(
                    f"  - {id_} ({self._id_to_name(id_)}): expected {expected}, found {found}"
                    for id_, (expected, found) in wrong_models.items()
                )

            error_lines.append("\nFull expected motor list (id: model_number):")
            error_lines.append(pformat(expected_models, indent=4, sort_dicts=False))
            error_lines.append("\nFull found motor list (id: model_number):")
            error_lines.append(pformat(found_models, indent=4, sort_dicts=False))

            raise RuntimeError("\n".join(error_lines))

    @abc.abstractmethod
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.port_handler.is_open

    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and initialise communication.

        Args:
            handshake (bool, optional): Pings every expected motor and performs additional
                integrity checks specific to the implementation. Defaults to `True`.

        Raises:
            DeviceAlreadyConnectedError: The port is already open.
            ConnectionError: The underlying SDK failed to open the port or the handshake did not succeed.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__}('{self.port}') is already connected. Do not call `{self.__class__.__name__}.connect()` twice."
            )

        self._connect(handshake)
        self.set_timeout()
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            elif handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e

    @abc.abstractmethod
    def _handshake(self) -> None:
        pass

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port (optionally disabling torque first).

        Args:
            disable_torque (bool, optional): If `True` (default) torque is disabled on every motor before
                closing the port. This can prevent damaging motors if they are left applying resisting torque
                after disconnect.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. Try running `{self.__class__.__name__}.connect()` first."
            )

        if disable_torque:
            self.port_handler.clearPort()
            self.port_handler.is_using = False
            self.disable_torque(num_retry=5)

        self.port_handler.closePort()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    @classmethod
    def scan_port(cls, port: str, *args, **kwargs) -> dict[int, list[int]]:
        """Probe *port* at every supported baud-rate and list responding IDs.

        Args:
            port (str): Serial/USB port to scan (e.g. ``"/dev/ttyUSB0"``).
            *args, **kwargs: Forwarded to the subclass constructor.

        Returns:
            dict[int, list[int]]: Mapping *baud-rate → list of motor IDs*
            for every baud-rate that produced at least one response.
        """
        bus = cls(port, {}, *args, **kwargs)
        bus._connect(handshake=False)
        baudrate_ids = {}
        for baudrate in tqdm(bus.available_baudrates, desc="Scanning port"):
            bus.set_baudrate(baudrate)
            ids_models = bus.broadcast_ping()
            if ids_models:
                tqdm.write(f"Motors found for {baudrate=}: {pformat(ids_models, indent=4)}")
                baudrate_ids[baudrate] = list(ids_models)

        bus.port_handler.closePort()
        return baudrate_ids

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:
        """Assign the correct ID and baud-rate to a single motor.

        This helper temporarily switches to the motor's current settings, disables torque, sets the desired
        ID, and finally programs the bus' default baud-rate.

        Args:
            motor (str): Key of the motor in :pyattr:`motors`.
            initial_baudrate (int | None, optional): Current baud-rate (skips scanning when provided).
                Defaults to None.
            initial_id (int | None, optional): Current ID (skips scanning when provided). Defaults to None.

        Raises:
            RuntimeError: The motor could not be found or its model number
                does not match the expected one.
            ConnectionError: Communication with the motor failed.
        """
        if not self.is_connected:
            self._connect(handshake=False)

        if initial_baudrate is None:
            initial_baudrate, initial_id = self._find_single_motor(motor)

        if initial_id is None:
            _, initial_id = self._find_single_motor(motor, initial_baudrate)

        model = self.motors[motor].model
        target_id = self.motors[motor].id
        self.set_baudrate(initial_baudrate)
        self._disable_torque(initial_id, model)

        # Set ID
        addr, length = get_address(self.model_ctrl_table, model, "ID")
        self._write(addr, length, initial_id, target_id)

        # Set Baudrate
        addr, length = get_address(self.model_ctrl_table, model, "Baud_Rate")
        baudrate_value = self.model_baudrate_table[model][self.default_baudrate]
        self._write(addr, length, target_id, baudrate_value)

        self.set_baudrate(self.default_baudrate)

    @abc.abstractmethod
    def _find_single_motor(self, motor: str, initial_baudrate: int | None) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def configure_motors(self) -> None:
        """Write implementation-specific recommended settings to every motor.

        Typical changes include shortening the return delay, increasing
        acceleration limits or disabling safety locks.
        """
        pass

    @abc.abstractmethod
    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors.

        Disabling Torque allows to write to the motors' permanent memory area (EPROM/EEPROM).

        Args:
            motors (int | str | list[str] | None, optional): Target motors.  Accepts a motor name, an ID, a
                list of names or `None` to affect every registered motor.  Defaults to `None`.
            num_retry (int, optional): Number of additional retry attempts on communication failure.
                Defaults to 0.
        """
        pass

    @abc.abstractmethod
    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors.

        Args:
            motor (int): Same semantics as :pymeth:`disable_torque`. Defaults to `None`.
            num_retry (int, optional): Number of additional retry attempts on communication failure.
                Defaults to 0.
        """
        pass

    @contextmanager
    def torque_disabled(self, motors: int | str | list[str] | None = None):
        """Context-manager that guarantees torque is re-enabled.

        This helper is useful to temporarily disable torque when configuring motors.

        Examples:
            >>> with bus.torque_disabled():
            ...     # Safe operations here
            ...     pass
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_timeout(self, timeout_ms: int | None = None):
        """Change the packet timeout used by the SDK.

        Args:
            timeout_ms (int | None, optional): Timeout in *milliseconds*. If `None` (default) the method falls
                back to :pyattr:`default_timeout`.
        """
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        self.port_handler.setPacketTimeoutMillis(timeout_ms)

    def get_baudrate(self) -> int:
        """Return the current baud-rate configured on the port.

        Returns:
            int: Baud-rate in bits / second.
        """
        return self.port_handler.getBaudRate()

    def set_baudrate(self, baudrate: int) -> None:
        """Set a new UART baud-rate on the port.

        Args:
            baudrate (int): Desired baud-rate in bits / second.

        Raises:
            RuntimeError: The SDK failed to apply the change.
        """
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            logger.info(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise RuntimeError("Failed to write bus baud rate.")

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """bool: ``True`` if the cached calibration matches the motors."""
        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration parameters from the motors.

        Returns:
            dict[str, MotorCalibration]: Mapping *motor name → calibration*.
        """
        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration parameters to the motors and optionally cache them.

        Args:
            calibration_dict (dict[str, MotorCalibration]): Calibration obtained from
                :pymeth:`read_calibration` or crafted by the user.
            cache (bool, optional): Save the calibration to :pyattr:`calibration`. Defaults to True.
        """
        pass

    def reset_calibration(self, motors: NameOrID | list[NameOrID] | None = None) -> None:
        """Restore factory calibration for the selected motors.

        Homing offset is set to ``0`` and min/max position limits are set to the full usable range.
        The in-memory :pyattr:`calibration` is cleared.

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): Selection of motors. `None` (default)
                resets every motor.
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str | int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        for motor in motors:
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(self, motors: NameOrID | list[NameOrID] | None = None) -> dict[NameOrID, Value]:
        """Centre each motor range around its current position.

        The function computes and writes a homing offset such that the present position becomes exactly one
        half-turn (e.g. `2047` on a 12-bit encoder).

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): Motors to adjust. Defaults to all motors (`None`).

        Returns:
            dict[NameOrID, Value]: Mapping *motor → written homing offset*.
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str | int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        self.reset_calibration(motors)
        actual_positions = self.sync_read("Present_Position", motors, normalize=False)
        homing_offsets = self._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    @abc.abstractmethod
    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        pass

    def record_ranges_of_motion(
        self, motors: NameOrID | list[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[NameOrID, Value], dict[NameOrID, Value]]:
        """Interactively record the min/max encoder values of each motor.

        Move the joints by hand (with torque disabled) while the method streams live positions. Press
        :kbd:`Enter` to finish.

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): Motors to record.
                Defaults to every motor (`None`).
            display_values (bool, optional): When `True` (default) a live table is printed to the console.

        Returns:
            tuple[dict[NameOrID, Value], dict[NameOrID, Value]]: Two dictionaries *mins* and *maxes* with the
                extreme values observed for each motor.
        """
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str | int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        start_positions = self.sync_read("Present_Position", motors, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motors, normalize=False)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motors:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                # Move cursor up to overwrite the previous output
                move_cursor_up(len(motors) + 3)

        same_min_max = [motor for motor in motors if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[id_] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[id_] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized_values[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    @abc.abstractmethod
    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        pass

    @abc.abstractmethod
    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        pass

    def _serialize_data(self, value: int, length: int) -> list[int]:
        """
        Converts an unsigned integer value into a list of byte-sized integers to be sent via a communication
        protocol. Depending on the protocol, split values can be in big-endian or little-endian order.

        Supported data length for both Feetech and Dynamixel:
            - 1 (for values 0 to 255)
            - 2 (for values 0 to 65,535)
            - 4 (for values 0 to 4,294,967,295)
        """
        if value < 0:
            raise ValueError(f"Negative values are not allowed: {value}")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {length}. Expected [1, 2, 4].")

        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum for {length} bytes ({max_value}).")

        return self._split_into_byte_chunks(value, length)

    @abc.abstractmethod
    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """Convert an integer into a list of byte-sized integers."""
        pass

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        """Ping a single motor and return its model number.

        Args:
            motor (NameOrID): Target motor (name or ID).
            num_retry (int, optional): Extra attempts before giving up. Defaults to `0`.
            raise_on_error (bool, optional): If `True` communication errors raise exceptions instead of
                returning `None`. Defaults to `False`.

        Returns:
            int | None: Motor model number or `None` on failure.
        """
        id_ = self._get_motor_id(motor)
        for n_try in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(self.port_handler, id_)
            if self._is_comm_success(comm):
                break
            logger.debug(f"ping failed for {id_=}: {n_try=} got {comm=} {error=}")

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            else:
                return
        if self._is_error(error):
            if raise_on_error:
                raise RuntimeError(self.packet_handler.getRxPacketError(error))
            else:
                return

        return model_number

    @abc.abstractmethod
    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """Ping every ID on the bus using the broadcast address.

        Args:
            num_retry (int, optional): Retry attempts.  Defaults to `0`.
            raise_on_error (bool, optional): When `True` failures raise an exception instead of returning
                `None`. Defaults to `False`.

        Returns:
            dict[int, int] | None: Mapping *id → model number* or `None` if the call failed.
        """
        pass

    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
        """Read a register from a motor.

        Args:
            data_name (str): Control-table key (e.g. `"Present_Position"`).
            motor (str): Motor name.
            normalize (bool, optional): When `True` (default) scale the value to a user-friendly range as
                defined by the calibration.
            num_retry (int, optional): Retry attempts.  Defaults to `0`.

        Returns:
            Value: Raw or normalised value depending on *normalize*.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        value, _, _ = self._read(addr, length, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

        id_value = self._decode_sign(data_name, {id_: value})

        if normalize and data_name in self.normalized_data:
            id_value = self._normalize(id_value)

        return id_value[id_]

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        if length == 1:
            read_fn = self.packet_handler.read1ByteTxRx
        elif length == 2:
            read_fn = self.packet_handler.read2ByteTxRx
        elif length == 4:
            read_fn = self.packet_handler.read4ByteTxRx
        else:
            raise ValueError(length)

        for n_try in range(1 + num_retry):
            value, comm, error = read_fn(self.port_handler, motor_id, address)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to read @{address=} ({length=}) on {motor_id=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return value, comm, error

    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        """Write a value to a single motor's register.

        Contrary to :pymeth:`sync_write`, this expects a response status packet emitted by the motor, which
        provides a guarantee that the value was written to the register successfully. In consequence, it is
        slower than :pymeth:`sync_write` but it is more reliable. It should typically be used when configuring
        motors.

        Args:
            data_name (str): Register name.
            motor (str): Motor name.
            value (Value): Value to write.  If *normalize* is `True` the value is first converted to raw
                units.
            normalize (bool, optional): Enable or disable normalisation. Defaults to `True`.
            num_retry (int, optional): Retry attempts.  Defaults to `0`.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        if normalize and data_name in self.normalized_data:
            value = self._unnormalize({id_: value})[id_]

        value = self._encode_sign(data_name, {id_: value})[id_]

        err_msg = f"Failed to write '{data_name}' on {id_=} with '{value}' after {num_retry + 1} tries."
        self._write(addr, length, id_, value, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        data = self._serialize_data(value, length)
        for n_try in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(self.port_handler, motor_id, addr, length, data)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) on id={motor_id} with {value=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return comm, error

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """Read the same register from several motors at once.

        Args:
            data_name (str): Register name.
            motors (str | list[str] | None, optional): Motors to query. `None` (default) reads every motor.
            normalize (bool, optional): Normalisation flag.  Defaults to `True`.
            num_retry (int, optional): Retry attempts.  Defaults to `0`.

        Returns:
            dict[str, Value]: Mapping *motor name → value*.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        self._assert_protocol_is_compatible("sync_read")

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]
        models = [self.motors[motor].model for motor in names]

        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        ids_values, _ = self._sync_read(
            addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )

        ids_values = self._decode_sign(data_name, ids_values)

        if normalize and data_name in self.normalized_data:
            ids_values = self._normalize(ids_values)

        return {self._id_to_name(id_): value for id_, value in ids_values.items()}

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[dict[int, int], int]:
        self._setup_sync_reader(motor_ids, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync read @{addr=} ({length=}) on {motor_ids=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
        return values, comm

    def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)

    # TODO(aliberts, pkooij): Implementing something like this could get even much faster read times if need be.
    # Would have to handle the logic of checking if a packet has been sent previously though but doable.
    # This could be at the cost of increase latency between the moment the data is produced by the motors and
    # the moment it is used by a policy.
    # def _async_read(self, motor_ids: list[int], address: int, length: int):
    #     if self.sync_reader.start_address != address or self.sync_reader.data_length != length or ...:
    #         self._setup_sync_reader(motor_ids, address, length)
    #     else:
    #         self.sync_reader.rxPacket()
    #         self.sync_reader.txPacket()

    #     for id_ in motor_ids:
    #         value = self.sync_reader.getData(id_, address, length)

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write the same register on multiple motors.

        Contrary to :pymeth:`write`, this *does not* expects a response status packet emitted by the motor, which
        can allow for lost packets. It is faster than :pymeth:`write` and should typically be used when
        frequency matters and losing some packets is acceptable (e.g. teleoperation loops).

        Args:
            data_name (str): Register name.
            values (Value | dict[str, Value]): Either a single value (applied to every motor) or a mapping
                *motor name → value*.
            normalize (bool, optional): If `True` (default) convert values from the user range to raw units.
            num_retry (int, optional): Retry attempts.  Defaults to `0`.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        ids_values = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in ids_values]
        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        if normalize and data_name in self.normalized_data:
            ids_values = self._unnormalize(ids_values)

        ids_values = self._encode_sign(data_name, ids_values)

        err_msg = f"Failed to sync write '{data_name}' with {ids_values=} after {num_retry + 1} tries."
        self._sync_write(addr, length, ids_values, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int:
        self._setup_sync_writer(ids_values, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) with {ids_values=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        return comm

    def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for id_, value in ids_values.items():
            data = self._serialize_data(value, length)
            self.sync_writer.addParam(id_, data)
