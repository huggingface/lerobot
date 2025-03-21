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

# TODO(aliberts): This noqa is for the PortHandler / PacketHandler Protocols
# Add block noqa when feature below is available
# https://github.com/astral-sh/ruff/issues/3711
# ruff: noqa: N802

import abc
import json
import time
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from pprint import pformat
from typing import Protocol, TypeAlias, overload

import serial
from deepdiff import DeepDiff

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

NameOrID: TypeAlias = str | int
Value: TypeAlias = int | float

MAX_ID_RANGE = 252


def get_group_sync_key(data_name: str, motor_ids: list[int]) -> str:
    group_key = f"{data_name}_" + "_".join([str(idx) for idx in motor_ids])
    return group_key


def get_log_name(var_name: str, fn_name: str, data_name: str, motor_names: list[str]) -> str:
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


def assert_same_address(model_ctrl_table: dict[str, dict], motor_models: list[str], data_name: str) -> None:
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = model_ctrl_table[model][data_name]
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_addr, strict=False))}). Contact a LeRobot maintainer."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_bytes, strict=False))}). Contact a LeRobot maintainer."
        )


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with liner motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


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


@dataclass
class Motor:
    id: int
    model: str


class MotorsBus(abc.ABC):
    """The main LeRobot class for implementing motors buses.

    There are currently two implementations of this abstract class:
        - DynamixelMotorsBus
        - FeetechMotorsBus

    Note: This class may evolve in the future should we add support for other manufacturers SDKs.

    A MotorsBus allows to efficiently read and write to the attached motors.
    It represents several motors daisy-chained together and connected through a serial port.

    A MotorsBus subclass instance requires a port (e.g. `FeetechMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run our utility script:
    ```bash
    python lerobot/scripts/find_motors_bus_port.py
    >>> Finding all available ports for the MotorsBus.
    >>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
    >>> Remove the usb cable from your MotorsBus and press Enter when done.
    >>> The port of this MotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 Feetech sts3215 motor connected to the bus:
    ```python
    motors_bus = FeetechMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={"gripper": (6, "sts3215")},
    )
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # Move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # When done, properly disconnect the port using
    motors_bus.disconnect()
    ```
    """

    model_ctrl_table: dict[str, dict]
    model_resolution_table: dict[str, int]
    model_baudrate_table: dict[str, dict]
    calibration_required: list[str]
    default_timeout: int

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
    ):
        self.port = port
        self.motors = motors
        self._validate_motors()

        self.port_handler: PortHandler
        self.packet_handler: PacketHandler
        self.reader: GroupSyncRead
        self.writer: GroupSyncWrite

        self.logs = {}  # TODO(aliberts): use subclass logger
        self.calibration = None

        self._id_to_model = {m.id: m.model for m in self.motors.values()}
        self._id_to_name = {m.id: name for name, m in self.motors.items()}

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8)},\n"
            ")',\n"
        )

    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        if len(self.models) < 2:
            return False

        first_table = self.model_ctrl_table[self.models[0]]
        return any(DeepDiff(first_table, self.model_ctrl_table[model]) for model in self.models[1:])

    def id_to_model(self, motor_id: int) -> str:
        return self._id_to_model[motor_id]

    def id_to_name(self, motor_id: int) -> str:
        return self._id_to_name[motor_id]

    @cached_property
    def names(self) -> list[str]:
        return list(self.motors)

    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _validate_motors(self) -> None:
        # TODO(aliberts): improve error messages for this (display problematics values)
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Some motors have the same id.")

        if len(self.names) != len(set(self.names)):
            raise ValueError("Some motors have the same name.")

        if any(model not in self.model_resolution_table for model in self.models):
            raise ValueError("Some motors models are not available.")

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__}('{self.port}') is already connected. Do not call `{self.__class__.__name__}.connect()` twice."
            )

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            print(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py`\n"
            )
            raise e

        self.set_timeout()

    def set_timeout(self, timeout_ms: int | None = None):
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        self.port_handler.setPacketTimeoutMillis(timeout_ms)

    @property
    def are_motors_configured(self) -> bool:
        """
        Only check the motor indices and not baudrate, since if the motor baudrates are incorrect, a
        ConnectionError will be raised anyway.
        """
        try:
            # TODO(aliberts): use ping instead
            return (self.ids == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        idx = self.get_motor_id(motor)
        for _ in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(self.port_handler, idx)
            if self._is_comm_success(comm):
                return model_number

        if raise_on_error:
            raise ConnectionError(f"Ping motor {motor} returned a {error} error code.")

    @abc.abstractmethod
    def broadcast_ping(
        self, num_retry: int = 0, raise_on_error: bool = False
    ) -> dict[int, list[int, int]] | None: ...

    def set_baudrate(self, baudrate) -> None:
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise OSError("Failed to write bus baud rate.")

    def set_calibration(self, calibration_fpath: Path) -> None:
        with open(calibration_fpath) as f:
            calibration = json.load(f)

        self.calibration = {int(idx): val for idx, val in calibration.items()}

    @abc.abstractmethod
    def calibrate_values(self, ids_values: dict[int, int]) -> dict[int, float]:
        pass

    @abc.abstractmethod
    def uncalibrate_values(self, ids_values: dict[int, float]) -> dict[int, int]:
        pass

    @abc.abstractmethod
    def _is_comm_success(self, comm: int) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def split_int_bytes(value: int, n_bytes: int) -> list[int]:
        """
        Splits an unsigned integer into a list of bytes in little-endian order.

        This function extracts the individual bytes of an integer based on the
        specified number of bytes (`n_bytes`). The output is a list of integers,
        each representing a byte (0-255).

        **Byte order:** The function returns bytes in **little-endian format**,
        meaning the least significant byte (LSB) comes first.

        Args:
            value (int): The unsigned integer to be converted into a byte list. Must be within
                the valid range for the specified `n_bytes`.
            n_bytes (int): The number of bytes to use for conversion. Supported values:
                - 1 (for values 0 to 255)
                - 2 (for values 0 to 65,535)
                - 4 (for values 0 to 4,294,967,295)

        Raises:
            ValueError: If `value` is negative or exceeds the maximum allowed for `n_bytes`.
            NotImplementedError: If `n_bytes` is not 1, 2, or 4.

        Returns:
            list[int]: A list of integers, each representing a byte in **little-endian order**.

        Examples:
            >>> split_int_bytes(0x12, 1)
            [18]
            >>> split_int_bytes(0x1234, 2)
            [52, 18]  # 0x1234 → 0x34 0x12 (little-endian)
            >>> split_int_bytes(0x12345678, 4)
            [120, 86, 52, 18]  # 0x12345678 → 0x78 0x56 0x34 0x12
        """
        pass

    def get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    @overload
    def read(self, data_name: str, motors: None = ...) -> dict[str, Value]: ...
    @overload
    def read(self, data_name: str, motors: NameOrID) -> dict[NameOrID, Value]: ...
    @overload
    def read(self, data_name: str, motors: list[NameOrID]) -> dict[NameOrID, Value]: ...
    def read(
        self, data_name: str, motors: NameOrID | list[NameOrID] | None = None, num_retry: int = 0
    ) -> dict[NameOrID, Value]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        start_time = time.perf_counter()

        id_key_map: dict[int, NameOrID] = {}
        if motors is None:
            id_key_map = {m.id: name for name, m in self.motors.items()}
        elif isinstance(motors, (str, int)):
            id_key_map = {self.get_motor_id(motors): motors}
        elif isinstance(motors, list):
            id_key_map = {self.get_motor_id(m): m for m in motors}
        else:
            raise TypeError(motors)

        motor_ids = list(id_key_map)
        if self._has_different_ctrl_tables:
            models = [self.id_to_model(idx) for idx in motor_ids]
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = self.id_to_model(next(iter(motor_ids)))
        addr, n_bytes = self.model_ctrl_table[model][data_name]

        comm, ids_values = self._read(motor_ids, addr, n_bytes, num_retry)
        if not self._is_comm_success(comm):
            raise ConnectionError(
                f"Failed to read {data_name} on port {self.port} for ids {motor_ids}:"
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        if data_name in self.calibration_required and self.calibration is not None:
            ids_values = self.calibrate_values(ids_values)

        keys_values = {id_key_map[idx]: val for idx, val in ids_values.items()}

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_ids)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_ids)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return keys_values

    def _read(
        self, motor_ids: list[str], address: int, n_bytes: int, num_retry: int = 0
    ) -> tuple[int, dict[int, int]]:
        self.reader.clearParam()
        self.reader.start_address = address
        self.reader.data_length = n_bytes

        # FIXME(aliberts, pkooij): We should probably not have to do this.
        # Let's try to see if we can do with better comm status handling instead.
        # self.port_handler.ser.reset_output_buffer()
        # self.port_handler.ser.reset_input_buffer()

        for idx in motor_ids:
            self.reader.addParam(idx)

        for _ in range(1 + num_retry):
            comm = self.reader.txRxPacket()
            if self._is_comm_success(comm):
                break

        values = {idx: self.reader.getData(idx, address, n_bytes) for idx in motor_ids}
        return comm, values

    # TODO(aliberts, pkooij): Implementing something like this could get much faster read times.
    # Note: this could be at the cost of increase latency between the moment the data is produced by the
    # motors and the moment it is used by a policy
    # def _async_read(self, motor_ids: list[str], address: int, n_bytes: int):
    #     self.reader.rxPacket()
    #     self.reader.txPacket()
    #     for idx in motor_ids:
    #         value = self.reader.getData(idx, address, n_bytes)

    def write(self, data_name: str, values: int | dict[NameOrID, int], num_retry: int = 0) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        start_time = time.perf_counter()

        if isinstance(values, int):
            ids_values = {id_: values for id_ in self.ids}
        elif isinstance(values, dict):
            ids_values = {self.get_motor_id(motor): val for motor, val in values.items()}
        else:
            raise ValueError(f"'values' is expected to be a single value or a dict. Got {values}")

        if self._has_different_ctrl_tables:
            models = [self.id_to_model(idx) for idx in ids_values]
            assert_same_address(self.model_ctrl_table, models, data_name)

        if data_name in self.calibration_required and self.calibration is not None:
            ids_values = self.uncalibrate_values(ids_values)

        model = self.id_to_model(next(iter(ids_values)))
        addr, n_bytes = self.model_ctrl_table[model][data_name]

        comm = self._write(ids_values, addr, n_bytes, num_retry)
        if not self._is_comm_success(comm):
            raise ConnectionError(
                f"Failed to write {data_name} on port {self.port} for ids {list(ids_values)}:"
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, list(ids_values))
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, list(ids_values))
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def _write(self, ids_values: dict[int, int], address: int, n_bytes: int, num_retry: int = 0) -> int:
        self.writer.clearParam()
        self.writer.start_address = address
        self.writer.data_length = n_bytes

        for idx, value in ids_values.items():
            data = self.split_int_bytes(value, n_bytes)
            self.writer.addParam(idx, data)

        for _ in range(1 + num_retry):
            comm = self.writer.txPacket()
            if self._is_comm_success(comm):
                break

        return comm

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. Try running `{self.__class__.__name__}.connect()` first."
            )

        self.port_handler.closePort()

    def __del__(self):
        if self.is_connected:
            self.disconnect()
