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
import time
import traceback
from enum import Enum
from typing import Protocol

import numpy as np
import tqdm

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

MAX_ID_RANGE = 252


def get_group_sync_key(data_name: str, motor_names: list[str]) -> str:
    group_key = f"{data_name}_" + "_".join(motor_names)
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
        self.ser: object

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


class MotorsBus(abc.ABC):
    """The main LeRobot class for implementing motors buses.

    There are currently two implementations of this abstract class:
        - DynamixelMotorsBus
        - FeetechMotorsBus

    Note: This class may evolve in the future should we add support for other manufacturers SDKs.

    A MotorsBus allows to efficiently read and write to the attached motors.
    It represents a several motors daisy-chained together and connected through a serial port.

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

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
    ):
        self.port = port
        self.motors = motors
        self.port_handler: PortHandler | None = None
        self.packet_handler: PacketHandler | None = None

        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

        self.calibration = None
        self.is_connected: bool = False

    def __len__(self):
        return len(self.motors)

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors)

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__name__}({self.port}) is already connected. Do not call `{self.__name__}.connect()` twice."
            )

        self._set_handlers()

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise

        self._set_timeout()

        # Allow to read and write
        self.is_connected = True

    @abc.abstractmethod
    def _set_handlers(self):
        pass

    @abc.abstractmethod
    def _set_timeout(self, timeout: int):
        pass

    def are_motors_configured(self):
        """
        Only check the motor indices and not baudrate, since if the motor baudrates are incorrect, a
        ConnectionError will be raised anyway.
        """
        try:
            return (self.motor_indices == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

    def find_motor_indices(self, possible_ids: list[str] = None, num_retry: int = 2):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        indices = []
        for idx in tqdm.tqdm(possible_ids):
            try:
                present_idx = self.read_with_motor_ids(self.motor_models, [idx], "ID", num_retry=num_retry)[0]
            except ConnectionError:
                continue

            if idx != present_idx:
                # sanity check
                raise OSError(
                    "Motor index used to communicate through the bus is not the same as the one present in the motor "
                    "memory. The motor memory might be damaged."
                )
            indices.append(idx)

        return indices

    def set_baudrate(self, baudrate):
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise OSError("Failed to write bus baud rate.")

    def set_calibration(self, calibration_dict: dict[str, list]):
        self.calibration = calibration_dict

    @abc.abstractmethod
    def apply_calibration(self):
        pass

    @abc.abstractmethod
    def revert_calibration(self):
        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__name__}({self.port}) is not connected. You need to run `{self.__name__}.connect()`."
            )

        start_time = time.perf_counter()
        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        values = self._read(data_name, motor_names)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    @abc.abstractmethod
    def _read(self, data_name: str, motor_names: list[str]):
        pass

    def write(
        self, data_name: str, values: int | float | np.ndarray, motor_names: str | list[str] | None = None
    ) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__name__}({self.port}) is not connected. You need to run `{self.__name__}.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        self._write(data_name, values, motor_names)

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    @abc.abstractmethod
    def _write(self, data_name: str, values: list[int], motor_names: list[str]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__name__}({self.port}) is not connected. Try running `{self.__name__}.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler.closePort()
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if self.is_connected:
            self.disconnect()
