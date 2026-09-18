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

from __future__ import annotations

import abc
import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pprint import pformat
from typing import TYPE_CHECKING, cast

from tqdm import tqdm

from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None  # type: ignore[assignment, misc]

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from .transport import MotorTransport, RustypotTransport

# What the transport raises when the bus does not answer. A motor that answers but
# reports a fault is a different thing, carried by the status error byte.
_TRANSPORT_ERRORS = (RuntimeError, OSError)

# Round-trip budget of a Model_Number read, in bits on the wire, and the floor under
# the timeout derived from it. Used to size an ID sweep -- see `_scan_timeout`.
SCAN_PACKET_BITS = 320
SCAN_TIMEOUT_MS = 5

type NameOrID = str | int
type Value = int | float

logger = logging.getLogger(__name__)


class MotorsBusBase(abc.ABC):
    """
    Base class for all motor bus implementations.

    This is a minimal interface that all motor buses must implement, regardless of their
    communication protocol (serial, CAN, etc.).
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

    @abc.abstractmethod
    def connect(self, handshake: bool = True) -> None:
        """Establish connection to the motors."""
        pass

    @abc.abstractmethod
    def disconnect(self, disable_torque: bool = True) -> None:
        """Disconnect from the motors."""
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the motors."""
        pass

    @abc.abstractmethod
    def read(self, data_name: str, motor: str) -> Value:
        """Read a value from a single motor."""
        pass

    @abc.abstractmethod
    def write(self, data_name: str, motor: str, value: Value) -> None:
        """Write a value to a single motor."""
        pass

    @abc.abstractmethod
    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:
        """Read a value from multiple motors."""
        pass

    @abc.abstractmethod
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        """Write values to multiple motors."""
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        pass

    @abc.abstractmethod
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration parameters from the motors."""
        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration parameters to the motors."""
        pass


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
    motor_type_str: str | None = None
    recv_id: int | None = None


def _resolve_motor_family(motors: dict[str, Motor] | None) -> type[SerialMotorsBus]:
    """Pick the bus implementation that knows every requested motor model."""
    # Imported here because each family module imports this one.
    from .dynamixel import DynamixelMotorsBus
    from .feetech import FeetechMotorsBus

    families: tuple[type[SerialMotorsBus], ...] = (FeetechMotorsBus, DynamixelMotorsBus)
    models = {motor.model for motor in motors.values()} if motors else set()
    if not models:
        raise ValueError(
            "Cannot tell which motor family to use without any motor. Name the implementation "
            "directly, e.g. FeetechMotorsBus(port, {}), when the bus starts out empty."
        )

    for family in families:
        if models <= set(family.model_ctrl_table):
            return family

    known = set().union(*(set(family.model_ctrl_table) for family in families))
    if unknown := sorted(models - known):
        raise ValueError(f"Unknown motor model(s) {unknown}. Known models: {sorted(known)}.")

    raise ValueError(f"Motor models {sorted(models)} belong to different families and cannot share one bus.")


class SerialMotorsBus(MotorsBusBase):
    """Read and write a chain of motors daisy-chained on one serial port.

    Constructing a `SerialMotorsBus` hands back the bus for the motor family it was
    given -- a `FeetechMotorsBus` or a `DynamixelMotorsBus` -- the way `pathlib.Path`
    hands back a `PosixPath`. Naming a family class directly also works, and skips the
    lookup. A bus carries the control tables, normalisation and calibration; the wire
    itself belongs to a `MotorTransport`.

    To find the port, run:
    ```bash
    lerobot-find-port
    >>> Finding all available ports for the MotorsBus.
    >>> ["/dev/tty.usbmodem575E0032081", "/dev/tty.usbmodem575E0031751"]
    >>> Remove the usb cable from your MotorsBus and press Enter when done.
    >>> The port of this MotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example for a single Feetech sts3215 on the bus:
    ```python
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    bus = SerialMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={"my_motor": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    bus.connect()

    position = bus.read("Present_Position", "my_motor", normalize=False)

    # Move a few motor steps as an example
    bus.write("Goal_Position", "my_motor", position + 30, normalize=False)

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
    model_number_address: tuple[int, int]
    max_id: int
    normalized_data: list[str]
    protocol: str

    def __new__(cls, *args, **kwargs) -> SerialMotorsBus:
        if cls is not SerialMotorsBus:
            return super().__new__(cls)

        motors = kwargs["motors"] if "motors" in kwargs else (args[1] if len(args) > 1 else None)
        return super().__new__(_resolve_motor_family(motors))

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        require_package("deepdiff", extra="deepdiff-dep")
        super().__init__(port, motors, calibration)

        self._io: MotorTransport = RustypotTransport(
            port, self.protocol, self.default_baudrate, self.default_timeout
        )

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

    def _get_motor_model(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motors_list(self, motors: NameOrID | Sequence[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, int):
            return [self._id_to_name(motors)]
        elif isinstance(motors, Sequence):
            return [m if isinstance(m, str) else self._id_to_name(m) for m in motors]
        else:
            raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> dict[int, Value]:
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
        return self._io.is_open

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and initialise communication.

        Args:
            handshake (bool, optional): Pings every expected motor and performs additional
                integrity checks specific to the implementation. Defaults to `True`.

        Raises:
            DeviceAlreadyConnectedError: The port is already open.
            ConnectionError: The port could not be opened, or the handshake did not succeed.
        """

        self._connect(handshake)
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        self._io.open()
        if not handshake:
            return
        try:
            self._handshake()
        except Exception:
            # Never leave the port open behind a failed handshake: the next
            # connect() would find it already taken.
            self._io.close()
            raise

    @abc.abstractmethod
    def _handshake(self) -> None:
        pass

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port (optionally disabling torque first).

        Safe to call on a bus that is already disconnected, and the port is released
        even if disabling torque fails.

        Args:
            disable_torque (bool, optional): If `True` (default) torque is disabled on every motor before
                closing the port. This can prevent damaging motors if they are left applying resisting torque
                after disconnect.
        """

        if not self._io.is_open:
            return

        try:
            if disable_torque:
                self.disable_torque(num_retry=5)
        finally:
            self._io.close()

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

        bus.disconnect(disable_torque=False)
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
    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def configure_motors(self) -> None:
        """Write implementation-specific recommended settings to every motor.

        Typical changes include shortening the return delay, increasing
        acceleration limits or disabling safety locks.
        """
        pass

    @abc.abstractmethod
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors.

        Disabling Torque allows to write to the motors' permanent memory area (EPROM/EEPROM).

        Args:
            motors ( str | list[str] | None, optional): Target motors.  Accepts a motor name, an ID, a
                list of names or `None` to affect every registered motor.  Defaults to `None`.
            num_retry (int, optional): Number of additional retry attempts on communication failure.
                Defaults to 0.
        """
        pass

    @abc.abstractmethod
    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors.

        Args:
            motors (int | str | list[str] | None, optional): Same semantics as :pymeth:`disable_torque`.
                Defaults to `None`.
            num_retry (int, optional): Number of additional retry attempts on communication failure.
                Defaults to 0.
        """
        pass

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
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

    def set_baudrate(self, baudrate: int) -> None:
        """Set a new UART baud-rate on the port.

        Args:
            baudrate (int): Desired baud-rate in bits / second.
        """
        if baudrate != self._io.baudrate:
            logger.info(f"Setting bus baud rate to {baudrate}. Previously {self._io.baudrate}.")
        self._io.set_baudrate(baudrate)

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

    def reset_calibration(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Restore factory calibration for the selected motors.

        Homing offset is set to ``0`` and min/max position limits are set to the full usable range.
        The in-memory :pyattr:`calibration` is cleared.

        Args:
            motors (NameOrID | Sequence[NameOrID] | None, optional): Selection of motors. `None` (default)
                resets every motor.
        """
        motor_names = self._get_motors_list(motors)

        for motor in motor_names:
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(
        self, motors: NameOrID | Sequence[NameOrID] | None = None
    ) -> dict[NameOrID, Value]:
        """Centre each motor range around its current position.

        The function computes and writes a homing offset such that the present position becomes exactly one
        half-turn (e.g. `2047` on a 12-bit encoder).

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): Motors to adjust. Defaults to all motors (`None`).

        Returns:
            dict[str, Value]: Mapping *motor name → written homing offset*.
        """
        motor_names = self._get_motors_list(motors)

        self.reset_calibration(motor_names)
        actual_positions = self.sync_read("Present_Position", motor_names, normalize=False)
        homing_offsets = self._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    @abc.abstractmethod
    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        pass

    def record_ranges_of_motion(
        self, motors: NameOrID | Sequence[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """Interactively record the min/max encoder values of each motor.

        Move the joints by hand (with torque disabled) while the method streams live positions. Press
        :kbd:`Enter` to finish.

        Args:
            motors (NameOrID | list[NameOrID] | None, optional): Motors to record.
                Defaults to every motor (`None`).
            display_values (bool, optional): When `True` (default) a live table is printed to the console.

        Returns:
            tuple[dict[str, Value], dict[str, Value]]: Two dictionaries *mins* and *maxes* with the
                extreme values observed for each motor.
        """
        motor_names = self._get_motors_list(motors)

        start_positions = self.sync_read("Present_Position", motor_names, normalize=False, num_retry=5)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motor_names, normalize=False, num_retry=5)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motor_names:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if not user_pressed_enter:
                if display_values:
                    # Move cursor up to overwrite the previous output
                    move_cursor_up(len(motor_names) + 3)
                # Throttle reads even when the live table is disabled.
                time.sleep(0.02)

        same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
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

    @abc.abstractmethod
    def _join_byte_chunks(self, data: bytes, length: int) -> int:
        """Convert register bytes back into an integer, inverse of :pymeth:`_split_into_byte_chunks`."""
        pass

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        """Ping a single motor and return its model number.

        Reads Model_Number rather than sending a ping: presence and identity then
        cost one round trip instead of two.

        Args:
            motor (NameOrID): Target motor (name or ID).
            num_retry (int, optional): Extra attempts before giving up. Defaults to `0`.
            raise_on_error (bool, optional): If `True` communication errors raise exceptions instead of
                returning `None`. Defaults to `False`.

        Returns:
            int | None: Motor model number or `None` on failure.
        """
        id_ = self._get_motor_id(motor)
        addr, length = self.model_number_address
        return self._read(addr, length, id_, num_retry=num_retry, raise_on_error=raise_on_error)

    def broadcast_ping(self, num_retry: int = 0) -> dict[int, int]:
        """Ping every ID on the bus and return the ones that answer.

        The transport exposes no broadcast ping, so this sweeps the ID space one
        motor at a time. Only bus scanning and motor setup call it, never a control
        loop.

        Args:
            num_retry (int, optional): Retry attempts per ID. Defaults to `0`.

        Returns:
            dict[int, int]: Mapping *id → model number* for every motor that answered.
        """
        self._assert_protocol_is_compatible("broadcast_ping")
        with self._scan_timeout():
            found = {id_: self.ping(id_, num_retry=num_retry) for id_ in range(self.max_id + 1)}

        return {id_: model for id_, model in found.items() if model is not None}

    @contextmanager
    def _scan_timeout(self):
        """Shorten the read timeout for the duration of an ID sweep.

        Every absent ID costs one full timeout, so a sweep at `default_timeout` would
        take minutes. Size it to what the current baud rate needs to shift a request
        and a status packet, with a floor for USB scheduling.
        """
        self._io.set_timeout(max(SCAN_TIMEOUT_MS, round(SCAN_PACKET_BITS * 1000 / self._io.baudrate)))
        try:
            yield
        finally:
            self._io.set_timeout(self.default_timeout)

    @check_if_not_connected
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

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        # raise_on_error=True, so a failure raises rather than returning None.
        value = cast(
            int, self._read(addr, length, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)
        )

        decoded = self._decode_sign(data_name, {id_: value})

        if normalize and data_name in self.normalized_data:
            normalized = self._normalize(decoded)
            return normalized[id_]

        return decoded[id_]

    def _read(
        self,
        addr: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int | None:
        """Read one register, or `None` if it failed and *raise_on_error* is `False`."""
        for n_try in range(1 + num_retry):
            try:
                data, status = self._io.read(motor_id, addr, length)
                break
            except _TRANSPORT_ERRORS as e:
                failure = e
                logger.debug(f"Failed to read @{addr=} ({length=}) on {motor_id=} ({n_try=}): {e}")
        else:
            if raise_on_error:
                raise ConnectionError(f"{err_msg} {failure}") from failure
            return None

        if status:
            if raise_on_error:
                raise RuntimeError(f"{err_msg} Motor {motor_id} returned error status 0x{status:02x}.")
            return None

        return self._join_byte_chunks(data, length)

    @check_if_not_connected
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

        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        int_value = int(value)
        if normalize and data_name in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]

        int_value = self._encode_sign(data_name, {id_: int_value})[id_]

        err_msg = f"Failed to write '{data_name}' on {id_=} with '{int_value}' after {num_retry + 1} tries."
        self._write(addr, length, id_, int_value, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

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
    ) -> None:
        data = bytes(self._serialize_data(value, length))
        for n_try in range(1 + num_retry):
            try:
                status = self._io.write(motor_id, addr, data)
                break
            except _TRANSPORT_ERRORS as e:
                failure = e
                logger.debug(
                    f"Failed to write @{addr=} ({length=}) on id={motor_id} with {value=} ({n_try=}): {e}"
                )
        else:
            if raise_on_error:
                raise ConnectionError(f"{err_msg} {failure}") from failure
            return

        if status and raise_on_error:
            raise RuntimeError(f"{err_msg} Motor {motor_id} returned error status 0x{status:02x}.")

    @check_if_not_connected
    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """Read the same register from several motors at once.

        Args:
            data_name (str): Register name.
            motors (NameOrID | Sequence[NameOrID] | None, optional): Motors to query. `None` (default) reads every motor.
            normalize (bool, optional): Normalisation flag.  Defaults to `True`.
            num_retry (int, optional): Retry attempts.  Defaults to `0`.

        Returns:
            dict[str, Value]: Mapping *motor name → value*.
        """

        self._assert_protocol_is_compatible("sync_read")

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]
        models = [self.motors[motor].model for motor in names]

        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        raw_ids_values = self._sync_read(
            addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )

        decoded = self._decode_sign(data_name, raw_ids_values)

        if normalize and data_name in self.normalized_data:
            normalized = self._normalize(decoded)
            return {self._id_to_name(id_): value for id_, value in normalized.items()}

        return {self._id_to_name(id_): value for id_, value in decoded.items()}

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> dict[int, int]:
        for n_try in range(1 + num_retry):
            try:
                frames = self._io.sync_read(motor_ids, addr, length)
                break
            except _TRANSPORT_ERRORS as e:
                failure = e
                logger.debug(f"Failed to sync read @{addr=} ({length=}) on {motor_ids=} ({n_try=}): {e}")
        else:
            if raise_on_error:
                raise ConnectionError(f"{err_msg} {failure}") from failure
            return {}

        return {
            id_: self._join_byte_chunks(frame, length) for id_, frame in zip(motor_ids, frames, strict=True)
        }

    @check_if_not_connected
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

        raw_ids_values = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in raw_ids_values]
        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        int_ids_values = {id_: int(val) for id_, val in raw_ids_values.items()}
        if normalize and data_name in self.normalized_data:
            int_ids_values = self._unnormalize(raw_ids_values)

        int_ids_values = self._encode_sign(data_name, int_ids_values)

        err_msg = f"Failed to sync write '{data_name}' with ids_values={int_ids_values} after {num_retry + 1} tries."
        self._sync_write(
            addr, length, int_ids_values, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> None:
        motor_ids = list(ids_values)
        data = [bytes(self._serialize_data(value, length)) for value in ids_values.values()]
        for n_try in range(1 + num_retry):
            try:
                self._io.sync_write(motor_ids, addr, data)
                return
            except _TRANSPORT_ERRORS as e:
                failure = e
                logger.debug(f"Failed to sync write @{addr=} ({length=}) with {ids_values=} ({n_try=}): {e}")

        if raise_on_error:
            raise ConnectionError(f"{err_msg} {failure}") from failure


# Backward compatibility alias
MotorsBus = SerialMotorsBus
