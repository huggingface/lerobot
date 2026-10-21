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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pprint import pformat
from typing import TYPE_CHECKING

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _rustypot_available, require_package
from lerobot.utils.utils import enter_pressed, move_cursor_up

if TYPE_CHECKING or _rustypot_available:
    import rustypot
else:
    rustypot = None

# What rustypot raises when the bus does not answer. A motor that answers
# but reports a fault is a different thing, carried by the status error byte.
_TRANSPORT_ERRORS = (RuntimeError, OSError)

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


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


class DriveMode(Enum):
    """Value of a Dynamixel `Drive_Mode` register, and of `MotorCalibration.drive_mode`."""

    NON_INVERTED = 0
    INVERTED = 1


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


class SerialMotorsBus(MotorsBusBase):
    """Read and write a chain of servos daisy-chained on one serial port, Feetech or Dynamixel.

    rustypot handles each model's metadata (control table, resolution, baud rates, operating
    modes) and the serial I/O; LeRobot handles normalisation, calibration and the recommended
    settings. Motors of several models, such as STS and SCS or XL430 and XL330, share the port.

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

    default_baudrate: int = 1_000_000
    # Milliseconds a status packet is waited for, per attempt: longer than the 16 ms latency
    # timer of a USB adapter. Each retry waits again.
    default_timeout: int = 50
    # Register names are matched without case, as rustypot does.
    normalized_data: frozenset[str] = frozenset({"goal_position", "present_position"})

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        super().__init__(port, motors, calibration)

        models = {m.model for m in self.motors.values()}
        self._models = {model: self._find_model(model) for model in models}
        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}

        self._definitions: dict[int, rustypot.ServoDefinition] = {
            m.id: self._models[m.model][1] for m in self.motors.values()
        }
        self._bus: rustypot.Bus | None = None  # open while connected

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
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    @staticmethod
    def _find_model(model: str) -> tuple[int, rustypot.ServoDefinition]:
        """The model number of `model` and the rustypot definition that covers it."""
        require_package("rustypot", extra="serial-motors")
        if (found := rustypot.find_model(model)) is None:
            raise ValueError(f"Unknown motor model '{model}': rustypot has no servo of that name.")
        return found

    def _open_bus(self) -> rustypot.Bus:
        """Open rustypot's `Bus` on the port at the default baud rate, with each motor's definition."""
        # LeRobot counts timeouts in milliseconds, rustypot in seconds.
        return rustypot.Bus(self.port, self.default_baudrate, self.default_timeout / 1000, self._definitions)

    @property
    def _connected_bus(self) -> rustypot.Bus:
        """The open rustypot `Bus`."""
        if self._bus is None:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        return self._bus

    def _definition(self, motor: NameOrID) -> rustypot.ServoDefinition:
        return self._models[self._get_motor_model(motor)][1]

    def _has_register(self, motor: NameOrID, data_name: str) -> bool:
        return self._definition(motor).register(data_name.lower()) is not None

    def resolution(self, model: str) -> int:
        """Encoder steps per turn of a motor model.

        Args:
            model (`str`):
                Model of one of the bus' motors, e.g. `"sts3215"`.

        Returns:
            `int`: The steps per turn, e.g. 4096 on a 12-bit encoder.

        Raises:
            ValueError: If the model does not count encoder steps.
        """
        if (resolution := self._models[model][1].resolution) is None:
            raise ValueError(f"Motor model '{model}' does not count encoder steps.")
        return resolution

    def _model_number(self, model: str) -> int:
        """The model number a motor of `model` answers with."""
        return self._models[model][0]

    def _inverts_in_software(self, motor: str) -> bool:
        """Whether the calibration's drive mode is applied when normalising. A Dynamixel
        inverts itself through its `Drive_Mode` register; a Feetech has none."""
        return not self._has_register(motor, "Drive_Mode")

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

    def _assert_motors_exist(self) -> None:
        expected_models = {m.id: self._model_number(m.model) for m in self.motors.values()}

        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found)
            for id_, found in found_models.items()
            if found != expected_models[id_]
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

    def _assert_same_firmware(self) -> None:
        """Feetech servos of one definition must run the same firmware: they report it in two
        registers that Dynamixel servos do not have. STS and SCS servos have firmware of their
        own, so each definition is compared on its own."""
        majors = self._sync_read_where("Firmware_Major_Version")
        minors = self._sync_read_where("Firmware_Minor_Version")
        versions: dict[str, dict[str, str]] = {}
        for motor in majors:
            versions.setdefault(self._definition(motor).name, {})[motor] = f"{majors[motor]}.{minors[motor]}"

        if mixed := {name: found for name, found in versions.items() if len(set(found.values())) > 1}:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(mixed)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self._bus is not None

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and check the motors.

        Args:
            handshake (`bool`, *optional*, defaults to `True`):
                Whether to check that every motor answers with the model number of its model, and
                that the Feetech motors of one definition run one firmware.

        Raises:
            DeviceAlreadyConnectedError: If the port is already open.
            ConnectionError: If the port could not be opened.
            RuntimeError: If the handshake fails; the port is closed again.
        """

        self._connect(handshake)
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            self._bus = self._open_bus()
        except OSError as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e
        if not handshake:
            return
        try:
            self._handshake()
        except Exception:
            # Never leave the port open behind a failed handshake: the next
            # connect() would find it already taken.
            self._close()
            raise

    def _close(self) -> None:
        self._connected_bus.close()
        self._bus = None

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port, by default with the torque of every motor off.

        Safe to call on a bus that is already disconnected, and the port is released even if
        disabling the torque fails.

        Args:
            disable_torque (`bool`, *optional*, defaults to `True`):
                Whether to disable the torque of every motor before closing the port, so that no
                motor is left resisting.
        """

        if not self.is_connected:
            return

        try:
            if disable_torque:
                self.disable_torque(num_retry=5)
        finally:
            self._close()

        logger.debug(f"{self.__class__.__name__} disconnected.")

    def _scan(self, definition: rustypot.ServoDefinition) -> dict[int, int]:
        """Every id answering at the current baud rate, with its model number: one broadcast
        ping when the servo answers one, which covers a USB adapter's latency once, else one
        read per id."""
        if definition.supports_broadcast_ping:
            return self._connected_bus.broadcast_scan(definition)
        return self._connected_bus.scan(definition)

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:
        """Give a single motor its id and the bus' default baud rate.

        The motor must be the only one on the bus. Unless both `initial_baudrate` and
        `initial_id` are given, it is found first: at `initial_baudrate` when only that is
        given, else across the baud rates of its model; an `initial_id` given alone is replaced
        by the id found. rustypot then turns its torque off, opens its EEPROM, and writes the id
        and the baud rate.

        Args:
            motor (`str`):
                Name of the motor.
            initial_baudrate (`int`, *optional*):
                Baud rate the motor answers at now.
            initial_id (`int`, *optional*):
                Id the motor answers at now, used with `initial_baudrate`.

        Raises:
            RuntimeError: If the motor is not found, is not of its model, or does not answer as
                configured once set up.
            ConnectionError: If communication with the motor fails.
        """
        if not self.is_connected:
            self._connect(handshake=False)

        definition = self._definition(motor)
        target_id = self.motors[motor].id
        # The search walks the baud rates too: the port goes back to the default whatever happens.
        try:
            if initial_baudrate is None:
                initial_baudrate, initial_id = self._find_single_motor(motor)

            if initial_id is None:
                _, initial_id = self._find_single_motor(motor, initial_baudrate)

            self.set_baudrate(initial_baudrate)
            try:
                # The motor may answer at an id the bus does not have, or has for another
                # motor: rustypot reaches it there through its definition.
                self._connected_bus.change_id(definition, initial_id, target_id)
                self._connected_bus.change_baudrate(definition, target_id, self.default_baudrate)
            except _TRANSPORT_ERRORS as e:
                raise ConnectionError(
                    f"Failed to set up '{motor}' (motor ID {initial_id} at {initial_baudrate} baud). {e}"
                ) from e
        finally:
            self.set_baudrate(self.default_baudrate)

        # A motor answers the id and baud rate writes before they take effect: check it now
        # answers as configured.
        expected = self._model_number(self.motors[motor].model)
        if (found := self.ping(target_id)) != expected:
            raise RuntimeError(
                f"'{motor}' does not answer as motor ID {target_id} at {self.default_baudrate} baud with "
                f"model number {expected} after its setup (got {found})."
            )

    def setup_motors(self, motors: Sequence[str] | None = None) -> None:
        """Set up the motors one at a time, asking for each to be alone on the bus.

        Args:
            motors (`Sequence[str]`, *optional*):
                Motors in the order they are set up. `None` goes from the last motor to the first,
                from the gripper down on an arm.
        """
        for motor in reversed(self.motors) if motors is None else motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.motors[motor].id}")

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        definition = self._definition(motor)
        if initial_baudrate is not None:
            search_baudrates = [initial_baudrate]
        else:
            # The default first, so a motor already set up answers on the first try, then
            # the factory rate a new one answers at.
            search_baudrates = sorted(
                definition.baudrates,
                key=lambda rate: (rate != self.default_baudrate, rate != definition.factory_baudrate, rate),
            )
        expected_model_nb = self._model_number(model)

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self._scan(definition)
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found a motor at {baudrate} baud with motor ID {found_id}, but its model number "
                        f"{found_model} is different than the one expected: {expected_model_nb}. Make sure "
                        f"only the intended motor, '{motor}' (model '{model}'), is connected."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(
        self, return_delay_time: int = 0, maximum_acceleration: int = 254, acceleration: int = 254
    ) -> None:
        """Write LeRobot's recommended settings to every motor.

        Args:
            return_delay_time (`int`, *optional*, defaults to `0`):
                Delay before a motor answers, in units of 2 µs; the motors leave the factory at
                250 (500 µs).
            maximum_acceleration (`int`, *optional*, defaults to `254`):
                Acceleration ceiling, on the servos that have one (Feetech STS), high to speed up
                acceleration and deceleration.
            acceleration (`int`, *optional*, defaults to `254`):
                Acceleration, on the servos that have it (Feetech).
        """
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)
            if self._has_register(motor, "Maximum_Acceleration"):
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            if self._has_register(motor, "Acceleration"):
                self.write("Acceleration", motor, acceleration)

            # STS3215: clear Phase bit 4 to keep positions within [0, resolution - 1].
            if self.motors[motor].model == "sts3215":
                phase = self.read("Phase", motor, normalize=False)
                if phase & 0x10:
                    self.write("Phase", motor, phase & ~0x10)

    def set_operating_mode(self, mode: str, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Put the selected motors in an operating mode, given by name.

        The modes come from each motor's definition, so they depend on the model, not only on
        the family: the Feetech SCS0009 has none. The mode must exist for every selected motor,
        which is checked before any motor changes mode. Write it with the torque off.

        - `position`: holds a goal position within one turn and the position limits.
        - `velocity`: turns endlessly at a goal velocity, as a wheel.
        - `pwm`: drives the motor voltage directly, open loop.
        - `step` (Feetech): moves by the number of steps written as the goal position.
        - `current` (Dynamixel): holds a goal current, that is a torque, whatever the position.
        - `extended_position` (Dynamixel): holds a goal position over several turns. The
          `Min_Position_Limit` and `Max_Position_Limit` registers are ignored in this mode.
        - `current_based_position` (Dynamixel): holds a goal position over several turns with
          the current capped by the goal current, as a gripper that must not crush what it holds.

        Args:
            mode (`str`):
                Name of the operating mode, e.g. `"position"`.
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to set, by name or id. `None` selects every motor.

        Raises:
            ValueError: If a selected motor has no such mode; no motor has changed mode then.
        """
        motor_names = self._get_motors_list(motors)
        for motor in motor_names:
            modes = self._definition(motor).operating_modes
            if mode not in modes:
                raise ValueError(
                    f"Motor '{motor}' ({self.motors[motor].model}) has no '{mode}' operating mode: {sorted(modes)}."
                )
        for motor in motor_names:
            self.write("Operating_Mode", motor, self._definition(motor).operating_modes[mode])

    def _set_torque(
        self, motors: NameOrID | Sequence[NameOrID] | None, enabled: bool, num_retry: int
    ) -> None:
        names = self._get_motors_list(motors)
        ids = [self._get_motor_id(motor) for motor in names]
        failed = self._connected_bus.set_torque(ids, enabled, retries=num_retry)
        if failed:
            details = ", ".join(
                f"'{self._id_to_name(id_)}' (motor ID {id_}): {error}" for id_, error in failed.items()
            )
            raise ConnectionError(
                f"Failed to {'enable' if enabled else 'disable'} torque after {num_retry + 1} tries on {details}."
            )

    @check_if_not_connected
    def disable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        """Disable the torque of the selected motors.

        With the torque off, a motor's EEPROM can be written, and a Feetech motor's lock opens
        for it. Every selected motor is attempted; failures are reported, and unresponsive
        motors may remain torque-enabled.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to free, by name or id. `None` selects every motor.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts on each motor after a communication failure.

        Raises:
            ConnectionError: If some motors could not be reached; the message names them.
        """
        self._set_torque(motors, False, num_retry)

    @check_if_not_connected
    def enable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        """Enable the torque of the selected motors, and close the lock of Feetech motors.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to hold, by name or id. `None` selects every motor.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts on each motor after a communication failure.

        Raises:
            ConnectionError: If some motors could not be reached; the message names them.
        """
        self._set_torque(motors, True, num_retry)

    @contextmanager
    def torque_disabled(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> Iterator[None]:
        """Disable the torque of the selected motors for the duration of a `with` block.

        The torque is enabled again on exit, even when the block raises, which suits configuring
        motors.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to free, by name or id. `None` selects every motor.

        Example:
            ```python
            >>> with bus.torque_disabled():  # doctest: +SKIP
            ...     bus.set_operating_mode("position")
            ```
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    @check_if_not_connected
    def set_baudrate(self, baudrate: int) -> None:
        """Change the baud rate of the open port.

        The port opens at `default_baudrate` on every connect.

        Args:
            baudrate (`int`):
                Baud rate, in bits per second.
        """
        self._connected_bus.set_baudrate(baudrate)

    def _same_calibration(self, motor: str, actual: MotorCalibration) -> bool:
        """Whether the `actual` calibration read from the motor matches the cached one of
        `motor`. The homing offset and the drive mode only count on motors with those
        registers: a Feetech's drive mode lives in the calibration file alone."""
        cached = self.calibration[motor]
        return (
            (cached.range_min, cached.range_max) == (actual.range_min, actual.range_max)
            and (
                cached.homing_offset == actual.homing_offset or not self._has_register(motor, "Homing_Offset")
            )
            and (cached.drive_mode == actual.drive_mode or not self._has_register(motor, "Drive_Mode"))
        )

    @property
    def is_calibrated(self) -> bool:
        """Whether the cached calibration matches the one in the motors.

        Calibrations are matched by motor name: the ids they hold are not compared. The homing
        offset and the drive mode count only on the motors with those registers.
        """
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            return False
        return all(self._same_calibration(motor, cal) for motor, cal in motors_calibration.items())

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read the calibration of every motor.

        A motor without a homing offset register (Feetech SCS) reports 0, and one without a
        `Drive_Mode` register (Feetech) reports drive mode 0.

        Returns:
            `dict[str, MotorCalibration]`: The calibration of each motor, by name.
        """
        mins = self.sync_read("Min_Position_Limit", normalize=False)
        maxes = self.sync_read("Max_Position_Limit", normalize=False)
        offsets = self._sync_read_where("Homing_Offset")
        drive_modes = self._sync_read_where("Drive_Mode")

        return {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=int(drive_modes.get(motor, 0)),
                homing_offset=int(offsets.get(motor, 0)),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )
            for motor, m in self.motors.items()
        }

    def _sync_read_where(self, data_name: str) -> dict[str, Value]:
        """`data_name` of the motors that have the register."""
        motors = [motor for motor in self.motors if self._has_register(motor, data_name)]
        return self.sync_read(data_name, motors, normalize=False) if motors else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write a calibration to the motors, and by default cache it.

        The position limits are written to every motor, and the homing offset to the motors
        that have one. Neither the id nor the drive mode is written: a Feetech's drive mode is
        applied in software when normalising, and a Dynamixel's `Drive_Mode` register is set on
        its own.

        Args:
            calibration_dict (`dict[str, MotorCalibration]`):
                Calibration of each motor, by name, as [`~SerialMotorsBus.read_calibration`]
                returns it.
            cache (`bool`, *optional*, defaults to `True`):
                Whether to keep the calibration as the bus' `calibration`, which normalises the
                positions.
        """
        for motor, calibration in calibration_dict.items():
            if self._has_register(motor, "Homing_Offset"):
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def reset_calibration(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Restore the factory calibration of the selected motors.

        The homing offset is set to 0 on the motors that have one, and the position limits span
        the whole encoder. The cached `calibration` is cleared.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to reset, by name or id. `None` selects every motor.
        """
        motor_names = self._get_motors_list(motors)

        for motor in motor_names:
            max_res = self.resolution(self._get_motor_model(motor)) - 1
            if self._has_register(motor, "Homing_Offset"):
                self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> dict[str, int]:
        """Make the present position of each selected motor a half turn.

        The calibration of the motors is reset, then each gets the homing offset that brings its
        present position to half its encoder, e.g. 2047 on a 12-bit one. Every selected motor
        must have a homing offset, which is checked before any register is written.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to home, by name or id. `None` selects every motor.

        Returns:
            `dict[str, int]`: The homing offset written to each motor, by name.

        Raises:
            ValueError: If a selected motor has no homing offset, as the Feetech SCS.
        """
        motor_names = self._get_motors_list(motors)
        signs = {
            motor: sign
            for motor in motor_names
            if (sign := self._definition(motor).homing_offset_sign) is not None
        }
        if without := [motor for motor in motor_names if motor not in signs]:
            raise ValueError(f"Motors {without} have no homing offset.")

        self.reset_calibration(motor_names)
        positions = self.sync_read("Present_Position", motor_names, normalize=False)
        homing_offsets = self._get_half_turn_homings(positions, signs)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    def _get_half_turn_homings(
        self, positions: Mapping[str, Value], signs: Mapping[str, int]
    ) -> dict[str, int]:
        """The homing offsets that bring each position to a half turn. The offset adds to the
        position the motor reports on Dynamixel servos and subtracts from it on Feetech ones:
        `present = actual + sign * offset`, the sign coming from the motor's definition.

        A sign-magnitude offset, as on Feetech servos, spans [-2047, 2047] on a 12-bit encoder
        while the half turn of position 4095 needs 2048. That one offset is clamped, which leaves
        the motor one step off the exact half turn instead of failing the write."""
        half_turn_homings = {}
        for motor, position in positions.items():
            offset = signs[motor] * ((self.resolution(self.motors[motor].model) - 1) // 2 - int(position))
            if (sign_bit := self._definition(motor).register("homing_offset").sign_bit) is not None:
                max_magnitude = (1 << sign_bit) - 1
                offset = max(-max_magnitude, min(max_magnitude, offset))
            half_turn_homings[motor] = offset
        return half_turn_homings

    def record_ranges_of_motion(
        self, motors: NameOrID | Sequence[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """Record the range each selected motor is moved through by hand, until ENTER.

        The joints are moved by hand, with the torque off, while a live table shows the
        positions.

        Args:
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to record, by name or id. `None` selects every motor.
            display_values (`bool`, *optional*, defaults to `True`):
                Whether to print the live table.

        Returns:
            `tuple[dict[str, int | float], dict[str, int | float]]`: The lowest and the highest
            raw position of each motor, by name.

        Raises:
            ValueError: If a motor did not move: its lowest and highest positions are equal.
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

    def record_calibration(
        self,
        mode: str,
        full_turn_motors: Sequence[str],
        inverted_motors: Sequence[str] = (),
        motors: Sequence[str] | None = None,
    ) -> dict[str, MotorCalibration]:
        """Calibrate motors moved by hand.

        The selected motors are freed, put in an operating mode and given their direction: a
        motor with a `Drive_Mode` register gets it written, inverted or not, while on the others,
        as Feetech servos, an inversion is applied in software when normalising. The user then
        holds the joints in the middle of their range of motion, which becomes a half turn of
        each motor, and moves each joint through its whole range, recorded until ENTER. A
        full-turn motor is not recorded: it spans the whole encoder of its model.

        The motors need the operating mode and a homing offset, which the Feetech SCS0009 has
        neither. Their homing offsets and position limits are rewritten, the cached
        `calibration` is cleared, and their torque is left off. The calibration returned is not
        written to the position limits nor saved: [`~SerialMotorsBus.write_calibration`] writes
        it, and the device saves it.

        Args:
            mode (`str`):
                Operating mode of the motors, e.g. `"position"`.
            full_turn_motors (`Sequence[str]`):
                Motors given the whole encoder range instead of the range they are moved through.
            inverted_motors (`Sequence[str]`, *optional*, defaults to `()`):
                Motors turning the other way.
            motors (`Sequence[str]`, *optional*):
                Motors to calibrate. `None` calibrates every motor.

        Returns:
            `dict[str, MotorCalibration]`: The calibration of each motor, in the order of the bus.

        Raises:
            ValueError: If a motor is not on the bus, or a full-turn or inverted motor is not one
                of the motors to calibrate; no motor has been touched then.
        """
        motor_names = self._get_motors_list(motors)
        if unknown := [motor for motor in motor_names if motor not in self.motors]:
            raise ValueError(f"Motors {unknown} are not on the bus.")
        if outside := [motor for motor in (*full_turn_motors, *inverted_motors) if motor not in motor_names]:
            raise ValueError(f"Motors {outside} are not among the motors to calibrate: {motor_names}.")

        self.disable_torque(motor_names)
        self.set_operating_mode(mode, motor_names)
        drive_modes = {
            motor: (DriveMode.INVERTED if motor in inverted_motors else DriveMode.NON_INVERTED).value
            for motor in motor_names
        }
        for motor, drive_mode in drive_modes.items():
            if self._has_register(motor, "Drive_Mode"):
                self.write("Drive_Mode", motor, drive_mode)

        input("Move the joints to the middle of their range of motion and press ENTER....")
        homing_offsets = self.set_half_turn_homings(motor_names)

        recorded = [motor for motor in motor_names if motor not in full_turn_motors]
        print(
            f"Move {recorded} sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.record_ranges_of_motion(recorded)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = self.resolution(self.motors[motor].model) - 1

        return {
            motor: MotorCalibration(
                id=self.motors[motor].id,
                drive_mode=drive_modes[motor],
                homing_offset=int(homing_offsets[motor]),
                range_min=int(range_mins[motor]),
                range_max=int(range_maxes[motor]),
            )
            for motor in motor_names
        }

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self._inverts_in_software(motor) and self.calibration[motor].drive_mode
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
                max_res = self.resolution(self._id_to_model(id_)) - 1
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
            drive_mode = self._inverts_in_software(motor) and self.calibration[motor].drive_mode
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
                max_res = self.resolution(self._id_to_model(id_)) - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    @check_if_not_connected
    def ping(self, motor: NameOrID, num_retry: int = 0) -> int | None:
        """The model number a motor answers with.

        Reads `Model_Number` rather than sending a ping: presence and identity then cost one
        round trip instead of two.

        Args:
            motor (`str | int`):
                Motor, by name or id, one of the bus' motors.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure.

        Returns:
            `int | None`: The model number, `None` when the motor does not answer or refuses the read.
        """
        try:
            return self._read("Model_Number", self._get_motor_id(motor), num_retry)
        except (ConnectionError, RuntimeError):
            return None

    @check_if_not_connected
    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:
        """Read a register of one motor.

        Args:
            data_name (`str`):
                Register name, e.g. `"Present_Position"`; the case does not matter.
            motor (`str`):
                Motor name.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether to scale a position to the range its calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure.

        Returns:
            `int | float`: The raw value, or the normalised one.

        Raises:
            ConnectionError: If the motor does not answer.
            RuntimeError: If the motor refuses the read. An alarm in its status, as an
                overload, does not raise.
        """

        id_ = self.motors[motor].id
        value = self._read(data_name, id_, num_retry)

        if normalize and data_name.lower() in self.normalized_data:
            return self._normalize({id_: value})[id_]

        return value

    def _read(self, data_name: str, motor_id: int, num_retry: int) -> int:
        """Read one register of a motor by id: `read` takes a name, `ping` any id."""
        try:
            value, status = self._connected_bus.read_register_with_error(
                motor_id, data_name.lower(), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to read '{data_name}' from motor ID {motor_id} after {num_retry + 1} tries. {e}"
            ) from e

        if self._definitions[motor_id].refuses(status):
            raise RuntimeError(
                f"Failed to read '{data_name}': motor ID {motor_id} refused it with error status 0x{status:02x}."
            )

        return value

    @check_if_not_connected
    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        """Write a register of one motor, and wait for its answer.

        Unlike [`~SerialMotorsBus.sync_write`], the motor acknowledges the write, so a failure is
        reported: slower, but suited to configuring motors.

        Args:
            data_name (`str`):
                Register name, e.g. `"Goal_Position"`; the case does not matter.
            motor (`str`):
                Motor name.
            value (`int | float`):
                Value to write, normalised or raw as `normalize` says.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether a position is given in the range its calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure.

        Raises:
            ConnectionError: If the motor does not answer.
            RuntimeError: If the motor refuses the write. An alarm in its status, as an
                overload, does not raise.
        """

        id_ = self.motors[motor].id
        int_value = int(value)
        if normalize and data_name.lower() in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]

        try:
            status = self._connected_bus.write_register_with_error(
                id_, data_name.lower(), int_value, retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to write {int_value} to '{data_name}' of motor ID {id_} after {num_retry + 1} tries. {e}"
            ) from e

        if self._definitions[id_].refuses(status):
            raise RuntimeError(
                f"Failed to write {int_value} to '{data_name}': motor ID {id_} refused it with error status "
                f"0x{status:02x}."
            )

    @check_if_not_connected
    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        """Read a register of several motors.

        rustypot sends one Sync Read per group of motors sharing a protocol and the register's
        address and size, and reads the motors of a servo without Sync Read, as the Feetech SCS,
        one after the other. The values are therefore not one simultaneous snapshot.

        Args:
            data_name (`str`):
                Register name, e.g. `"Present_Position"`; the case does not matter.
            motors (`str | int | Sequence[str | int]`, *optional*):
                Motors to read, by name or id. `None` reads every motor.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether to scale positions to the range their calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a communication failure. A retry sends every group
                again, those that had succeeded included; an unknown motor or register is never
                retried.

        Returns:
            `dict[str, int | float]`: The value of each motor, by name.

        Raises:
            ConnectionError: If a motor does not answer.
            RuntimeError: If a motor refuses the read. An alarm in a status, as an overload,
                does not raise.
        """

        names = self._get_motors_list(motors)
        ids = [self._get_motor_id(motor) for motor in names]

        try:
            read = self._connected_bus.sync_read_register_with_error(
                ids, data_name.lower(), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to sync read '{data_name}' from motor IDs {ids} after {num_retry + 1} tries. {e}"
            ) from e
        if errors := [
            f"motor ID {id_} refused it with error status 0x{status:02x}"
            for id_, (_, status) in zip(ids, read, strict=True)
            if self._definitions[id_].refuses(status)
        ]:
            raise RuntimeError(f"Failed to sync read '{data_name}': {', '.join(errors)}.")
        ids_values = {id_: value for id_, (value, _) in zip(ids, read, strict=True)}

        if normalize and data_name.lower() in self.normalized_data:
            return {self._id_to_name(id_): value for id_, value in self._normalize(ids_values).items()}

        return {self._id_to_name(id_): value for id_, value in ids_values.items()}

    @check_if_not_connected
    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        """Write a register of several motors, without waiting for an answer.

        Unlike [`~SerialMotorsBus.write`], no motor answers a sync write: faster, suited to
        control loops, but a lost packet goes unnoticed. rustypot sends one Sync Write per group
        of motors sharing a protocol and the register's address and size.

        Args:
            data_name (`str`):
                Register name, e.g. `"Goal_Position"`; the case does not matter.
            values (`int | float | dict[str, int | float]`):
                One value for every motor, or a value per motor name.
            normalize (`bool`, *optional*, defaults to `True`):
                Whether positions are given in the range their calibration and norm mode define.
            num_retry (`int`, *optional*, defaults to `0`):
                Additional attempts after a failure of the port. A retry sends every group again.

        Raises:
            ConnectionError: If the port fails.
        """

        raw_ids_values = self._get_ids_values_dict(values)
        int_ids_values = {id_: int(val) for id_, val in raw_ids_values.items()}
        if normalize and data_name.lower() in self.normalized_data:
            int_ids_values = self._unnormalize(raw_ids_values)

        try:
            self._connected_bus.sync_write_register(
                list(int_ids_values), data_name.lower(), list(int_ids_values.values()), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to sync write '{data_name}' to motor IDs {list(int_ids_values)} after "
                f"{num_retry + 1} tries. {e}"
            ) from e
