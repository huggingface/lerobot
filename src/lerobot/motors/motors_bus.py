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
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
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

    Each motor's model (`sts3215`, `xl330-m288`, ...) is looked up in rustypot, which gives
    its model number, its control table and the facts around it: resolution, baud rates,
    operating modes, whether its homing offset adds to or subtracts from the position, which
    registers exist. The bus carries what is LeRobot's: normalisation, calibration and the
    recommended settings. The wire, and the byte order and sign encoding of what goes over
    it, belong to a rustypot `Bus` opened on connect with each motor's definition, so motors
    of several models (STS and SCS, or XL430 and XL330) share the port.

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
    # Milliseconds per status packet: enough for a USB adapter's 16 ms latency timer, short
    # enough that a lost reply costs a 30 Hz control loop one tick instead of stalling it.
    default_timeout: int = 50
    normalized_data: list[str] = ["Goal_Position", "Present_Position"]

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        super().__init__(port, motors, calibration)

        self._models = {m.model: self._find_model(m.model) for m in self.motors.values()}
        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}

        self._definitions: dict[int, rustypot.ServoDefinition] = {
            m.id: self._models[m.model][1] for m in self.motors.values()
        }
        self._bus: Any = None  # the open rustypot.Bus, None while disconnected

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

    @staticmethod
    def _bus_class() -> type[rustypot.Bus]:
        """rustypot's `Bus`: a serial port, a protocol handler and each motor's definition."""
        return rustypot.Bus

    def _definition(self, motor: NameOrID) -> rustypot.ServoDefinition:
        return self._models[self._get_motor_model(motor)][1]

    def _has_register(self, motor: NameOrID, data_name: str) -> bool:
        return self._definition(motor).register(data_name.lower()) is not None

    def resolution(self, model: str) -> int:
        """Encoder steps per turn of `model`."""
        return self._models[model][1].resolution

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
        """Feetech servos of one bus must run the same firmware: they report it in two
        registers that Dynamixel servos do not have."""
        firmware_versions = {}
        for motor in self.motors:
            if self._has_register(motor, "Firmware_Major_Version"):
                major = self.read("Firmware_Major_Version", motor, normalize=False)
                minor = self.read("Firmware_Minor_Version", motor, normalize=False)
                firmware_versions[motor] = f"{major}.{minor}"

        if len(set(firmware_versions.values())) > 1:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(firmware_versions)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self._bus is not None

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and initialise communication.

        Args:
            handshake (bool, optional): Checks that every expected motor answers with the
                model number of its model, and that Feetech motors share one firmware.
                Defaults to `True`.

        Raises:
            DeviceAlreadyConnectedError: The port is already open.
            ConnectionError: The port could not be opened, or the handshake did not succeed.
        """

        self._connect(handshake)
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            self._bus = self._bus_class()(
                self.port,
                self.default_baudrate,
                # LeRobot counts timeouts in milliseconds, rustypot in seconds.
                self.default_timeout / 1000,
                self._definitions,
            )
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
        self._bus.close()
        self._bus = None

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port (optionally disabling torque first).

        Safe to call on a bus that is already disconnected, and the port is released
        even if disabling torque fails.

        Args:
            disable_torque (bool, optional): If `True` (default) torque is disabled on every motor before
                closing the port. This can prevent damaging motors if they are left applying resisting torque
                after disconnect.
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
            return self._bus.broadcast_scan(definition)
        return self._bus.scan(definition)

    @classmethod
    def scan_port(cls, port: str, model: str) -> dict[int, list[int]]:
        """Probe *port* at every baud rate a *model* motor can be set to and list responding IDs.

        Args:
            port (str): Serial/USB port to scan (e.g. ``"/dev/ttyUSB0"``).
            model (str): Model of the motors looked for (e.g. ``"sts3215"``); its protocol and
                baud rates set what is tried.

        Returns:
            dict[int, list[int]]: Mapping *baud-rate → list of motor IDs*
            for every baud-rate that produced at least one response.
        """
        _, definition = cls._find_model(model)
        bus = cls(port, {})
        bus._connect(handshake=False)
        # Another model may order the bytes of its model number the other way, so only the
        # IDs are reported.
        baudrate_ids = {}
        try:
            for baudrate in tqdm(definition.baudrates, desc="Scanning port"):
                bus.set_baudrate(baudrate)
                ids = list(bus._scan(definition))
                if ids:
                    tqdm.write(f"Motors found for {baudrate=}: {ids}")
                    baudrate_ids[baudrate] = ids
        finally:
            bus.disconnect(disable_torque=False)
        return baudrate_ids

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:
        """Assign the correct ID and baud-rate to a single motor.

        This helper finds the motor at its current settings, then has rustypot turn its torque
        off, open its EEPROM, write the desired ID and program the bus' default baud-rate.

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

        definition = self._definition(motor)
        target_id = self.motors[motor].id
        self.set_baudrate(initial_baudrate)
        try:
            # The motor may answer at an id the bus does not have, or has for another
            # motor: rustypot reaches it there through its definition.
            self._bus.change_id(definition, initial_id, target_id)
            self._bus.change_baudrate(definition, target_id, self.default_baudrate)
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(
                f"Failed to set up '{motor}' (id {initial_id} at {initial_baudrate}). {e}"
            ) from e
        finally:
            self.set_baudrate(self.default_baudrate)

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
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: {expected_model_nb}. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(
        self, return_delay_time: int = 0, maximum_acceleration: int = 254, acceleration: int = 254
    ) -> None:
        """Write LeRobot's recommended settings to every motor.

        Args:
            return_delay_time (int, optional): Delay before a motor answers, in units of 2 µs.
                Defaults to `0`: the motors leave the factory at 250 (500 µs).
            maximum_acceleration (int, optional): Acceleration ceiling, on the servos that have one
                (Feetech STS). Defaults to `254`, to speed up acceleration and deceleration.
            acceleration (int, optional): Acceleration, on the servos that have it (Feetech).
                Defaults to `254`.
        """
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)
            if self._has_register(motor, "Maximum_Acceleration"):
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            if self._has_register(motor, "Acceleration"):
                self.write("Acceleration", motor, acceleration)

            # Clear bit 4 (0x10) of the Phase register (0x12) to set angle feedback mode to 0.
            # This forces position readings to be in the range [0, resolution - 1] and prevents overflow or negative values.
            # Only known to be necessary for the STS3215.
            if self.motors[motor].model == "sts3215":
                phase = self.read("Phase", motor, normalize=False)
                if phase & 0x10:
                    self.write("Phase", motor, phase & ~0x10)

    def set_operating_mode(self, mode: str, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Put the selected motors in operating mode *mode*, given by name.

        `position`, `velocity` and `pwm` exist on every family, at a different register value
        on each; Dynamixel servos add `current`, `extended_position` and
        `current_based_position`, Feetech servos `step`. Write it with the torque off.

        Args:
            mode (str): Name of the operating mode.
            motors (NameOrID | Sequence[NameOrID] | None, optional): Target motors. `None` (default)
                selects every motor.

        Raises:
            ValueError: A selected motor has no such mode.
        """
        for motor in self._get_motors_list(motors):
            modes = self._definition(motor).operating_modes
            if mode not in modes:
                raise ValueError(
                    f"Motor '{motor}' ({self.motors[motor].model}) has no '{mode}' operating mode: {sorted(modes)}."
                )
            self.write("Operating_Mode", motor, modes[mode])

    def _set_torque(
        self, motors: NameOrID | Sequence[NameOrID] | None, enabled: bool, num_retry: int
    ) -> None:
        names = self._get_motors_list(motors)
        failed = self._bus.set_torque([self.motors[motor].id for motor in names], enabled, retries=num_retry)
        if failed:
            details = ", ".join(
                f"'{self._id_to_name(id_)}' (id {id_}): {error}" for id_, error in failed.items()
            )
            raise ConnectionError(
                f"Failed to {'enable' if enabled else 'disable'} torque after {num_retry + 1} tries on {details}."
            )

    @check_if_not_connected
    def disable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors.

        Disabling Torque allows to write to the motors' permanent memory area (EPROM/EEPROM),
        and opens the lock of Feetech motors for it. Every motor is tried even when one fails,
        so a motor that does not answer leaves no other one under torque.

        Args:
            motors (NameOrID | Sequence[NameOrID] | None, optional): Target motors. Accepts a motor name, an
                ID, a list of names or `None` to affect every registered motor. Defaults to `None`.
            num_retry (int, optional): Number of additional retry attempts on communication failure.
                Defaults to 0.

        Raises:
            ConnectionError: Some motors could not be reached; the message lists them.
        """
        self._set_torque(motors, False, num_retry)

    @check_if_not_connected
    def enable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors, and close the lock of Feetech motors.

        Args:
            motors (NameOrID | Sequence[NameOrID] | None, optional): Same semantics as
                :pymeth:`disable_torque`. Defaults to `None`.
            num_retry (int, optional): Number of additional retry attempts on communication failure.
                Defaults to 0.

        Raises:
            ConnectionError: Some motors could not be reached; the message lists them.
        """
        self._set_torque(motors, True, num_retry)

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
        """Set a new UART baud-rate on the open port.

        The port opens at :pyattr:`default_baudrate` on every connect.

        Args:
            baudrate (int): Desired baud-rate in bits / second.
        """
        self._bus.set_baudrate(baudrate)

    def _same_calibration(self, motor: str, read: MotorCalibration) -> bool:
        """Whether *read* from the motor matches the cached calibration of *motor*. The
        homing offset and the drive mode only count on motors with those registers: a
        Feetech's drive mode lives in the calibration file alone."""
        cached = self.calibration[motor]
        return (
            (cached.range_min, cached.range_max) == (read.range_min, read.range_max)
            and (cached.homing_offset == read.homing_offset or not self._has_register(motor, "Homing_Offset"))
            and (cached.drive_mode == read.drive_mode or not self._has_register(motor, "Drive_Mode"))
        )

    @property
    def is_calibrated(self) -> bool:
        """bool: ``True`` if the cached calibration matches the motors."""
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            return False
        return all(self._same_calibration(motor, cal) for motor, cal in motors_calibration.items())

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration parameters from the motors.

        A motor without a homing offset register (Feetech SCS) reports 0, and one without a
        `Drive_Mode` register (Feetech) reports drive mode 0.

        Returns:
            dict[str, MotorCalibration]: Mapping *motor name → calibration*.
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
        """Write calibration parameters to the motors and optionally cache them.

        Args:
            calibration_dict (dict[str, MotorCalibration]): Calibration obtained from
                :pymeth:`read_calibration` or crafted by the user.
            cache (bool, optional): Save the calibration to :pyattr:`calibration`. Defaults to True.
        """
        for motor, calibration in calibration_dict.items():
            if self._has_register(motor, "Homing_Offset"):
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def reset_calibration(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        """Restore factory calibration for the selected motors.

        Homing offset is set to ``0`` (on motors that have one) and min/max position limits are set to the
        full usable range. The in-memory :pyattr:`calibration` is cleared.

        Args:
            motors (NameOrID | Sequence[NameOrID] | None, optional): Selection of motors. `None` (default)
                resets every motor.
        """
        motor_names = self._get_motors_list(motors)

        for motor in motor_names:
            max_res = self.resolution(self._get_motor_model(motor)) - 1
            if self._has_register(motor, "Homing_Offset"):
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

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """The homing offsets that bring each position to a half turn. The offset adds to the
        position the motor reports on Dynamixel servos and subtracts from it on Feetech ones:
        `present = actual + sign * offset`, the sign coming from the motor's definition."""
        half_turn_homings: dict[NameOrID, Value] = {}
        for motor, pos in positions.items():
            sign = self._definition(motor).homing_offset_sign
            if sign is None:
                raise ValueError(f"Motor '{motor}' has no homing offset.")
            max_res = self.resolution(self._get_motor_model(motor)) - 1
            half_turn_homings[motor] = sign * (int(max_res / 2) - pos)

        return half_turn_homings

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

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        """Ping a single motor of the bus and return its model number.

        Reads Model_Number rather than sending a ping: presence and identity then
        cost one round trip instead of two. :pymeth:`scan_port` finds motors at
        ids the bus does not have.

        Args:
            motor (NameOrID): Target motor (name or ID), one of :pyattr:`motors`.
            num_retry (int, optional): Extra attempts before giving up. Defaults to `0`.
            raise_on_error (bool, optional): If `True` communication errors raise exceptions instead of
                returning `None`. Defaults to `False`.

        Returns:
            int | None: Motor model number or `None` on failure.
        """
        id_ = self._get_motor_id(motor)
        return self._read("Model_Number", id_, num_retry=num_retry, raise_on_error=raise_on_error)

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
        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        # raise_on_error=True, so a failure raises rather than returning None.
        value = cast(
            int, self._read(data_name, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)
        )

        if normalize and data_name in self.normalized_data:
            return self._normalize({id_: value})[id_]

        return value

    def _read(
        self,
        data_name: str,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int | None:
        """Read one register, or `None` if it failed and *raise_on_error* is `False`."""
        try:
            value, status = self._bus.read_register_with_error(motor_id, data_name.lower(), retries=num_retry)
        except _TRANSPORT_ERRORS as e:
            if raise_on_error:
                raise ConnectionError(f"{err_msg} {e}") from e
            return None

        if status:
            if raise_on_error:
                raise RuntimeError(f"{err_msg} Motor {motor_id} returned error status 0x{status:02x}.")
            return None

        return value

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
        int_value = int(value)
        if normalize and data_name in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]

        self._write(data_name, id_, int_value, num_retry=num_retry)

    def _write(self, data_name: str, motor_id: int, value: int, *, num_retry: int = 0) -> None:
        err_msg = (
            f"Failed to write '{data_name}' on id_={motor_id} with '{value}' after {num_retry + 1} tries."
        )
        try:
            status = self._bus.write_register_with_error(
                motor_id, data_name.lower(), value, retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(f"{err_msg} {e}") from e

        if status:
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

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]

        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        ids_values = self._sync_read(data_name, ids, num_retry=num_retry, err_msg=err_msg)

        if normalize and data_name in self.normalized_data:
            return {self._id_to_name(id_): value for id_, value in self._normalize(ids_values).items()}

        return {self._id_to_name(id_): value for id_, value in ids_values.items()}

    def _sync_read(
        self,
        data_name: str,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        err_msg: str = "",
    ) -> dict[int, int]:
        try:
            values = self._bus.sync_read_register(motor_ids, data_name.lower(), retries=num_retry)
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(f"{err_msg} {e}") from e

        return dict(zip(motor_ids, values, strict=True))

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
        int_ids_values = {id_: int(val) for id_, val in raw_ids_values.items()}
        if normalize and data_name in self.normalized_data:
            int_ids_values = self._unnormalize(raw_ids_values)

        err_msg = f"Failed to sync write '{data_name}' with ids_values={int_ids_values} after {num_retry + 1} tries."
        self._sync_write(data_name, int_ids_values, num_retry=num_retry, err_msg=err_msg)

    def _sync_write(
        self,
        data_name: str,
        ids_values: dict[int, int],
        num_retry: int = 0,
        err_msg: str = "",
    ) -> None:
        try:
            self._bus.sync_write_register(
                list(ids_values), data_name.lower(), list(ids_values.values()), retries=num_retry
            )
        except _TRANSPORT_ERRORS as e:
            raise ConnectionError(f"{err_msg} {e}") from e
