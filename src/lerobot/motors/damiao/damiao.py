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
from copy import deepcopy
from enum import Enum

from lerobot.motors.encoding_utils import decode_twos_complement, encode_twos_complement

logger = logging.getLogger(__name__)

class OperatingMode(Enum):
    MIT = 0


class DamiaoMotorsBus(MotorsBus):
    """
    The Damiao implementation for a MotorsBus. It relies on the python-can library to communicate with
    the motors. For more info, see the python-can documentation: https://python-can.readthedocs.io/en/stable/, seedstudio documentation: https://wiki.seeedstudio.com/damiao_series/ and DM_Control_Python repo: https://github.com/cmjang/DM_Control_Python
    https://wiki.seeedstudio.com/damiao_series/ and DM_Control_Python repo: https://github.com/cmjang/DM_Control_Python
    """

    def __init__(
        self,
        port: str,
    ):
        self.port = port

    def configure_motors(self) -> None:
        for motor in self.motors

    
    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

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


    @property
    def is_calibrated(self) -> bool:
        return self.calibration == self.read_calibration()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets = self.sync_read("Homing_Offset", normalize=False)
        mins = self.sync_read("Min_Position_Limit", normalize=False)
        maxes = self.sync_read("Max_Position_Limit", normalize=False)
        drive_modes = self.sync_read("Drive_Mode", normalize=False)

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=offsets[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        for motor, calibration in calibration_dict.items():
            self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)

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

