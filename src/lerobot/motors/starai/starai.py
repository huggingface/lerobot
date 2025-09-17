import logging
from copy import deepcopy
from enum import Enum

from fashionstar_uart_sdk.uart_pocket_handler import (
    PortHandler as starai_PortHandler,
    SyncPositionControlOptions,
)

from lerobot.errors import DeviceNotConnectedError
from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

from ..motors_bus import Motor, MotorCalibration, MotorNormMode, MotorsBus, NameOrID, Value
from .tables import (
    MODEL_NUMBER_TABLE,
)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

DEFAULT_ACC_TIME = 50
DEFAULT_DEC_TIME = 50
DEFAULT_MOTION_TIME = 100

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class StaraiMotorsBus(MotorsBus):
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
        default_motion_time: int = DEFAULT_MOTION_TIME,
    ):
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version
        self.apply_drive_mode = True
        self.port_handler = starai_PortHandler(port, 1000000)

        self.default_motion_time = default_motion_time

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.port_handler.is_open

    def set_half_turn_homings(self, motors: NameOrID | list[NameOrID] | None = None) -> dict[NameOrID, Value]:
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        list_of_homing_offsets = [0 for motor in motors]
        homing_offsets = dict(zip(motors, list_of_homing_offsets, strict=False))
        return homing_offsets

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        return

    def _handshake(self) -> None:
        self._assert_motors_exist()

    def connect(self, handshake: bool = True) -> None:
        self.port_handler.openPort()
        for motor in self.motors:
            if not self.port_handler.ping(self.motors[motor].id):
                raise Exception(f"motor not found id:{self.motors[motor].id}")
        self.disable_torque()
        self.port_handler.ResetLoop(0xFF)

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        raise NotImplementedError("this function should never be called")

    def configure_motors(self, return_delay_time=0, maximum_acceleration=254, acceleration=254) -> None:
        raise NotImplementedError("this function should never be called")

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]

        read_data = {}
        if data_name == "Monitor" or data_name == "Present_Position":
            servos_id = dict(zip(names, ids, strict=False))
            monitor_data = self.port_handler.sync_read["Monitor"](servos_id)
            for name in names:
                if monitor_data[name].current_position >= 180:
                    monitor_data[name].current_position = 180
                elif monitor_data[name].current_position <= -180:
                    monitor_data[name].current_position = -180
                monitor_data[name].current_position = (
                    int(monitor_data[name].current_position + 180) / 360.0 * 4096
                )
            for name in names:
                read_data[name] = int(monitor_data[name].current_position)

            if normalize:
                if not self.calibration:
                    raise RuntimeError(f"{self} has no calibration registered.")

                normalized_values = {}
                for name, val in read_data.items():
                    motor = name
                    min_ = self.calibration[motor].range_min
                    max_ = self.calibration[motor].range_max
                    drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
                    if max_ == min_:
                        raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

                    bounded_val = min(max_, max(min_, val))
                    if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                        norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                        normalized_values[name] = -norm if drive_mode else norm
                    elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                        norm = ((bounded_val - min_) / (max_ - min_)) * 100
                        normalized_values[name] = 100 - norm if drive_mode else norm
                    elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

                read_data = normalized_values

        return read_data

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
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

        write_data = {}
        values_ = {}
        if data_name == "Goal_Position":
            if normalize and data_name in self.normalized_data:
                for motor, _ in values.items():
                    min_ = self.calibration[motor].range_min
                    max_ = self.calibration[motor].range_max
                    drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
                    if max_ == min_:
                        raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

                    if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                        values_[motor] = -values[motor] if drive_mode else values[motor]
                        bounded_val = min(100.0, max(-100.0, values_[motor]))
                        values_[motor] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
                    elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                        values_[motor] = 100 - values[motor] if drive_mode else values[motor]
                        bounded_val = min(100.0, max(0.0, values_[motor]))
                        values_[motor] = int((bounded_val / 100) * (max_ - min_) + min_)
                    elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

            for motor in values_:
                data = SyncPositionControlOptions(
                    self.motors[motor].id,
                    int(((values_[motor] / 4096 * 360) - 180) * 10),
                    self.default_motion_time,
                    0,
                    DEFAULT_ACC_TIME,
                    DEFAULT_DEC_TIME,
                )
                write_data[motor] = data

            if self.motors["gripper"].model == "rx8-u50":
                write_data["gripper"].power = 100
            else:
                write_data["gripper"].power = 1000

            self.port_handler.sync_write["Goal_Position"](write_data)

    @property
    def is_calibrated(self) -> bool:
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            return False

        same_ranges = all(
            self.calibration[motor].range_min == cal.range_min
            and self.calibration[motor].range_max == cal.range_max
            for motor, cal in motors_calibration.items()
        )
        if self.protocol_version == 1:
            return same_ranges

        same_offsets = all(
            self.calibration[motor].homing_offset == cal.homing_offset
            for motor, cal in motors_calibration.items()
        )
        return same_ranges and same_offsets

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets, mins, maxes = {}, {}, {}
        for motor in self.motors:
            mins[motor] = -1024 * 360.0
            maxes[motor] = 1024 * 360.0
            offsets[motor] = 0

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=offsets[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        return

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        return

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.port_handler.write["Stop_On_Control_Mode"](self.motors[motor].id, "unlocked", 0)

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        return

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.port_handler.write["Stop_On_Control_Mode"](motor, "locked", 0)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                sign_bit = encoding_table[data_name]
                ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                sign_bit = encoding_table[data_name]
                ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        tmp = []
        return tmp

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        return None

    def _read_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        model_numbers = {}

        return model_numbers
