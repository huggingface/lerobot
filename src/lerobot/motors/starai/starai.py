

import logging
import time
from copy import deepcopy
from enum import Enum
from pprint import pformat
from typing import Any

from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from fashionstar_uart_sdk import *

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address
from .tables import (
#     FIRMWARE_MAJOR_VERSION,
#     FIRMWARE_MINOR_VERSION,
#     MODEL_BAUDRATE_TABLE,
    # MODEL_CONTROL_TABLE,
#     MODEL_ENCODING_TABLE,
#     MODEL_NUMBER,
    MODEL_NUMBER_TABLE,
#     MODEL_PROTOCOL,
#     MODEL_RESOLUTION,
#     SCAN_BAUDRATES,
)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

DEFAULT_ACC_TIME =100
DEFAULT_DEC_TIME =100
DEFAULT_MOTION_TIME = 500

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class OperatingMode(Enum):
    # position servo mode
    POSITION = 0
    # The motor is in constant speed mode, which is controlled by parameter 0x2e, and the highest bit 15 is
    # the direction bit
    VELOCITY = 1
    # PWM open-loop speed regulation mode, with parameter 0x2c running time parameter control, bit11 as
    # direction bit
    PWM = 2
    # In step servo mode, the number of step progress is represented by parameter 0x2a, and the highest bit 15
    # is the direction bit
    STEP = 3


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    import scservo_sdk as scs

    if length == 1:
        data = [value]
    elif length == 2:
        data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
    elif length == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    return data


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """
    HACK: This patches the PortHandler behavior to set the correct packet timeouts.

    It fixes https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6
    The bug is fixed on the official Feetech SDK repo (https://gitee.com/ftservo/FTServo_Python)
    but because that version is not published on PyPI, we rely on the (unofficial) on that is, which needs
    patching.
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class StaraiMotorsBus(MotorsBus):


    # apply_drive_mode = True
    # available_baudrates = deepcopy(SCAN_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    # model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    # model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    # model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    # model_resolution_table = deepcopy(MODEL_RESOLUTION)
    # normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,

    ):
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version


        self.port_handler = PortHandler(port,1000000)

        # # HACK: monkeypatch

        
        # self.packet_handler = scs.PacketHandler(protocol_version)
        # self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        # self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        # self._comm_success = scs.COMM_SUCCESS
        # self._no_error = 0x00

        # if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
        #     raise ValueError(f"Some motors are incompatible with protocol_version={self.protocol_version}")
    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.port_handler.is_open

    # def write(self,  data_name:str, motor:str, value, *, normalize = True, num_retry = 0):
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(
    #             f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
    #         )
    #     id_ = self.motors[motor].id
    #     # model = self.motors[motor].model
    def set_half_turn_homings(self, motors: NameOrID | list[NameOrID] | None = None) -> dict[NameOrID, Value]:
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        list_of_homing_offsets = [0 for motor in motors]
        

        homing_offsets = dict(zip(motors,list_of_homing_offsets))
        # for motor, offset in homing_offsets.items():
        #     self.write("Homing_Offset", motor, offset)

        return homing_offsets



    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass
        if instruction_name == "sync_read" and self.protocol_version != 0:
            raise NotImplementedError(
                "'Sync Read' is not available with Feetech motors using Protocol 1. Use 'Read' sequentially instead."
            )
        if instruction_name == "broadcast_ping" and self.protocol_version == 0:
            raise NotImplementedError(
                "'Broadcast Ping' is not available with Feetech motors using Protocol 1. Use 'Ping' sequentially instead."
            )

    # def _assert_same_firmware(self) -> None:
    #     firmware_versions = self._read_firmware_version(self.ids, raise_on_error=True)
    #     if len(set(firmware_versions.values())) != 1:
    #         raise RuntimeError(
    #             "Some Motors use different firmware versions:"
    #             f"\n{pformat(firmware_versions)}\n"
    #             "Update their firmware first using Feetech's software. "
    #             "Visit https://www.feetechrc.com/software."
    #         )

    def _handshake(self) -> None:
        self._assert_motors_exist()
        # self._assert_same_firmware()

    def connect(self, handshake: bool = True) -> None:
        self.port_handler.openPort()
        for motor in self.motors:
            if (self.port_handler.ping(self.motors[motor].id)!= True):
                raise Exception(f"motor not found id:{self.motors[motor].id}")
        self.port_handler.ResetLoop(0xff)

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        raise NotImplementedError(f"this function should never be called")

    def configure_motors(self, return_delay_time=0, maximum_acceleration=254, acceleration=254) -> None:
        raise NotImplementedError(f"this function should never be called")
    
    def sync_read(self, data_name, motors = None, *, normalize = True, num_retry = 0):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__}('{self.port}') is not connected. You need to run `{self.__class__.__name__}.connect()`."
            )

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]

        read_data = {}
        if data_name == "Monitor" or data_name == "Present_Position":
            servos_id = dict(zip(names, ids))
            monitor_data = self.port_handler.sync_read["Monitor"](servos_id)
            for name in names:
                if monitor_data[name].current_position >=180:
                    monitor_data[name].current_position = 180
                elif monitor_data[name].current_position <=-180:
                    monitor_data[name].current_position = -180
                monitor_data[name].current_position = int(monitor_data[name].current_position+180)/360.0*4096
            for name in names:
                read_data[name]=int(monitor_data[name].current_position)
        

        return read_data
    
        # models = [self.motors[motor].model for motor in names]

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
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
        # models = [self._id_to_model(id_) for id_ in ids_values]

        write_data = {} 
        if data_name == "Goal_Position":
            for motor in values:
                data=SyncPositionControlOptions(self.motors[motor].id,
                                                int((values[motor]/4096.0*360.0-180)*10),
                                                DEFAULT_MOTION_TIME,
                                                0,
                                                DEFAULT_ACC_TIME,
                                                DEFAULT_DEC_TIME)
                write_data[motor] = data
            self.port_handler.sync_write["Goal_Position"](write_data)
        
        # model = next(iter(models))
        # addr, length = get_address(self.model_ctrl_table, model, data_name)

        # if normalize and data_name in self.normalized_data:
        #     ids_values = self._unnormalize(ids_values)

        # ids_values = self._encode_sign(data_name, ids_values)

        # err_msg = f"Failed to sync write '{data_name}' with {ids_values=} after {num_retry + 1} tries."
        # self._sync_write(addr, length, ids_values, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)



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
            mins[motor] = self.port_handler.read["Min_Position_Limit"](self.motors[motor].id)
            maxes[motor] = self.port_handler.read["Max_Position_Limit"]( self.motors[motor].id )
            offsets[motor] = (
                0
            )

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
        """
        On Feetech Motors:
        Present_Position = Actual_Position - Homing_Offset
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            half_turn_homings[motor] = pos - int(max_res / 2)

        return half_turn_homings

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.port_handler.write["Stop_On_Control_Mode"](self.motors[motor].id,"unlocked",0)


    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        pass
    #     addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
    #     self._write(addr, length, motor_id, TorqueMode.DISABLED.value, num_retry=num_retry)
    #     addr, length = get_address(self.model_ctrl_table, model, "Lock")
    #     self._write(addr, length, motor_id, 0, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.port_handler.write["Stop_On_Control_Mode"](motor, "locked",0)

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
        return _split_into_byte_chunks(value, length)

    # def _broadcast_ping(self) -> tuple[dict[int, int], int]:
    #     import scservo_sdk as scs

    #     data_list = {}

    #     status_length = 6

    #     rx_length = 0
    #     wait_length = status_length * scs.MAX_ID

    #     txpacket = [0] * 6

    #     tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

    #     txpacket[scs.PKT_ID] = scs.BROADCAST_ID
    #     txpacket[scs.PKT_LENGTH] = 2
    #     txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING

    #     result = self.packet_handler.txPacket(self.port_handler, txpacket)
    #     if result != scs.COMM_SUCCESS:
    #         self.port_handler.is_using = False
    #         return data_list, result

    #     # set rx timeout
    #     self.port_handler.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * scs.MAX_ID) + 16.0)

    #     rxpacket = []
    #     while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
    #         rxpacket += self.port_handler.readPort(wait_length - rx_length)
    #         rx_length = len(rxpacket)

    #     self.port_handler.is_using = False

    #     if rx_length == 0:
    #         return data_list, scs.COMM_RX_TIMEOUT

    #     while True:
    #         if rx_length < status_length:
    #             return data_list, scs.COMM_RX_CORRUPT

    #         # find packet header
    #         for idx in range(0, (rx_length - 1)):
    #             if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF):
    #                 break

    #         if idx == 0:  # found at the beginning of the packet
    #             # calculate checksum
    #             checksum = 0
    #             for idx in range(2, status_length - 1):  # except header & checksum
    #                 checksum += rxpacket[idx]

    #             checksum = ~checksum & 0xFF
    #             if rxpacket[status_length - 1] == checksum:
    #                 result = scs.COMM_SUCCESS
    #                 data_list[rxpacket[scs.PKT_ID]] = rxpacket[scs.PKT_ERROR]

    #                 del rxpacket[0:status_length]
    #                 rx_length = rx_length - status_length

    #                 if rx_length == 0:
    #                     return data_list, result
    #             else:
    #                 result = scs.COMM_RX_CORRUPT
    #                 # remove header (0xFF 0xFF)
    #                 del rxpacket[0:2]
    #                 rx_length = rx_length - 2
    #         else:
    #             # remove unnecessary packets
    #             del rxpacket[0:idx]
    #             rx_length = rx_length - idx

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        self._assert_protocol_is_compatible("broadcast_ping")
        for n_try in range(1 + num_retry):
            ids_status, comm = self._broadcast_ping()
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast ping failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return

        ids_errors = {id_: status for id_, status in ids_status.items() if self._is_error(status)}
        if ids_errors:
            display_dict = {id_: self.packet_handler.getRxPacketError(err) for id_, err in ids_errors.items()}
            logger.error(f"Some motors found returned an error status:\n{pformat(display_dict, indent=4)}")

        return self._read_model_number(list(ids_status), raise_on_error)

    # def _read_firmware_version(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, str]:
    #     firmware_versions = {}
    #     for id_ in motor_ids:
    #         firm_ver_major, comm, error = self._read(
    #             *FIRMWARE_MAJOR_VERSION, id_, raise_on_error=raise_on_error
    #         )
    #         if not self._is_comm_success(comm) or self._is_error(error):
    #             continue

    #         firm_ver_minor, comm, error = self._read(
    #             *FIRMWARE_MINOR_VERSION, id_, raise_on_error=raise_on_error
    #         )
    #         if not self._is_comm_success(comm) or self._is_error(error):
    #             continue

    #         firmware_versions[id_] = f"{firm_ver_major}.{firm_ver_minor}"

    #     return firmware_versions

    def _read_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        model_numbers = {}
        for id_ in motor_ids:
            model_nb, comm, error = self._read(*MODEL_NUMBER, id_, raise_on_error=raise_on_error)
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            model_numbers[id_] = model_nb

        return model_numbers
