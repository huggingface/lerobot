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

import logging
from copy import deepcopy
from enum import Enum
from pprint import pformat

from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address
from .tables import (
    FIRMWARE_MAJOR_VERSION,
    FIRMWARE_MINOR_VERSION,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER,
    MODEL_NUMBER_TABLE,
    MODEL_PROTOCOL,
    MODEL_RESOLUTION,
    SCAN_BAUDRATES,
)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

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


class FeetechMotorsBus(MotorsBus):
    """
    The FeetechMotorsBus class allows to efficiently read and write to the attached motors. It relies on the
    python feetech sdk to communicate with the motors, which is itself based on the dynamixel sdk.
    """

    apply_drive_mode = True
    available_baudrates = deepcopy(SCAN_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ):
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version
        self._assert_same_protocol()
        import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        # HACK: monkeypatch
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
            self.port_handler, scs.PortHandler
        )
        self.packet_handler = scs.PacketHandler(protocol_version)
        self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        self._comm_success = scs.COMM_SUCCESS
        self._no_error = 0x00

        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise ValueError(f"Some motors are incompatible with protocol_version={self.protocol_version}")

    def _assert_same_protocol(self) -> None:
        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise RuntimeError("Some motors use an incompatible protocol.")

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        if instruction_name == "sync_read" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Sync Read' is not available with Feetech motors using Protocol 1. Use 'Read' sequentially instead."
            )
        if instruction_name == "broadcast_ping" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Broadcast Ping' is not available with Feetech motors using Protocol 1. Use 'Ping' sequentially instead."
            )

    def _assert_same_firmware(self) -> None:
        firmware_versions = self._read_firmware_version(self.ids, raise_on_error=True)
        if len(set(firmware_versions.values())) != 1:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(firmware_versions)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        if self.protocol_version == 0:
            return self._find_single_motor_p0(motor, initial_baudrate)
        else:
            return self._find_single_motor_p1(motor, initial_baudrate)

    def _find_single_motor_p0(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def _find_single_motor_p1(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        import scservo_sdk as scs

        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            for id_ in range(scs.MAX_ID + 1):
                found_model = self.ping(id_)
                if found_model is not None:
                    if found_model != expected_model_nb:
                        raise RuntimeError(
                            f"Found one motor on {baudrate=} with id={id_} but it has a "
                            f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                            f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                        )
                    return baudrate, id_

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(self) -> None:
        for motor in self.motors:
            # By default, Feetech motors have a 500µs delay response time (corresponding to a value of 250 on
            # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
            self.write("Return_Delay_Time", motor, 0)
            # Set 'Maximum_Acceleration' to 254 to speedup acceleration and deceleration of the motors.
            # Note: this address is not in the official STS3215 Memory Table
            self.write("Maximum_Acceleration", motor, 254)
            self.write("Acceleration", motor, 254)

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
            mins[motor] = self.read("Min_Position_Limit", motor, normalize=False)
            maxes[motor] = self.read("Max_Position_Limit", motor, normalize=False)
            offsets[motor] = (
                self.read("Homing_Offset", motor, normalize=False) if self.protocol_version == 0 else 0
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

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        for motor, calibration in calibration_dict.items():
            if self.protocol_version == 0:
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        self.calibration = calibration_dict

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
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)
            self.write("Lock", motor, 0, num_retry=num_retry)

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
        self._write(addr, length, motor_id, TorqueMode.DISABLED.value, num_retry=num_retry)
        addr, length = get_address(self.model_ctrl_table, model, "Lock")
        self._write(addr, length, motor_id, 0, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)
            self.write("Lock", motor, 1, num_retry=num_retry)

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

    def _broadcast_ping(self) -> tuple[dict[int, int], int]:
        import scservo_sdk as scs

        data_list = {}

        status_length = 6

        rx_length = 0
        wait_length = status_length * scs.MAX_ID

        txpacket = [0] * 6

        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

        txpacket[scs.PKT_ID] = scs.BROADCAST_ID
        txpacket[scs.PKT_LENGTH] = 2
        txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING

        result = self.packet_handler.txPacket(self.port_handler, txpacket)
        if result != scs.COMM_SUCCESS:
            self.port_handler.is_using = False
            return data_list, result

        # set rx timeout
        self.port_handler.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * scs.MAX_ID) + 16.0)

        rxpacket = []
        while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

        self.port_handler.is_using = False

        if rx_length == 0:
            return data_list, scs.COMM_RX_TIMEOUT

        while True:
            if rx_length < status_length:
                return data_list, scs.COMM_RX_CORRUPT

            # find packet header
            for idx in range(0, (rx_length - 1)):
                if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF):
                    break

            if idx == 0:  # found at the beginning of the packet
                # calculate checksum
                checksum = 0
                for idx in range(2, status_length - 1):  # except header & checksum
                    checksum += rxpacket[idx]

                checksum = ~checksum & 0xFF
                if rxpacket[status_length - 1] == checksum:
                    result = scs.COMM_SUCCESS
                    data_list[rxpacket[scs.PKT_ID]] = rxpacket[scs.PKT_ERROR]

                    del rxpacket[0:status_length]
                    rx_length = rx_length - status_length

                    if rx_length == 0:
                        return data_list, result
                else:
                    result = scs.COMM_RX_CORRUPT
                    # remove header (0xFF 0xFF)
                    del rxpacket[0:2]
                    rx_length = rx_length - 2
            else:
                # remove unnecessary packets
                del rxpacket[0:idx]
                rx_length = rx_length - idx

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

    def _read_firmware_version(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, str]:
        firmware_versions = {}
        for id_ in motor_ids:
            firm_ver_major, comm, error = self._read(
                *FIRMWARE_MAJOR_VERSION, id_, raise_on_error=raise_on_error
            )
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            firm_ver_minor, comm, error = self._read(
                *FIRMWARE_MINOR_VERSION, id_, raise_on_error=raise_on_error
            )
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            firmware_versions[id_] = f"{firm_ver_major}.{firm_ver_minor}"

        return firmware_versions

    def _read_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        model_numbers = {}
        for id_ in motor_ids:
            model_nb, comm, error = self._read(*MODEL_NUMBER, id_, raise_on_error=raise_on_error)
            if not self._is_comm_success(comm) or self._is_error(error):
                continue

            model_numbers[id_] = model_nb

        return model_numbers
