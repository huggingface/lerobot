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

from lerobot.common.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value
from .tables import (
    FIRMWARE_VERSION,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER,
    MODEL_NUMBER_TABLE,
    MODEL_RESOLUTION,
    SCAN_BAUDRATES,
)

DEFAULT_PROTOCOL_VERSION = 0
BAUDRATE = 1_000_000
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

    available_baudrates = deepcopy(SCAN_BAUDRATES)
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

    def configure_motors(self) -> None:
        # By default, Feetech motors have a 500µs delay response time (corresponding to a value of 250 on the
        # 'Return_Delay' address). We ensure this is reduced to the minimum of 2µs (value of 0).
        for id_ in self.ids:
            self.write("Return_Delay_Time", id_, 0)

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

    def _disable_torque(self, motors: list[NameOrID]) -> None:
        for motor in motors:
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value)
            self.write("Lock", motor, 0)

    def _enable_torque(self, motors: list[NameOrID]) -> None:
        for motor in motors:
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value)
            self.write("Lock", motor, 1)

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

    @staticmethod
    def _split_int_to_bytes(value: int, n_bytes: int) -> list[int]:
        # Validate input
        if value < 0:
            raise ValueError(f"Negative values are not allowed: {value}")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(n_bytes)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {n_bytes}. Expected [1, 2, 4].")

        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum for {n_bytes} bytes ({max_value}).")

        import scservo_sdk as scs

        if n_bytes == 1:
            data = [value]
        elif n_bytes == 2:
            data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        elif n_bytes == 4:
            data = [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
            ]
        return data

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
        while True:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

            if self.port_handler.isPacketTimeout():  # or rx_length >= wait_length
                break

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

                checksum = scs.SCS_LOBYTE(~checksum)
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

        return self._get_model_number(list(ids_status), raise_on_error)

    def _get_firmware_version(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        # comm, major = self._sync_read(*FIRMWARE_MAJOR_VERSION, motor_ids)
        # if not self._is_comm_success(comm):
        #     if raise_on_error:
        #         raise ConnectionError(self.packet_handler.getTxRxResult(comm))
        #     return

        # comm, minor = self._sync_read(*FIRMWARE_MINOR_VERSION, motor_ids)
        # if not self._is_comm_success(comm):
        #     if raise_on_error:
        #         raise ConnectionError(self.packet_handler.getTxRxResult(comm))
        #     return

        # return {id_: f"{major[id_]}.{minor[id_]}" for id_ in motor_ids}

        comm, firmware_versions = self._sync_read(*FIRMWARE_VERSION, motor_ids)
        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return

        return firmware_versions

    def _get_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        # comm, major = self._sync_read(*MODEL_MAJOR_VERSION, motor_ids)
        # if not self._is_comm_success(comm):
        #     if raise_on_error:
        #         raise ConnectionError(self.packet_handler.getTxRxResult(comm))
        #     return

        # comm, minor = self._sync_read(*MODEL_MINOR_VERSION, motor_ids)
        # if not self._is_comm_success(comm):
        #     if raise_on_error:
        #         raise ConnectionError(self.packet_handler.getTxRxResult(comm))
        #     return

        # return {id_: f"{major[id_]}.{minor[id_]}" for id_ in motor_ids}

        comm, model_numbers = self._sync_read(*MODEL_NUMBER, motor_ids)
        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return

        return model_numbers
