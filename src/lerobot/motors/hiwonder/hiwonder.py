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

from lerobot.motors.feetech.feetech import (
    DriveMode,
    FeetechMotorsBus,
    OperatingMode,
    TorqueMode,
    patch_setPacketTimeout,
)
from lerobot.motors.feetech.tables import SCAN_BAUDRATES
from lerobot.motors.motors_bus import Motor, MotorCalibration

from . import hiwonder_sdk as hw
from .tables import (
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_PROTOCOL,
    MODEL_RESOLUTION,
)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

logger = logging.getLogger(__name__)


class HiwonderMotorsBus(FeetechMotorsBus):
    """
    MotorsBus implementation for Hiwonder serial bus servos (e.g. HX-30HM).

    The HX-30HM uses the same STS/SMS serial protocol as Feetech motors and is
    fully compatible with the hiwonder_sdk bundled in this module. This class
    registers the Hiwonder model names so they can be used as the ``model``
    argument when constructing :class:`~lerobot.motors.motors_bus.Motor` objects.

    Example::

        from lerobot.motors.motors_bus import Motor
        from lerobot.motors.hiwonder import HiwonderMotorsBus

        bus = HiwonderMotorsBus(
            port="/dev/ttyUSB0",
            motors={
                "shoulder_pan": Motor(1, "hx30hm"),
                "shoulder_lift": Motor(2, "hx30hm"),
            },
        )
    """

    available_baudrates = deepcopy(SCAN_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ):
        # Call grandparent (SerialMotorsBus) directly to avoid FeetechMotorsBus.__init__
        # importing scservo_sdk, then set up hiwonder_sdk equivalents.
        from lerobot.motors.motors_bus import SerialMotorsBus

        SerialMotorsBus.__init__(self, port, motors, calibration)
        self.protocol_version = protocol_version
        self._assert_same_protocol()

        self.port_handler = hw.PortHandler(self.port)
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(  # type: ignore[method-assign]
            self.port_handler, hw.PortHandler
        )
        self.packet_handler = hw.PacketHandler(self.port_handler, endianness=0)
        self.sync_reader = hw.GroupSyncRead(self.packet_handler, 0, 0)
        self.sync_writer = hw.GroupSyncWrite(self.packet_handler, 0, 0)
        self._comm_success = hw.COMM_SUCCESS
        self._no_error = 0x00

        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise ValueError(f"Some motors are incompatible with protocol_version={self.protocol_version}")

    def _assert_same_protocol(self) -> None:
        if any(MODEL_PROTOCOL[model] != self.protocol_version for model in self.models):
            raise RuntimeError("Some motors use an incompatible protocol.")

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        if length == 1:
            return [value]
        elif length == 2:
            return [value & 0xFF, (value >> 8) & 0xFF]
        elif length == 4:
            lo = value & 0xFFFF
            hi = (value >> 16) & 0xFFFF
            return [lo & 0xFF, (lo >> 8) & 0xFF, hi & 0xFF, (hi >> 8) & 0xFF]
        raise ValueError(f"Unsupported data length: {length}")

    def _broadcast_ping(self) -> tuple[dict[int, int], int]:
        data_list: dict[int, int] = {}
        status_length = 6
        rx_length = 0
        wait_length = status_length * hw.MAX_ID

        txpacket = [0] * 6
        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

        txpacket[hw.PKT_ID] = hw.BROADCAST_ID
        txpacket[hw.PKT_LENGTH] = 2
        txpacket[hw.PKT_INSTRUCTION] = hw.INST_PING

        result = self.packet_handler.txPacket(txpacket)
        if result != hw.COMM_SUCCESS:
            self.port_handler.is_using = False
            return data_list, result

        self.port_handler.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * hw.MAX_ID) + 16.0)

        rxpacket = []
        while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

        self.port_handler.is_using = False

        if rx_length == 0:
            return data_list, hw.COMM_RX_TIMEOUT

        while True:
            if rx_length < status_length:
                return data_list, hw.COMM_RX_CORRUPT

            for idx in range(0, rx_length - 1):
                if rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF:
                    break

            if idx == 0:
                checksum = 0
                for idx in range(2, status_length - 1):
                    checksum += rxpacket[idx]
                checksum = ~checksum & 0xFF

                if rxpacket[status_length - 1] == checksum:
                    result = hw.COMM_SUCCESS
                    data_list[rxpacket[hw.PKT_ID]] = rxpacket[hw.PKT_ERROR]
                    del rxpacket[0:status_length]
                    rx_length -= status_length
                    if rx_length == 0:
                        return data_list, result
                else:
                    result = hw.COMM_RX_CORRUPT
                    del rxpacket[0:2]
                    rx_length -= 2
            else:
                del rxpacket[0:idx]
                rx_length -= idx

    def _find_single_motor_p1(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            for id_ in range(hw.MAX_ID + 1):
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


__all__ = ["HiwonderMotorsBus", "DriveMode", "OperatingMode", "TorqueMode"]
