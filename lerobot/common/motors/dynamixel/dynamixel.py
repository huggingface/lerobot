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

# TODO(aliberts): Should we implement FastSyncRead/Write?
# https://github.com/ROBOTIS-GIT/DynamixelSDK/pull/643
# https://github.com/ROBOTIS-GIT/DynamixelSDK/releases/tag/3.8.2
# https://emanual.robotis.com/docs/en/dxl/protocol2/#fast-sync-read-0x8a
# -> Need to check compatibility across models

import logging
from copy import deepcopy
from enum import Enum

from lerobot.common.utils.encoding_utils import decode_twos_complement, encode_twos_complement

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value
from .tables import (
    AVAILABLE_BAUDRATES,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_NUMBER,
    MODEL_RESOLUTION,
)

PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

NORMALIZATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class OperatingMode(Enum):
    # DYNAMIXEL only controls current(torque) regardless of speed and position. This mode is ideal for a
    # gripper or a system that only uses current(torque) control or a system that has additional
    # velocity/position controllers.
    CURRENT = 0

    # This mode controls velocity. This mode is identical to the Wheel Mode(endless) from existing DYNAMIXEL.
    # This mode is ideal for wheel-type robots.
    VELOCITY = 1

    # This mode controls position. This mode is identical to the Joint Mode from existing DYNAMIXEL. Operating
    # position range is limited by the Max Position Limit(48) and the Min Position Limit(52). This mode is
    # ideal for articulated robots that each joint rotates less than 360 degrees.
    POSITION = 3

    # This mode controls position. This mode is identical to the Multi-turn Position Control from existing
    # DYNAMIXEL. 512 turns are supported(-256[rev] ~ 256[rev]). This mode is ideal for multi-turn wrists or
    # conveyer systems or a system that requires an additional reduction gear. Note that Max Position
    # Limit(48), Min Position Limit(52) are not used on Extended Position Control Mode.
    EXTENDED_POSITION = 4

    # This mode controls both position and current(torque). Up to 512 turns are supported (-256[rev] ~
    # 256[rev]). This mode is ideal for a system that requires both position and current control such as
    # articulated robots or grippers.
    CURRENT_POSITION = 5

    # This mode directly controls PWM output. (Voltage Control Mode)
    PWM = 16


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class DynamixelMotorsBus(MotorsBus):
    """
    The Dynamixel implementation for a MotorsBus. It relies on the python dynamixel sdk to communicate with
    the motors. For more info, see the Dynamixel SDK Documentation:
    https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20
    """

    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalization_required = deepcopy(NORMALIZATION_REQUIRED)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        super().__init__(port, motors, calibration)
        import dynamixel_sdk as dxl

        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self.sync_reader = dxl.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        self._comm_success = dxl.COMM_SUCCESS
        self._no_error = 0x00

    def configure_motors(self) -> None:
        # By default, Dynamixel motors have a 500µs delay response time (corresponding to a value of 250 on
        # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
        for id_ in self.ids:
            self.write("Return_Delay_Time", id_, 0)

    def _disable_torque(self, motors: list[NameOrID]) -> None:
        for motor in motors:
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value)

    def _enable_torque(self, motors: list[NameOrID]) -> None:
        for motor in motors:
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value)

    def _encode_value(self, value: int, data_name: str | None = None, n_bytes: int | None = None) -> int:
        return encode_twos_complement(value, n_bytes)

    def _decode_value(self, value: int, data_name: str | None = None, n_bytes: int | None = None) -> int:
        return decode_twos_complement(value, n_bytes)

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        On Dynamixel Motors:
        Present_Position = Actual_Position + Homing_Offset
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            half_turn_homings[motor] = int(max_res / 2) - pos

        return half_turn_homings

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

        import dynamixel_sdk as dxl

        # Note: No need to convert back into unsigned int, since this byte preprocessing
        # already handles it for us.
        if n_bytes == 1:
            data = [value]
        elif n_bytes == 2:
            data = [dxl.DXL_LOBYTE(value), dxl.DXL_HIBYTE(value)]
        elif n_bytes == 4:
            data = [
                dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
            ]
        return data

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        for n_try in range(1 + num_retry):
            data_list, comm = self.packet_handler.broadcastPing(self.port_handler)
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))

            return data_list if data_list else None

        return {id_: data[0] for id_, data in data_list.items()}
