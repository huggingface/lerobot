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

from lerobot.utils.encoding_utils import decode_twos_complement, encode_twos_complement

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address
from .tables import (
    AVAILABLE_BAUDRATES,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_RESOLUTION,
)

PROTOCOL_VERSION = 2.0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

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


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    import dynamixel_sdk as dxl

    if length == 1:
        data = [value]
    elif length == 2:
        data = [dxl.DXL_LOBYTE(value), dxl.DXL_HIBYTE(value)]
    elif length == 4:
        data = [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
        ]
    return data


class DynamixelMotorsBus(MotorsBus):
    """
    The Dynamixel implementation for a MotorsBus. It relies on the python dynamixel sdk to communicate with
    the motors. For more info, see the Dynamixel SDK Documentation:
    https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20
    """

    apply_drive_mode = False
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
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
    ):
        super().__init__(port, motors, calibration)
        import dynamixel_sdk as dxl

        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self.sync_reader = dxl.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        self._comm_success = dxl.COMM_SUCCESS
        self._no_error = 0x00

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    def _handshake(self) -> None:
        self._assert_motors_exist()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                expected_model_nb = self.model_number_table[model]
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(self) -> None:
        # By default, Dynamixel motors have a 500µs delay response time (corresponding to a value of 250 on
        # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, 0)

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

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        for motor, calibration in calibration_dict.items():
            self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        self.calibration = calibration_dict

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
        self._write(addr, length, motor_id, TorqueMode.DISABLED.value, num_retry=num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                n_bytes = encoding_table[data_name]
                ids_values[id_] = encode_twos_complement(ids_values[id_], n_bytes)

        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                n_bytes = encoding_table[data_name]
                ids_values[id_] = decode_twos_complement(ids_values[id_], n_bytes)

        return ids_values

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

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return _split_into_byte_chunks(value, length)

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        for n_try in range(1 + num_retry):
            data_list, comm = self.packet_handler.broadcastPing(self.port_handler)
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast ping failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))

            return

        return {id_: data[0] for id_, data in data_list.items()}
