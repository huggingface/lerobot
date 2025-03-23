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

from copy import deepcopy

from ..motors_bus import Motor, MotorsBus
from .tables import MODEL_BAUDRATE_TABLE, MODEL_CONTROL_TABLE, MODEL_RESOLUTION

PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000
MAX_ID_RANGE = 252

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]


class DynamixelMotorsBus(MotorsBus):
    """
    The Dynamixel implementation for a MotorsBus. It relies on the python dynamixel sdk to communicate with
    the motors. For more info, see the Dynamixel SDK Documentation:
    https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20
    """

    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    calibration_required = deepcopy(CALIBRATION_REQUIRED)
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
    ):
        super().__init__(port, motors)
        import dynamixel_sdk as dxl

        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self.sync_reader = dxl.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        self._comm_success = dxl.COMM_SUCCESS
        self._error = 0x00

    def broadcast_ping(
        self, num_retry: int = 0, raise_on_error: bool = False
    ) -> dict[int, list[int, int]] | None:
        for _ in range(1 + num_retry):
            data_list, comm = self.packet_handler.broadcastPing(self.port_handler)
            if self._is_comm_success(comm):
                return data_list

        if raise_on_error:
            raise ConnectionError(f"Broadcast ping returned a {comm} comm error.")

    def calibrate_values(self, ids_values: dict[int, int]) -> dict[int, float]:
        # TODO
        return ids_values

    def uncalibrate_values(self, ids_values: dict[int, float]) -> dict[int, int]:
        # TODO
        return ids_values

    @staticmethod
    def split_int_bytes(value: int, n_bytes: int) -> list[int]:
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
