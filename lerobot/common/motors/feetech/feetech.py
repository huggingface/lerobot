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

from copy import deepcopy

from ..motors_bus import MotorsBus
from .tables import (
    CALIBRATION_REQUIRED,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_RESOLUTION,
)

PROTOCOL_VERSION = 0
BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

MAX_ID_RANGE = 252


class FeetechMotorsBus(MotorsBus):
    """
    The FeetechMotorsBus class allows to efficiently read and write to the attached motors. It relies on the
    python feetech sdk to communicate with the motors, which is itself based on the dynamixel sdk.
    """

    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    calibration_required = deepcopy(CALIBRATION_REQUIRED)
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
    ):
        super().__init__(port, motors)
        import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)
        self.reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)

    def broadcast_ping(self, num_retry: int | None = None):
        raise NotImplementedError  # TODO

    def calibrate_values(self, ids_values: dict[int, int]) -> dict[int, float]:
        # TODO
        return ids_values

    def uncalibrate_values(self, ids_values: dict[int, float]) -> dict[int, int]:
        # TODO
        return ids_values

    def _is_comm_success(self, comm: int) -> bool:
        import scservo_sdk as scs

        return comm == scs.COMM_SUCCESS

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

        import scservo_sdk as scs

        # Note: No need to convert back into unsigned int, since this byte preprocessing
        # already handles it for us.
        if n_bytes == 1:
            data = [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            ]
        elif n_bytes == 2:
            data = [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            ]
        elif n_bytes == 4:
            data = [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
            ]
        return data
