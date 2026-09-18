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

from lerobot.motors.motors_bus import (
    Motor,
    MotorsBus,
    MotorsBusBase,
)

DUMMY_CTRL_TABLE_1 = {
    "Firmware_Version": (0, 1),
    "Model_Number": (1, 2),
    "Present_Position": (3, 4),
    "Goal_Position": (11, 2),
}

DUMMY_CTRL_TABLE_2 = {
    "Model_Number": (0, 2),
    "Firmware_Version": (2, 1),
    "Present_Position": (3, 4),
    "Present_Velocity": (7, 4),
    "Goal_Position": (11, 4),
    "Goal_Velocity": (15, 4),
    "Lock": (19, 1),
}

DUMMY_MODEL_CTRL_TABLE = {
    "model_1": DUMMY_CTRL_TABLE_1,
    "model_2": DUMMY_CTRL_TABLE_2,
    "model_3": DUMMY_CTRL_TABLE_2,
}

DUMMY_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
}

DUMMY_MODEL_BAUDRATE_TABLE = {
    "model_1": DUMMY_BAUDRATE_TABLE,
    "model_2": DUMMY_BAUDRATE_TABLE,
    "model_3": DUMMY_BAUDRATE_TABLE,
}

DUMMY_ENCODING_TABLE = {
    "Present_Position": 8,
    "Goal_Position": 10,
}

DUMMY_MODEL_ENCODING_TABLE = {
    "model_1": DUMMY_ENCODING_TABLE,
    "model_2": DUMMY_ENCODING_TABLE,
    "model_3": DUMMY_ENCODING_TABLE,
}

DUMMY_MODEL_NUMBER_TABLE = {
    "model_1": 1234,
    "model_2": 5678,
    "model_3": 5799,
}

DUMMY_MODEL_RESOLUTION_TABLE = {
    "model_1": 4096,
    "model_2": 1024,
    "model_3": 4096,
}


class MockTransport:
    """In-memory stand-in for the serial transport.

    Holds one byte per (motor id, register address), so a test seeds a value or
    reads back what the bus wrote without ever building a packet. Framing,
    checksums and sync instructions are the transport's business and are tested
    where they are implemented.
    """

    def __init__(self, baudrate: int = 1_000_000, timeout_ms: int = 1000):
        self.baudrate = baudrate
        self.timeout_ms = timeout_ms
        self._open = False

        self.registers: dict[tuple[int, int], int] = {}
        self.status: dict[int, int] = {}  # id -> status error byte
        self.absent: set[int] = set()  # ids that never answer
        self.fail_times: int = 0  # make the next N calls time out

        self.reads: list[tuple[int, int, int]] = []
        self.writes: list[tuple[int, int, bytes]] = []
        self.sync_reads: list[tuple[list[int], int, int]] = []
        self.sync_writes: list[tuple[list[int], int, list[bytes]]] = []

    # -- test-side helpers -------------------------------------------------

    def seed(self, motor_id: int, addr: int, data: bytes) -> None:
        for offset, byte in enumerate(data):
            self.registers[(motor_id, addr + offset)] = byte

    def stored(self, motor_id: int, addr: int, length: int) -> bytes:
        return bytes(self.registers.get((motor_id, addr + i), 0) for i in range(length))

    # -- MotorTransport ----------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def set_baudrate(self, baudrate: int) -> None:
        self.baudrate = baudrate

    def set_timeout(self, timeout_ms: int) -> None:
        self.timeout_ms = timeout_ms

    def _answer(self, motor_id: int) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("Timeout error")
        if motor_id in self.absent:
            raise RuntimeError("Timeout error")

    def read(self, motor_id: int, addr: int, length: int) -> tuple[bytes, int]:
        self._answer(motor_id)
        self.reads.append((motor_id, addr, length))
        return self.stored(motor_id, addr, length), self.status.get(motor_id, 0)

    def write(self, motor_id: int, addr: int, data: bytes) -> int:
        self._answer(motor_id)
        self.writes.append((motor_id, addr, data))
        self.seed(motor_id, addr, data)
        return self.status.get(motor_id, 0)

    def sync_read(self, motor_ids: list[int], addr: int, length: int) -> list[bytes]:
        for motor_id in motor_ids:
            self._answer(motor_id)
        self.sync_reads.append((list(motor_ids), addr, length))
        return [self.stored(motor_id, addr, length) for motor_id in motor_ids]

    def sync_write(self, motor_ids: list[int], addr: int, data: list[bytes]) -> None:
        for motor_id in motor_ids:
            self._answer(motor_id)
        self.sync_writes.append((list(motor_ids), addr, list(data)))
        for motor_id, payload in zip(motor_ids, data, strict=True):
            self.seed(motor_id, addr, payload)


class MockMotorsBus(MotorsBus):
    """Bus wired to a `MockTransport` instead of a serial port."""

    available_baudrates = [500_000, 1_000_000]
    default_timeout = 1000
    model_baudrate_table = DUMMY_MODEL_BAUDRATE_TABLE
    model_ctrl_table = DUMMY_MODEL_CTRL_TABLE
    model_encoding_table = DUMMY_MODEL_ENCODING_TABLE
    model_number_table = DUMMY_MODEL_NUMBER_TABLE
    model_resolution_table = DUMMY_MODEL_RESOLUTION_TABLE
    model_number_address = DUMMY_CTRL_TABLE_2["Model_Number"]
    max_id = 3
    normalized_data = ["Present_Position", "Goal_Position"]

    def __init__(self, port: str, motors: dict[str, Motor]):
        # Skip SerialMotorsBus.__init__: it guards deepdiff and builds a real transport.
        MotorsBusBase.__init__(self, port, motors)
        self._io = MockTransport()
        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: name for name, m in self.motors.items()}
        self._model_nb_to_model_dict = {v: k for k, v in self.model_number_table.items()}

    def _assert_protocol_is_compatible(self, instruction_name): ...
    def _handshake(self): ...
    def _find_single_motor(self, motor, initial_baudrate): ...
    def configure_motors(self): ...
    def is_calibrated(self): ...
    def read_calibration(self): ...
    def write_calibration(self, calibration_dict): ...
    def disable_torque(self, motors, num_retry): ...
    def _disable_torque(self, motor, model, num_retry): ...
    def enable_torque(self, motors, num_retry): ...
    def _get_half_turn_homings(self, positions): ...
    def _encode_sign(self, data_name, ids_values): ...
    def _decode_sign(self, data_name, ids_values): ...
    def _split_into_byte_chunks(self, value, length):
        return list(value.to_bytes(length, "little"))

    def _join_byte_chunks(self, data, length):
        return int.from_bytes(data, "little")
