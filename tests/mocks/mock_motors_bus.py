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

from functools import cached_property

from lerobot.motors.motors_bus import SerialMotorsBus


class DummyServo:
    """Stands in for a rustypot controller class: the models it covers, the registers it
    has, its resolution and baud rates. It also stands in for its own definition, which
    is what a `MockBus` is given, so the tests need no rustypot.
    """

    def __init__(self, models: dict[str, int], registers: tuple[str, ...]):
        self._models = models
        self._registers = {name.lower() for name in registers}

    def definition(self) -> "DummyServo":
        return self

    def models(self) -> dict[str, int]:
        return dict(self._models)

    def register(self, name: str) -> str | None:
        return name if name in self._registers else None

    def resolution(self) -> int:
        return 4096

    def baudrates(self) -> dict[int, int]:
        return {1_000_000: 0, 500_000: 1, 250_000: 2}


DUMMY_SERVO_1 = DummyServo(
    {"model_1": 1234}, ("Model_Number", "Firmware_Version", "Present_Position", "Goal_Position")
)
DUMMY_SERVO_2 = DummyServo(
    {"model_2": 5678, "model_3": 5799},
    (
        "Model_Number",
        "Firmware_Version",
        "Present_Position",
        "Present_Velocity",
        "Goal_Position",
        "Goal_Velocity",
        "Lock",
    ),
)


class MockBus:
    """In-memory stand-in for an open `rustypot.Bus`.

    Holds one integer per (motor id, register name), so a test seeds a value or
    reads back what the bus wrote without ever building a packet. Framing, byte
    order and sign encoding are rustypot's business and are tested there. Like
    rustypot, it refuses an id it was not opened with.
    """

    def __init__(
        self,
        serial_port: str = "/dev/dummy-port",
        baudrate: int = 1_000_000,
        timeout: float = 1.0,
        motors: dict | None = None,
    ):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.motors = dict(motors or {})  # id -> definition
        self.closed = False

        self.registers: dict[tuple[int, str], int] = {}
        self.status: dict[int, int] = {}  # id -> status error byte
        self.absent: set[int] = set()  # ids that never answer
        self.retries: list[int] = []  # the retries each access was given

        self.reads: list[tuple[int, str]] = []
        self.writes: list[tuple[int, str, int]] = []
        self.sync_reads: list[tuple[list[int], str]] = []
        self.sync_writes: list[tuple[list[int], str, list[int]]] = []
        self.scans: list = []  # the definition each scan read model numbers with

    # -- test-side helpers -------------------------------------------------

    def seed(self, motor_id: int, register: str, value: int) -> None:
        self.registers[(motor_id, register)] = value

    def stored(self, motor_id: int, register: str) -> int:
        return self.registers.get((motor_id, register), 0)

    # -- rustypot.Bus ------------------------------------------------------

    def close(self) -> None:
        self.closed = True

    def set_baudrate(self, baudrate: int) -> None:
        self.baudrate = baudrate

    def _answer(self, motor_ids: list[int], retries: int) -> None:
        if unknown := [motor_id for motor_id in motor_ids if motor_id not in self.motors]:
            raise ValueError(f"no motor with id {unknown[0]} on this bus")
        self.retries.append(retries)
        if any(motor_id in self.absent for motor_id in motor_ids):
            raise RuntimeError("Timeout error")

    def read_register_with_error(self, motor_id: int, register: str, retries: int = 0) -> tuple[int, int]:
        self._answer([motor_id], retries)
        self.reads.append((motor_id, register))
        return self.stored(motor_id, register), self.status.get(motor_id, 0)

    def write_register_with_error(self, motor_id: int, register: str, value: int, retries: int = 0) -> int:
        self._answer([motor_id], retries)
        self.writes.append((motor_id, register, value))
        self.seed(motor_id, register, value)
        return self.status.get(motor_id, 0)

    def sync_read_register(self, motor_ids: list[int], register: str, retries: int = 0) -> list[int]:
        self._answer(motor_ids, retries)
        self.sync_reads.append((list(motor_ids), register))
        return [self.stored(motor_id, register) for motor_id in motor_ids]

    def sync_write_register(
        self, motor_ids: list[int], register: str, values: list[int], retries: int = 0
    ) -> None:
        self._answer(motor_ids, retries)
        self.sync_writes.append((list(motor_ids), register, list(values)))
        for motor_id, value in zip(motor_ids, values, strict=True):
            self.seed(motor_id, register, value)

    def scan(self, definition) -> dict[int, int]:
        """Every id holding a model number, as a sweep of the whole protocol range finds them."""
        self.scans.append(definition)
        return {
            motor_id: value
            for (motor_id, register), value in self.registers.items()
            if register == "model_number" and motor_id not in self.absent
        }


class MockMotorsBus(SerialMotorsBus):
    """Bus over dummy servo definitions. It opens a `MockBus` where the real one opens a
    `rustypot.Bus`, and keeps every one it opened in `opened`, oldest first."""

    @staticmethod
    def _servos() -> tuple[DummyServo, ...]:
        return (DUMMY_SERVO_1, DUMMY_SERVO_2)

    @cached_property
    def opened(self) -> list[MockBus]:
        return []

    def _bus_class(self):
        def open_bus(*args) -> MockBus:
            self.opened.append(MockBus(*args))
            return self.opened[-1]

        return open_bus

    def configure_motors(self): ...
    def is_calibrated(self): ...
    def read_calibration(self): ...
    def write_calibration(self, calibration_dict): ...
    def disable_torque(self, motors, num_retry): ...
    def _disable_torque(self, motor, num_retry=0):
        self._write("Torque_Enable", motor, 0, num_retry=num_retry)

    def enable_torque(self, motors, num_retry): ...
    def _get_half_turn_homings(self, positions): ...
