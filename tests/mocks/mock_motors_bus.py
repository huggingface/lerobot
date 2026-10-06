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


class DummyDefinition:
    """Stands in for a rustypot `ServoDefinition`: the models it covers, the registers it
    has and the facts the bus reads from it, so the tests need no rustypot.
    """

    def __init__(
        self,
        name: str,
        models: dict[str, int],
        registers: tuple[str, ...],
        *,
        operating_modes: dict[str, int] | None = None,
        homing_offset_sign: int | None = None,
        supports_broadcast_ping: bool = False,
        eeprom_lock: bool = False,
    ):
        self.name = name
        self.models = models
        self._registers = {register.lower() for register in registers}
        self.resolution = 4096
        # rustypot hands these out slowest first.
        self.baudrates = {250_000: 2, 500_000: 1, 1_000_000: 0}
        self.factory_baudrate = 500_000
        self.operating_modes = operating_modes or {}
        self.homing_offset_sign = homing_offset_sign
        self.supports_broadcast_ping = supports_broadcast_ping
        self.eeprom_lock = eeprom_lock

    def register(self, name: str) -> str | None:
        return name if name in self._registers else None

    def __repr__(self) -> str:
        return f"DummyDefinition('{self.name}')"


DUMMY_1 = DummyDefinition(
    "DUMMY1", {"model_1": 1234}, ("Model_Number", "Firmware_Version", "Present_Position", "Goal_Position")
)
DUMMY_2 = DummyDefinition(
    "DUMMY2",
    {"model_2": 5678, "model_3": 5799},
    (
        "Model_Number",
        "Firmware_Version",
        "Present_Position",
        "Present_Velocity",
        "Goal_Position",
        "Goal_Velocity",
        "Operating_Mode",
        "Lock",
    ),
    operating_modes={"position": 3, "velocity": 1},
    homing_offset_sign=1,
    supports_broadcast_ping=True,
    eeprom_lock=True,
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
        self.scans: list = []  # the definition each sweep read model numbers with
        self.broadcast_scans: list = []  # the definition each broadcast scan read them with
        self.torques: list[tuple[list[int], bool, int]] = []  # (ids, enabled, retries)
        self.setups: list[tuple] = []  # change_id / change_baudrate calls, in order

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

    def _found(self) -> dict[int, int]:
        """Every id holding a model number, as a scan of the whole protocol range finds them."""
        return {
            motor_id: value
            for (motor_id, register), value in self.registers.items()
            if register == "model_number" and motor_id not in self.absent
        }

    def scan(self, definition, ids=None) -> dict[int, int]:
        self.scans.append(definition)
        return self._found()

    def broadcast_scan(self, definition, ids=None) -> dict[int, int]:
        self.broadcast_scans.append(definition)
        return self._found()

    def set_torque(self, motor_ids: list[int], enabled: bool, retries: int = 0) -> dict[int, str]:
        """Like rustypot: every motor tried, the ones that failed returned with why."""
        self.torques.append((list(motor_ids), enabled, retries))
        failed = {}
        for motor_id in motor_ids:
            if motor_id not in self.motors:
                failed[motor_id] = f"no motor with id {motor_id} on this bus"
            elif motor_id in self.absent:
                failed[motor_id] = "Timeout error"
            else:
                self.seed(motor_id, "torque_enable", int(enabled))
        return failed

    def change_id(self, definition, motor_id: int, new_id: int) -> None:
        if motor_id in self.absent:
            raise RuntimeError("Timeout error")
        self.setups.append(("change_id", definition, motor_id, new_id))

    def change_baudrate(self, definition, motor_id: int, baudrate: int) -> None:
        if motor_id in self.absent:
            raise RuntimeError("Timeout error")
        self.setups.append(("change_baudrate", definition, motor_id, baudrate))


def find_dummy_model(model: str) -> tuple[int, DummyDefinition]:
    for definition in (DUMMY_1, DUMMY_2):
        if model in definition.models:
            return definition.models[model], definition
    raise ValueError(f"Unknown motor model '{model}'.")


class MockMotorsBus(SerialMotorsBus):
    """Bus over dummy servo definitions. It opens a `MockBus` where the real one opens a
    `rustypot.Bus`, and keeps every one it opened in `opened`, oldest first."""

    _find_model = staticmethod(find_dummy_model)

    @cached_property
    def opened(self) -> list[MockBus]:
        return []

    def _bus_class(self):
        def open_bus(*args) -> MockBus:
            self.opened.append(MockBus(*args))
            return self.opened[-1]

        return open_bus
