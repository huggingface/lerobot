#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Serial transport for the motor buses.

The bus owns register names, control tables, normalisation and calibration; the
transport owns the wire. Everything between the two is an address, a length and
some bytes.
"""

from typing import Any, Protocol

from lerobot.utils.import_utils import require_package

PROTOCOL_V1 = "v1"
PROTOCOL_V2 = "v2"


class MotorTransport(Protocol):
    """Register access on a daisy chain of motors, addressed by byte offset."""

    baudrate: int

    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def set_baudrate(self, baudrate: int) -> None: ...

    def set_timeout(self, timeout_ms: int) -> None: ...

    def read(self, motor_id: int, addr: int, length: int) -> tuple[bytes, int]:
        """Read `length` bytes and the motor's status error byte."""
        ...

    def write(self, motor_id: int, addr: int, data: bytes) -> int:
        """Write `data` and return the motor's status error byte."""
        ...

    def sync_read(self, motor_ids: list[int], addr: int, length: int) -> list[bytes]: ...

    def sync_write(self, motor_ids: list[int], addr: int, data: list[bytes]) -> None: ...


class RustypotTransport:
    """`MotorTransport` backed by rustypot.

    rustypot generates one controller class per servo model, but only its raw
    address-based API is used here, and that API varies by protocol version alone.
    So one controller stands in for a whole motor family, LeRobot keeps its own
    control tables and units, and a single bus can address mixed models -- which
    the typed per-model API could not do, since a controller owns its port.
    """

    def __init__(self, port: str, protocol: str, baudrate: int, timeout_ms: int) -> None:
        require_package("rustypot", extra="rustypot-dep")
        if protocol not in (PROTOCOL_V1, PROTOCOL_V2):
            raise ValueError(f"Unknown protocol {protocol!r}, expected {PROTOCOL_V1!r} or {PROTOCOL_V2!r}.")

        self.port = port
        self.protocol = protocol
        self.baudrate = baudrate
        self.timeout_ms = timeout_ms
        # rustypot generates its controllers at import time, so there is no type to name.
        self._controller: Any = None

    def _controller_class(self):
        import rustypot

        # Any v1 controller speaks to any v1 motor over the raw API, likewise v2.
        return {
            PROTOCOL_V1: rustypot.Sts3215PyController,
            PROTOCOL_V2: rustypot.Xl330PyController,
        }[self.protocol]

    @property
    def is_open(self) -> bool:
        return self._controller is not None

    def open(self) -> None:
        if self._controller is not None:
            return
        try:
            self._controller = self._controller_class()(
                serial_port=self.port,
                baudrate=self.baudrate,
                # LeRobot counts timeouts in milliseconds, rustypot in seconds.
                timeout=self.timeout_ms / 1000,
            )
        except OSError as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e

    def close(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def set_baudrate(self, baudrate: int) -> None:
        self._reopen(baudrate=baudrate)

    def set_timeout(self, timeout_ms: int) -> None:
        self._reopen(timeout_ms=timeout_ms)

    def _reopen(self, baudrate: int | None = None, timeout_ms: int | None = None) -> None:
        """rustypot fixes both at build time, so changing either means a new controller."""
        baudrate = self.baudrate if baudrate is None else baudrate
        timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        if baudrate == self.baudrate and timeout_ms == self.timeout_ms and self.is_open:
            return
        was_open = self.is_open
        self.close()
        self.baudrate, self.timeout_ms = baudrate, timeout_ms
        if was_open:
            self.open()

    def read(self, motor_id: int, addr: int, length: int) -> tuple[bytes, int]:
        data, error = self._controller.read_raw_data_with_error(motor_id, addr, length)
        return bytes(data), error

    def write(self, motor_id: int, addr: int, data: bytes) -> int:
        return self._controller.write_raw_data_with_error(motor_id, addr, list(data))

    def sync_read(self, motor_ids: list[int], addr: int, length: int) -> list[bytes]:
        return [bytes(v) for v in self._controller.sync_read_raw_data(motor_ids, addr, length)]

    def sync_write(self, motor_ids: list[int], addr: int, data: list[bytes]) -> None:
        self._controller.sync_write_raw_data(motor_ids, addr, [list(d) for d in data])
