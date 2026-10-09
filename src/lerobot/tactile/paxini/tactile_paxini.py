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

"""PaXini fingertip tactile sensor implementation.

Speaks the PaXini serial hub wire format directly (pyserial), no vendor SDK
required. The hub streams auto frames containing one resultant 3D force vector
per connected fingertip sensor.

Wire format (little-endian):
    Register request:  55 AA | 00 | func | addr(2) | count(2) | LRC
    Register response: AA 55 | 00 | func | addr(2) | nbytes(2) | data | LRC
    Auto stream frame: AA 56 | 00 | eff_len(2) | payload | LRC
    LRC = two's complement of the low byte of the sum of all preceding bytes.
    Resultant payload: 6 bytes per sensor; each axis occupies a 2-byte slot with
    data in the low byte (int8 Fx, int8 Fy, uint8 Fz), 0.1 N per LSB.
"""

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.import_utils import is_package_available

from ..tactile import TactileSensor
from .configuration_paxini import PaxiniTactileConfig

logger = logging.getLogger(__name__)

HEADER_REQUEST = bytes([0x55, 0xAA])
HEADER_RESPONSE = bytes([0xAA, 0x55])
HEADER_AUTO = bytes([0xAA, 0x56])
FUNC_READ = 0x03
FUNC_WRITE = 0x10
ADDR_CONNECTED_SENSORS = 0x0010
ADDR_AUTO_DATA_TYPE = 0x0016
ADDR_AUTO_ENABLE = 0x0017
AUTO_DATA_RESULTANT = 0x01
BYTES_PER_RESULTANT = 6
NEWTON_PER_LSB = 0.1
# Hardware-fixed (byte_idx, bit_pos) of each fingertip slot in the
# connected-sensors register.
SLOT_CONNECTED_BITS = [(0, 2), (0, 6), (1, 2), (1, 6), (2, 2)]


def checksum(frame: bytes) -> int:
    """LRC checksum: two's complement of the low byte of the sum."""
    return (0x100 - (sum(frame) & 0xFF)) & 0xFF


def build_read_request(address: int, count: int) -> bytes:
    body = HEADER_REQUEST + bytes([0x00, FUNC_READ]) + address.to_bytes(2, "little")
    body += count.to_bytes(2, "little")
    return body + bytes([checksum(body)])


def build_write_request(address: int, data: bytes) -> bytes:
    body = HEADER_REQUEST + bytes([0x00, FUNC_WRITE]) + address.to_bytes(2, "little")
    body += len(data).to_bytes(2, "little") + data
    return body + bytes([checksum(body)])


def decode_resultant_payload(payload: bytes, num_sensors: int) -> NDArray[np.float64]:
    """Decode an auto-stream resultant payload into an (N, 3) force array in Newtons."""
    expected = num_sensors * BYTES_PER_RESULTANT
    if len(payload) != expected:
        raise ValueError(f"Resultant payload size mismatch: expected {expected}, got {len(payload)}")
    out = np.empty((num_sensors, 3), dtype=np.float64)
    for i in range(num_sensors):
        off = i * BYTES_PER_RESULTANT
        fx, fy, fz = payload[off], payload[off + 2], payload[off + 4]
        out[i, 0] = (fx - 256 if fx > 127 else fx) * NEWTON_PER_LSB
        out[i, 1] = (fy - 256 if fy > 127 else fy) * NEWTON_PER_LSB
        out[i, 2] = fz * NEWTON_PER_LSB
    return out


def parse_connected_slots(register: bytes) -> list[int]:
    """Return indices of fingertip slots reported connected by the hub."""
    slots = []
    for slot, (byte_idx, bit_pos) in enumerate(SLOT_CONNECTED_BITS):
        if byte_idx < len(register) and register[byte_idx] & (1 << bit_pos):
            slots.append(slot)
    return slots


class PaxiniTactile(TactileSensor):
    """PaXini fingertip force sensor hub implementation.

    Streams per-fingertip resultant 3D forces from up to five taxel-array
    sensors connected to one serial hub.

    Example:
        ```python
        from lerobot.tactile.paxini import PaxiniTactile, PaxiniTactileConfig

        config = PaxiniTactileConfig(port="/dev/ttyACM1", num_points=5)
        with PaxiniTactile(config) as sensor:
            data = sensor.read()  # (5, 3) array
        ```
    """

    def __init__(self, config: PaxiniTactileConfig):
        super().__init__(config)
        self.config = config
        self._serial: Any = None
        self._is_connected = False
        self._frame_queue: Queue[NDArray[np.float64]] = Queue(maxsize=2)
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame: NDArray[np.float64] | None = None
        self._latest_ts: float = 0.0
        self._tare_offset: NDArray[np.float64] | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect serial ports that may host a PaXini hub."""
        if not is_package_available("serial"):
            logger.warning("pyserial is not installed. Cannot detect PaXini sensors.")
            return []
        from serial.tools import list_ports

        found = []
        for p in list_ports.comports():
            found.append({"id": p.device, "type": "paxini", "description": p.description})
        return found

    def connect(self, warmup: bool = True) -> None:
        """Open the serial port, enable the resultant auto stream and start capture.

        Raises:
            ImportError: If pyserial is not installed.
            ConnectionError: If already connected or the hub does not respond.
        """
        if self._is_connected:
            raise ConnectionError("Sensor is already connected.")
        try:
            import serial
        except ImportError as e:
            raise ImportError(
                "pyserial is required for PaXini sensors. Install with: pip install 'lerobot[paxini]'"
            ) from e

        self._serial = serial.Serial(self.config.port, self.config.baudrate, timeout=0.1)
        # Stop any previous stream, then configure resultant-only auto streaming.
        self._write_register(ADDR_AUTO_ENABLE, bytes([0x00]))
        connected = parse_connected_slots(self._read_register(ADDR_CONNECTED_SENSORS, 4))
        if len(connected) < self.config.num_points:
            self._serial.close()
            raise ConnectionError(
                f"Hub reports {len(connected)} connected fingertip sensors "
                f"({connected}), but num_points={self.config.num_points}."
            )
        self._write_register(ADDR_AUTO_DATA_TYPE, bytes([AUTO_DATA_RESULTANT]))
        self._write_register(ADDR_AUTO_ENABLE, bytes([0x01]))

        self._is_connected = True
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        if warmup:
            deadline = time.perf_counter() + self.config.timeout_ms / 1000.0
            while self._latest_frame is None and time.perf_counter() < deadline:
                time.sleep(0.01)
        if self.config.tare_on_connect:
            self.tare()
        logger.info(f"PaXini hub connected on {self.config.port} ({self.config.num_points} sensors)")

    def disconnect(self) -> None:
        """Stop streaming and release the serial port."""
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if self._serial is not None:
            try:
                self._write_register(ADDR_AUTO_ENABLE, bytes([0x00]))
            except Exception:  # nosec B110
                pass
            self._serial.close()
            self._serial = None
        self._is_connected = False
        logger.info("PaXini hub disconnected")

    def read(self) -> NDArray[np.float64]:
        """Capture a single tactile frame synchronously.

        Returns:
            np.ndarray: Force data with shape (num_points, 3), Newtons.
        """
        if not self._is_connected:
            raise ConnectionError("Sensor is not connected. Call connect() first.")
        try:
            frame = self._frame_queue.get(timeout=self.config.timeout_ms / 1000.0)
        except Empty as e:
            raise TimeoutError(f"No frame received within {self.config.timeout_ms}ms") from e
        return self._apply_tare(frame)

    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Return the most recent new tactile frame."""
        if not self._is_connected:
            raise ConnectionError("Sensor is not connected.")
        try:
            frame = self._frame_queue.get(timeout=timeout_ms / 1000.0)
        except Empty as e:
            raise TimeoutError(f"No frame received within {timeout_ms}ms") from e
        return self._apply_tare(frame)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.float64]:
        """Return the most recent frame immediately (non-blocking)."""
        if not self._is_connected:
            raise ConnectionError("Sensor is not connected.")
        if self._latest_frame is None:
            raise RuntimeError("No frames captured yet.")
        age_ms = (time.perf_counter() - self._latest_ts) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(f"Latest frame is {age_ms:.0f}ms old (max {max_age_ms}ms)")
        return self._apply_tare(self._latest_frame)

    def tare(self, num_samples: int = 10) -> None:
        """Zero the readings by averaging the next ``num_samples`` frames."""
        if not self._is_connected:
            raise ConnectionError("Cannot tare: sensor is not connected")
        samples = []
        for i in range(num_samples):
            try:
                samples.append(self._frame_queue.get(timeout=self.config.timeout_ms / 1000.0))
            except Empty as e:
                raise RuntimeError(f"Failed to capture tare sample {i}") from e
        self._tare_offset = np.mean(samples, axis=0)

    def _apply_tare(self, frame: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._tare_offset is not None:
            return frame - self._tare_offset
        return frame

    def _read_register(self, address: int, count: int) -> bytes:
        self._serial.reset_input_buffer()
        self._serial.write(build_read_request(address, count))
        frame = self._read_response_frame()
        nbytes = int.from_bytes(frame[6:8], "little")
        return frame[8 : 8 + nbytes]

    def _write_register(self, address: int, data: bytes) -> None:
        self._serial.reset_input_buffer()
        self._serial.write(build_write_request(address, data))
        self._read_response_frame()

    def _read_response_frame(self) -> bytes:
        """Read one AA 55 response frame, skipping any interleaved auto frames."""
        deadline = time.perf_counter() + self.config.timeout_ms / 1000.0
        while time.perf_counter() < deadline:
            header = self._resync_to_header()
            if header == HEADER_AUTO:
                meta = self._read_exact(3)
                eff_len = int.from_bytes(meta[1:3], "little")
                self._read_exact(eff_len + 1)  # payload + LRC
                continue
            meta = self._read_exact(6)
            nbytes = int.from_bytes(meta[4:6], "little")
            body = self._read_exact(nbytes + 1)
            frame = header + meta + body
            if frame[-1] != checksum(frame[:-1]):
                raise OSError("Register response LRC mismatch")
            return frame
        raise TimeoutError("No register response from PaXini hub")

    def _resync_to_header(self) -> bytes:
        """Scan the byte stream until a response or auto-frame header is found."""
        prev = b""
        deadline = time.perf_counter() + self.config.timeout_ms / 1000.0
        while time.perf_counter() < deadline:
            b1 = self._serial.read(1)
            if not b1:
                continue
            pair = prev + b1
            if pair in (HEADER_RESPONSE, HEADER_AUTO):
                return pair
            prev = b1
        raise TimeoutError("Could not sync to PaXini frame header")

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        deadline = time.perf_counter() + self.config.timeout_ms / 1000.0
        while len(buf) < n and time.perf_counter() < deadline:
            chunk = self._serial.read(n - len(buf))
            if chunk:
                buf += chunk
        if len(buf) < n:
            raise TimeoutError(f"Expected {n} bytes, got {len(buf)}")
        return buf

    def _capture_loop(self) -> None:
        """Background thread: parse auto-stream frames into the queue."""
        expected = self.config.num_points * BYTES_PER_RESULTANT
        while not self._stop_event.is_set():
            try:
                header = self._resync_to_header()
                if header != HEADER_AUTO:
                    # Register response consumed by main thread paths only pre-stream;
                    # here we just skip it.
                    meta = self._read_exact(6)
                    nbytes = int.from_bytes(meta[4:6], "little")
                    self._read_exact(nbytes + 1)
                    continue
                meta = self._read_exact(3)
                eff_len = int.from_bytes(meta[1:3], "little")
                payload_and_lrc = self._read_exact(eff_len + 1)
                payload, lrc = payload_and_lrc[:-1], payload_and_lrc[-1]
                if checksum(header + meta + payload) != lrc:
                    logger.debug("Auto frame LRC mismatch, dropping frame")
                    continue
                if len(payload) < expected:
                    continue
                frame = decode_resultant_payload(payload[:expected], self.config.num_points)
                self._latest_frame = frame
                self._latest_ts = time.perf_counter()
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:  # nosec B110
                        pass
                self._frame_queue.put_nowait(frame)
            except TimeoutError:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning(f"PaXini capture error: {e}")
