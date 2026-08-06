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

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

from .exo_calib import ExoskeletonCalibration, exo_raw_to_angles, run_exo_calibration

logger = logging.getLogger(__name__)


def parse_raw16(line: bytes) -> list[int] | None:
    """Parse one line of exoskeleton telemetry into 16 raw ADC channel readings.

    Args:
        line (`bytes`):
            One raw line read from the exoskeleton's serial port, expected to contain 16
            whitespace-separated integers (sin/cos pairs for each sensed joint, plus joystick channels).

    Returns:
        `list[int] | None`: The 16 raw ADC values in channel order, or `None` if the line is malformed or
        has fewer than 16 values.
    """
    try:
        parts = line.decode("utf-8", errors="ignore").split()
        if len(parts) < 16:
            return None
        return [int(x) for x in parts[:16]]
    except (ValueError, IndexError):
        return None


def read_raw_from_serial(ser) -> list[int] | None:
    """Read the latest sample from serial; if the input buffer is backed up, keep only the newest.

    Draining the buffer down to the newest line keeps teleoperation responsive to the exoskeleton's
    current pose instead of replaying a queue of stale samples.

    Args:
        ser (`serial.Serial`):
            Open serial connection to the exoskeleton's sensor board.

    Returns:
        `list[int] | None`: The most recently parsed sample, or `None` if no valid line was available.
    """
    try:
        last = None
        while ser.in_waiting > 0:
            b = ser.readline()
            if not b:
                break
            raw16 = parse_raw16(b)
            if raw16 is not None:
                last = raw16
        if last is None:
            b = ser.readline()
            if b:
                last = parse_raw16(b)
        return last
    except serial.SerialException as e:
        logger.warning(f"Serial read error: {e}")
        return None


@dataclass
class ExoskeletonArm:
    """Serial link and calibration state for one exoskeleton arm (left or right).

    Wraps the raw serial connection to the arm's sensor board and converts its hall-effect sensor readings
    into calibrated joint angles via `get_angles`, once a calibration has been loaded or produced by
    `calibrate`.

    Args:
        port (`str`):
            Serial port the arm's sensor board is connected to, e.g. `/dev/ttyUSB0`.
        calibration_fpath (`Path`):
            Path to the JSON file used to load and save this arm's calibration.
        side (`str`):
            Which arm this is, `"left"` or `"right"`. Used to label saved calibration data and log
            messages.
        baud_rate (`int`, *optional*, defaults to 115200):
            Baud rate for the serial connection.
        calibration (`ExoskeletonCalibration | None`, *optional*):
            Calibration data for this arm. Loaded automatically from `calibration_fpath` if that file
            exists; otherwise populated by calling `calibrate`.
    """

    port: str
    calibration_fpath: Path
    side: str
    baud_rate: int = 115200

    _ser: serial.Serial | None = None
    calibration: ExoskeletonCalibration | None = None

    def __post_init__(self):
        """Check that `pyserial` is installed and load an existing calibration file, if any."""
        require_package("pyserial", extra="unitree_g1", import_name="serial")
        if self.calibration_fpath.is_file():
            self._load_calibration()

    @property
    def is_connected(self) -> bool:
        """Whether the serial connection to the arm's sensor board is open.

        Returns:
            `bool`: `True` if the serial port has been opened and not yet closed.
        """
        return self._ser is not None and getattr(self._ser, "is_open", False)

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration data is available for this arm.

        Returns:
            `bool`: `True` if a calibration has been loaded from disk or produced by `calibrate`.
        """
        return self.calibration is not None

    def connect(self, calibrate: bool = True) -> None:
        """Open the serial connection to the arm's sensor board.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to run `calibrate` automatically after connecting if no calibration is loaded yet.

        Raises:
            ConnectionError: If the serial port cannot be opened.
        """
        if self.is_connected:
            return
        try:
            self._ser = serial.Serial(self.port, self.baud_rate, timeout=0.02)
            self._ser.reset_input_buffer()
            logger.info(f"connected: {self.port}")
        except serial.SerialException as e:
            raise ConnectionError(f"failed to connect to {self.port}: {e}") from e

        if calibrate and not self.is_calibrated:
            self.calibrate()

    def disconnect(self) -> None:
        """Close the serial connection to the arm's sensor board, if open."""
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def _load_calibration(self) -> None:
        try:
            data = json.loads(self.calibration_fpath.read_text())
            self.calibration = ExoskeletonCalibration.from_dict(data)
            logger.info(f"loaded calibration: {self.calibration_fpath}")
        except Exception as e:
            logger.warning(f"failed to load calibration: {e}")

    def read_raw(self) -> list[int] | None:
        """Read the arm's latest raw ADC sample.

        Returns:
            `list[int] | None`: The 16 raw ADC channel values, or `None` if the arm is not connected or no
            valid sample was available.
        """
        if not self._ser:
            return None
        return read_raw_from_serial(self._ser)

    def get_angles(self) -> dict[str, float]:
        """Read the arm's current sensor sample and convert it to calibrated joint angles.

        Returns:
            `dict[str, float]`: Joint name to angle in radians, or an empty dict if no sample was
            available on the serial link.

        Raises:
            RuntimeError: If the arm has not been calibrated yet.
        """
        if not self.calibration:
            raise RuntimeError("exoskeleton not calibrated")
        raw = self.read_raw()
        return {} if raw is None else exo_raw_to_angles(raw, self.calibration)

    def calibrate(self) -> None:
        """Run the interactive per-joint calibration procedure and store its result.

        Delegates to `run_exo_calibration`, which walks the operator through moving each joint through
        its range and holding a zero pose, then saves the resulting ellipse fits and zero offsets to
        `calibration_fpath`.

        Raises:
            RuntimeError: If the arm is not connected.
        """
        if not self.is_connected:
            raise RuntimeError("Cannot calibrate: exoskeleton not connected")
        self.calibration = run_exo_calibration(self._ser, self.side, self.calibration_fpath)
