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

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import serial

from .exo_calib import ExoskeletonCalibration, exo_raw_to_angles, run_exo_calibration

logger = logging.getLogger(__name__)


def parse_raw16(line: bytes) -> list[int] | None:
    try:
        parts = line.decode("utf-8", errors="ignore").split()
        if len(parts) < 16:
            return None
        return [int(x) for x in parts[:16]]
    except Exception:
        return None


def read_raw_from_serial(ser) -> list[int] | None:
    """Read latest sample from serial; if buffer is backed up, keep only the newest."""
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


@dataclass
class ExoskeletonArm:
    port: str
    calibration_fpath: Path
    side: str
    baud_rate: int = 115200

    _ser: serial.Serial | None = None
    calibration: ExoskeletonCalibration | None = None

    def __post_init__(self):
        if self.calibration_fpath.is_file():
            self._load_calibration()

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", False)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def connect(self, calibrate: bool = True) -> None:
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
        if not self._ser:
            return None
        return read_raw_from_serial(self._ser)

    def get_angles(self) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError("exoskeleton not calibrated")
        raw = self.read_raw()
        return {} if raw is None else exo_raw_to_angles(raw, self.calibration)

    def calibrate(self) -> None:
        ser = self._ser
        self.calibration = run_exo_calibration(ser, self.side, self.calibration_fpath)
