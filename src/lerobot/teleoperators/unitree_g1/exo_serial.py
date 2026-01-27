"""Exoskeleton serial communication and minimal ExoskeletonArm class."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import serial

from .exo_calib import ExoskeletonCalibration, exo_raw_to_angles, run_exo_calibration

logger = logging.getLogger(__name__)

SENSOR_COUNT = 16


def parse_raw16(line: bytes) -> list[int] | None:
    try:
        parts = line.decode("utf-8", errors="ignore").split()
        if len(parts) < SENSOR_COUNT:
            return None
        return [int(x) for x in parts[:SENSOR_COUNT]]
    except Exception:
        return None


@dataclass
class ExoskeletonArm:
    port: str
    baud_rate: int = 115200
    calibration_fpath: Path | None = None
    side: str = ""

    _ser: serial.Serial | None = None
    calibration: ExoskeletonCalibration | None = None

    def __post_init__(self):
        if self.calibration_fpath and self.calibration_fpath.is_file():
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
        if not self.calibration_fpath:
            return
        try:
            data = json.loads(self.calibration_fpath.read_text())
            self.calibration = ExoskeletonCalibration.from_dict(data)
            logger.info(f"loaded calibration: {self.calibration_fpath}")
        except Exception as e:
            logger.warning(f"failed to load calibration: {e}")

    def read_raw(self) -> list[int] | None:
        """read latest sample; if buffer is backed up, keep only the newest."""
        ser = self._ser
        if not ser:
            return None

        last = None
        while ser.in_waiting:
            b = ser.readline()
            if not b:
                break
            v = parse_raw16(b)
            if v is not None:
                last = v

        if last is None:
            b = ser.readline()
            if b:
                last = parse_raw16(b)

        return last

    def get_angles(self) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError("exoskeleton not calibrated")
        raw = self.read_raw()
        return {} if raw is None else exo_raw_to_angles(raw, self.calibration)

    def calibrate(self) -> None:
        ser = self._ser
        if not ser:
            raise RuntimeError("connect before calibrating")
        self.calibration = run_exo_calibration(ser, self.side, save_path=self.calibration_fpath)
