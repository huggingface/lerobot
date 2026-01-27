"""Exoskeleton serial communication and minimal ExoskeletonArm class."""

import json
import logging
from pathlib import Path

import serial

from .exo_calib import (
    SENSOR_COUNT,
    ExoskeletonCalibration,
    exo_raw_to_angles,
    run_exo_calibration,
)

logger = logging.getLogger(__name__)


def parse_raw16(line: str) -> list[int] | None:
    """Parse a line of 16 space-separated ADC values."""
    parts = line.strip().split()
    if len(parts) < SENSOR_COUNT:
        return None
    try:
        return [int(x) for x in parts[:SENSOR_COUNT]]
    except ValueError:
        return None


class ExoskeletonArm:
    """
    Reads raw sensor data from an exoskeleton arm over serial
    and converts to joint angles using calibration.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        calibration_fpath: Path | None = None,
        side: str = "",
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.side = side
        self.calibration_fpath = calibration_fpath
        self.calibration: ExoskeletonCalibration | None = None

        self._ser: serial.Serial | None = None
        self._connected = False
        self._calibrated = False

        if calibration_fpath and calibration_fpath.is_file():
            self._load_calibration()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated and self.calibration is not None

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the exoskeleton serial port."""
        if self._connected:
            return

        try:
            self._ser = serial.Serial(self.port, self.baud_rate, timeout=0.02)
            self._ser.reset_input_buffer()
            self._connected = True
            logger.info(f"Connected to exoskeleton at {self.port}")
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

        if calibrate and not self.is_calibrated:
            self.calibrate()

    def disconnect(self) -> None:
        """Disconnect from the serial port."""
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self._connected = False

    def _load_calibration(self) -> None:
        """Load calibration from file."""
        if self.calibration_fpath is None:
            return
        try:
            with open(self.calibration_fpath) as f:
                data = json.load(f)
            self.calibration = ExoskeletonCalibration.from_dict(data)
            self._calibrated = True
            logger.info(f"Loaded calibration from {self.calibration_fpath}")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")
            self._calibrated = False

    def read_raw(self) -> list[int] | None:
        """Read latest raw16 sample, draining the buffer."""
        if self._ser is None:
            return None

        last = None
        while self._ser.in_waiting > 0:
            b = self._ser.readline()
            if not b:
                break
            raw16 = parse_raw16(b.decode("utf-8", errors="ignore"))
            if raw16 is not None:
                last = raw16

        if last is None:
            b = self._ser.readline()
            if b:
                last = parse_raw16(b.decode("utf-8", errors="ignore"))

        return last

    def get_angles(self) -> dict[str, float]:
        """Get current joint angles in radians."""
        if not self.is_calibrated or self.calibration is None:
            raise RuntimeError("Exoskeleton not calibrated")

        raw = self.read_raw()
        if raw is None:
            return {}

        return exo_raw_to_angles(raw, self.calibration)

    def calibrate(self) -> None:
        """Run interactive calibration."""
        if not self._connected or self._ser is None:
            raise RuntimeError("Must be connected before calibrating")

        self.calibration = run_exo_calibration(
            self._ser,
            self.side,
            save_path=self.calibration_fpath,
        )
        self._calibrated = True

