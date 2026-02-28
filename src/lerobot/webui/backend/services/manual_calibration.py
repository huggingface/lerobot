"""Manual calibration service for SO101 robot arms.

Provides direct motor bus access for the two-step manual calibration flow:
1. Homing — user places arm in middle position, then homing offsets are set
2. Range recording — user moves joints while encoder min/max/current are tracked live
"""

import json
import logging
from pathlib import Path
from typing import Optional

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

logger = logging.getLogger(__name__)

# Motor definitions matching SO101Follower / SO101Leader
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}


class ManualCalibrationService:
    """Service for step-by-step manual calibration via WebSocket."""

    def __init__(self):
        self.calibration_base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"

    def create_bus(self, port: str) -> FeetechMotorsBus:
        """Create and connect a FeetechMotorsBus with all 6 SO101 motors."""
        motors = {}
        for name in MOTOR_NAMES:
            norm_mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
            motors[name] = Motor(MOTOR_IDS[name], "sts3215", norm_mode)
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()
        bus.disable_torque()
        for motor in MOTOR_NAMES:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        return bus

    def set_homing(self, bus: FeetechMotorsBus) -> dict[str, int]:
        """Set homing offsets so current position maps to encoder midpoint.

        This is step 1 of calibration — the arm should be in its middle position.

        Returns:
            Dict of motor_name → homing_offset value.
        """
        return bus.set_half_turn_homings()

    def read_positions(self, bus: FeetechMotorsBus) -> dict[str, int]:
        """Read current encoder positions for all motors.

        Returns:
            Dict of motor_name → raw encoder value.
        """
        return bus.sync_read("Present_Position", list(bus.motors.keys()), normalize=False)

    def save_calibration(
        self,
        calibration_data: dict,
        device_type: str,
        robot_type: str,
        device_id: str,
    ) -> Path:
        """Save calibration results to the standard calibration file.

        Merges into existing file if present.

        Returns:
            Path to the saved calibration file.
        """
        category = "robots" if device_type == "robot" else "teleoperators"
        cal_dir = self.calibration_base_path / category / robot_type
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_path = cal_dir / f"{device_id}.json"

        existing = {}
        if cal_path.exists():
            with open(cal_path) as f:
                existing = json.load(f)

        existing.update(calibration_data)

        with open(cal_path, "w") as f:
            json.dump(existing, f, indent=4)

        logger.info(f"Calibration saved to {cal_path}")
        return cal_path

    def write_calibration_to_motors(self, port: str, calibration_data: dict):
        """Write calibration values to motor EEPROM."""
        motor_names = list(calibration_data.keys())
        motors = {}
        for name in motor_names:
            norm_mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
            motors[name] = Motor(MOTOR_IDS[name], "sts3215", norm_mode)
        bus = FeetechMotorsBus(port=port, motors=motors)

        try:
            bus.connect()
            cal_dict = {}
            for name, data in calibration_data.items():
                cal_dict[name] = MotorCalibration(
                    id=data["id"],
                    drive_mode=data["drive_mode"],
                    homing_offset=data["homing_offset"],
                    range_min=data["range_min"],
                    range_max=data["range_max"],
                )
            bus.write_calibration(cal_dict)
            logger.info(f"Calibration written to motors: {motor_names}")
        finally:
            if bus.is_connected:
                bus.disconnect()
