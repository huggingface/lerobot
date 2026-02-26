"""Auto-calibration service for SO101 robot arms.

Instead of requiring the user to manually move joints through their range,
this service drives each servo programmatically to find its physical limits
by monitoring encoder position changes.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

logger = logging.getLogger(__name__)

# STS3215 servo constants
ENCODER_RESOLUTION = 4096  # 12-bit encoder, 0-4095
HALF_TURN = ENCODER_RESOLUTION // 2 - 1  # 2047

# Motor definitions matching SO101Follower
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}

# Auto-calibration tuning
STALL_THRESHOLD = 2  # If position doesn't change by more than this, consider stalled
STALL_COUNT_LIMIT = 5  # Number of consecutive stall reads before declaring a limit
POLL_INTERVAL = 0.05  # Seconds between position polls while waiting for stall
GRIPPER_MAX_TORQUE = 500  # 50% torque limit for gripper safety


@dataclass
class CalibrationProgress:
    motor_name: str
    phase: str  # "finding_min", "finding_max", "done"
    current_position: int
    range_min: Optional[int] = None
    range_max: Optional[int] = None
    progress_pct: float = 0.0


class AutoCalibrationService:
    """Service that auto-calibrates servos by driving them to their physical limits."""

    def __init__(self):
        self.calibration_base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
        self._bus: Optional[FeetechMotorsBus] = None
        self._cancel_requested = False

    def _create_bus(self, port: str, motor_names: list[str]) -> FeetechMotorsBus:
        """Create a FeetechMotorsBus for the given motors."""
        motors = {}
        for name in motor_names:
            norm_mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
            motors[name] = Motor(MOTOR_IDS[name], "sts3215", norm_mode)
        return FeetechMotorsBus(port=port, motors=motors)

    async def auto_calibrate_motor(
        self,
        port: str,
        motor_name: str,
        on_progress: Optional[Callable[[CalibrationProgress], None]] = None,
        **kwargs,
    ) -> dict:
        """Auto-calibrate a single motor by finding its physical limits.

        Algorithm:
        1. Disable torque, reset calibration, set homing offset
        2. Enable torque, command goal to 0 — servo drives to physical min, poll until stall
        3. Command goal to 4095 — servo drives to physical max, poll until stall
        4. Return to midpoint, disable torque

        Args:
            port: Serial port path (e.g. /dev/tty.usbmodemXXX)
            motor_name: Name of the motor to calibrate (e.g. "gripper")
            on_progress: Optional callback for progress updates

        Returns:
            Dict with calibration results: {motor_name: {id, drive_mode, homing_offset, range_min, range_max}}
        """
        if motor_name not in MOTOR_IDS:
            raise ValueError(f"Unknown motor: {motor_name}. Must be one of {MOTOR_NAMES}")

        self._cancel_requested = False
        bus = self._create_bus(port, [motor_name])
        self._bus = bus

        try:
            bus.connect()

            # Setup: disable torque, position mode, reset calibration
            bus.disable_torque(motor_name)
            bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)
            bus.reset_calibration([motor_name])

            # Set homing offset so current position maps to midpoint
            actual_position = bus.read("Present_Position", motor_name, normalize=False)
            homing_offset = actual_position - HALF_TURN
            bus.write("Homing_Offset", motor_name, homing_offset)

            # Configure motor safety & PID
            if motor_name == "gripper":
                bus.write("Max_Torque_Limit", motor_name, GRIPPER_MAX_TORQUE)
                bus.write("Protection_Current", motor_name, 250)
                bus.write("Overload_Torque", motor_name, 25)
            bus.write("P_Coefficient", motor_name, 16)
            bus.write("I_Coefficient", motor_name, 0)
            bus.write("D_Coefficient", motor_name, 32)

            # Enable torque
            bus.enable_torque(motor_name)

            # Find min: command to 0 and wait for stall
            bus.write("Goal_Position", motor_name, 0, normalize=False)
            range_min = await self._wait_for_stall(bus, motor_name, "finding_min", on_progress)

            if self._cancel_requested:
                return {"cancelled": True}

            # Find max: command to 4095 and wait for stall
            bus.write("Goal_Position", motor_name, ENCODER_RESOLUTION - 1, normalize=False)
            range_max = await self._wait_for_stall(bus, motor_name, "finding_max", on_progress)

            if self._cancel_requested:
                return {"cancelled": True}

            # Return to midpoint and release
            midpoint = (range_min + range_max) // 2
            bus.write("Goal_Position", motor_name, midpoint, normalize=False)
            await asyncio.sleep(0.3)
            bus.disable_torque(motor_name)

            self._report_progress(on_progress, CalibrationProgress(
                motor_name=motor_name, phase="done",
                current_position=midpoint,
                range_min=range_min, range_max=range_max,
                progress_pct=100.0,
            ))

            result = {
                motor_name: {
                    "id": MOTOR_IDS[motor_name],
                    "drive_mode": 0,
                    "homing_offset": homing_offset,
                    "range_min": range_min,
                    "range_max": range_max,
                }
            }
            return result

        finally:
            try:
                if bus.is_connected:
                    bus.disable_torque(motor_name)
                    bus.disconnect()
            except Exception:
                pass
            self._bus = None

    async def _wait_for_stall(
        self,
        bus: FeetechMotorsBus,
        motor_name: str,
        phase: str,
        on_progress: Optional[Callable] = None,
    ) -> int:
        """Poll position until the servo stops moving (stall = physical limit reached).

        The goal position should already be set before calling this.
        Waits for the servo to actually start moving before beginning stall detection.

        Returns:
            The encoder position where the servo stalled.
        """
        start_pos = bus.read("Present_Position", motor_name, normalize=False)
        prev_pos = start_pos
        moving = False
        stall_count = 0

        while stall_count < STALL_COUNT_LIMIT:
            if self._cancel_requested:
                return prev_pos

            await asyncio.sleep(POLL_INTERVAL)
            pos = bus.read("Present_Position", motor_name, normalize=False)

            # Wait until servo has actually started moving before detecting stalls
            if not moving:
                if abs(pos - start_pos) > STALL_THRESHOLD * 2:
                    moving = True
                prev_pos = pos
                self._report_progress(on_progress, CalibrationProgress(
                    motor_name=motor_name, phase=phase,
                    current_position=pos,
                    progress_pct=0.0 if phase == "finding_min" else 50.0,
                ))
                continue

            if abs(pos - prev_pos) <= STALL_THRESHOLD:
                stall_count += 1
            else:
                stall_count = 0

            prev_pos = pos

            self._report_progress(on_progress, CalibrationProgress(
                motor_name=motor_name, phase=phase,
                current_position=pos,
                progress_pct=stall_count / STALL_COUNT_LIMIT * 50.0 + (0 if phase == "finding_min" else 50.0),
            ))

        return prev_pos

    def _report_progress(
        self,
        callback: Optional[Callable],
        progress: CalibrationProgress,
    ):
        if callback:
            callback(progress)

    def cancel(self):
        """Request cancellation of the current auto-calibration."""
        self._cancel_requested = True

    def save_calibration(
        self,
        calibration_data: dict,
        device_type: str,
        robot_type: str,
        device_id: str,
    ) -> Path:
        """Save calibration results to the standard calibration file.

        If the file already exists, merges the new motor data into it.
        Otherwise creates a new file.

        Args:
            calibration_data: Dict of {motor_name: {id, drive_mode, homing_offset, range_min, range_max}}
            device_type: "robot" or "teleoperator"
            robot_type: e.g. "so101_follower"
            device_id: e.g. "left_follower"

        Returns:
            Path to the saved calibration file
        """
        category = "robots" if device_type == "robot" else "teleoperators"
        cal_dir = self.calibration_base_path / category / robot_type
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_path = cal_dir / f"{device_id}.json"

        # Load existing calibration if present
        existing = {}
        if cal_path.exists():
            with open(cal_path) as f:
                existing = json.load(f)

        # Merge new data
        existing.update(calibration_data)

        with open(cal_path, "w") as f:
            json.dump(existing, f, indent=4)

        logger.info(f"Calibration saved to {cal_path}")
        return cal_path

    def write_calibration_to_motors(
        self,
        port: str,
        calibration_data: dict,
    ):
        """Write calibration values to motor EEPROM.

        Args:
            port: Serial port
            calibration_data: Dict of {motor_name: {id, drive_mode, homing_offset, range_min, range_max}}
        """
        motor_names = list(calibration_data.keys())
        bus = self._create_bus(port, motor_names)

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
