import json
import logging
from pathlib import Path
from typing import Any, Dict

from lerobot.motors.motors_bus import MotorCalibration


logger = logging.getLogger(__name__)


class SO100LeaderCalibrator:
    """Handles calibration operations for SO100 leader teleoperators."""

    def __init__(self, teleop):
        self.teleop = teleop

    def default_calibrate(self, reversed: bool = False) -> Dict[str, MotorCalibration]:
        """Perform default calibration."""

        homing_offsets = self._initialize_calibration(reversed)

        min_ranges = {}
        max_ranges = {}
        default_calibration = self._load_default_calibration(reversed)
        for motor, m in self.teleop.bus.motors.items():
            min_ranges[motor] = default_calibration[motor]["range_min"]
            max_ranges[motor] = default_calibration[motor]["range_max"]

        self.teleop.calibration = self._create_calibration_dict(homing_offsets, min_ranges, max_ranges)
        self.teleop.bus.write_calibration(self.teleop.calibration)
        self._save_calibration()
        logger.info(f"Default calibration completed and saved to {self.teleop.calibration_fpath}")
        return self.teleop.calibration

    def _initialize_calibration(self, reversed: bool = False) -> Dict[str, int]:
        """Initialize the calibration of the teleop."""

        homing_offsets = self.teleop.bus.set_position_homings({
            "shoulder_pan": 2047 if reversed else 2048,
            "shoulder_lift": 3325 if reversed else 770,
            "elbow_flex": 1000 if reversed else 3095,
            "wrist_flex": 1335 if reversed else 2760,
            "wrist_roll": 2020 if reversed else 2085,
            "gripper": 1190 if reversed else 2905
        })
        return homing_offsets

    def _create_calibration_dict(self, homing_offsets: Dict[str, int],
                                range_mins: Dict[str, Any], range_maxes: Dict[str, int] = None) -> Dict[str, MotorCalibration]:

        calibration = {}
        for motor, m in self.teleop.bus.motors.items():
            drive_mode = 0

            range_min = range_mins[motor]
            range_max = range_maxes[motor]

            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_mode,
                homing_offset=homing_offsets[motor],
                range_min=range_min,
                range_max=range_max,
            )
        return calibration

    def _load_default_calibration(self, reversed: bool = False) -> Dict[str, Any]:
        """Load default calibration from file."""
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        calibration_dir = current_dir.parent / "so100_leader"
        calibration_file = calibration_dir / "default_calibration.json"

        # Create the calibration directory if it doesn't exist
        calibration_dir.mkdir(parents=True, exist_ok=True)

        # If the calibration file doesn't exist, create it with default values
        if not calibration_file.exists():
            logger.info(f"Calibration file {calibration_file} not found. Creating default calibration...")
            default_calibration = self._create_default_calibration(reversed)
            with open(calibration_file, "w") as f:
                json.dump(default_calibration, f, indent=4)
            logger.info(f"Created default calibration file: {calibration_file}")

        with open(calibration_file, "r") as f:
            return json.load(f)

    def _create_default_calibration(self, reversed: bool = False) -> Dict[str, Any]:
        """Create default calibration data for the robot."""

        return {
            "shoulder_pan": {
                "id": 1,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 1000,
                "range_max": 3095
            },
            "shoulder_lift": {
                "id": 2,
                "drive_mode": 1,
                "homing_offset": 0,
                "range_min": 800,
                "range_max": 3295
            },
            "elbow_flex": {
                "id": 3,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 850,
                "range_max": 3345
            },
            "wrist_flex": {
                "id": 4,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 750,
                "range_max": 3245
            },
            "wrist_roll": {
                "id": 5,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095
            },
            "gripper": {
                "id": 6,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 2023,
                "range_max": 3500
            }
        }

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        self.teleop.bus.write_calibration(self.teleop.calibration)
        self.teleop._save_calibration()
