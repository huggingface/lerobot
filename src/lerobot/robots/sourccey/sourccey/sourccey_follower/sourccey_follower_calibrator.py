# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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
import time
from pathlib import Path
from typing import Any, Dict
from venv import logger

from lerobot.motors.feetech.feetech import OperatingMode
from lerobot.motors.motors_bus import MotorCalibration

class SourcceyFollowerCalibrator:
    """Handles calibration operations for Sourccey robots."""

    def __init__(self, robot):
        self.robot = robot

    def default_calibrate(self, reversed: bool = False) -> Dict[str, MotorCalibration]:
        """Perform default calibration."""

        homing_offsets = self._initialize_calibration(reversed)

        min_ranges = {}
        max_ranges = {}
        default_calibration = self._load_default_calibration(reversed)
        for motor, m in self.robot.bus.motors.items():
            min_ranges[motor] = default_calibration[motor]["range_min"]
            max_ranges[motor] = default_calibration[motor]["range_max"]

        self.robot.calibration = self._create_calibration_dict(homing_offsets, min_ranges, max_ranges)
        self.robot.bus.write_calibration(self.robot.calibration)
        self._save_calibration()
        logger.info(f"Default calibration completed and saved to {self.robot.calibration_fpath}")
        return self.robot.calibration

    def manual_calibrate(self) -> Dict[str, MotorCalibration]:
        """Perform manual calibration with user interaction."""
        if self.robot.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.robot.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.robot.id} to the motors")
                self.robot.bus.write_calibration(self.robot.calibration)
                return self.robot.calibration

        logger.info(f"\nRunning calibration of robot {self.robot.id}")
        self.robot.bus.disable_torque()
        for motor in self.robot.bus.motors:
            self.robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.robot.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.robot.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.robot.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.robot.calibration = self._create_calibration_dict(homing_offsets, range_mins, range_maxes)
        self.robot.bus.write_calibration(self.robot.calibration)
        self._save_calibration()
        print("Calibration saved to", self.robot.calibration_fpath)
        return self.robot.calibration

    def auto_calibrate(self, reversed: bool = False) -> Dict[str, MotorCalibration]:
        """Automatically calibrate the robot using current monitoring to detect mechanical limits.

        This method performs automatic calibration by:
        1. Adjusting calibration so current physical positions become desired logical positions
        2. Detecting mechanical limits using current monitoring (if full_reset=True)
        3. Setting homing offsets to center the range around the middle of detected limits
        4. Writing calibration to motors and saving to file

        WARNING: This process involves moving the robot to find limits.
        Ensure the robot arm is clear of obstacles and people during calibration.
        """
        logger.info(f"Starting automatic calibration of robot {self.robot.id}")
        if (not self.robot.is_calibrated):
            logger.info("Performing preliminary default calibration...")
            self.default_calibrate(reversed)

        # Set all motors to half turn homings except shoulder_lift
        logger.warning("WARNING: Robot will move to detect mechanical limits. Ensure clear workspace!")

        # Step 1: Adjust calibration so current positions become desired logical positions
        logger.info("Adjusting calibration to align current positions with desired logical positions...")
        homing_offsets = self._initialize_calibration(reversed)

        # Step 2: Detect actual mechanical limits using current monitoring
        # Note: Torque will be enabled during limit detection
        logger.info("Detecting mechanical limits using current monitoring...")
        detected_ranges = self._detect_mechanical_limits(reversed)

        # Step 3: Disable torque for safety before setting homing offsets
        logger.info("Disabling torque for safety...")
        self.robot.bus.disable_torque()

        # Step 4: Create calibration dictionary
        self.robot.calibration = self._create_calibration_dict(homing_offsets, detected_ranges["min"], detected_ranges["max"])

        # Step 5: Write calibration to motors and save
        self.robot.bus.write_calibration(self.robot.calibration)
        self._save_calibration()
        logger.info(f"Automatic calibration completed and saved to {self.robot.calibration_fpath}")
        return self.robot.calibration

    def _create_calibration_dict(self, homing_offsets: Dict[str, int],
                                range_mins: Dict[str, Any], range_maxes: Dict[str, int] = None) -> Dict[str, MotorCalibration]:
        calibration = {}
        for motor, m in self.robot.bus.motors.items():
            drive_mode = 1 if motor == "shoulder_lift" or (self.robot.config.orientation == "right" and motor == "gripper") else 0

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

    def _initialize_calibration(self, reversed: bool = False) -> Dict[str, int]:
        """Initialize the calibration of the robot."""
        # Set all motors to half turn homings except shoulder_lift
        homing_offsets = self.robot.bus.set_half_turn_homings()
        shoulder_lift_homing_offset = self.robot.bus.set_position_homings({"shoulder_lift": 296 if reversed else 3800})
        homing_offsets["shoulder_lift"] = shoulder_lift_homing_offset["shoulder_lift"]
        return homing_offsets

    def _load_default_calibration(self, reversed: bool = False) -> Dict[str, Any]:
        """Load default calibration from file."""
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        # Navigate to the sourccey directory where the calibration files are located
        calibration_dir = current_dir.parent / "sourccey"

        if reversed:
            calibration_file = calibration_dir / "left_arm_default_calibration.json"
        else:
            calibration_file = calibration_dir / "right_arm_default_calibration.json"

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
        if reversed:
            # Left arm calibration (IDs 1-6)
            return {
                "shoulder_pan": {
                    "id": 1,
                    "drive_mode": 0,
                    "homing_offset": 2035,
                    "range_min": 970,
                    "range_max": 3095
                },
                "shoulder_lift": {
                    "id": 2,
                    "drive_mode": 1,
                    "homing_offset": 660,
                    "range_min": 0,
                    "range_max": 3800
                },
                "elbow_flex": {
                    "id": 3,
                    "drive_mode": 0,
                    "homing_offset": -1515,
                    "range_min": 200,
                    "range_max": 3800
                },
                "wrist_flex": {
                    "id": 4,
                    "drive_mode": 0,
                    "homing_offset": -1100,
                    "range_min": 850,
                    "range_max": 3350
                },
                "wrist_roll": {
                    "id": 5,
                    "drive_mode": 0,
                    "homing_offset": 975,
                    "range_min": 0,
                    "range_max": 4095
                },
                "gripper": {
                    "id": 6,
                    "drive_mode": 0,
                    "homing_offset": 1640,
                    "range_min": 425,
                    "range_max": 2150
                }
            }
        else:
            # Right arm calibration (IDs 7-12)
            return {
                "shoulder_pan": {
                    "id": 7,
                    "drive_mode": 0,
                    "homing_offset": 2000,
                    "range_min": 925,
                    "range_max": 3050
                },
                "shoulder_lift": {
                    "id": 8,
                    "drive_mode": 1,
                    "homing_offset": 1645,
                    "range_min": 296,
                    "range_max": 4095
                },
                "elbow_flex": {
                    "id": 9,
                    "drive_mode": 0,
                    "homing_offset": 20,
                    "range_min": 275,
                    "range_max": 3850
                },
                "wrist_flex": {
                    "id": 10,
                    "drive_mode": 0,
                    "homing_offset": 1960,
                    "range_min": 770,
                    "range_max": 3350
                },
                "wrist_roll": {
                    "id": 11,
                    "drive_mode": 0,
                    "homing_offset": 1025,
                    "range_min": 75,
                    "range_max": 4020
                },
                "gripper": {
                    "id": 12,
                    "drive_mode": 1,
                    "homing_offset": 515,
                    "range_min": 425,
                    "range_max": 2047
                }
            }

    def _detect_mechanical_limits(self, reversed: bool = False) -> Dict[str, Dict[str, float]]:
        """Detect the mechanical limits of the robot using current monitoring.

        This function moves each motor incrementally while monitoring current draw.
        When a motor hits a mechanical limit, the current will spike, indicating
        the limit has been reached.

        Search ranges:
        - shoulder_lift: 4096 steps in negative direction only, double step size and current threshold
        - All other motors: 2048 steps in both positive and negative directions

        Returns:
            dict[str, dict[str, float]]: Dictionary mapping motor names to their
            detected min/max position limits.
        """
        logger.info("Starting mechanical limit detection...")

        # Enable torque for all motors to allow movement
        self.robot.bus.enable_torque()

        # Get current positions as starting points
        start_positions = self.robot.bus.sync_read("Present_Position", normalize=False)
        reset_positions = start_positions.copy()
        reset_positions['shoulder_lift'] = 1792 if reversed else 2304 # Manually set shoulder_lift to half way position

        # Initialize results dictionary
        detected_ranges = {}

        # Base parameters
        base_step_size = 50
        settle_time = 0.1

        # Motor-specific configuration
        motor_configs = {
            "shoulder_lift": {
                "search_range": 3800,
                "search_step": base_step_size * 2,
                "max_current": self.robot.config.max_current_calibration_threshold * 5,
                "search_positive": reversed,
                "search_negative": not reversed
            },
            "gripper": {
                "search_range": 1664,
                "search_step": base_step_size,
                "max_current": self.robot.config.max_current_calibration_threshold,
                "search_positive": False,
                "search_negative": True
            }
        }

        # Default configuration for all other motors
        default_config = {
            "search_range": 2048,
            "search_step": base_step_size,
            "max_current": self.robot.config.max_current_calibration_threshold,
            "search_positive": True,
            "search_negative": True
        }

        min_ranges = {}
        max_ranges = {}
        detected_ranges = {
            "min": min_ranges,
            "max": max_ranges
        }
        for motor_name in self.robot.bus.motors:
            logger.info(f"Detecting limits for motor: {motor_name}")

            # Get motor-specific configuration or use default
            config = motor_configs.get(motor_name, default_config)

            # Get current position
            start_pos = start_positions[motor_name]
            reset_pos = reset_positions[motor_name]
            min_pos = start_pos
            max_pos = start_pos

            # Determine search order based on reversed flag
            if reversed:
                search_order = [("negative", config["search_negative"]), ("positive", config["search_positive"])]
            else:
                search_order = [("positive", config["search_positive"]), ("negative", config["search_negative"])]

            # Execute searches in the determined order
            for direction, should_search in search_order:
                if not should_search:
                    logger.info(f"  Skipping {direction} direction for {motor_name}")
                    continue

                logger.info(f"  Testing {direction} direction for {motor_name} (range: {config['search_range']})")
                current_pos = start_pos
                steps_taken = 0
                max_steps = config["search_range"] // config["search_step"]

                while steps_taken < max_steps:
                    if direction == "positive":
                        target_pos = current_pos + config["search_step"]
                    else:
                        target_pos = current_pos - config["search_step"]

                    self.robot.bus.write("Goal_Position", motor_name, target_pos, normalize=False)

                    # Wait for movement to settle
                    time.sleep(settle_time)

                    # Check current draw with retry logic
                    current, limit_reached = self._read_calibration_current(motor_name)
                    if current > config["max_current"]:
                        actual_pos = self.robot.bus.read("Present_Position", motor_name, normalize=False)
                        logger.info(f"    Hit {direction} limit for {motor_name} at position {actual_pos} (current: {current}mA)")
                        if direction == "positive":
                            max_pos = actual_pos
                        else:
                            min_pos = actual_pos
                        break

                    current_pos = target_pos
                    steps_taken += 1
                else:
                    logger.info(f"    Reached search range limit ({config['search_range']}) for {motor_name} {direction} direction")
                    actual_pos = self.robot.bus.read("Present_Position", motor_name, normalize=False)
                    if direction == "positive":
                        max_pos = actual_pos
                    else:
                        min_pos = actual_pos

                # Reset to middle position after each direction test
                self._move_calibration_slow(motor_name, reset_pos, duration=3.0)
                time.sleep(settle_time * 5)

            # Store detected range
            detected_ranges["min"][motor_name] = int(min_pos)
            detected_ranges["max"][motor_name] = int(max_pos)

            logger.info(f"  Detected range for {motor_name}: {min_pos} to {max_pos}")

        # Reset all motors to their start positions (Just the shoulder lift is out of position)
        reset_motor = "shoulder_lift"
        self._move_calibration_slow(reset_motor, start_positions[reset_motor], duration=3.0)

        logger.info("Mechanical limit detection completed")
        return detected_ranges

    def _read_calibration_current(self, motor_name: str, max_retries: int = 3, base_delay: float = 0.1) -> tuple[float, bool]:
        """Read the calibration current of the robot with exponential backoff retry.

        Args:
            motor_name: Name of the motor to read current from
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 0.1s)

        Returns:
            Current reading in mA, and a boolean indicating if the limit was reached
        """
        for attempt in range(max_retries + 1):
            try:
                current = self.robot.bus.read("Present_Current", motor_name, normalize=False)
                return current, False
            except Exception as e:
                if "Overload error" in str(e):
                    # Overload error indicates mechanical limit reached
                    logger.info(f"    Hit limit for {motor_name} (overload error)")
                    return 1001, True

                if attempt == max_retries:
                    # Final attempt failed, log error and return default value
                    logger.error(f"Error reading calibration current for {motor_name} after {max_retries + 1} attempts: {e}")
                    return 1001, False
                else:
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {motor_name}: {e}. Retrying in {delay:.3f}s...")
                    time.sleep(delay)

        # This should never be reached, but just in case
        return 1001, False

    def _move_calibration_slow(self, motor_name: str, target_position: float, duration: float = 3.0,
                             steps_per_second: float = 10.0, max_retries: int = 3) -> bool:
        """Move a single motor slowly to target position during calibration.

        This function moves the motor in small steps over the specified duration
        to prevent high g-forces that could damage the robot during calibration.

        Args:
            motor_name: Name of the motor to move
            target_position: Target position to move to
            duration: Time in seconds to complete the movement (default: 3.0s)
            steps_per_second: Number of position updates per second (default: 10Hz)
            max_retries: Maximum retries for each position write (default: 3)

        Returns:
            True if movement completed successfully, False otherwise
        """
        try:
            # Get current position
            current_position = self.robot.bus.read("Present_Position", motor_name, normalize=False)

            # Calculate movement parameters
            total_steps = int(duration * steps_per_second)
            step_time = duration / total_steps

            logger.info(f"Moving {motor_name} from {current_position:.1f} to {target_position:.1f} over {duration}s")

            # Move in small steps
            for step in range(total_steps + 1):  # +1 to include final position
                # Calculate interpolation factor (0 to 1)
                t = step / total_steps

                # Interpolate between current and target position
                interpolated_position = current_position + t * (target_position - current_position)

                # Convert to integer for motor bus
                interpolated_position_int = int(round(interpolated_position))

                # Write position with retry logic
                self.robot.bus.write("Goal_Position", motor_name, interpolated_position_int, normalize=False)

                # Wait for next step (except on final step)
                if step < total_steps:
                    time.sleep(step_time)

            logger.info(f"Successfully moved {motor_name} to {target_position:.1f}")
            return True

        except Exception as e:
            logger.error(f"Error during slow movement of {motor_name}: {e}")
            return False

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        self.robot.bus.write_calibration(self.robot.calibration)
        self.robot._save_calibration()
