#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_mini import OpenArmMiniConfig

logger = logging.getLogger(__name__)

# Motors whose direction is inverted on the leader side.
LEFT_MOTORS_TO_FLIP = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_7"}
RIGHT_MOTORS_TO_FLIP = {"joint_1", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"}
# Leader(OpenArmMini) -> Follower(OpenArms) joint remap
JOINT_REMAP_TO_OPENARMS = {"joint_6": "joint_7", "joint_7": "joint_6"}
# Follower(OpenArms) -> Leader(OpenArmMini) joint remap
JOINT_REMAP_TO_MINI = {"joint_7": "joint_6", "joint_6": "joint_7"}
OPENARMS_GRIPPER_MIN = -65.0
OPENARMS_GRIPPER_MAX = 0.0
MINI_GRIPPER_MIN = 0.0
MINI_GRIPPER_MAX = 100.0


class OpenArmMini(Teleoperator):
    """
    OpenArm Mini Teleoperator with dual Feetech-based arms (8 motors per arm).

    Each arm has 7 joints plus a gripper, using Feetech STS3215 servos.
    """

    config_class = OpenArmMiniConfig
    name = "openarm_mini"

    def __init__(self, config: OpenArmMiniConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES

        motors_right = {
            "joint_1": Motor(1, "sts3215", norm_mode_body),
            "joint_2": Motor(2, "sts3215", norm_mode_body),
            "joint_3": Motor(3, "sts3215", norm_mode_body),
            "joint_4": Motor(4, "sts3215", norm_mode_body),
            "joint_5": Motor(5, "sts3215", norm_mode_body),
            "joint_6": Motor(6, "sts3215", norm_mode_body),
            "joint_7": Motor(7, "sts3215", norm_mode_body),
            "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
        }

        motors_left = {
            "joint_1": Motor(1, "sts3215", norm_mode_body),
            "joint_2": Motor(2, "sts3215", norm_mode_body),
            "joint_3": Motor(3, "sts3215", norm_mode_body),
            "joint_4": Motor(4, "sts3215", norm_mode_body),
            "joint_5": Motor(5, "sts3215", norm_mode_body),
            "joint_6": Motor(6, "sts3215", norm_mode_body),
            "joint_7": Motor(7, "sts3215", norm_mode_body),
            "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
        }

        cal_right = {
            k.replace("right_", ""): v for k, v in (self.calibration or {}).items() if k.startswith("right_")
        }
        cal_left = {
            k.replace("left_", ""): v for k, v in (self.calibration or {}).items() if k.startswith("left_")
        }

        self.bus_right = FeetechMotorsBus(
            port=self.config.port_right,
            motors=motors_right,
            calibration=cal_right,
        )

        self.bus_left = FeetechMotorsBus(
            port=self.config.port_left,
            motors=motors_left,
            calibration=cal_left,
        )

    @staticmethod
    def _mini_gripper_to_openarms(value: float) -> float:
        """Convert OpenArmMini gripper range [0, 100] to OpenArms gripper range [-65, 0]."""
        mapped = OPENARMS_GRIPPER_MAX + (
            (value - MINI_GRIPPER_MIN)
            * (OPENARMS_GRIPPER_MIN - OPENARMS_GRIPPER_MAX)
            / (MINI_GRIPPER_MAX - MINI_GRIPPER_MIN)
        )
        return max(min(mapped, OPENARMS_GRIPPER_MAX), OPENARMS_GRIPPER_MIN)

    @staticmethod
    def _openarms_gripper_to_mini(value: float) -> float:
        """Convert OpenArms gripper range [-65, 0] to OpenArmMini gripper range [0, 100]."""
        clipped = max(min(value, OPENARMS_GRIPPER_MAX), OPENARMS_GRIPPER_MIN)
        return MINI_GRIPPER_MIN + (
            (OPENARMS_GRIPPER_MAX - clipped)
            * (MINI_GRIPPER_MAX - MINI_GRIPPER_MIN)
            / (OPENARMS_GRIPPER_MAX - OPENARMS_GRIPPER_MIN)
        )

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for motor in self.bus_right.motors:
            features[f"right_{motor}.pos"] = float
        for motor in self.bus_left.motors:
            features[f"left_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus_right.is_connected and self.bus_left.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting right arm on {self.config.port_right}...")
        self.bus_right.connect()
        logger.info(f"Connecting left arm on {self.config.port_left}...")
        self.bus_left.connect()

        if calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus_right.is_calibrated and self.bus_left.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArm Mini.

        1. Disable torque
        2. Ask user to position arms in hanging position with grippers closed
        3. Set this as zero position via half-turn homing
        4. Interactive gripper calibration (open/close positions)
        5. Save calibration
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration for {self.id}, "
                f"or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                cal_right = {
                    k.replace("right_", ""): v for k, v in self.calibration.items() if k.startswith("right_")
                }
                cal_left = {
                    k.replace("left_", ""): v for k, v in self.calibration.items() if k.startswith("left_")
                }
                self.bus_right.write_calibration(cal_right)
                self.bus_left.write_calibration(cal_left)
                return

        logger.info(f"\nRunning calibration for {self}")

        self._calibrate_arm("right", self.bus_right)
        self._calibrate_arm("left", self.bus_left)

        self._save_calibration()
        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def _calibrate_arm(self, arm_name: str, bus: FeetechMotorsBus) -> None:
        """Calibrate a single arm with Feetech motors."""
        logger.info(f"\n=== Calibrating {arm_name.upper()} arm ===")

        bus.disable_torque()

        logger.info(f"Setting Phase to 12 for all motors in {arm_name.upper()} arm...")
        for motor in bus.motors:
            bus.write("Phase", motor, 12)

        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            f"\nCalibration: Zero Position ({arm_name.upper()} arm)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        homing_offsets = bus.set_half_turn_homings()
        logger.info(f"{arm_name.capitalize()} arm zero position set.")

        print(f"\nSetting motor ranges for {arm_name.upper()} arm\n")

        if self.calibration is None:
            self.calibration = {}

        motor_resolution = bus.model_resolution_table[list(bus.motors.values())[0].model]
        max_res = motor_resolution - 1

        for motor_name, motor in bus.motors.items():
            prefixed_name = f"{arm_name}_{motor_name}"

            if motor_name == "gripper":
                input(
                    f"\nGripper Calibration ({arm_name.upper()} arm)\n"
                    f"Step 1: CLOSE the gripper fully\n"
                    f"Press ENTER when gripper is closed..."
                )
                closed_pos = bus.read("Present_Position", motor_name, normalize=False)
                logger.info(f"  Gripper closed position recorded: {closed_pos}")

                input("\nStep 2: OPEN the gripper fully\nPress ENTER when gripper is fully open...")
                open_pos = bus.read("Present_Position", motor_name, normalize=False)
                logger.info(f"  Gripper open position recorded: {open_pos}")

                if closed_pos < open_pos:
                    range_min = int(closed_pos)
                    range_max = int(open_pos)
                    drive_mode = 0
                else:
                    range_min = int(open_pos)
                    range_max = int(closed_pos)
                    drive_mode = 1

                logger.info(
                    f"  {prefixed_name}: range set to [{range_min}, {range_max}] "
                    f"(0=closed, 100=open, drive_mode={drive_mode})"
                )
            else:
                range_min = 0
                range_max = max_res
                drive_mode = 0
                logger.info(f"  {prefixed_name}: range set to [0, {max_res}] (full motor range)")

            self.calibration[prefixed_name] = MotorCalibration(
                id=motor.id,
                drive_mode=drive_mode,
                homing_offset=homing_offsets[motor_name],
                range_min=range_min,
                range_max=range_max,
            )

        cal_for_bus = {
            k.replace(f"{arm_name}_", ""): v
            for k, v in self.calibration.items()
            if k.startswith(f"{arm_name}_")
        }
        bus.write_calibration(cal_for_bus)

    def configure(self) -> None:
        self.bus_right.disable_torque()
        self.bus_right.configure_motors()
        for motor in self.bus_right.motors:
            self.bus_right.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        self.bus_left.disable_torque()
        self.bus_left.configure_motors()
        for motor in self.bus_left.motors:
            self.bus_left.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        print("\nSetting up RIGHT arm motors...")
        for motor in reversed(self.bus_right.motors):
            input(f"Connect the controller board to the RIGHT '{motor}' motor only and press enter.")
            self.bus_right.setup_motor(motor)
            print(f"RIGHT '{motor}' motor id set to {self.bus_right.motors[motor].id}")

        print("\nSetting up LEFT arm motors...")
        for motor in reversed(self.bus_left.motors):
            input(f"Connect the controller board to the LEFT '{motor}' motor only and press enter.")
            self.bus_left.setup_motor(motor)
            print(f"LEFT '{motor}' motor id set to {self.bus_left.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Get current action from both arms (read positions from all motors)."""
        start = time.perf_counter()

        right_positions = self.bus_right.sync_read("Present_Position")
        left_positions = self.bus_left.sync_read("Present_Position")

        action: dict[str, Any] = {}
        for motor, val in right_positions.items():
            target_motor = JOINT_REMAP_TO_OPENARMS.get(motor, motor)
            mapped_val = -val if motor in RIGHT_MOTORS_TO_FLIP else val
            if target_motor == "gripper":
                mapped_val = self._mini_gripper_to_openarms(mapped_val)
            action[f"right_{target_motor}.pos"] = mapped_val
        for motor, val in left_positions.items():
            target_motor = JOINT_REMAP_TO_OPENARMS.get(motor, motor)
            mapped_val = -val if motor in LEFT_MOTORS_TO_FLIP else val
            if target_motor == "gripper":
                mapped_val = self._mini_gripper_to_openarms(mapped_val)
            action[f"left_{target_motor}.pos"] = mapped_val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def enable_torque(self) -> None:
        """Enable torque on both arms for active motion commands."""
        self.bus_right.enable_torque()
        self.bus_left.enable_torque()

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable torque on both arms for manual teleoperation."""
        self.bus_right.disable_torque()
        self.bus_left.disable_torque()

    @check_if_not_connected
    def write_goal_positions(self, action: dict[str, float]) -> None:
        """Send normalized bilateral goal positions to the underlying Feetech buses."""
        right_goals: dict[str, float] = {}
        left_goals: dict[str, float] = {}

        for key, value in action.items():
            if not key.endswith(".pos"):
                continue

            if key.startswith("right_"):
                openarms_motor = key.removeprefix("right_").removesuffix(".pos")
                mini_motor = JOINT_REMAP_TO_MINI.get(openarms_motor, openarms_motor)
                mapped_val = self._openarms_gripper_to_mini(value) if openarms_motor == "gripper" else value
                right_goals[mini_motor] = -mapped_val if mini_motor in RIGHT_MOTORS_TO_FLIP else mapped_val
            elif key.startswith("left_"):
                openarms_motor = key.removeprefix("left_").removesuffix(".pos")
                mini_motor = JOINT_REMAP_TO_MINI.get(openarms_motor, openarms_motor)
                mapped_val = self._openarms_gripper_to_mini(value) if openarms_motor == "gripper" else value
                left_goals[mini_motor] = -mapped_val if mini_motor in LEFT_MOTORS_TO_FLIP else mapped_val

        if right_goals:
            self.bus_right.sync_write("Goal_Position", right_goals)
        if left_goals:
            self.bus_left.sync_write("Goal_Position", left_goals)

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Route feedback position commands through the same OpenArms/OpenArmMini mapping."""
        self.write_goal_positions(feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus_right.disconnect()
        self.bus_left.disconnect()
        logger.info(f"{self} disconnected.")
