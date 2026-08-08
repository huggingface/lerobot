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

import logging
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_v1_dynamixel import OpenArmV1DynamixelConfig

logger = logging.getLogger(__name__)

MODEL = "xl330-m288"

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]

# First servo id of each arm. Both arms are daisy-chained onto one OpenRB-150 bus:
# right arm ids 1-8, left arm ids 9-16 (7 joints + gripper each).
SIDE_BASE_ID = {"right": 1, "left": 9}

# Per-side motor direction flips applied during readout. Some servos are mounted
# mirrored, so the leader reads the opposite sign from what the follower expects.
# These are fixed by the mechanics, not measured during calibration.
SIDE_MOTORS_TO_FLIP: dict[str, list[str]] = {
    "right": ["joint_2", "joint_4", "joint_7"],
    "left": ["joint_2", "joint_7"],
}

# Gripper goes out as follower degrees: teleop 0 (closed) -> 0 deg,
# teleop 100 (fully open) -> -65 deg. Same convention as `openarm_mini`,
# so either leader can drive the same follower.
GRIPPER_TELEOP_TO_DEGREES = -0.65


class OpenArmV1Dynamixel(Teleoperator):
    """OpenARM v1 Dynamixel leader (XL330-M288, both arms on one OpenRB-150 bus).

    This is the bimanual device itself, not a single arm: all 16 servos share one
    serial port, so a single instance reads both arms and emits
    ``{side}_{joint}.pos`` actions for a ``bi_openarm_follower``.

    For the Feetech variant, which uses one port per arm, see :class:`OpenArmMini`
    and :class:`BiOpenArmMini`.
    """

    config_class = OpenArmV1DynamixelConfig
    name = "openarm_v1_dynamixel"

    def __init__(self, config: OpenArmV1DynamixelConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        motors: dict[str, Motor] = {}
        for side, base_id in SIDE_BASE_ID.items():
            for offset, joint in enumerate(ARM_JOINTS):
                motors[f"{side}_{joint}"] = Motor(base_id + offset, MODEL, norm_mode_body)
            motors[f"{side}_gripper"] = Motor(base_id + 7, MODEL, MotorNormMode.RANGE_0_100)

        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
        )

    @property
    def _motors_to_flip(self) -> set[str]:
        return {f"{side}_{j}" for side, joints in SIDE_MOTORS_TO_FLIP.items() for j in joints}

    @property
    def _emitted_motors(self) -> list[str]:
        if self.config.include_gripper:
            return list(self.bus.motors)
        return [m for m in self.bus.motors if not m.endswith("_gripper")]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._emitted_motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Run calibration for both arms.

        1. Disable torque so the arms can be moved by hand
        2. Ask the operator for the home pose (both arms hanging, grippers closed)
        3. Set that as zero via half-turn homing
        4. Capture each gripper's closed/open range
        5. Save

        The leader is never driven, so torque stays off for the whole procedure.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration for {self.id}, "
                f"or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")

        self.bus.disable_torque()
        # EXTENDED_POSITION, not POSITION: in single-turn position mode a Dynamixel
        # silently ignores a Homing_Offset outside +/-1024 pulses, which would leave the
        # arm calibrated to the wrong zero with no error. The other Dynamixel leaders
        # upstream do the same (`koch_leader`, `omx_leader`); the Feetech ones
        # (`so_leader`, `openarm_mini`) use POSITION because that limit is Dynamixel-only.
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(
            "\nCalibration: Zero Position\n"
            "Position both arms in the following configuration:\n"
            "  - Both arms hanging straight down, elbows not bent\n"
            "  - Both grippers closed\n"
            "  - Let go of the arms before pressing ENTER\n"
            "Press ENTER when ready..."
        )

        homing_offsets = self.bus.set_half_turn_homings()
        logger.info("Zero position set for both arms.")

        if self.calibration is None:
            self.calibration = {}

        motor_resolution = self.bus.model_resolution_table[MODEL]
        max_res = motor_resolution - 1

        for motor_name, motor in self.bus.motors.items():
            if motor_name.endswith("_gripper"):
                side = motor_name.rsplit("_", 1)[0]
                input(
                    f"\nGripper Calibration ({side})\n"
                    f"Step 1: CLOSE the {side} gripper fully\n"
                    "Press ENTER when it is closed..."
                )
                closed_pos = self.bus.read("Present_Position", motor_name, normalize=False)

                input(
                    f"\nStep 2: OPEN the {side} gripper fully\n"
                    "Open it all the way -- this travel becomes the whole gripper range.\n"
                    "Press ENTER when it is fully open..."
                )
                open_pos = self.bus.read("Present_Position", motor_name, normalize=False)

                if closed_pos < open_pos:
                    range_min, range_max, drive_mode = int(closed_pos), int(open_pos), 0
                else:
                    range_min, range_max, drive_mode = int(open_pos), int(closed_pos), 1

                if range_min == range_max:
                    raise ValueError(
                        f"{motor_name}: closed and open positions are identical ({range_min}). "
                        "The gripper did not move -- check its servo and calibrate again."
                    )
                logger.info(
                    f"  {motor_name}: range set to [{range_min}, {range_max}] "
                    f"(0=closed, 100=open, drive_mode={drive_mode})"
                )
            else:
                range_min, range_max, drive_mode = 0, max_res, 0
                logger.info(f"  {motor_name}: range set to [0, {max_res}] (full motor range)")

            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=drive_mode,
                homing_offset=homing_offsets[motor_name],
                range_min=range_min,
                range_max=range_max,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Leave the leader limp so the operator can move it by hand.

        Torque is never enabled on this device -- there is deliberately no
        ``enable_torque``. If the arms feel stiff, another program is holding the bus.
        """
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(list(self.bus.motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()

        positions = self.bus.sync_read("Present_Position")

        emitted = set(self._emitted_motors)
        to_flip = self._motors_to_flip
        action: dict[str, Any] = {}
        for motor, val in positions.items():
            if motor not in emitted:
                continue
            if motor.endswith("_gripper"):
                # A gripper whose encoder counts *down* as it opens is recorded with
                # drive_mode=1 during calibration. `DynamixelMotorsBus` sets
                # `apply_drive_mode = False` (unlike the Feetech bus), so the RANGE_0_100
                # normalisation never applies that flip for us -- undo it here, otherwise
                # a fully open leader gripper would command the follower to clamp shut.
                if self.calibration and self.calibration[motor].drive_mode:
                    val = 100.0 - val
                # teleop 0-100 -> follower degrees: 0 -> 0 deg, 100 -> -65 deg
                action[f"{motor}.pos"] = val * GRIPPER_TELEOP_TO_DEGREES
            else:
                action[f"{motor}.pos"] = -val if motor in to_flip else val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("The OpenARM v1 Dynamixel leader has no force feedback -- it is read-only.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
