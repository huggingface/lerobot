#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_koch_screwdriver_leader import KochScrewdriverLeaderConfig

logger = logging.getLogger(__name__)


class KochScrewdriverLeader(Teleoperator):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochScrewdriverLeaderConfig
    name = "koch_screwdriver_leader"

    def __init__(self, config: KochScrewdriverLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
                
                # @TODO(jackvial) - might need to configure this to map correctly the follower wheel mode screwdriver servo
                "screwdriver": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        """Describe the leader action space.

        All joints except the screwdriver are forwarded as target *positions* (normalized) while the
        screwdriver, which is used like a trigger, is mapped to a *velocity* command that should be
        applied to the follower screwdriver.
        """
        # TODO(jackvial) needs review
        # Old implementation for reference:
        # return {f"{motor}.pos": float for motor in self.bus.motors}
        return {
            (f"{motor}.vel" if motor == "screwdriver" else f"{motor}.pos"): float
            for motor in self.bus.motors
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)


        # TODO(jackvial) - inverting the elbow seems like it should not be hardcoded here and should instead be a config option
        # self.bus.write("Drive_Mode", "elbow_flex", DriveMode.NON_INVERTED.value)
        # drive_modes = {motor: 1 if motor == "elbow_flex" else 0 for motor in self.bus.motors}
        drive_modes = {motor: 0 for motor in self.bus.motors}

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "screwdriver":
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        # self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        self.bus.write("Operating_Mode", "screwdriver", OperatingMode.CURRENT_POSITION.value)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        # self.bus.enable_torque("gripper")
        self.bus.enable_torque("screwdriver")
        if self.is_calibrated:
            # self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)
            self.bus.write("Goal_Position", "screwdriver", self.config.screwdriver_open_pos)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        # TODO(jackvial) needs review
        # Old implementation for reference:
        # start = time.perf_counter()
        # action = self.bus.sync_read("Present_Position")
        # action = {f"{motor}.pos": val for motor, val in action.items()}
        # dt_ms = (time.perf_counter() - start) * 1e3
        # logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        # return action
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read all joint positions once.
        pos_dict = self.bus.sync_read("Present_Position")

        # Build the action dictionary, converting the screwdriver position into a velocity command.
        action = {}
        for motor, pos in pos_dict.items():
            if motor == "screwdriver":
                # Map the positional deviation from the neutral (open) pose to a velocity value.
                # @TODO(jackvial) - lift this to the config
                delta = pos - self.config.screwdriver_open_pos
                GAIN     = 10.0
                MAX_VEL  = 700
                vel_cmd = max(min(-delta * GAIN,  MAX_VEL), -MAX_VEL)

                # Small jitters around the neutral point are ignored.
                if abs(vel_cmd) < 2.0:
                    vel_cmd = 0.0

                action[f"{motor}.vel"] = vel_cmd
            else:
                action[f"{motor}.pos"] = pos

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
