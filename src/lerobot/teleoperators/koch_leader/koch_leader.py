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

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_koch_leader import KochLeaderConfig

logger = logging.getLogger(__name__)


class KochLeader(Teleoperator):
    """The Koch leader arm, in either of its two revisions, held by an operator to teleoperate a follower arm.

    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
      expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com).
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1), developed by Jess Moss.

    Actions are keyed `"<motor>.pos"`. See [`~teleoperators.Teleoperator`] for the contract every method
    here implements.

    Example:
        ```python
        >>> from lerobot.teleoperators.koch_leader import KochLeader, KochLeaderConfig
        >>> teleop = KochLeader(KochLeaderConfig(port="/dev/ttyACM0"))  # doctest: +SKIP
        >>> with teleop:  # doctest: +SKIP
        ...     action = teleop.get_action()
        ```
    """

    config_class = KochLeaderConfig
    name = "koch_leader"

    def __init__(self, config: KochLeaderConfig):
        """Build the teleoperator from its configuration.

        Args:
            config (`KochLeaderConfig`):
                The teleoperator's configuration. Its `port` determines what is connected.
        """
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
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        """The arm's joint positions.

        Returns:
            `dict[str, type]`: `"<motor>.pos"` keys mapped to `float`.
        """
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Same as [`~teleoperators.Teleoperator.feedback_features`].

        This arm does not support feedback; [`~teleoperators.koch_leader.KochLeader.send_feedback`] always
        raises `NotImplementedError`.

        Returns:
            `dict[str, type]`: Always empty.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """Same as [`~teleoperators.Teleoperator.is_connected`]."""
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect the motor bus, calibrating and configuring the arm.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to run calibration when the motors disagree with the calibration file, or no file
                exists yet. Calibration is interactive and prompts on stdin.

        Raises:
            DeviceAlreadyConnectedError: If the teleoperator is already connected.
        """
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Same as [`~teleoperators.Teleoperator.is_calibrated`]."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Calibrate the arm, writing the result to the motors and the calibration file.

        This is interactive: it prompts on stdin to reuse an existing calibration file, and otherwise asks
        you to move the arm to its middle position and then through each joint's full range. The
        `elbow_flex` motor is inverted, and `shoulder_pan` and `wrist_roll` are treated as full-turn joints.
        """
        self.bus.disable_torque()
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        self.bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
        drive_modes = {motor: 1 if motor == "elbow_flex" else 0 for motor in self.bus.motors}

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
        """Write the operating modes to every motor, including the gripper's spring-back trigger behavior.

        All motors except the gripper are set to extended position mode. The gripper is set to
        current-based position control and driven to `gripper_open_pos`, with torque enabled, so it springs
        back to that position when released and can be used as a physical trigger.
        """
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
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
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        self.bus.enable_torque("gripper")
        if self.is_calibrated:
            self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    def setup_motors(self) -> None:
        """Assign each motor its bus ID, one at a time.

        Run this once when building an arm. It is interactive: it prompts you to connect the controller
        board to a single motor at a time, working from the gripper back to the base.
        """
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        """Same as [`~teleoperators.Teleoperator.get_action`].

        Returns:
            `dict[str, float]`: `"<motor>.pos"` keys mapped to the arm's current joint positions.

        Raises:
            DeviceNotConnectedError: If the teleoperator is not connected.
        """
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Not implemented for this arm.

        Raises:
            NotImplementedError: Always. This arm does not support force feedback.
        """
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        """Same as [`~teleoperators.Teleoperator.disconnect`].

        Raises:
            DeviceNotConnectedError: If the teleoperator is not connected.
        """
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
