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
from .config_omx_leader import OmxLeaderConfig

logger = logging.getLogger(__name__)


class OmxLeader(Teleoperator):
    """The OMX leader arm, held by an operator to teleoperate a follower arm.

    [OMX](https://github.com/ROBOTIS-GIT/open_manipulator), developed by Woojin Wie and Junha Cha from
    [ROBOTIS](https://ai.robotis.com/).

    Actions are keyed `"<motor>.pos"`. See [`~teleoperators.Teleoperator`] for the contract every method
    here implements.

    Args:
        config (`OmxLeaderConfig`): The teleoperator's configuration. Its `port` determines what is
            connected.

    Example:
        ```python
        >>> from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
        >>> teleop = OmxLeader(OmxLeaderConfig(port="/dev/ttyACM0"))  # doctest: +SKIP
        >>> with teleop:  # doctest: +SKIP
        ...     action = teleop.get_action()
        ```
    """

    config_class = OmxLeaderConfig
    name = "omx_leader"

    def __init__(self, config: OmxLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
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

        This arm does not support feedback; [`~teleoperators.omx_leader.OmxLeader.send_feedback`] always
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
                Whether to write the factory default calibration when the motors disagree with the
                calibration file, or no file exists yet.

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
        """Write the factory default calibration to the motors and the calibration file.

        Unlike other SO/Koch-family arms, this is not interactive: the OMX arm's homing offsets and ranges
        of motion are fixed factory defaults, so no manual positioning is required.
        """
        self.bus.disable_torque()
        logger.info(f"\nUsing factory default calibration values for {self}")
        logger.info(f"\nWriting default configuration of {self} to the motors")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        for motor in self.bus.motors:
            if motor == "gripper":
                self.bus.write("Drive_Mode", motor, DriveMode.INVERTED.value)
            else:
                self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)
        drive_modes = {motor: 1 if motor == "gripper" else 0 for motor in self.bus.motors}

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=0 if motor != "gripper" else 100,
                range_min=0,
                range_max=4095,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Write the operating and drive modes to every motor, including the gripper's spring-back trigger.

        All motors except the gripper are set to extended position mode with a non-inverted drive mode. The
        gripper's drive mode is inverted, and it is set to current-based position control with a reduced
        current limit and driven to `gripper_open_pos`, with torque enabled, so it springs back to that
        position when released and can be used as a physical trigger.
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

            if motor == "gripper":
                self.bus.write("Drive_Mode", motor, DriveMode.INVERTED.value)
            else:
                self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        self.bus.write("Current_Limit", "gripper", 100)
        self.bus.write("Goal_Current", "gripper", 100)
        self.bus.write("Homing_Offset", "gripper", 100)
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
