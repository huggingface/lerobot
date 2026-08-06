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
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_omx_follower import OmxFollowerConfig

logger = logging.getLogger(__name__)


class OmxFollower(Robot):
    """The [OpenMANIPULATOR-X](https://github.com/ROBOTIS-GIT/open_manipulator) follower arm.

    Developed by Woojin Wie and Junha Cha at [ROBOTIS](https://ai.robotis.com/).
    """

    config_class = OmxFollowerConfig
    name = "omx_follower"

    def __init__(self, config: OmxFollowerConfig):
        """Build the robot from its configuration.

        Args:
            config (`OmxFollowerConfig`):
                The robot's configuration. Its `port` and `cameras` determine what is connected.
        """
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(11, "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(12, "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(13, "xl430-w250", norm_mode_body),
                "wrist_flex": Motor(14, "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(15, "xl330-m288", norm_mode_body),
                "gripper": Motor(16, "xl330-m288", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """The values this robot reports, and their types or shapes.

        Returns:
            `dict`: Keys as returned by [`~robots.Robot.get_observation`], mapped to a scalar type for
            proprioceptive values or to a `(height, width, channels)` shape for images.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """The values this robot accepts, and their types.

        Returns:
            `dict`: Keys accepted by [`~robots.Robot.send_action`], mapped to their type.
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Whether every device this robot uses is connected.

        Returns:
            `bool`: `True` only when the robot and all its cameras are connected.
        """
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect the motor bus and cameras, handling the pre-calibrated case.

        OMX arms ship calibrated, so this avoids asking for a manual calibration where possible:

        - if the packaged default calibration does not match the motors, the motors' own values are read
          and saved;
        - if no calibration file exists, factory defaults are used (`homing_offset=0`, `range_min=0`,
          `range_max=4095`).

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to calibrate if the arm is not already calibrated.

        Raises:
            DeviceAlreadyConnectedError: If the robot is already connected.
        """
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is calibrated.

        Returns:
            `bool`: `True` when no calibration is needed before use.
        """
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Calibrate the robot and store the result.

        Interactive: prompts on stdin and asks you to move the robot through the required positions.
        """
        self.bus.disable_torque()
        logger.info(f"\nUsing factory default calibration values for {self}")
        logger.info(f"\nWriting default configuration of {self} to the motors")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        for motor in self.bus.motors:
            self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Apply the operating mode, gains and limits from the configuration to the robot."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
            # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling
            # the arm, you could end up with a servo with a position 0 or 4095 at a crucial point
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # Use 'position control current based' for gripper to be limited by the limit of the current. For
            # the follower gripper, it means it can grasp an object without forcing too much even tho, its
            # goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with
            # our finger to make it move, and it will move back to its original target position when we
            # release the force.
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)

    def setup_motors(self) -> None:
        """Assign each motor its bus ID, one at a time.

        Run this once when building the robot. Interactive: prompts you to connect the controller board to a
        single motor at a time.
        """
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position
        """Read the robot's current state and a frame from each camera.

        Returns:
            `dict[str, Any]`: Keys matching [`~robots.Robot.observation_features`].

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (RobotAction): The goal positions for the motors.

        Returns:
            RobotAction: The action sent to the motors, potentially clipped.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """Disconnect from the robot and its cameras.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
