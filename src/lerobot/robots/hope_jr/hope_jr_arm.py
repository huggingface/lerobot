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
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.calibration_gui import RangeFinderGUI
from lerobot.motors.feetech import (
    FeetechMotorsBus,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_hope_jr import HopeJrArmConfig

logger = logging.getLogger(__name__)


class HopeJrArm(Robot):
    """One arm of the Hope Jr humanoid.

    The arm and the hand are separate robots; pair this with [`~robots.hope_jr.HopeJrHand`] for a full
    limb. See [`~robots.Robot`] for the contract every method here implements.

    Args:
        config (`HopeJrArmConfig`):
            The robot's configuration. Its `port` and `cameras` determine what is connected.
    """

    config_class = HopeJrArmConfig
    name = "hope_jr_arm"

    def __init__(self, config: HopeJrArmConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pitch": Motor(1, "sm8512bl", MotorNormMode.RANGE_M100_100),
                "shoulder_yaw": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
                "shoulder_roll": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_yaw": Motor(6, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_pitch": Motor(7, "sts3250", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # HACK
        self.shoulder_pitch = "shoulder_pitch"
        self.other_motors = [m for m in self.bus.motors if m != "shoulder_pitch"]

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
        """Connect the motor bus and cameras, calibrating and configuring the arm.

        > [!WARNING]
        > The arm is assumed to be at rest when this is called, because torque is disabled to run
        > calibration.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to run calibration if the arm is not already calibrated.

        Raises:
            DeviceAlreadyConnectedError: If the robot is already connected.
        """
        self.bus.connect(handshake=False)
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect the cameras
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
        groups = {
            "all": list(self.bus.motors.keys()),
            "shoulder": ["shoulder_pitch", "shoulder_yaw", "shoulder_roll"],
            "elbow": ["elbow_flex"],
            "wrist": ["wrist_roll", "wrist_yaw", "wrist_pitch"],
        }

        self.calibration = RangeFinderGUI(self.bus, groups).run()
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        """Apply the operating mode, gains and limits from the configuration to the robot."""
        with self.bus.torque_disabled():
            self.bus.configure_motors(maximum_acceleration=30, acceleration=30)

    def setup_motors(self) -> None:
        # TODO: add docstring
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
        obs_dict = self.bus.sync_read("Present_Position", self.other_motors)
        obs_dict[self.shoulder_pitch] = self.bus.read("Present_Position", self.shoulder_pitch)
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
        """Command the robot to move towards a target configuration.

        Args:
            action (`dict[str, Any]`):
                Target values, keyed as in [`~robots.Robot.action_features`].

        Returns:
            `dict[str, Any]`: The action actually sent, which may be clipped by `max_relative_target`.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

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
