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
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from .config_so101_follower_t import SO101FollowerTConfig

logger = logging.getLogger(__name__)


class SO101FollowerT(Robot):
    """
    SO-101 Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101FollowerTConfig
    name = "so101_follower"

    def __init__(self, config: SO101FollowerTConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "hls3625", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "hls3625", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "hls3625", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "hls3625", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "hls3625", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "hls3625", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"{m}.pos": float for m in self.bus.motors},
            **{f"{m}.effort": int for m in self.bus.motors},
        }

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                phase = self.bus.read("Phase", motor, normalize=False)
                if phase & 0x10:  # bit-4 set = multi-turn
                    new_phase = phase & ~0x10
                    print(f"Switching {motor} to single-turn: 0x{phase:02X} → 0x{new_phase:02X}")
                    self.bus.write("Phase", motor, new_phase, normalize=False)

                self.bus.write("Operating_Mode", motor, 2)  # Set to current mode
                self.bus.write("Target_Torque", motor, 0)
                self.bus.write("Torque_Limit", motor, 1000)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        pos = self.bus.sync_read("Present_Position", normalize=False, num_retry=5)
        curr = self.bus.sync_read("Present_Current", normalize=False, num_retry=5)
        pos_dict = {f"{motor}.pos": val for motor, val in pos.items()}
        curr_dict = {f"{motor}.effort": val for motor, val in curr.items()}
        obs_dict = {**pos_dict, **curr_dict}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Send goal position to the arm
        effort = {k.removesuffix(".effort"): int(v) for k, v in action.items() if k.endswith(".effort")}
        if effort:
            # current (torque) mode write
            self.bus.sync_write("Target_Torque", effort, normalize=False, num_retry=2)
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
