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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.calibration_gui import RangeFinderGUI
from lerobot.motors.feetech import (
    FeetechMotorsBus,
)

from ..robot import Robot
from .config_hope_jr import HopeJrHandConfig

logger = logging.getLogger(__name__)

RIGHT_HAND_INVERSIONS = [
    "thumb_mcp",
    "thumb_dip",
    "index_ulnar_flexor",
    "middle_ulnar_flexor",
    "ring_ulnar_flexor",
    "ring_pip_dip",
    "pinky_ulnar_flexor",
    "pinky_pip_dip",
]

LEFT_HAND_INVERSIONS = [
    "thumb_cmc",
    "thumb_mcp",
    "thumb_dip",
    "index_radial_flexor",
    "index_pip_dip",
    "middle_radial_flexor",
    "middle_pip_dip",
    "ring_radial_flexor",
    "ring_pip_dip",
    "pinky_radial_flexor",
    # "pinky_pip_dip",
]


class HopeJrHand(Robot):
    config_class = HopeJrHandConfig
    name = "hope_jr_hand"

    def __init__(self, config: HopeJrHandConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # Thumb
                "thumb_cmc": Motor(1, "scs0009", MotorNormMode.RANGE_0_100),
                "thumb_mcp": Motor(2, "scs0009", MotorNormMode.RANGE_0_100),
                "thumb_pip": Motor(3, "scs0009", MotorNormMode.RANGE_0_100),
                "thumb_dip": Motor(4, "scs0009", MotorNormMode.RANGE_0_100),
                # Index
                "index_radial_flexor": Motor(5, "scs0009", MotorNormMode.RANGE_0_100),
                "index_ulnar_flexor": Motor(6, "scs0009", MotorNormMode.RANGE_0_100),
                "index_pip_dip": Motor(7, "scs0009", MotorNormMode.RANGE_0_100),
                # Middle
                "middle_radial_flexor": Motor(8, "scs0009", MotorNormMode.RANGE_0_100),
                "middle_ulnar_flexor": Motor(9, "scs0009", MotorNormMode.RANGE_0_100),
                "middle_pip_dip": Motor(10, "scs0009", MotorNormMode.RANGE_0_100),
                # Ring
                "ring_radial_flexor": Motor(11, "scs0009", MotorNormMode.RANGE_0_100),
                "ring_ulnar_flexor": Motor(12, "scs0009", MotorNormMode.RANGE_0_100),
                "ring_pip_dip": Motor(13, "scs0009", MotorNormMode.RANGE_0_100),
                # Pinky
                "pinky_radial_flexor": Motor(14, "scs0009", MotorNormMode.RANGE_0_100),
                "pinky_ulnar_flexor": Motor(15, "scs0009", MotorNormMode.RANGE_0_100),
                "pinky_pip_dip": Motor(16, "scs0009", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
            protocol_version=1,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self.inverted_motors = RIGHT_HAND_INVERSIONS if config.side == "right" else LEFT_HAND_INVERSIONS

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
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        fingers = {}
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            fingers[finger] = [motor for motor in self.bus.motors if motor.startswith(finger)]

        self.calibration = RangeFinderGUI(self.bus, fingers).run()
        for motor in self.inverted_motors:
            self.calibration[motor].drive_mode = 1
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        # TODO: add docstring
        for motor in self.bus.motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read hand position
        start = time.perf_counter()
        for motor in self.bus.motors:
            obs_dict[f"{motor}.pos"] = self.bus.read("Present_Position", motor)
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
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal_pos)
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
