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

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from .config_hope_jr import HopeJrHandConfig

logger = logging.getLogger(__name__)


class HopeJrHand(Robot):
    config_class = HopeJrHandConfig
    name = "hope_jr_hand"

    def __init__(self, config: HopeJrHandConfig):
        super().__init__(config)
        self.config = config
        self.hand = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # Thumb
                "thumb_basel_rotation": Motor(1, "scs0009", MotorNormMode.RANGE_M100_100),
                "thumb_mcp": Motor(2, "scs0009", MotorNormMode.RANGE_M100_100),
                "thumb_pip": Motor(3, "scs0009", MotorNormMode.RANGE_M100_100),
                "thumb_dip": Motor(4, "scs0009", MotorNormMode.RANGE_M100_100),
                # Index
                "index_thumb_side": Motor(5, "scs0009", MotorNormMode.RANGE_M100_100),
                "index_pinky_side": Motor(6, "scs0009", MotorNormMode.RANGE_M100_100),
                "index_flexor": Motor(7, "scs0009", MotorNormMode.RANGE_M100_100),
                # Middle
                "middle_thumb_side": Motor(8, "scs0009", MotorNormMode.RANGE_M100_100),
                "middle_pinky_side": Motor(9, "scs0009", MotorNormMode.RANGE_M100_100),
                "middle_flexor": Motor(10, "scs0009", MotorNormMode.RANGE_M100_100),
                # Ring
                "ring_thumb_side": Motor(11, "scs0009", MotorNormMode.RANGE_M100_100),
                "ring_pinky_side": Motor(12, "scs0009", MotorNormMode.RANGE_M100_100),
                "ring_flexor": Motor(13, "scs0009", MotorNormMode.RANGE_M100_100),
                # Pinky
                "pinky_thumb_side": Motor(14, "scs0009", MotorNormMode.RANGE_M100_100),
                "pinky_pinky_side": Motor(15, "scs0009", MotorNormMode.RANGE_M100_100),
                "pinky_flexor": Motor(16, "scs0009", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
            protocol_version=1,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def state_feature(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.hand),),
            "names": {"motors": list(self.hand.motors)},
        }

    @property
    def action_feature(self) -> dict:
        return self.state_feature

    @property
    def camera_features(self) -> dict[str, dict]:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft[cam_key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def is_connected(self) -> bool:
        # TODO(aliberts): add cam.is_connected for cam in self.cameras
        return self.hand.is_connected

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.hand.connect()
        if not self.is_calibrated:
            self.calibrate()

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.hand.is_calibrated

    def calibrate(self) -> None:
        raise NotImplementedError  # TODO(aliberts): adapt code below (copied from koch)
        logger.info(f"\nRunning calibration of {self}")
        self.hand.disable_torque()
        for name in self.hand.names:
            self.hand.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.hand.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [name for name in self.hand.names if name != full_turn_motor]
        logger.info(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.hand.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for name, motor in self.hand.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.hand.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.hand.torque_disabled():
            self.hand.configure_motors()
            # TODO

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read hand position
        start = time.perf_counter()
        for motor in self.hand.motors:
            obs_dict[f"{OBS_STATE}.{motor}"] = self.hand.read("Present_Position", motor, normalize=False)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.hand.sync_write("Goal_Position", action)
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.hand.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
