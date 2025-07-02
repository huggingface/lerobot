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
import math
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
    name = "so101_follower_t"

    _CURRENT_STEP_A: float = 6.5e-3  # 6.5 mA per register LSB #http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0
    _KT_NM_PER_AMP: float = 0.814  # Torque constant Kt [N·m/A] #https://www.feetechrc.com/811177.html
    _MAX_CURRENT_A: float = 3.0  # Safe driver limit for this model

    _COUNT_TO_RAD: float = math.radians(0.087)  # 1 pos count to rad

    def _current_to_torque_nm(self, raw: dict[str, int]) -> dict[str, float]:
        """Convert "Present_Current" register counts (±2047) → torque [Nm].
        Values are clamped to ±3A before conversion for protection.
        """
        max_cnt = int(round(self._MAX_CURRENT_A / self._CURRENT_STEP_A))  # ≈ 462
        coef = self._CURRENT_STEP_A * self._KT_NM_PER_AMP
        return {k: max(min(v, max_cnt), -max_cnt) * coef for k, v in raw.items()}

    def _torque_nm_to_current(self, torque: dict[str, float]) -> dict[str, int]:
        """Convert torque [Nm] to register counts, clamped to ±3A (2.44 Nm)."""
        inv_coef = 1.0 / (self._CURRENT_STEP_A * self._KT_NM_PER_AMP)
        max_cnt = int(round(self._MAX_CURRENT_A / self._CURRENT_STEP_A))
        max_torque = self._MAX_CURRENT_A * self._KT_NM_PER_AMP
        return {
            k: max(-max_cnt, min(max_cnt, int(round(max(-max_torque, min(max_torque, float(t))) * inv_coef))))
            for k, t in torque.items()
        }

    def _deg_to_rad(self, deg: dict[str, float | int]) -> dict[str, float]:
        """GDegrees to radians."""
        return {m: math.radians(float(v)) for m, v in deg.items()}

    def __init__(self, config: SO101FollowerTConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "hls3625", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "hls3625", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "hls3625", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "hls3625", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "hls3625", MotorNormMode.DEGREES),
                "gripper": Motor(6, "hls3625", MotorNormMode.DEGREES),
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
                self.bus.write("Torque_Limit", motor, 1000)  # 100%

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Positions
        pos_deg = self.bus.sync_read("Present_Position", num_retry=10)
        pos_rad = self._deg_to_rad(pos_deg)
        obs_dict = {f"{m}.pos": r for m, r in pos_rad.items()}

        # Currents to torque (Nm)
        curr_raw = self.bus.sync_read("Present_Current", normalize=False, num_retry=10)
        torque_nm = self._current_to_torque_nm({f"{m}.effort": v for m, v in curr_raw.items()})
        obs_dict.update(torque_nm)

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
        """Command arm to move to a target torque for a joint.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract torque commands
        torque_cmd = {k: v for k, v in action.items() if k.endswith(".effort")}
        if torque_cmd:
            counts = self._torque_nm_to_current(torque_cmd)
            # remove the .effort suffix
            counts = {k.removesuffix(".effort"): v for k, v in counts.items()}
            self.bus.sync_write("Target_Torque", counts, normalize=False, num_retry=2)

        # pass back the other keys (.pos) untouched
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
