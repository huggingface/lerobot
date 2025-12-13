# Copyright 2025 The Phantson Technologies Inc. team. All rights reserved.
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
import os
import subprocess
import sys
import time
from functools import cached_property
from typing import Any
import shutil

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_STATE
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_ned2 import Ned2Config

logger = logging.getLogger(__name__)

class Ned2(Robot):
    """
    New implementation of the Niryo Ned2 robot by Phantson Technologies Inc. (August 2025 -- Unofficial)
    """
    config_class = Ned2Config
    name = "ned2"

    def __init__(self, config: Ned2Config):
        super().__init__(config)
        self.config = config
        self._socat_process: subprocess.Popen | None = None
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "base_to_arm": Motor(2, "stepper", norm_mode),
                "arm_to_elbow": Motor(3, "stepper", norm_mode),
                "elbow_to_forearm": Motor(4, "stepper", norm_mode),
                "forearm_to_hand": Motor(5, "xl430-w250", norm_mode),
                "hand_to_shoulder": Motor(6, "xl430-w250", norm_mode),
                "shoulder_to_wrist": Motor(7, "xl330-m288", norm_mode),
                "gripper": Motor(11, "xl330-m288", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    def _start_socat_tunnel(self) -> None:
        if self.config.ip is None:
            return
        if self._socat_process is not None and self._socat_process.poll() is None:
            return
        if shutil.which("socat") is None:
            logger.error("Please install socat to use IP connection.")
            return

        link_path = self.config.port
        remote = f"TCP:{self.config.ip}:7777"
        cmd = [
            "socat",
            "-d",
            "-d",
            f"PTY,link={link_path},raw,echo=0",
            remote,
        ]

        try:
            self._socat_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.error(f"Failed to start socat: {exc}")
            self._socat_process = None
            return

        start_time = time.time()
        timeout_s = 5.0
        while not os.path.exists(link_path):
            if self._socat_process.poll() is not None:
                logger.error("Socat process stopped prematurely.")
                self._socat_process = None
                return
            if time.time() - start_time > timeout_s:
                logger.error("Timeout waiting for virtual device creation by socat.")
                self._socat_process.terminate()
                try:
                    self._socat_process.wait(timeout=1)
                except Exception:
                    self._socat_process.kill()
                self._socat_process = None
                return
            time.sleep(0.1)
        logger.info(f"Tunnel socat started to {remote} and linked to {link_path}.")

    def _stop_socat_tunnel(self) -> None:
        if self._socat_process is None:
            return
        if self._socat_process.poll() is None:
            try:
                self._socat_process.terminate()
                self._socat_process.wait(timeout=2)
            except Exception:
                self._socat_process.kill()
        self._socat_process = None
        logger.info("Tunnel socat stopped.")

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

        self._start_socat_tunnel()

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        steppers = [name for name, m in self.bus.motors.items() if m.model == "stepper"]
        if len(steppers) != 3:
            return False  
        # The ned2 is considered "calibrated" if the motor firmwares of the 3 steppers return "calibrated" (i.e. ==2)
        statuses = self.bus.sync_read("Homing_Status", steppers, normalize=False)
        return all(statuses[motor] == 2 for motor in steppers)

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        script_path = os.path.join(os.path.dirname(__file__), 'configure_all_steppers.py')
        script_path = os.path.abspath(script_path)
        subprocess.run([sys.executable, script_path], check=True)
        self.bus.disable_torque()

        input(f"Move {self} to the middle of its range of motion (with gripper open) and press ENTER....")

        actual_positions = self.bus.sync_read("Present_Position", normalize=False)

        homing_offsets = self.bus._get_half_turn_homings(actual_positions)

        range_mins = {
            "base_to_arm": 0,
            "arm_to_elbow": 0,
            "elbow_to_forearm": 0,
            "forearm_to_hand": 680,
            "hand_to_shoulder": 536,
            "shoulder_to_wrist": 0,
            "gripper": 800
        }

        range_maxes = {
            "base_to_arm": 3795,
            "arm_to_elbow": 1665,
            "elbow_to_forearm": 1800,
            "forearm_to_hand": 3406,
            "hand_to_shoulder": 3668,
            "shoulder_to_wrist": 4096,
            "gripper": 1895
        }

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            for motor, m in self.bus.motors.items():
                if m.model != "stepper":
                    self.bus.write("Return_Delay_Time", motor, 0)

    def setup_motors(self) -> None:
        """unimplemented"""
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read arm position
        start = time.perf_counter()
        positions = self.bus.sync_read("Present_Position", num_retry=10)
        for motor, val in positions.items():
            obs_dict[f"{motor}.pos"] = val
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        self._stop_socat_tunnel()

        logger.info(f"{self} disconnected.") 
