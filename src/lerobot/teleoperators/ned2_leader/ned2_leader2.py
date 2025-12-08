#!/usr/bin/env python

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
import shutil

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_ned2_leader import Ned2LeaderConfig

logger = logging.getLogger(__name__)

class Ned2Leader(Teleoperator):
    """
    The ned2 that will lead the robot (Phantson Technologies Inc.) -- Unofficial
    """
    config_class = Ned2LeaderConfig
    name = "ned2_leader"

    def __init__(self, config: Ned2LeaderConfig):
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

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        #TODO (@atphantson) : Implement feedback features
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

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

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._start_socat_tunnel()

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        steppers = [name for name, m in self.bus.motors.items() if m.model == "stepper"]
        if len(steppers) != 3:
            return False
        statuses = self.bus.sync_read("Homing_Status", steppers, normalize=False)
        return all(statuses[motor] == 2 for motor in steppers)

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configure_all_steppers_of_teleop.py'))
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
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position", num_retry=10)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect()
        self._stop_socat_tunnel()
        logger.info(f"{self} disconnected.") 
