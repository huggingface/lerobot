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

# Docs:
# hebi: https://docs.hebi.us/tools.html#mobile-io
# teleop: https://github.com/SpesRobotics/teleop

import logging
import threading
import time

import hebi
import numpy as np
from scipy.spatial.transform import Rotation
from teleop import Teleop

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_phone import PhoneConfig, PhoneOS

logger = logging.getLogger(__name__)


class Phone(Teleoperator):
    """
    Phone-based teleoperator using ARKit (iOS via HEBI Mobile I/O App) or the teleop Python package (Android via WebXR API).
    For HEBI Mobile I/O we also expose 8 analog (a1-a8) and 8 digital (b1-b8) inputs.

    After calibration, ``position.{x,y,z}`` report absolute offsets from the calibration
    pose, while ``delta_{x,y,z}`` report incremental changes between consecutive frames.
    """

    config_class = PhoneConfig
    name = "phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self.source = config.phone_os
        self._group = None
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None

        # Previous position sample (for deltas)
        self._prev_position: np.ndarray | None = None
        # Fixed reference position captured at calibration time (for absolute values)
        self._calib_position: np.ndarray | None = None

    @property
    def action_features(self) -> dict[str, type]:
        # Absolute position (relative to calibration), incremental deltas, orientation, analog & digital IO
        features = {
            "position.x": float,
            "position.y": float,
            "position.z": float,
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "orientation.x": float,
            "orientation.y": float,
            "orientation.z": float,
            "orientation.w": float,
        }
        # Analog inputs normalized between -1 and 1
        for i in range(1, 9):
            features[f"a{i}"] = float
        # Digital inputs 0 or 1
        for i in range(1, 9):
            features[f"b{i}"] = int
        return features

    @property
    def is_connected(self) -> bool:
        if self.source == PhoneOS.IOS:
            return self._group is not None
        elif self.source == PhoneOS.ANDROID:
            return self._teleop is not None
        return False

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.source == PhoneOS.IOS:
            logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
            lookup = hebi.Lookup()
            time.sleep(2.0)
            group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
            if group is None:
                raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
            self._group = group
            logger.info(f"{self} connected to HEBI group with {group.size} module(s).")
        elif self.source == PhoneOS.ANDROID:
            logger.info("Starting teleop stream for Android...")
            self._teleop = Teleop()
            self._teleop.subscribe(self._android_callback)
            self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
            self._teleop_thread.start()
            logger.info(f"{self} connected, teleop stream started.")
        else:
            raise ValueError(f"Invalid config phone_os: {self.source}")

        if not self.is_calibrated and calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._calib_position is not None

    def calibrate(self) -> None:
        """
        Reads a single phone pose sample and sets it as the reference position for future delta computations.
        """
        input(
            "Please hold the phone still at the desired origin and press Enter to calibrate reference pose..."
        )
        logger.info(f"{self} calibrating reference pose...")
        if self.source == PhoneOS.IOS:
            fbk = None
            while fbk is None:
                fbk = self._group.get_next_feedback()
            pose = fbk[0]
            pos = getattr(pose, "ar_position", None)
        else:
            while self._latest_pose is None:
                time.sleep(0.01)
            pose = self._latest_pose
            pos = pose[:3, 3]
        if pos is None:
            raise RuntimeError("Failed to read position during calibration.")
        self._calib_position = np.array(pos, dtype=float)
        self._prev_position = self._calib_position.copy()
        logger.info(f"{self} reference position set to {self._calib_position}.")

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic or other feedback implemented yet
        pass

    def configure(self) -> None:
        # No additional configuration required for phone teleop
        pass

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        self._latest_pose = pose
        time.sleep(0.001)  # 1ms delay to avoid race condition

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()

        if self.source == PhoneOS.IOS:
            fbk = self._group.get_next_feedback()
            pose = fbk[0]
            position = getattr(pose, "ar_position", None)
            orientation_quat = getattr(pose, "ar_orientation", None)
            analogs = getattr(pose, "analog_input", None)
            digitals = getattr(pose, "digital_input", None)
        else:
            pose = self._latest_pose
            if pose is None:
                return {}
            position = pose[:3, 3]
            orientation_matrix = pose[:3, :3]
            r = Rotation.from_matrix(orientation_matrix)
            xyzw = r.as_quat()
            orientation_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            analogs = getattr(pose, "analog_input", None)
            digitals = getattr(pose, "digital_input", None)

        if position is None or orientation_quat is None:
            return {}

        curr_pos = np.array(position, dtype=float)

        # Absolute offset from calibration
        absolute = curr_pos - self._calib_position
        # Incremental delta from previous frame
        delta = curr_pos - self._prev_position
        self._prev_position = curr_pos.copy()

        action: dict[str, float] = {
            "position.x": float(absolute[0]),
            "position.y": float(absolute[1]),
            "position.z": float(absolute[2]),
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
            "orientation.x": float(orientation_quat[1]),
            "orientation.y": float(orientation_quat[2]),
            "orientation.z": float(orientation_quat[3]),
            "orientation.w": float(orientation_quat[0]),
        }

        # TODO(pepijn): add analog and digital inputs for Android
        action.update({f"a{i}": 0.0 for i in range(1, 9)})
        if analogs is not None:
            for i in range(min(8, len(analogs))):
                action[f"a{i + 1}"] = float(analogs[i])

        action.update({f"b{i}": 0 for i in range(1, 9)})
        if digitals is not None:
            for i in range(min(8, len(digitals))):
                action[f"b{i + 1}"] = int(digitals[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.info(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # We could add haptic feedback (phonevibrations) here, but it's not implemented yet
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # TODO(pepijn): Do this in a more clean way
        if self.source == PhoneOS.IOS:
            self._group = None  # HEBI has no explicit disconnect
        else:
            self._teleop = None  # Teleop thread will exit on program end

        logger.info(f"{self} disconnected.")
