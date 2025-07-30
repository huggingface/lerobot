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

# TODO(pepijn): Make sure each time enabled is clicked we calibrate again and set the axis of the target joint equal to the normal vector of the phone
# TODO(pepijn): Train pick place with phone teleop and check if code is still easy to use when recording etc now that we have a robot pipeline
# TODO(pepijn): Add to docs with image etc


class Phone(Teleoperator):
    """
    Phone-based teleoperator using ARKit (iOS via HEBI Mobile I/O App) or the teleop Python package (Android via WebXR API).
    For HEBI Mobile I/O we also expose 8 analog (a1-a8) and 8 digital (b1-b8) inputs.

    Press and hold **B1** to enable teleoperation. While enabled, the first B1 press
    captures a reference pose. Motion is mapped relative to that reference pose using
    a reference-frame approach (similar to TidyBot++ https://tidybot2.github.io)
    """

    config_class = PhoneConfig
    name = "phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._group = None
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._enabled: bool = False

        # Calibration origins
        self._calib_pos: np.ndarray | None = None
        self._calib_rot_inv: Rotation | None = None

    @property
    def is_connected(self) -> bool:
        return (self.config.phone_os == PhoneOS.IOS and self._group is not None) or (
            self.config.phone_os == PhoneOS.ANDROID and self._teleop is not None
        )

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.config.phone_os == PhoneOS.IOS:
            logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
            lookup = hebi.Lookup()
            time.sleep(2.0)
            group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
            if group is None:
                raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
            self._group = group
            logger.info(f"{self} connected to HEBI group with {group.size} module(s).")
        elif self.config.phone_os == PhoneOS.ANDROID:
            logger.info("Starting teleop stream for Android...")
            self._teleop = Teleop()
            self._teleop.subscribe(self._android_callback)
            self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
            self._teleop_thread.start()
            logger.info(f"{self} connected, teleop stream started.")
        else:
            raise ValueError(f"Invalid config phone_os: {self.config.phone_os}")

        self.calibrate()

    def calibrate(self) -> None:
        # Calibrate when the user explicitly triggers capture.
        print("Hold the phone so that: top edge points forward (robot +x) and screen points up (robot +z)")
        if self.config.phone_os == PhoneOS.IOS:
            print("Press and hold B1 in the HEBI Mobile I/O app to capture this pose...\n")
        else:
            print("Touch and move on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()

        # Store calibration transform: phone frame is now robot frame
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    @property
    def is_calibrated(self) -> bool:
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    @property
    def action_features(self) -> dict[str, type]:
        features = {
            "enabled": bool,  # if true the "enabled" the phone button is pressed
            "target_x": float,  # meters
            "target_y": float,
            "target_z": float,
            "target_qx": float,
            "target_qy": float,
            "target_qz": float,
            "target_qw": float,
            "gripper": float,  # gripper velocity in m/s
            "x": float,  # possible forward/backward
            "y": float,  # possible left/right
            "theta": float,  # possible rotation (in radians)
        }
        return features

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """Wait trigger for calibration: iOS: B1. Android: 'move'."""
        while True:
            ok, pos, rot, pose = self._read_current_pose()
            if not ok:
                time.sleep(0.01)
                continue

            if self.config.phone_os == PhoneOS.IOS:
                io = getattr(pose, "io", None)
                b = getattr(io, "b", None) if io is not None else None
                b1 = False
                if b is not None:
                    b1 = bool(b.get_bool(1))
                if b1:
                    return pos, rot
            else:
                msg = self._latest_message or {}
                if bool(msg.get("move", False)):
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        if self.config.phone_os == PhoneOS.IOS:
            fbk = self._group.get_next_feedback()
            pose = fbk[0]
            ar_pos = getattr(pose, "ar_position", None)
            ar_quat = getattr(pose, "ar_orientation", None)
            if ar_pos is None or ar_quat is None:
                return False, None, None, None
            quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz to xyzw
            rot = Rotation.from_quat(quat_xyzw)
            pos = ar_pos - rot.apply(self.config.camera_offset)
            return True, pos, rot, pose
        else:
            p = self._latest_pose
            if p is None:
                return False, None, None, None
            rot = Rotation.from_matrix(p[:3, :3])
            pos = p[:3, 3] - rot.apply(self.config.camera_offset)
            pose = self._latest_pose
            return True, pos, rot, pose

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic or other feedback implemented yet
        pass

    def configure(self) -> None:
        # No additional configuration required for phone teleop
        pass

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        self._latest_pose = pose
        self._latest_message = message
        time.sleep(0.001)  # 1ms delay to avoid race condition

    def get_action(self) -> dict:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok or not self.is_calibrated:
            return {}

        # Collect raw inputs (B1 / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        if self.config.phone_os == PhoneOS.IOS:
            io = getattr(pose, "io", None)
            if io is not None:
                bank_a, bank_b = io.a, io.b
                if bank_a:
                    for ch in range(1, 9):
                        if bank_a.has_float(ch):
                            raw_inputs[f"a{ch}"] = float(bank_a.get_float(ch))
                if bank_b:
                    for ch in range(1, 9):
                        if bank_b.has_int(ch):
                            raw_inputs[f"b{ch}"] = int(bank_b.get_int(ch))
                        elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                            raw_inputs[f"b{ch}"] = int(bank_b.get_bool(ch))
        else:
            msg = self._latest_message or {}
            raw_inputs["move"] = bool(msg.get("move", False))
            raw_inputs["scale"] = float(msg.get("scale", 1.0))

        # Apply calibration here
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        if self.config.phone_os == PhoneOS.IOS:
            b1 = bool(raw_inputs.get("b1", 0))
            enabled = b1
        else:
            enabled = bool(raw_inputs.get("move", False))
        self._enabled = bool(enabled)

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # We could add haptic feedback (vibrations) here, but it's not implemented yet
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.phone_os == PhoneOS.IOS:
            self._group = None
        else:
            self._teleop = None
            if self._teleop_thread and self._teleop_thread.is_alive():
                self._teleop_thread.join(timeout=1.0)
                self._teleop_thread = None
                self._latest_pose = None
