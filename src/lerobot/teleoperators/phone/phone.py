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

# TODO(pepijn): specify the phone calibration point as axis of a robot "frame" that is going to be aligned with the normal axis of the back of the phone. at calibration and then each time b1 is pressed again.


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

        # Always calibrate on connect
        self.calibrate()

    def calibrate(self) -> None:
        # Ask the human to hold the phone in its neutral pose and press B1.
        print("\n Phone <-> Robot calibration")
        print("Hold the phone so that: top edge points forward (robot +x) and screen points up (robot +z)")
        print("Press and hold B1 to capture this pose...\n")

        pos, rot = self._wait_for_stable_pose()
        self._wait_for_button_press("b1", 0.3)

        # Store calibration transform: phone frame  is now robot frame
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
            "enabled": bool,
            "target_x": float,  # meters
            "target_y": float,
            "target_z": float,
            "target_qx": float,
            "target_qy": float,
            "target_qz": float,
            "target_qw": float,
        }
        for i in range(1, 9):  # Analog inputs between -1 and 1
            features[f"a{i}"] = float

        for i in range(1, 9):  # Digital inputs 0 or 1
            features[f"b{i}"] = int
        return features

    def _wait_for_stable_pose(self) -> tuple[np.ndarray, Rotation]:
        while True:
            ok, pos, rot, _ = self._read_current_pose()
            if ok:
                return pos, rot
            time.sleep(0.01)

    def _wait_for_button_press(self, button: str, hold_duration: float) -> None:
        required_count = int(hold_duration / 0.01)
        consecutive = 0
        while consecutive < required_count:
            act = self.get_action()
            if act.get(button, 0):
                consecutive += 1
            else:
                consecutive = 0
            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None]:
        if self.config.phone_os == PhoneOS.IOS:
            fbk = self._group.get_next_feedback()
            pose = fbk[0]
            ar_pos = getattr(pose, "ar_position", None)
            ar_quat = getattr(pose, "ar_orientation", None)  # wxyz
            if ar_pos is None or ar_quat is None:
                return False, None, None, None
            quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz to xyzw
            rot = Rotation.from_quat(quat_xyzw)
            pos = ar_pos - rot.apply(self.config.camera_offset)
            return True, pos, rot, pose
        else:  # Android
            p = self._latest_pose
            if p is None:
                return False, None, None, None
            webxr_pos = p[:3, 3]
            webxr_rot = Rotation.from_matrix(p[:3, :3])
            webxr_wxyz = np.array([webxr_rot.as_quat()[3], *webxr_rot.as_quat()[:3]])  # wxyz to xyzw
            pose = self._latest_pose
            pos, rot = self._map_webxr_to_robot(np.array(webxr_pos, dtype=float), webxr_wxyz)
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
        time.sleep(0.001)  # 1ms delay to avoid race condition

    def _apply_calibration(self, pos: np.ndarray, rot: Rotation) -> tuple[np.ndarray, Rotation]:
        if not self.is_calibrated:
            raise RuntimeError("Phone not calibrated, cannot apply calibration.")
        pos_cal = self._calib_rot_inv.apply(pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * rot
        return pos_cal, rot_cal

    def get_action(self) -> dict[str, float]:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok:
            return {}

        action: dict[str, float | int] = {}
        if self.config.phone_os == PhoneOS.IOS:
            io = getattr(pose, "io", None)
            if io is not None:
                bank_a = getattr(io, "a", None)
                bank_b = getattr(io, "b", None)

                if bank_a is not None:
                    for ch in range(1, 9):
                        if bank_a.has_float(ch):
                            action[f"a{ch}"] = float(bank_a.get_float(ch))

                if bank_b is not None:
                    for ch in range(1, 9):
                        if bank_b.has_int(ch):
                            action[f"b{ch}"] = int(bank_b.get_int(ch))
                        elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                            action[f"b{ch}"] = int(bank_b.get_bool(ch))

        else:
            # Android TODO: implement analog/digital inputs
            pass

        for i in range(1, 9):
            action.setdefault(f"a{i}", 0.0)
        for i in range(1, 9):
            action.setdefault(f"b{i}", 0)

        # Digital button B1 (index 1 in our dictionary). If absent, treat as 0.
        b1_pressed = bool(action.get("b1", 0))
        # Rising edge: capture reference
        if b1_pressed and not self._enabled:
            self._enabled = True

        # Falling edge: disable
        if (not b1_pressed) and self._enabled:
            self._enabled = False

        if not self._enabled or not self.is_calibrated:
            action.update(
                {
                    "enabled": 0,
                    "target_x": 0.0,
                    "target_y": 0.0,
                    "target_z": 0.0,
                    "target_qx": 0.0,
                    "target_qy": 0.0,
                    "target_qz": 0.0,
                    "target_qw": 0.0,
                }
            )
            return action

        curr_pos, r_curr = self._apply_calibration(raw_pos, raw_rot)
        quat_xyzw = r_curr.as_quat()

        # Canonicalize
        if quat_xyzw[3] < 0:
            quat_xyzw = -quat_xyzw

        action.update(
            {
                "enabled": int(self._enabled),
                "target_x": -1.0 * float(curr_pos[1]),
                "target_y": float(curr_pos[0]),
                "target_z": float(curr_pos[2]),
                "target_qx": float(quat_xyzw[1]),
                "target_qy": float(quat_xyzw[0]),
                "target_qz": -1.0 * float(quat_xyzw[2]),
                "target_qw": float(quat_xyzw[3]),
            }
        )
        return action

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
