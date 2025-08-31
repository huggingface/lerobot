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
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class BasePhone:
    _enabled: bool = False
    _calib_pos: np.ndarray | None = None
    _calib_rot_inv: Rotation | None = None

    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        self._calib_pos = pos.copy()

    @property
    def is_calibrated(self) -> bool:
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "phone.pos": np.ndarray,  # shape (3,)
            "phone.rot": Rotation,  # scipy.spatial.transform.Rotation
            "phone.raw_inputs": dict,  # analogs/buttons or webXR meta
            "phone.enabled": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic or other feedback implemented yet
        pass

    def configure(self) -> None:
        # No additional configuration required for phone teleop
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # We could add haptic feedback (vibrations) here, but it's not implemented yet
        raise NotImplementedError


class IOSPhone(BasePhone, Teleoperator):
    name = "ios_phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._group = None

    @property
    def is_connected(self) -> bool:
        return self._group is not None

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
        lookup = hebi.Lookup()
        time.sleep(2.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        if group is None:
            raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
        self._group = group
        logger.info(f"{self} connected to HEBI group with {group.size} module(s).")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Press and hold B1 in the HEBI Mobile I/O app to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """Wait trigger for calibration: iOS: B1. Android: 'move'."""
        while True:
            ok, pos, rot, pose = self._read_current_pose()
            if not ok:
                time.sleep(0.01)
                continue

            io = getattr(pose, "io", None)
            b = getattr(io, "b", None) if io is not None else None
            b1 = False
            if b is not None:
                b1 = bool(b.get_int(1))
            if b1:
                return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        fbk = self._group.get_next_feedback()
        pose = fbk[0]
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return False, None, None, None
        # HEBI provides orientation in w, x, y, z format.
        # Scipy's Rotation expects x, y, z, w.
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz to xyzw
        rot = Rotation.from_quat(quat_xyzw)
        pos = ar_pos - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def get_action(self) -> dict:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok or not self.is_calibrated:
            return {}

        # Collect raw inputs (B1 / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
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

        enable = bool(raw_inputs.get("b1", 0))

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        # Apply calibration
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._group = None


class AndroidPhone(BasePhone, Teleoperator):
    name = "android_phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._teleop is not None

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Starting teleop stream for Android...")
        self._teleop = Teleop()
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        logger.info(f"{self} connected, teleop stream started.")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Touch and move on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """Wait trigger for calibration: iOS: B1. Android: 'move'."""
        while True:
            with self._android_lock:
                msg = self._latest_message or {}

            if bool(msg.get("move", False)):
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    def get_action(self) -> dict:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok or not self.is_calibrated:
            return {}

        # Collect raw inputs (B1 / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        msg = self._latest_message or {}
        raw_inputs["move"] = bool(msg.get("move", False))
        raw_inputs["scale"] = float(msg.get("scale", 1.0))
        raw_inputs["reservedButtonA"] = bool(msg.get("reservedButtonA", False))
        raw_inputs["reservedButtonB"] = bool(msg.get("reservedButtonB", False))

        enable = bool(raw_inputs.get("move", False))

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        # Apply calibration
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None


class Phone(Teleoperator):
    """
    Phone-based teleoperator using ARKit (iOS via HEBI Mobile I/O App) or the teleop Python package (Android via WebXR API).
    For HEBI Mobile I/O we also expose 8 analog (a1-a8) and 8 digital (b1-b8) inputs.

    Press and hold **B1** to enable teleoperation. While enabled, the first B1 press
    captures a reference pose and rotation, when disabled and pressed again the position is reapplied.
    """

    config_class = PhoneConfig
    name = "phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config

        self._phone_impl: Teleoperator

        if self.config.phone_os == PhoneOS.IOS:
            self._phone_impl = IOSPhone(config)
        elif self.config.phone_os == PhoneOS.ANDROID:
            self._phone_impl = AndroidPhone(config)
        else:
            raise ValueError(f"Invalid config phone_os: {self.config.phone_os}")

    @property
    def is_connected(self) -> bool:
        return self._phone_impl.is_connected

    def connect(self) -> None:
        return self._phone_impl.connect()

    def calibrate(self) -> None:
        return self._phone_impl.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._phone_impl.is_calibrated

    @property
    def action_features(self) -> dict[str, type]:
        return self._phone_impl.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return self._phone_impl.feedback_features

    def configure(self) -> None:
        return self._phone_impl.configure()

    def get_action(self) -> dict:
        return self._phone_impl.get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return self._phone_impl.send_feedback(feedback)

    def disconnect(self) -> None:
        return self._phone_impl.disconnect()
