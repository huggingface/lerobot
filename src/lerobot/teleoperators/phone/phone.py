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

DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # tune per device

# ARKit world: +x right, +y up, +z forward
# Robot world: +x forward, +y left, +z up
_R_AR2R = (
    Rotation.from_euler("z", -90, degrees=True)  # right → fwd
    * Rotation.from_euler("x", 180, degrees=True)
)  # fix handedness (det = +1)

# WebXR world: +x right, +y up, +z back (-forward)
_R_WXR2R = Rotation.from_euler("z", -90, degrees=True)  # right → fwd


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
        self.source = config.phone_os
        self._group = None
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None

        # Arm control (reference-frame) state
        self._enabled: bool = False
        self._arm_ref_pos: np.ndarray | None = None
        self._arm_ref_rot_inv: Rotation | None = None

    def update_base_pose(self, x: float, y: float, yaw: float) -> None:
        """
        Update current base pose (in robot/world frame) for teleop compensation.
        Call this once per loop *before* get_action().
        """
        self._base_pose[:] = [x, y, yaw]

    @property
    def action_features(self) -> dict[str, type]:
        features = {
            "enabled": bool,
            "target_x": float,  # meters
            "target_y": float,
            "target_z": float,
            # Relative quaternion (delta) from reference -> current, XYZW
            "target_qx": float,
            "target_qy": float,
            "target_qz": float,
            "target_qw": float,
        }
        # Analog inputs normalized between -1 and 1
        for i in range(1, 9):
            features[f"a{i}"] = float
        # Digital inputs 0 or 1
        for i in range(1, 9):
            features[f"b{i}"] = int
        return features

    def _map_arkit_to_robot(self, pos_xyz: np.ndarray, quat_wxyz: np.ndarray):
        r_arkit = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])  # wxyz ➜ xyzw
        r_robot = _R_AR2R * r_arkit
        robot_pos = _R_AR2R.apply(pos_xyz) + r_robot.apply(DEVICE_CAMERA_OFFSET)
        return robot_pos, r_robot

    def _map_webxr_to_robot(self, pos_xyz: np.ndarray, quat_wxyz: np.ndarray):
        r_wxr = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])  # wxyz ➜ xyzw
        r_robot = _R_WXR2R * r_wxr
        robot_pos = _R_WXR2R.apply(pos_xyz) + r_robot.apply(DEVICE_CAMERA_OFFSET)
        return robot_pos, r_robot

    @property
    def is_connected(self) -> bool:
        if self.source == PhoneOS.IOS:
            return self._group is not None
        elif self.source == PhoneOS.ANDROID:
            return self._teleop is not None
        return False

    def connect(self) -> None:
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

    @property
    def is_calibrated(self) -> bool:
        # No calibration needed for phone teleop
        return True

    def calibrate(self) -> None:
        # No calibration needed for phone teleop
        pass

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
        if self.source == PhoneOS.IOS:
            fbk = self._group.get_next_feedback()
            pose = fbk[0]
            ar_pos = getattr(pose, "ar_position", None)
            ar_quat = getattr(pose, "ar_orientation", None)  # [w,x,y,z]
            if ar_pos is None or ar_quat is None:
                return {}
            curr_pos, r_curr = self._map_arkit_to_robot(
                np.array(ar_pos, dtype=float), np.array(ar_quat, dtype=float)
            )
        else:
            pose = self._latest_pose
            if pose is None:
                return {}
            # teleop library pose: 4x4 matrix in WebXR frame
            webxr_pos = pose[:3, 3]
            webxr_r = Rotation.from_matrix(pose[:3, :3])
            # Convert Rotation back to wxyz for a unified interface
            webxr_quat_wxyz = np.array([webxr_r.as_quat()[3], *webxr_r.as_quat()[:3]])
            curr_pos, r_curr = self._map_webxr_to_robot(
                np.array(webxr_pos, dtype=float), np.array(webxr_quat_wxyz, dtype=float)
            )
        # Build action with raw IO (so caller can still read buttons/analogs)
        action: dict[str, float | int] = {}

        if self.source == PhoneOS.IOS:
            io = getattr(pose, "io", None)
            if io is not None:
                bank_a = getattr(io, "a", None)
                bank_b = getattr(io, "b", None)

                if bank_a is not None:
                    for ch in range(1, 9):  # 1..8 inclusive
                        if bank_a.has_float(ch):
                            action[f"a{ch}"] = float(bank_a.get_float(ch))

                if bank_b is not None:
                    for ch in range(1, 9):  # 1..8 inclusive
                        if bank_b.has_int(ch):
                            action[f"b{ch}"] = int(bank_b.get_int(ch))
                        elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                            action[f"b{ch}"] = int(bank_b.get_bool(ch))

                    if not hasattr(self, "_printed_b_debug"):
                        logger.info(
                            "MobileIO bank_b debug: "
                            + ", ".join(f"b{n}={action.get(f'b{n}', 'NA')}" for n in range(1, 9))
                        )
                        self._printed_b_debug = True
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
            self._arm_ref_pos = curr_pos.copy()
            self._arm_ref_rot_inv = r_curr.inv()
            logger.info("Phone teleop enabled (B1 pressed) – reference pose captured.")

        # Falling edge: disable
        if (not b1_pressed) and self._enabled:
            self._enabled = False
            self._reset_arm_reference()
            logger.info("Phone teleop disabled (B1 released).")

        # Always include enabled flag
        action["enabled"] = int(self._enabled)

        if not self._enabled:
            # Zero absolute offsets when disabled
            action.update(
                {
                    "target_x": 0.0,
                    "target_y": 0.0,
                    "target_z": 0.0,
                    "target_qx": 0.0,
                    "target_qy": 0.0,
                    "target_qz": 0.0,
                    "target_qw": 1.0,
                }
            )
            return action

        # Absolute offsets relative to reference
        delta_world = curr_pos - self._arm_ref_pos
        # Express in reference frame so translation is phone‑centric
        pos_offset_total = self._arm_ref_rot_inv.apply(delta_world)

        # Relative rotation quaternion (reference -> current)
        rot_offset_total = r_curr * self._arm_ref_rot_inv  # Rotation object
        quat_xyzw = rot_offset_total.as_quat()  # xyzw
        # Canonicalize so qw >= 0
        if quat_xyzw[3] < 0:
            quat_xyzw = -quat_xyzw

        action.update(
            {
                "target_x": float(pos_offset_total[0]),
                "target_y": float(pos_offset_total[1]),
                "target_z": float(pos_offset_total[2]),
                "target_qx": float(quat_xyzw[0]),
                "target_qy": float(quat_xyzw[1]),
                "target_qz": float(quat_xyzw[2]),
                "target_qw": float(quat_xyzw[3]),
            }
        )
        return action

    def _reset_arm_reference(self):
        self._arm_ref_pos = None
        self._arm_ref_rot_inv = None

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
