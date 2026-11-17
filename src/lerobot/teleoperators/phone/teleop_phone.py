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
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _hebi_available, _teleop_available, require_package
from lerobot.utils.rotation import Rotation

if TYPE_CHECKING or _hebi_available:
    import hebi
else:
    hebi = None

if TYPE_CHECKING or _teleop_available:
    from teleop import Teleop
else:
    Teleop = None

if TYPE_CHECKING:
    from hebi._internal.ffi._message_types import GroupFeedback as HebiGroupFeedback
    from hebi._internal.group import Group as HebiGroup

from ..teleoperator import Teleoperator
from .config_phone import PhoneConfig, PhoneOS

logger = logging.getLogger(__name__)

_PHONE_ORIENTATION_DIAGRAM = r"""
  ── Calibration pose: hold the phone flat, screen up, top edge pointing FORWARD ──

    seen from above: +x forward is up the page, +y left is to the left

      robot                          phone (screen faces the ceiling)

          +x forward                     ╭───────────────╮
              ▲                          │▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔│ ← top edge, pointing
              │                          │               │   the same way as
       ╭──────┴──────╮                   │   screen up   │   the gripper (+x)
  ◄────┤   base ◉    │                   │               │
  +y   ╰─────────────╯                   ╰───────────────╯
  left

      +z is up, away from the table
"""

# The Mobile I/O app does not put every pin in every feedback packet, for the analog sliders as well
# as the buttons, so a pin's value is latched between the packets that carry it.
_PIN_STALE_AFTER_S = 0.5

# HEBI Mobile I/O pins: "B1" is the dead-man switch, "A3" the gripper slider.
_ENABLE_BUTTON_PIN = 1
_GRIPPER_SLIDER_PIN = 3


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
            "phone.gripper_vel": float,  # normalized gripper velocity, negative closes
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic or other feedback implemented yet
        return {}

    def configure(self) -> None:
        # No additional configuration required for phone teleop
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # We could add haptic feedback (vibrations) here, but it's not implemented yet
        raise NotImplementedError

    def _apply_calibration(self, raw_pos: np.ndarray, raw_rot: Rotation) -> tuple[np.ndarray, Rotation]:
        """Express a raw phone pose in the reference frame captured by `calibrate()`."""
        if self._calib_pos is None or self._calib_rot_inv is None:
            raise RuntimeError(f"{self} is not calibrated. Run `.calibrate()` first.")
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot
        return pos_cal, rot_cal


class IOSPhone(BasePhone, Teleoperator):
    name = "ios_phone"

    def __init__(self, config: PhoneConfig):
        require_package("hebi-py", extra="phone", import_name="hebi")
        require_package("teleop", extra="phone")
        super().__init__(config)
        self.config = config
        self._group: HebiGroup | None = None
        self._feedback: HebiGroupFeedback | None = None
        self._digital_state: dict[int, int] = {}
        self._digital_seen_at: dict[int, float] = {}
        self._analog_state: dict[int, float] = {}
        self._analog_seen_at: dict[int, float] = {}

    @property
    def is_connected(self) -> bool:
        return self._group is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
        lookup = hebi.Lookup()
        time.sleep(2.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        if group is None:
            raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
        self._group = group
        # Reuse one feedback object across polls, as the HEBI SDK recommends for repetitive reads.
        self._feedback = hebi.GroupFeedback(group.size)
        logger.info(f"{self} connected to HEBI group with {group.size} module(s).")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        enable_control = f"B{_ENABLE_BUTTON_PIN}"
        gripper_slider = f"A{_GRIPPER_SLIDER_PIN}"
        print(_PHONE_ORIENTATION_DIAGRAM)
        print(
            f"  {enable_control}  hold to move the arm, release to freeze it\n"
            f"  {gripper_slider}  slider drives the gripper: up opens, down closes\n"
            f"  both hands: one thumb on {enable_control}, the other on {gripper_slider}\n"
        )
        print(f"Hold the phone as shown, then press and hold {enable_control} to capture it...")

        position, rotation = self._wait_for_capture_trigger()
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._enabled = False
        print(f"Calibrated. Hold {enable_control} to move the robot.\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        Blocks execution until the calibration trigger is detected from the iOS device.

        This method enters a loop, continuously reading the phone's state. It waits for the user to
        press and hold B1 in the HEBI Mobile I/O app. Once it is pressed, the loop breaks and returns
        the phone's pose at that moment.

        Returns:
            A tuple containing the position (np.ndarray) and rotation (Rotation) of the phone at the
            moment the trigger was activated.
        """
        while True:
            pose = self._read_current_pose()
            if pose is None:
                time.sleep(0.01)
                continue
            position, rotation, fb_pose = pose

            enabled, _, _ = self._enable_signal(fb_pose)
            if enabled:
                return position, rotation

            time.sleep(0.01)

    def _enable_signal(self, fb_pose: object) -> tuple[bool, dict[int, int], dict[int, float]]:
        """Returns whether the dead-man switch is held, plus the latched digital and analog states."""
        digital = self._latched_digital_inputs(fb_pose)
        analog = self._latched_analog_inputs(fb_pose)
        return bool(digital.get(_ENABLE_BUTTON_PIN, 0)), digital, analog

    def _latched_analog_inputs(self, fb_pose: object) -> dict[int, float]:
        """Returns every analog pin's value (bank A), carried across packets that omit it.

        A slider unreported for longer than `_PIN_STALE_AFTER_S` reads as 0.0 rather than keeping
        its last value: a stale gripper velocity would keep driving the jaws after you let go.
        """
        io = getattr(fb_pose, "io", None)
        bank_a = getattr(io, "a", None) if io is not None else None
        now = time.perf_counter()

        if bank_a:
            for ch in range(1, 9):
                if bank_a.has_float(ch):
                    self._analog_state[ch] = float(bank_a.get_float(ch))
                    self._analog_seen_at[ch] = now

        return {
            ch: (value if now - self._analog_seen_at[ch] <= _PIN_STALE_AFTER_S else 0.0)
            for ch, value in self._analog_state.items()
        }

    def _latched_digital_inputs(self, fb_pose: object) -> dict[int, int]:
        """Returns every digital pin's state (bank B), carried across packets that omit it.

        The app sends a different subset of pins in each packet, so reading one packet in isolation
        makes a held button look released most of the time. Pins present in this packet update the
        latch; pins unreported for longer than `_PIN_STALE_AFTER_S` read as released.
        """
        io = getattr(fb_pose, "io", None)
        bank_b = getattr(io, "b", None) if io is not None else None
        now = time.perf_counter()

        if bank_b:
            for ch in range(1, 9):
                if bank_b.has_int(ch):
                    self._digital_state[ch] = int(bank_b.get_int(ch))
                    self._digital_seen_at[ch] = now
                elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                    self._digital_state[ch] = int(bank_b.get_bool(ch))
                    self._digital_seen_at[ch] = now

        return {
            ch: (value if now - self._digital_seen_at[ch] <= _PIN_STALE_AFTER_S else 0)
            for ch, value in self._digital_state.items()
        }

    def _read_current_pose(self) -> tuple[np.ndarray, Rotation, object] | None:
        """
        Reads the instantaneous 6-DoF pose from the connected iOS device via the HEBI SDK.

        This method fetches the latest feedback packet from the HEBI group, extracts the ARKit
        position and orientation, and converts them into a standard format. It also applies a
        configured camera offset to adjust the pose from the camera's frame to the phone's
        physical frame.

        Returns:
            None if no valid pose could be read, otherwise a tuple containing:
            - The 3D position as a NumPy array.
            - The orientation as a `Rotation` object.
            - The raw HEBI feedback object for accessing other data like button presses.
        """
        if self._group is None:
            raise DeviceNotConnectedError(f"{self} is not connected. Run `.connect()` first.")

        fbk = self._group.get_next_feedback(reuse_fbk=self._feedback)
        if fbk is None:
            # No packet arrived before the timeout; degrade like a missing pose.
            return None

        pose = fbk[0]
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return None
        # HEBI provides orientation in w, x, y, z format.
        # Scipy's Rotation expects x, y, z, w.
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz to xyzw
        # ARKit can emit zero/NaN quaternions before tracking is ready or on a
        # dropped packet. Rotation.from_quat now rejects those; degrade the same
        # way as a missing pose so teleop stays alive mid-session.
        try:
            rot = Rotation.from_quat(quat_xyzw)
        except ValueError:
            return None
        pos = ar_pos - rot.apply(self.config.camera_offset)
        return pos, rot, pose

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        pose = self._read_current_pose()
        if pose is None or not self.is_calibrated:
            return {}
        raw_position, raw_rotation, fb_pose = pose

        # Collect raw inputs (buttons / analogs on iOS, move/scale on Android)
        enable, digital, analog = self._enable_signal(fb_pose)
        raw_inputs: dict[str, float | int | bool] = {f"a{ch}": value for ch, value in analog.items()}
        raw_inputs.update({f"b{ch}": value for ch, value in digital.items()})

        gripper_vel = float(analog.get(_GRIPPER_SLIDER_PIN, 0.0))

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_position)

        pos_cal, rot_cal = self._apply_calibration(raw_position, raw_rotation)

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
            "phone.gripper_vel": gripper_vel,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._group = None
        self._feedback = None
        self._digital_state.clear()
        self._digital_seen_at.clear()
        self._analog_state.clear()
        self._analog_seen_at.clear()


class AndroidPhone(BasePhone, Teleoperator):
    name = "android_phone"

    def __init__(self, config: PhoneConfig):
        require_package("hebi-py", extra="phone", import_name="hebi")
        require_package("teleop", extra="phone")
        super().__init__(config)
        self.config = config
        self._teleop: Teleop | None = None
        self._teleop_thread: threading.Thread | None = None
        self._latest_pose: np.ndarray | None = None
        self._latest_message: dict[str, Any] | None = None
        self._android_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._teleop is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info("Starting teleop stream for Android...")
        self._teleop = Teleop()
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        logger.info(f"{self} connected, teleop stream started.")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        print(_PHONE_ORIENTATION_DIAGRAM)
        print(
            "  Move   hold to move the arm, release to freeze it\n"
            "  A / B  buttons drive the gripper: A opens, B closes\n"
        )
        print("Hold the phone as shown, then touch and move on the WebXR page to capture it...")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibrated. Hold Move to drive the robot.\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        Blocks execution until the calibration trigger is detected from the Android device.

        This method enters a loop, continuously checking the latest message received from the WebXR
        session. It waits for the user to touch and move their finger on the screen, which generates
        a `move` event. Once this event is detected, the loop breaks and returns the phone's current
        pose.

        Returns:
            A tuple containing the position (np.ndarray) and rotation (Rotation) of the phone at the
            moment the trigger was activated.
        """
        while True:
            with self._android_lock:
                msg = self._latest_message or {}

            if bool(msg.get("move", False)):
                pose = self._read_current_pose()
                if pose is not None:
                    pos, rot, _raw_pose = pose
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[np.ndarray, Rotation, np.ndarray] | None:
        """
        Reads the latest 6-DoF pose received from the Android device's WebXR session.

        This method accesses the most recent pose data stored by the `_android_callback`. It uses a
        thread lock to safely read the shared `_latest_pose` variable. The pose, a 4x4 matrix, is
        then decomposed into position and rotation, and the configured camera offset is applied.

        Returns:
            None if no pose has been received yet, otherwise a tuple containing:
            - The 3D position as a NumPy array.
            - The orientation as a `Rotation` object.
            - The raw 4x4 pose matrix as received from the teleop stream.
        """
        with self._android_lock:
            if self._latest_pose is None:
                return None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        return pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict[str, Any]) -> None:
        """
        Callback function to handle incoming data from the Android teleop stream.

        This method is executed by the `teleop` package's subscriber thread whenever a new
        pose and message are received from the WebXR session on the Android phone. It updates
        the internal state (`_latest_pose` and `_latest_message`) with the new data.
        A thread lock is used to ensure that these shared variables are updated atomically,
        preventing race conditions with the main thread that reads them.

        Args:
            pose: A 4x4 NumPy array representing the phone's transformation matrix.
            message: A dictionary containing additional data, such as button presses or touch events.
        """
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        pose = self._read_current_pose()
        if pose is None or not self.is_calibrated:
            return {}
        raw_pos, raw_rot, _raw_pose = pose

        # Collect raw inputs (buttons / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        msg = self._latest_message or {}
        raw_inputs["move"] = bool(msg.get("move", False))
        raw_inputs["scale"] = float(msg.get("scale", 1.0))
        raw_inputs["reservedButtonA"] = bool(msg.get("reservedButtonA", False))
        raw_inputs["reservedButtonB"] = bool(msg.get("reservedButtonB", False))

        enable = bool(raw_inputs.get("move", False))
        # Positive if A is pressed, negative if B is pressed, 0 if both or neither are pressed.
        gripper_vel = float(raw_inputs["reservedButtonA"]) - float(raw_inputs["reservedButtonB"])

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        pos_cal, rot_cal = self._apply_calibration(raw_pos, raw_rot)

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
            "phone.gripper_vel": gripper_vel,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None


class Phone(Teleoperator):
    """
    Phone-based teleoperator using ARKit (iOS via HEBI Mobile I/O App) or the teleop Python package (Android via WebXR API).
    For HEBI Mobile I/O we also expose 8 analog (a1-a8) and 8 digital (b1-b8) inputs.

    Teleoperation is gated by a switch that has to be **held down** for the whole
    motion: B1 on iOS, `Move` on Android. Releasing it
    freezes the robot; pressing again re-anchors the phone to the arm's current pose. The gripper
    runs off A3 on iOS and the reserved A/B buttons on
    Android, and stays live while the dead-man switch is released.
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

    def connect(self, calibrate: bool = True) -> None:
        return self._phone_impl.connect(calibrate)

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

    def get_action(self) -> RobotAction:
        return self._phone_impl.get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return self._phone_impl.send_feedback(feedback)

    def disconnect(self) -> None:
        return self._phone_impl.disconnect()
