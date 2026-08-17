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
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
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

from ..teleoperator import Teleoperator
from .config_phone import PhoneConfig, PhoneOS

logger = logging.getLogger(__name__)


class BasePhone:
    """Shared calibration state and `Teleoperator` interface parts common to both phone backends.

    `IOSPhone` and `AndroidPhone` mix this in alongside `Teleoperator` so that the action/feedback feature
    schemas, calibration status, and the no-op configuration step only need to be written once. Each
    backend implements the parts that genuinely differ: connecting, reading the raw pose, and capturing a
    calibration reference.
    """

    _enabled: bool = False
    _calib_pos: np.ndarray | None = None
    _calib_rot_inv: Rotation | None = None

    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        self._calib_pos = pos.copy()

    @property
    def is_calibrated(self) -> bool:
        """Whether a calibration reference pose has been captured.

        Returns:
            `bool`: `True` once both a reference position and inverse rotation have been recorded by
            `calibrate`.
        """
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    @property
    def action_features(self) -> dict[str, type]:
        """Describe the action dictionary returned by `get_action`.

        Returns:
            `dict[str, type]`: Maps `"phone.pos"` (3D position, shape `(3,)`), `"phone.rot"` (orientation,
            a `scipy.spatial.transform.Rotation`), `"phone.raw_inputs"` (device-specific analog/button or
            WebXR values), and `"phone.enabled"` (whether the teleoperation trigger is currently held) to
            their value types.
        """
        return {
            "phone.pos": np.ndarray,  # shape (3,)
            "phone.rot": Rotation,  # scipy.spatial.transform.Rotation
            "phone.raw_inputs": dict,  # analogs/buttons or webXR meta
            "phone.enabled": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback schema accepted by `send_feedback`.

        No haptic or other feedback channel is implemented for phone teleoperators yet.

        Returns:
            `dict[str, type]`: Currently always `None`, since `feedback_features` has no implementation
            yet; this deviates from the declared return type and should not be relied on.
        """
        # No haptic or other feedback implemented yet
        pass

    def configure(self) -> None:
        """No-op. Phone teleoperators require no runtime configuration.

        See [`~teleoperators.Teleoperator.configure`] for the base contract.
        """
        # No additional configuration required for phone teleop
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Not implemented. Phone teleoperators do not support feedback yet.

        Args:
            feedback (`dict[str, float]`):
                Feedback values; see [`~teleoperators.Teleoperator.send_feedback`] for the base contract.

        Raises:
            NotImplementedError: Always. Haptic feedback (phone vibration) is not implemented yet.
        """
        # We could add haptic feedback (vibrations) here, but it's not implemented yet
        raise NotImplementedError


class IOSPhone(BasePhone, Teleoperator):
    """ARKit-based teleoperator backend for iOS, driven through the HEBI Mobile I/O app.

    Reads the phone's 6-DoF pose (position and orientation) captured by ARKit and relayed over the HEBI
    SDK, along with the app's 8 analog (`a1`-`a8`) and 8 digital (`b1`-`b8`) inputs. `Phone` instantiates
    this internally when `PhoneConfig.phone_os` is `PhoneOS.IOS`; use `Phone` directly rather than this
    class.

    Initializing an instance checks for the optional dependencies this backend needs.

    Args:
        config (`PhoneConfig`): Configuration shared with the parent `Phone` teleoperator.

    Raises:
        ImportError: If the `hebi-py` or `teleop` packages are not installed.
    """

    name = "ios_phone"

    def __init__(self, config: PhoneConfig):
        require_package("hebi-py", extra="phone", import_name="hebi")
        require_package("teleop", extra="phone")
        super().__init__(config)
        self.config = config
        self._group = None

    @property
    def is_connected(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_connected`].

        Returns:
            `bool`: `True` once a HEBI feedback group has been acquired by `connect`.
        """
        return self._group is not None

    @check_if_already_connected
    def connect(self) -> None:
        """Look up the HEBI Mobile I/O group over the network, then calibrate.

        Waits briefly for the HEBI lookup service to discover the phone running the Mobile I/O app under
        the `"HEBI"` family / `"mobileIO"` name, then immediately runs `calibrate`, which blocks until the
        user captures a reference pose in the app. Unlike
        [`~teleoperators.Teleoperator.connect`], this method always calibrates; there is no way to skip it.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            RuntimeError: If no matching Mobile I/O group is found on the network.
        """
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
        """Block until the user captures a reference pose via the HEBI Mobile I/O app.

        Prompts the user to hold the phone so its top edge points along the robot's +x axis and its
        screen faces the robot's +z axis, then to press and hold button `B1` in the app to capture that
        pose as the calibration reference. See [`~teleoperators.Teleoperator.calibrate`] for the base
        contract.
        """
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Press and hold B1 in the HEBI Mobile I/O app to capture this pose...\n")
        position, rotation = self._wait_for_capture_trigger()
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """Blocks execution until the calibration trigger is detected from the iOS device.

        This method enters a loop, continuously reading the phone's state. It waits for the user to press
        and hold the 'B1' button in the HEBI Mobile I/O app. Once B1 is pressed, the loop breaks and
        returns the phone's pose at that exact moment.

        Returns:
            A tuple containing the position (np.ndarray) and rotation (Rotation) of the phone at the
            moment the trigger was activated.
        """
        while True:
            has_pose, position, rotation, fb_pose = self._read_current_pose()
            if not has_pose:
                time.sleep(0.01)
                continue

            io = getattr(fb_pose, "io", None)
            button_b = getattr(io, "b", None) if io is not None else None
            button_b1_pressed = False
            if button_b is not None:
                button_b1_pressed = bool(button_b.get_int(1))
            if button_b1_pressed:
                return position, rotation

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """Reads the instantaneous 6-DoF pose from the connected iOS device via the HEBI SDK.

        This method fetches the latest feedback packet from the HEBI group, extracts the ARKit
        position and orientation, and converts them into a standard format. It also applies a
        configured camera offset to adjust the pose from the camera's frame to the phone's
        physical frame.

        Returns:
            A tuple containing:
            - A boolean indicating if a valid pose was successfully read.
            - The 3D position as a NumPy array, or None if not available.
            - The orientation as a `Rotation` object, or None if not available.
            - The raw HEBI feedback object for accessing other data like button presses.
        """
        fbk = self._group.get_next_feedback()
        pose = fbk[0]
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return False, None, None, None
        # HEBI provides orientation in w, x, y, z format.
        # Scipy's Rotation expects x, y, z, w.
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz to xyzw
        # ARKit can emit zero/NaN quaternions before tracking is ready or on a
        # dropped packet. Rotation.from_quat now rejects those; degrade the same
        # way as a missing pose so teleop stays alive mid-session.
        try:
            rot = Rotation.from_quat(quat_xyzw)
        except ValueError:
            return False, None, None, None
        pos = ar_pos - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    @check_if_not_connected
    def get_action(self) -> dict:
        """Read the phone's current calibrated pose and raw HEBI inputs.

        Applies the calibration captured by `calibrate` to the raw ARKit pose, and re-anchors the
        reference position on the rising edge of the `b1` "enable" button so that moving the phone while
        disabled does not cause a jump once teleoperation resumes.

        Returns:
            `dict`: Matches `action_features`: `"phone.pos"`, `"phone.rot"`, `"phone.raw_inputs"` (the
            app's analog/digital channel values, keyed e.g. `"a1"`, `"b1"`), and `"phone.enabled"`. An
            empty `dict` if no pose has been received yet or the teleoperator has not been calibrated.

        Raises:
            DeviceNotConnectedError: If `connect` has not been called.
        """
        has_pose, raw_position, raw_rotation, fb_pose = self._read_current_pose()
        if not has_pose or not self.is_calibrated:
            return {}

        # Collect raw inputs (B1 / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        io = getattr(fb_pose, "io", None)
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
            self._reapply_position_calibration(raw_position)

        # Apply calibration
        pos_cal = self._calib_rot_inv.apply(raw_position - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rotation

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        """See [`~teleoperators.Teleoperator.disconnect`].

        Raises:
            DeviceNotConnectedError: If `connect` has not been called.
        """
        self._group = None


class AndroidPhone(BasePhone, Teleoperator):
    """WebXR-based teleoperator backend for Android, driven through the `teleop` Python package.

    Runs the `teleop` package's local WebXR server on a background thread and reads the pose and touch
    events posted by the phone's browser session. `Phone` instantiates this internally when
    `PhoneConfig.phone_os` is `PhoneOS.ANDROID`; use `Phone` directly rather than this class.

    Initializing an instance checks for the optional dependencies this backend needs.

    Args:
        config (`PhoneConfig`): Configuration shared with the parent `Phone` teleoperator.

    Raises:
        ImportError: If the `hebi-py` or `teleop` packages are not installed.
    """

    name = "android_phone"

    def __init__(self, config: PhoneConfig):
        require_package("hebi-py", extra="phone", import_name="hebi")
        require_package("teleop", extra="phone")
        super().__init__(config)
        self.config = config
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_connected`].

        Returns:
            `bool`: `True` once the `teleop` background thread has been started by `connect`.
        """
        return self._teleop is not None

    @check_if_already_connected
    def connect(self) -> None:
        """Start the `teleop` WebXR server on a background thread, then calibrate.

        Subscribes to pose/message updates from the `teleop` package and starts its server loop on a
        daemon thread, then immediately runs `calibrate`, which blocks until the user captures a reference
        pose from the phone's browser session. Unlike
        [`~teleoperators.Teleoperator.connect`], this method always calibrates; there is no way to skip
        it.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
        """
        logger.info("Starting teleop stream for Android...")
        self._teleop = Teleop()
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        logger.info(f"{self} connected, teleop stream started.")

        self.calibrate()

    def calibrate(self) -> None:
        """Block until the user captures a reference pose via touch on the WebXR page.

        Prompts the user to hold the phone so its top edge points along the robot's +x axis and its
        screen faces the robot's +z axis, then to touch and move a finger on the WebXR page to capture
        that pose as the calibration reference. See [`~teleoperators.Teleoperator.calibrate`] for the base
        contract.
        """
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
        """Blocks execution until the calibration trigger is detected from the Android device.

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
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """Reads the latest 6-DoF pose received from the Android device's WebXR session.

        This method accesses the most recent pose data stored by the `_android_callback`. It uses a
        thread lock to safely read the shared `_latest_pose` variable. The pose, a 4x4 matrix, is
        then decomposed into position and rotation, and the configured camera offset is applied.

        Returns:
            A tuple containing:
            - A boolean indicating if a valid pose was available.
            - The 3D position as a NumPy array, or None if no pose has been received yet.
            - The orientation as a `Rotation` object, or None if no pose has been received.
            - The raw 4x4 pose matrix as received from the teleop stream.
        """
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        """Callback function to handle incoming data from the Android teleop stream.

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
    def get_action(self) -> dict:
        """Read the phone's current calibrated pose and raw touch/button state.

        Applies the calibration captured by `calibrate` to the latest pose received from the `teleop`
        background thread, and re-anchors the reference position on the rising edge of the `"move"` touch
        event so that moving the phone while disabled does not cause a jump once teleoperation resumes.

        Returns:
            `dict`: Matches `action_features`: `"phone.pos"`, `"phone.rot"`, `"phone.raw_inputs"`
            (`"move"`, `"scale"`, `"reservedButtonA"`, `"reservedButtonB"`), and `"phone.enabled"`. An
            empty `dict` if no pose has been received yet or the teleoperator has not been calibrated.

        Raises:
            DeviceNotConnectedError: If `connect` has not been called.
        """
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

    @check_if_not_connected
    def disconnect(self) -> None:
        """Stop the `teleop` background thread.

        Raises:
            DeviceNotConnectedError: If `connect` has not been called.
        """
        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None


class Phone(Teleoperator):
    """Phone-based teleoperator: iOS via ARKit and the HEBI Mobile I/O app, Android via WebXR.

    Reads the phone's 6-DoF pose and, for the HEBI Mobile I/O app, 8 analog (`a1`-`a8`) and 8 digital
    (`b1`-`b8`) inputs. Which backend is used is picked at construction time from
    `config.phone_os` and delegated to internally: [`~teleoperators.Teleoperator`] method calls on `Phone`
    forward to either an `IOSPhone` or an `AndroidPhone` instance.

    Press and hold **B1** (iOS) or touch and move on the WebXR page (Android) to enable teleoperation.
    The first press/touch while enabled captures a reference pose; releasing and re-triggering re-anchors
    the reference position to wherever the phone currently is, so motion is always relative to where
    teleoperation was last resumed.

    Initializing an instance picks and constructs the backend matching `config.phone_os`.

    Args:
        config (`PhoneConfig`): Configuration selecting the phone platform (`config.phone_os`) and
            forwarded to the chosen backend.

    Raises:
        ValueError: If `config.phone_os` is not a valid `PhoneOS` member.
        ImportError: If the `hebi-py` or `teleop` packages are not installed.

    Example:
        ```python
        >>> from lerobot.teleoperators.phone import Phone, PhoneConfig
        >>> teleop = Phone(PhoneConfig())  # doctest: +SKIP
        >>> teleop.connect()  # doctest: +SKIP
        >>> teleop.get_action()  # doctest: +SKIP
        ```
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
        """See [`~teleoperators.Teleoperator.is_connected`].

        Returns:
            `bool`: `True` if the underlying `IOSPhone` or `AndroidPhone` backend is connected.
        """
        return self._phone_impl.is_connected

    def connect(self) -> None:
        """Connect and calibrate through the underlying backend.

        Unlike [`~teleoperators.Teleoperator.connect`], this always calibrates; there is no `calibrate`
        argument to opt out.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            RuntimeError: If the iOS backend cannot find the Mobile I/O group on the network.
        """
        return self._phone_impl.connect()

    def calibrate(self) -> None:
        """See [`~teleoperators.Teleoperator.calibrate`]. Delegates to the underlying backend."""
        return self._phone_impl.calibrate()

    @property
    def is_calibrated(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_calibrated`].

        Returns:
            `bool`: `True` once a calibration reference pose has been captured.
        """
        return self._phone_impl.is_calibrated

    @property
    def action_features(self) -> dict[str, type]:
        """See [`~teleoperators.Teleoperator.action_features`].

        Returns:
            `dict[str, type]`: `"phone.pos"`, `"phone.rot"`, `"phone.raw_inputs"`, and `"phone.enabled"`
            mapped to their value types; see `get_action` for what each holds.
        """
        return self._phone_impl.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        """See [`~teleoperators.Teleoperator.feedback_features`].

        Returns:
            `dict[str, type]`: Currently always `None`, since no feedback channel is implemented yet.
        """
        return self._phone_impl.feedback_features

    def configure(self) -> None:
        """No-op. See [`~teleoperators.Teleoperator.configure`]."""
        return self._phone_impl.configure()

    def get_action(self) -> dict:
        """Read the phone's current calibrated pose and raw inputs from the underlying backend.

        Returns:
            `dict`: Matches `action_features`. An empty `dict` if no pose has been received yet or the
            teleoperator has not been calibrated.

        Raises:
            DeviceNotConnectedError: If `connect` has not been called.
        """
        return self._phone_impl.get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Not implemented. See [`~teleoperators.Teleoperator.send_feedback`].

        Args:
            feedback (`dict[str, float]`):
                Feedback values; unused.

        Raises:
            NotImplementedError: Always. Haptic feedback is not implemented yet.
        """
        return self._phone_impl.send_feedback(feedback)

    def disconnect(self) -> None:
        """See [`~teleoperators.Teleoperator.disconnect`]. Delegates to the underlying backend.

        Raises:
            DeviceNotConnectedError: If `connect` has not been called.
        """
        return self._phone_impl.disconnect()
