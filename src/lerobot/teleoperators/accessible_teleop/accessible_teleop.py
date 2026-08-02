# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_accessible_teleop import (
    AccessibleTeleopConfig,
    ChannelCalibration,
    InputSource,
    JointBinding,
)
from .control import InputFrame, JointController
from .web_bridge import ControlBridge

logger = logging.getLogger(__name__)


class AccessibleTeleop(Teleoperator):
    """Joint-space teleoperation driven by whatever movement the operator actually has.

    This teleoperator exists for people who cannot use a leader arm. Instead of assuming a
    pair of working hands, it lets the operator bind each robot joint, one at a time, to any
    small input they can produce reliably: a facial movement picked up by the webcam, a
    single key, or an on-screen joystick they can reach with a mouse, a trackball, or a head
    pointer. Each binding is calibrated to that person's own comfortable range and tuned
    separately for dead zone, gain, smoothing, and speed.

    Control is joint-space and velocity-integrated. Deflecting an input moves a joint while
    it is held and stops when the input returns to neutral, which is what makes a few
    millimetres of usable movement enough to reach across the robot's whole range.

    A browser page served on loopback is the input surface. It reports raw sensor readings
    and lets the operator edit their mapping; the conditioning, integration, and limiting all
    happen here, so a stalled tab or a dropped socket stops the robot rather than surprising
    it.

    Example:
        ```python
        from lerobot.teleoperators.accessible_teleop import AccessibleTeleop, AccessibleTeleopConfig

        teleop = AccessibleTeleop(AccessibleTeleopConfig(id="my_profile"))
        teleop.connect()
        action = teleop.get_action()
        ```

    Safety:
        The teleoperator returns an empty action until the operator engages the clutch on the
        page. The first engaged action commands :pyattr:`AccessibleTeleopConfig.start_pose`,
        which the follower may be far away from. Either call :pymeth:`send_feedback` with the
        robot's current positions first, or run the follower with ``--robot.max_relative_target``
        so that the approach is rate limited by the robot as well.
    """

    config_class = AccessibleTeleopConfig
    name = "accessible_teleop"

    def __init__(self, config: AccessibleTeleopConfig):
        self.config = config
        self.bindings: dict[str, JointBinding] = {
            joint: config.bindings.get(joint, JointBinding()) for joint in config.joints
        }
        self.channel_calibrations: dict[str, ChannelCalibration] = {}

        # Populates bindings and channel calibrations from the saved profile, if there is one.
        super().__init__(config)

        self.controller = JointController(
            joints=config.joints,
            bindings=self.bindings,
            joint_limits={j: tuple(v) for j, v in config.joint_limits.items()},
            start_pose=dict(config.start_pose),
            calibrations=self.channel_calibrations,
            max_step_s=config.max_step_s,
        )
        self._bridge: ControlBridge | None = None
        self._last_step: float | None = None
        self._armed = False
        self._stale_logged = False

    # ── features ─────────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joints}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._bridge is not None and self._bridge.is_running

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._bridge = ControlBridge(
            config=self.config,
            bindings=self.bindings,
            calibrations=self.channel_calibrations,
            on_profile_change=self._on_profile_change,
        )
        self._bridge.start()

        if self.config.open_browser:
            webbrowser.open(self._bridge.url)

        print(f"\nOpen the accessible teleop control page at {self._bridge.url}")
        print("Waiting for the page to connect...")
        if not self._bridge.wait_for_client(self.config.connect_timeout_s):
            self._bridge.stop()
            self._bridge = None
            raise TimeoutError(
                f"No control page connected within {self.config.connect_timeout_s:.0f}s. "
                f"Open {self.config.host}:{self.config.web_port} in a browser and retry."
            )

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        self._warn_about_start_pose()
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self._bridge.release_clutch()
        self._bridge.stop()
        self._bridge = None
        self._armed = False
        self._last_step = None
        logger.info(f"{self} disconnected.")

    def configure(self) -> None:
        self.controller.reset()

    # ── calibration ──────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """True when every face channel the operator has bound has a captured range.

        Keyboard and joystick bindings need no calibration, so a profile that uses only
        those is calibrated by definition.
        """
        return all(
            self.channel_calibrations.get(binding.channel, ChannelCalibration()).ready
            for binding in self.bindings.values()
            if binding.source is InputSource.FACE and binding.channel
        )

    def calibrate(self) -> None:
        """Wait for the operator to capture a range for each face channel they have bound.

        Calibration happens in the page, where the operator can see their own movement, so
        this only blocks until the page reports that the work is done.
        """
        pending = self._uncalibrated_channels()
        if not pending:
            logger.info(f"{self} has nothing to calibrate.")
            return

        print(f"\nRunning calibration of {self}")
        print("In the control page, relax, press Calibrate on each channel below, then move")
        print("that feature through the range you can hold comfortably:")
        for channel in pending:
            print(f"  - {channel}")
        print("Press Ctrl+C to keep the ranges captured so far.\n")

        try:
            while self._uncalibrated_channels():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nCalibration interrupted; uncalibrated channels will not move the robot.")
            return

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def _uncalibrated_channels(self) -> list[str]:
        return sorted(
            {
                binding.channel
                for binding in self.bindings.values()
                if binding.source is InputSource.FACE
                and binding.channel
                and not self.channel_calibrations.get(binding.channel, ChannelCalibration()).ready
            }
        )

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """Load the operator's bindings and channel ranges.

        This teleoperator has no motors, so it stores a control profile where other
        teleoperators store motor offsets.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f:
            profile = json.load(f)

        for joint, raw in (profile.get("bindings") or {}).items():
            if joint in self.bindings:
                self.bindings[joint] = _binding_from_profile(raw)
        for channel, raw in (profile.get("calibrations") or {}).items():
            self.channel_calibrations[channel] = ChannelCalibration(**raw)
        logger.info(f"Loaded accessible teleop profile from {fpath}")

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        profile = {
            "bindings": {joint: _binding_to_profile(b) for joint, b in self.bindings.items()},
            "calibrations": {ch: asdict(c) for ch, c in self.channel_calibrations.items()},
        }
        with open(fpath, "w") as f:
            json.dump(profile, f, indent=4)

    def _on_profile_change(self) -> None:
        """Persist a mapping the operator edited in the page, so it survives a restart."""
        try:
            self._save_calibration()
        except OSError as exc:
            logger.warning(f"Could not save the control profile: {exc}")

    # ── control ──────────────────────────────────────────────────────────

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        frame, age = self._bridge.read_frame()

        if age is None or age > self.config.input_timeout_s:
            # Losing the page mid-motion is the failure this teleoperator has to survive
            # gracefully, so treat silence as a released clutch rather than as the last
            # command repeating forever.
            if frame.engaged and not self._stale_logged:
                logger.warning("No input from the control page; holding position.")
                self._stale_logged = True
            self._bridge.release_clutch()
            frame = InputFrame()
        else:
            self._stale_logged = False

        if frame.engaged:
            self._armed = True

        now = time.perf_counter()
        dt_s = 0.0 if self._last_step is None else now - self._last_step
        self._last_step = now

        pose = self.controller.step(frame, dt_s)
        self._publish_state(frame, age)

        if not self._armed:
            # Nothing has been engaged yet, so there is no pose this teleoperator is
            # entitled to command.
            return {}

        return {f"{joint}.pos": pose[joint] for joint in self.config.joints}

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Anchor the commanded pose on the robot's measured positions.

        Feeding the follower's observation back while the clutch is open removes the jump
        that would otherwise happen the first time the operator engages, and keeps the
        commanded pose honest if the robot was moved by hand.
        """
        positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in feedback.items()
            if key.endswith(".pos") and isinstance(value, (int, float))
        }
        if not positions:
            return

        frame, _ = self._bridge.read_frame()
        if frame.engaged:
            return

        self.controller.reset(positions)

    def sync_to(self, pose: dict[str, float]) -> None:
        """Re-anchor the commanded pose without going through the feedback dict format."""
        self.controller.reset(pose)

    def _publish_state(self, frame: InputFrame, age: float | None) -> None:
        limits = self.config.joint_limits
        pose = self.controller.pose
        self._bridge.publish_state(
            {
                "pose": pose,
                "velocity": self.controller.velocity,
                "atLimit": {
                    joint: bool(
                        joint in limits
                        and (pose[joint] <= limits[joint][0] + 1e-6 or pose[joint] >= limits[joint][1] - 1e-6)
                    )
                    for joint in pose
                },
                "engaged": frame.engaged,
                "armed": self._armed,
                "inputAgeMs": None if age is None else round(age * 1000),
            }
        )

    def _warn_about_start_pose(self) -> None:
        pose = ", ".join(f"{j}={v:g}" for j, v in self.controller.pose.items())
        print("\n" + "=" * 78)
        print("The robot will move to the start pose the first time you engage the clutch.")
        print(f"  start pose: {pose}")
        print("Clear the workspace, and consider running the follower with a relative-target")
        print("limit, for example --robot.max_relative_target=5.")
        print("=" * 78 + "\n")


def _binding_to_profile(binding: JointBinding) -> dict[str, Any]:
    payload = asdict(binding)
    payload["source"] = binding.source.value
    return payload


def _binding_from_profile(raw: dict[str, Any]) -> JointBinding:
    payload = dict(raw)
    payload["source"] = InputSource(payload.get("source", InputSource.NONE.value))
    fields = JointBinding.__dataclass_fields__
    return JointBinding(**{k: v for k, v in payload.items() if k in fields})
