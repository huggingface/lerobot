#!/usr/bin/env python

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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lerobot.lerobot_types import RobotAction
from lerobot.robots.almond_axol.almond_axol import _require_almond_axol

from ..teleoperator import Teleoperator
from .config_almond_axol_vr import AlmondAxolVRConfig

if TYPE_CHECKING:
    from almond_axol.lerobot.teleop import AxolVRTeleop


class AlmondAxolVR(Teleoperator):
    """
    VR teleoperator for the [Almond Axol](https://docs.almond.bot) dual-arm
    robot.

    Wraps the LeRobot-native teleoperator that ships in the
    [almond-axol SDK](https://github.com/almond-bot/axol): a WebSocket server
    receives 6-DoF hand poses from a VR headset (via the WebXR app at
    https://axol.almond.bot) and an inverse-kinematics subprocess turns them
    into joint-position actions matching the ``almond_axol`` robot's action
    features. Episode control (start/stop/re-record) is driven from the
    headset controllers and exposed through ``get_teleop_events()``.
    """

    config_class = AlmondAxolVRConfig
    name = "almond_axol_vr"

    def __init__(self, config: AlmondAxolVRConfig):
        _require_almond_axol()
        super().__init__(config)
        self.config = config
        self._teleop: AxolVRTeleop = self._make_sdk_teleop(config)

    @staticmethod
    def _make_sdk_config(config: AlmondAxolVRConfig):
        from almond_axol.lerobot.teleop import AxolVRTeleopConfig
        from almond_axol.vr.config import VRServerConfig

        return AxolVRTeleopConfig(
            id=config.id,
            calibration_dir=config.calibration_dir,
            vr_server_config=VRServerConfig(port=config.port),
            has_gripper=config.has_gripper,
        )

    @classmethod
    def _make_sdk_teleop(cls, config: AlmondAxolVRConfig) -> AxolVRTeleop:
        from almond_axol.lerobot.teleop import AxolVRTeleop

        return AxolVRTeleop(cls._make_sdk_config(config))

    @property
    def action_features(self) -> dict[str, Any]:
        """Joint-position keys matching the `almond_axol` robot's action features."""
        return self._teleop.action_features

    @property
    def feedback_features(self) -> dict[str, Any]:
        """Joint-position keys accepted by `send_feedback`."""
        return self._teleop.feedback_features

    @property
    def is_connected(self) -> bool:
        """Whether the VR server and IK subprocess are running."""
        return self._teleop.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Always `True`: the teleoperator needs no calibration."""
        return self._teleop.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Start the VR WebSocket server and the inverse-kinematics subprocess."""
        self._teleop.connect(calibrate)

    def calibrate(self) -> None:
        """No-op: the teleoperator needs no calibration."""
        self._teleop.calibrate()

    def configure(self) -> None:
        """No-op: session parameters are applied on connect."""
        self._teleop.configure()

    def get_action(self) -> RobotAction:
        """Return the latest smoothed joint-position action from the tracked hand poses."""
        return self._teleop.get_action()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Forward the robot's current joint state to seed the IK solver."""
        self._teleop.send_feedback(feedback)

    def get_teleop_events(self) -> dict[Any, Any]:
        """Episode-control events driven from the headset controllers."""
        return self._teleop.get_teleop_events()

    def disconnect(self) -> None:
        self._teleop.disconnect()
