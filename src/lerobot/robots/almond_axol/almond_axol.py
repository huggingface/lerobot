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

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from .config_almond_axol import AlmondAxolConfig

if TYPE_CHECKING:
    from almond_axol.lerobot.robot import AxolRobot


def _require_almond_axol() -> None:
    require_package("almond-axol", extra="almond_axol", import_name="almond_axol")


class AlmondAxol(Robot):
    """
    [Almond Axol](https://docs.almond.bot), by Almond Bot: a dual-arm robot with
    two 7-DoF torque-controlled arms plus grippers, driven over CAN.

    This class wraps the LeRobot-native robot that ships in the
    [almond-axol SDK](https://github.com/almond-bot/axol), which owns the
    hardware driver (impedance control with gravity/friction feedforward,
    telemetry caching, safety limits) and the ZED camera capture. Observations
    are the 16 joint positions (8 per arm, optionally with torques, or per-arm
    Cartesian end-effector poses) plus camera frames; actions are joint-position
    targets (or Cartesian poses resolved through the SDK's inverse kinematics).
    """

    config_class = AlmondAxolConfig
    name = "almond_axol"

    def __init__(self, config: AlmondAxolConfig):
        _require_almond_axol()
        super().__init__(config)
        self.config = config
        self._robot: AxolRobot = self._make_sdk_robot(config)

    @staticmethod
    def _make_sdk_config(config: AlmondAxolConfig):
        from almond_axol.lerobot.camera.configuration_zed import ZedCameraConfig
        from almond_axol.lerobot.robot import AxolRobotConfig

        cameras = {
            name: ZedCameraConfig(
                serial=cam.serial,
                fps=cam.fps,
                width=cam.width,
                height=cam.height,
                stereo=cam.stereo,
                eyes=cam.eyes,
            )
            for name, cam in config.cameras.items()
        }
        return AxolRobotConfig(
            id=config.id,
            calibration_dir=config.calibration_dir,
            cameras=cameras,
            telemetry_hz=config.telemetry_hz,
            observe_torques=config.observe_torques,
            observe_cartesian=config.observe_cartesian,
            left_channel=config.left_channel,
            right_channel=config.right_channel,
            # The SDK's GStreamer capture path is tuned for its own
            # collect-data relay; the ZED SDK path is the right one under the
            # stock LeRobot tools.
            video_backend="sdk",
        )

    @classmethod
    def _make_sdk_robot(cls, config: AlmondAxolConfig) -> AxolRobot:
        from almond_axol.lerobot.robot import AxolRobot

        return AxolRobot(cls._make_sdk_config(config))

    @property
    def observation_features(self) -> dict[str, Any]:
        """Joint-position keys (`left_*.pos` / `right_*.pos`) plus one key per camera."""
        return self._robot.observation_features

    @property
    def action_features(self) -> dict[str, Any]:
        """Joint-position keys matching the observation's non-camera features."""
        return self._robot.action_features

    @property
    def cameras(self) -> dict[str, Any]:
        """Connected camera objects keyed by slot name."""
        return self._robot.cameras

    @property
    def is_connected(self) -> bool:
        """Whether the CAN buses and cameras are open."""
        return self._robot.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Always `True`: encoder zeros are managed by the Axol hardware setup."""
        return self._robot.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Open the CAN buses, enable the motors, start telemetry, and connect cameras."""
        self._robot.connect(calibrate)

    def calibrate(self) -> None:
        """No-op: encoder zeros are managed by the Axol hardware setup."""
        self._robot.calibrate()

    def configure(self) -> None:
        """No-op: motor gains are applied by the SDK driver on connect."""
        self._robot.configure()

    def get_observation(self) -> RobotObservation:
        """Return the cached joint state and timestamp-aligned camera frames.

        Returns:
            `dict[str, Any]`: Values keyed by `observation_features` — joint positions in
            radians (gripper normalized to `[0, 1]`) and one image array per camera.
        """
        return self._robot.get_observation()

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send joint-position targets to both arms.

        Args:
            action (`dict[str, float]`):
                Target values keyed by `action_features`, in radians (gripper normalized
                to `[0, 1]`). Arm joints are driven by impedance control, grippers by
                position-force control.

        Returns:
            `dict[str, float]`: The action as sent, unmodified.
        """
        return self._robot.send_action(action)

    def disconnect(self) -> None:
        """Disable the motors, stop telemetry, and close the CAN buses and cameras."""
        self._robot.disconnect()
