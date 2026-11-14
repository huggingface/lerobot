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

"""Bimanual YAM composed from two single-arm followers."""

import logging
import threading
from copy import deepcopy
from dataclasses import fields
from functools import cached_property

from lerobot.cameras import CameraConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils import DeviceNotConnectedError, check_if_already_connected, check_if_not_connected
from lerobot.utils.bimanual import BimanualMixin

from ..robot import Robot
from ..yam_follower import YamFollower, YamFollowerConfig, YamFollowerRobotConfig
from .config_bi_yam_follower import BiYamFollowerConfig

logger = logging.getLogger(__name__)


class BiYamFollower(BimanualMixin, Robot):
    """Two YAM followers with prefixed actions and observations."""

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config

        # Top-level cameras are opened by `left_arm` for convenience, but their
        # keys stay unprefixed in observations (tracked via `_top_level_cam_keys`).
        self._top_level_cam_keys = set(config.cameras)
        collisions = self._top_level_cam_keys & (
            set(config.left_arm_config.cameras) | set(config.right_arm_config.cameras)
        )
        if collisions:
            raise ValueError(
                f"Top-level camera names collide with per-arm camera names: {sorted(collisions)}"
            )
        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}

        # One stop event for both controllers, so a fault on either arm stops both.
        self._stop = threading.Event()
        self.left_arm = YamFollower(
            self._arm_robot_config("left", config.left_arm_config, left_arm_cameras), stop_event=self._stop
        )
        self.right_arm = YamFollower(
            self._arm_robot_config("right", config.right_arm_config, config.right_arm_config.cameras),
            stop_event=self._stop,
        )
        self.arms = {"left": self.left_arm, "right": self.right_arm}
        self.supports_position_hold = (
            self.left_arm.supports_position_hold and self.right_arm.supports_position_hold
        )
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for k, v in self.left_arm._cameras_ft.items():
            out[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm._cameras_ft.items():
            out[f"right_{k}"] = v
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect both arms and start control.

        Neither arm gets torque until both are calibrated and in their initial position.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Measure the gripper limits of an arm if none are saved.
        """
        if any(arm.is_connected for arm in self.arms.values()):
            raise RuntimeError("Disconnect both YAM arms before reconnecting")
        try:
            for arm in self.arms.values():
                arm.connect_bus()
            if not self.is_calibrated and calibrate:
                logger.info(f"No gripper calibration found for {self}")
                self.calibrate()
            for arm in self.arms.values():
                for camera in arm.cameras.values():
                    camera.connect()
            self.configure()
            if self.is_calibrated:
                # Check both initial positions before either arm gets torque. This also clears the stop event
                # both arms share, so it must come before either controller starts: clearing it later could
                # erase a fault of the arm that started first.
                for arm in self.arms.values():
                    arm.prepare_controller()
                for arm in self.arms.values():
                    arm.controller.start()
        except BaseException:
            self._stop.set()
            if any(arm.is_connected for arm in self.arms.values()):
                try:
                    self.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect YAM after bimanual connection failure")
            raise
        logger.info(f"{self} connected." if self.is_calibrated else f"{self} connected with torque off.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return observations from both arms with side-prefixed keys."""
        observation: RobotObservation = {}
        for name, value in self.left_arm.get_observation().items():
            observation[name if name in self._top_level_cam_keys else f"left_{name}"] = value
        for name, value in self.right_arm.get_observation().items():
            observation[f"right_{name}"] = value
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send each arm its side-prefixed targets. A fault on either arm stops both controllers."""
        sent = {
            side: arm.send_action({name: action[f"{side}_{name}"] for name in arm.action_features})
            for side, arm in self.arms.items()
        }
        return {f"{side}_{name}": value for side, values in sent.items() for name, value in values.items()}

    def disconnect(self) -> None:
        """Stop both arms, even if only one is connected or the first fails to stop."""
        if not any(arm.is_connected for arm in self.arms.values()):
            raise DeviceNotConnectedError("BiYamFollower is not connected. Run `.connect()` first.")
        self._stop.set()
        first_error: BaseException | None = None
        for side, arm in self.arms.items():
            if not arm.is_connected:
                continue
            try:
                arm.disconnect()
            except BaseException as exc:
                logger.exception("Failed to disconnect %s YAM arm", side)
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _arm_robot_config(
        self, side: str, arm_config: YamFollowerConfig, cameras: dict[str, CameraConfig]
    ) -> YamFollowerRobotConfig:
        values = {
            field.name: deepcopy(getattr(arm_config, field.name)) for field in fields(YamFollowerConfig)
        }
        values["cameras"] = deepcopy(cameras)
        return YamFollowerRobotConfig(
            id=f"{self.config.id}_{side}" if self.config.id else None,
            calibration_dir=self.config.calibration_dir,
            **values,
        )
