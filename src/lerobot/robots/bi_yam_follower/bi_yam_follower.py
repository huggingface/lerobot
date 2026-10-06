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

"""Bimanual YAM composed from two single-arm followers."""

import logging
import threading
from copy import deepcopy
from dataclasses import fields

from lerobot.cameras import CameraConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..yam_follower import YamArmConfig, YamFollower, YamFollowerConfig
from ..yam_follower.yam_follower import action_to_target
from .config_bi_yam_follower import BiYamFollowerConfig

logger = logging.getLogger(__name__)


class BiYamFollower(BimanualMixin, Robot):
    """Two YAM followers with prefixed actions and observations."""

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self._top_level_cam_keys = set(config.cameras)
        # One stop event for both servos, so a fault on either arm stops both.
        self._stop = threading.Event()
        self.left_arm = YamFollower(
            self._arm_robot_config("left", config.left_arm, config.cameras), stop_event=self._stop
        )
        self.right_arm = YamFollower(
            self._arm_robot_config("right", config.right_arm, {}), stop_event=self._stop
        )
        self.arms = {"left": self.left_arm, "right": self.right_arm}
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    def _arm_robot_config(
        self, side: str, arm_config: YamArmConfig, cameras: dict[str, CameraConfig]
    ) -> YamFollowerConfig:
        values = {field.name: deepcopy(getattr(arm_config, field.name)) for field in fields(YamArmConfig)}
        return YamFollowerConfig(
            id=f"{self.config.id}_{side}" if self.config.id else None,
            calibration_dir=self.config.calibration_dir,
            cameras=deepcopy(cameras),
            read_only=self.config.read_only,
            control_frequency=self.config.control_frequency,
            feedback_timeout_s=self.config.feedback_timeout_s,
            command_timeout_s=self.config.command_timeout_s,
            **values,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{name}": value for name, value in self.left_arm.action_features.items()},
            **{f"right_{name}": value for name, value in self.right_arm.action_features.items()},
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for name, value in self.left_arm.observation_features.items():
            features[name if name in self._top_level_cam_keys else f"left_{name}"] = value
        for name, value in self.right_arm.observation_features.items():
            features[f"right_{name}"] = value
        return features

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if calibrate and not self.is_calibrated:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure both gripper endpoints")
        self._stop.clear()
        try:
            if not calibrate:
                for arm in self.arms.values():
                    arm.connect(calibrate=False)
            else:
                # Open and pose-check both arms, and switch both to MIT mode, before either gets torque.
                for arm in self.arms.values():
                    arm.open()
                self.configure()
                for arm in self.arms.values():
                    arm.start()
        except BaseException:
            self._stop.set()
            for arm in reversed(tuple(self.arms.values())):
                if not arm.is_connected:
                    continue
                try:
                    arm.disconnect()
                except Exception:
                    logger.exception("Failed to close YAM arm after bimanual connection failure")
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = {}
        for name, value in self.left_arm.get_observation().items():
            observation[name if name in self._top_level_cam_keys else f"left_{name}"] = value
        for name, value in self.right_arm.get_observation().items():
            observation[f"right_{name}"] = value
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.read_only or not self.left_arm.servo.active:
            raise RuntimeError("YAM read-only/calibration connection forbids motor commands")
        if set(action) != set(self.action_features):
            raise ValueError("YAM requires all 14 absolute joint/gripper targets; Cartesian actions need IK")
        arm_actions = {
            side: {name: action[f"{side}_{name}"] for name in arm.action_features}
            for side, arm in self.arms.items()
        }
        # Validate and health-check both arms before moving either; a later fault on one
        # stops both servos through the shared stop event.
        for arm_action in arm_actions.values():
            action_to_target(arm_action)
        for arm in self.arms.values():
            arm.servo.check_healthy()
        sent = {side: arm.send_action(arm_actions[side]) for side, arm in self.arms.items()}
        return {f"{side}_{name}": value for side, values in sent.items() for name, value in values.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop.set()
        first_error: BaseException | None = None
        for side, arm in self.arms.items():
            try:
                arm.disconnect()
            except BaseException as exc:
                logger.exception("Failed to disconnect %s YAM arm", side)
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
