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

import logging
from dataclasses import fields
from functools import cached_property

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.robots.openarm_follower import (
    OpenArmFollower,
    OpenArmFollowerConfig,
    OpenArmFollowerConfigBase,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_openarm_gripette_follower import OpenArmGripetteFollowerConfig
from .gripette_gripper import MOTORS, GripetteGripper

logger = logging.getLogger(__name__)


class OpenArmGripetteFollower(Robot):
    """7-joint OpenArm (Damiao/CAN) arm + networked Gripette gripper (gRPC)."""

    config_class = OpenArmGripetteFollowerConfig
    name = "openarm_gripette_follower"

    def __init__(self, config: OpenArmGripetteFollowerConfig):
        super().__init__(config)
        self.config = config

        # The gripper is driven over gRPC, so it is removed from the CAN bus.
        base = config.arm_config
        arm_cfg = OpenArmFollowerConfig(
            id=f"{config.id}_arm" if config.id else None,
            calibration_dir=config.calibration_dir,
            **{f.name: getattr(base, f.name) for f in fields(OpenArmFollowerConfigBase)},
        )
        arm_cfg.motor_config = {k: v for k, v in base.motor_config.items() if k != "gripper"}
        self.arm = OpenArmFollower(arm_cfg)

        self._gripper = GripetteGripper(
            addr=config.gripette_addr,
            motor_max=config.gripper_motor_max,
            camera_name=config.gripper_camera_name if config.use_gripper_camera else None,
            camera_shape=(config.gripper_camera_height, config.gripper_camera_width),
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self.arm.action_features, **{f"{m}.pos": float for m in MOTORS}}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {
            **self.arm.observation_features,
            **{f"{m}.pos": float for m in MOTORS},
        }
        if self.config.use_gripper_camera:
            features[self.config.gripper_camera_name] = (
                self.config.gripper_camera_height,
                self.config.gripper_camera_width,
                3,
            )
        return features

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self._gripper.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.arm.connect(calibrate=calibrate)
        self._gripper.connect()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated

    def calibrate(self) -> None:
        self.arm.calibrate()

    def configure(self) -> None:
        self.arm.configure()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        return {**self.arm.get_observation(), **self._gripper.read_observation()}

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        gripper_keys = {f"{m}.pos" for m in MOTORS}
        sent = self.arm.send_action({k: v for k, v in action.items() if k not in gripper_keys})
        goals = {k.removesuffix(".pos"): v for k, v in action.items() if k in gripper_keys}
        if goals:
            sent.update({f"{m}.pos": v for m, v in self._gripper.send_goal(goals).items()})
        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        self._gripper.disconnect(disable_torque=self.config.arm_config.disable_torque_on_disconnect)
        self.arm.disconnect()
        logger.info(f"{self} disconnected.")
