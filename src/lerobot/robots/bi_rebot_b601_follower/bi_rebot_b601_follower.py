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
from functools import cached_property

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
from ..robot import Robot
from .config_bi_rebot_b601_follower import BiRebotB601FollowerConfig

logger = logging.getLogger(__name__)


class BiRebotB601Follower(BimanualMixin, Robot):
    """Bimanual Seeed Studio reBot B601-DM follower.

    Composes two single-arm :class:`RebotB601Follower` instances. Observation and
    action keys of each arm are namespaced with a ``left_`` / ``right_`` prefix.
    """

    config_class = BiRebotB601FollowerConfig
    name = "bi_rebot_b601_follower"

    def __init__(self, config: BiRebotB601FollowerConfig):
        super().__init__(config)
        self.config = config

        # Top-level cameras are opened by `left_arm` for convenience, but their
        # keys stay unprefixed in observations (tracked via `_top_level_cam_keys`).
        self._top_level_cam_keys = set(config.cameras)
        _collisions = self._top_level_cam_keys & set(
            config.left_arm_config.cameras
        ) | self._top_level_cam_keys & set(config.right_arm_config.cameras)
        if _collisions:
            raise ValueError(
                f"Top-level camera names collide with per-arm camera names: {sorted(_collisions)}"
            )
        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}

        left_arm_config = RebotB601FollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            can_adapter=config.left_arm_config.can_adapter,
            dm_serial_baud=config.left_arm_config.dm_serial_baud,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.left_arm_config.max_relative_target,
            cameras=left_arm_cameras,
            motor_can_ids=config.left_arm_config.motor_can_ids,
            pos_vel_velocity=config.left_arm_config.pos_vel_velocity,
            gripper_torque_ratio=config.left_arm_config.gripper_torque_ratio,
            joint_limits=config.left_arm_config.joint_limits,
        )

        right_arm_config = RebotB601FollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            can_adapter=config.right_arm_config.can_adapter,
            dm_serial_baud=config.right_arm_config.dm_serial_baud,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.right_arm_config.max_relative_target,
            cameras=config.right_arm_config.cameras,
            motor_can_ids=config.right_arm_config.motor_can_ids,
            pos_vel_velocity=config.right_arm_config.pos_vel_velocity,
            gripper_torque_ratio=config.right_arm_config.gripper_torque_ratio,
            joint_limits=config.right_arm_config.joint_limits,
        )

        self.left_arm = RebotB601Follower(left_arm_config)
        self.right_arm = RebotB601Follower(right_arm_config)

        # Only for compatibility with parts of the codebase that expect `robot.cameras`.
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
        return self._motors_ft

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}
        for k, v in self.left_arm.get_observation().items():
            obs_dict[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm.get_observation().items():
            obs_dict[f"right_{k}"] = v
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_action_left = self.left_arm.send_action(left_action)
        sent_action_right = self.right_arm.send_action(right_action)

        return {
            **{f"left_{k}": v for k, v in sent_action_left.items()},
            **{f"right_{k}": v for k, v in sent_action_right.items()},
        }
