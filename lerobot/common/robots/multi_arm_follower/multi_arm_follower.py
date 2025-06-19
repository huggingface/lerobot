#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.robots.utils import make_robot_from_config

from ..robot import Robot
from .config_multi_arm_follower import MultiArmFollowerConfig

logger = logging.getLogger(__name__)


class MultiArmFollower(Robot):
    """
    Multiple Arms Follower.

    For example, how to run the teleoperate script with multi-arm leader and follower
    being composed of two SO101 arms:
    ```bash
    export arm1="{type: so101_follower, port: /dev/ttyACM0}"
    export arm2="{type: so101_follower, port: /dev/ttyACM1}"
    export teleop1="{type: so101_leader, port: /dev/ttyACM2}"
    export teleop2="{type: so101_leader, port: /dev/ttyACM3}"

    python -m lerobot.teleoperate \
        --robot.type=multi_arm_follower \
        --robot.arms=["$arm1","$arm2"] \
        --robot.id=two-so101-follower \
        --teleop.type=multi_arm_leader \
        --teleop.arms=["$teleop1","$teleop2"] \
        --teleop.id=two-so101-leader
    ```
    """

    config_class = MultiArmFollowerConfig
    name = "multi_arm_follower"

    def __init__(self, config: MultiArmFollowerConfig):
        super().__init__(config)
        self.config = config

        self.arms = [make_robot_from_config(arm_config) for arm_config in config.arms]

        self.cameras = make_cameras_from_configs(config.cameras)

    def _encode_arm_index(self, key: str, index: int) -> str:
        return f"arm{index}__{key}"

    def _decode_arm_index(self, key: str) -> int:
        arm_id, *remaining = key.split("__")
        assert arm_id.startswith("arm"), (arm_id, key)
        return int(arm_id[len("arm") :]), "__".join(remaining)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Get quickly all observation_features
        # assuming minimal latency due the loop
        all_observations = [arm.observation_features for arm in self.arms]
        # Post-process the results:
        all_observations = [
            {self._encode_arm_index(key, i): value for key, value in obs.items()}
            for i, obs in enumerate(all_observations)
        ]
        return {k: v for obs_ft in all_observations for k, v in obs_ft.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Get quickly all action_features
        # assuming minimal latency due the loop
        all_actions = [arm.action_features for arm in self.arms]
        # Post-process the results:
        all_actions = [
            {self._encode_arm_index(key, i): value for key, value in actions.items()}
            for i, actions in enumerate(all_actions)
        ]
        return {k: v for actions in all_actions for k, v in actions.items()}

    @property
    def is_connected(self) -> bool:
        all_arms_connected = all(arm.is_connected for arm in self.arms)
        return all_arms_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arms are in a rest position,
        and torque can be safely disabled to run calibration.
        """
        for arm in self.arms:
            arm.connect(calibrate=calibrate)

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return all(arm.is_calibrated for arm in self.arms)

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        for arm in self.arms:
            arm.calibrate()

    def configure(self) -> None:
        for arm in self.arms:
            arm.configure()

    def setup_motors(self) -> None:
        for arm in self.arms:
            arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        # Get quickly all observations
        # assuming minimal latency due the loop
        all_observations = [arm.get_observation() for arm in self.arms]
        # Post-process the results:
        all_observations = [
            {self._encode_arm_index(key, i): value for key, value in obs.items()}
            for i, obs in enumerate(all_observations)
        ]
        obs_dict = {k: v for obs in all_observations for k, v in obs.items()}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        action_per_arm = [None] * len(self.arms)
        for key in action:
            index, base_key = self._decode_arm_index(key)
            if action_per_arm[index] is None:
                action_per_arm[index] = {base_key: action[key]}
            else:
                action_per_arm[index][base_key] = action[key]

        output = [
            arm.send_action(action_per_arm)
            for arm, action_per_arm in zip(self.arms, action_per_arm, strict=False)
        ]
        output = [
            {self._encode_arm_index(key, i): value for key, value in action.items()}
            for i, action in enumerate(output)
        ]
        return {k: v for action in output for k, v in action.items()}

    def disconnect(self):
        for arm in self.arms:
            arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
