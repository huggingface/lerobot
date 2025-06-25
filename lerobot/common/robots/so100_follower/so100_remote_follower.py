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

import logging
from functools import cached_property
from typing import Any

from ..remote_robot import RemoteRobot
from .config_so100_follower import SO100RemoteFollowerConfig

logger = logging.getLogger(__name__)


class SO100RemoteFollower(RemoteRobot):
    """
    Remote SO-100 Follower Arm via WebRTC.

    This robot enables controlling SO-100 follower robots over the internet with low latency
    by receiving action commands via WebRTC data channels and publishing observations.
    """

    config_class = SO100RemoteFollowerConfig
    name = "so100_remote_follower"

    def __init__(self, config: SO100RemoteFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = {}  # make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for SO-100 follower arm."""
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features based on configured cameras."""
        return {}
        # return {
        #     cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
        #     for cam in self.cameras
        # }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        Return the observation features combining motor positions and camera images.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        Return the same action features as the SO100Follower to maintain compatibility.
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected to LiveKit and cameras are connected.
        """
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return super().is_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the LiveKit server and connect cameras.

        Args:
            calibrate (bool): Ignored for remote robot
        """
        # Connect to LiveKit server
        super().connect(calibrate=calibrate)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected with {len(self.cameras)} cameras.")

    def configure(self) -> None:
        """
        No-op for remote robot.
        """
        pass

    def setup_motors(self) -> None:
        """
        No-op for remote robot.
        """
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot including motor positions and camera images.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
        """
        if not self.is_connected:
            from lerobot.common.errors import DeviceNotConnectedError

            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get base observation from RemoteRobot (motor positions from remote state)
        obs_dict = super().get_observation()

        # Capture images from local cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the remote SO-100 follower robot via LiveKit.

        Args:
            action (dict[str, Any]): Dictionary representing the desired joint positions.

        Returns:
            dict[str, Any]: The action that was sent (same as input for remote robot).
        """
        return super().send_action(action)

    def disconnect(self) -> None:
        """
        Disconnect from the LiveKit server and cameras.
        """
        # Disconnect cameras first
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        # Disconnect from LiveKit server
        super().disconnect()

        logger.info(f"{self} disconnected from LiveKit server and cameras.")
