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
from typing import Any

import numpy as np

# Make sure to install the reachy-mini dependency from pyproject.toml
# pip install -e .[reachy_mini]
from reachy_mini import ReachyMini as ReachyMiniSDK
from reachy_mini.utils import create_head_pose

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .configuration_reachy_mini import ReachyMiniConfig


class ReachyMini(Robot):
    """
    LeRobot driver for the Pollen Robotics Reachy Mini robot.
    Note: Reachy Mini does not have arms. It's a Stewart platform (head)
    with antennas and a rotating base.
    """

    config_class = ReachyMiniConfig
    name = "reachy_mini"

    def __init__(self, config: ReachyMiniConfig):
        super().__init__(config)
        self.config = config
        self.robot: ReachyMini | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self.last_head_z_pos = 0.0

    @property
    def is_connected(self) -> bool:
        return self.robot is not None and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True):
        try:
            self.robot = ReachyMiniSDK()
            # The 'with' context manager is recommended, but for persistent connection
            # in this class, we manually manage the resource.
            # The SDK doesn't have an explicit connect method; instantiation handles it.
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Reachy Mini at {self.config.ip_address}") from e

        for cam in self.cameras.values():
            cam.connect()
        print(f"Successfully connected to Reachy Mini at {self.config.ip_address}.")

    def disconnect(self):
        if self.robot is not None:
            # The SDK uses a __exit__ method for cleanup when used with 'with'.
            # We call it manually here to ensure resources are released.
            self.robot.__exit__(None, None, None)
            self.robot = None
            print("Disconnected from Reachy Mini.")

        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def observation_features(self) -> dict:
        obs_features = {
            # Head Z position (Stewart Platform)
            "head.z.pos": float,
            # Body Yaw
            "body.yaw.pos": float,
            # Antennas (Left and Right)
            "antennas.left.pos": float,
            "antennas.right.pos": float,
            **{cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()},
        }
        return obs_features

    @property
    def action_features(self) -> dict:
        # Actions mirror the controllable parts of the observation space.
        return {
            "head.z.pos": (self.config.head_z_pos_limits_mm[0], self.config.head_z_pos_limits_mm[1]),
            "body.yaw.pos": (self.config.body_yaw_limits_deg[0], self.config.body_yaw_limits_deg[1]),
            "antennas.left.pos": (
                self.config.antennas_pos_limits_deg[0],
                self.config.antennas_pos_limits_deg[1],
            ),
            "antennas.right.pos": (
                self.config.antennas_pos_limits_deg[0],
                self.config.antennas_pos_limits_deg[1],
            ),
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self.robot is None:
            raise ConnectionError("Reachy Mini is not connected.")

        # NOTE: The reachy_mini SDK does not provide a direct way to get the
        # current pose of the head (Stewart platform). It's write-only via set_target/goto_target.
        # As a fallback, we return the last commanded position.
        obs = {
            "head.z.pos": self.last_head_z_pos,
        }
        try:
            joint_positions = self.robot.get_current_joint_positions()
            antenna_positions = self.robot.get_present_antenna_joint_positions()
            obs.update(
                {
                    "body.yaw.pos": np.rad2deg(joint_positions[0][0]),
                    "antennas.left.pos": np.rad2deg(antenna_positions[0]),
                    "antennas.right.pos": np.rad2deg(antenna_positions[1]),
                }
            )
        except (AttributeError, KeyError):
            obs.update(
                {
                    "body.yaw.pos": np.rad2deg(self.robot.body_yaw.present_position),
                    "antennas.left.pos": np.rad2deg(self.robot.antennas.left.present_position),
                    "antennas.right.pos": np.rad2deg(self.robot.antennas.right.present_position),
                }
            )

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self.robot is None:
            raise ConnectionError("Reachy Mini is not connected.")

        # Clamp actions to defined limits
        head_z_pos = np.clip(
            action["head.z.pos"],
            self.config.head_z_pos_limits_mm[0],
            self.config.head_z_pos_limits_mm[1],
        )
        self.last_head_z_pos = head_z_pos

        body_yaw_deg = np.clip(
            action["body.yaw.pos"],
            self.config.body_yaw_limits_deg[0],
            self.config.body_yaw_limits_deg[1],
        )
        antennas_left_deg = np.clip(
            action["antennas.left.pos"],
            self.config.antennas_pos_limits_deg[0],
            self.config.antennas_pos_limits_deg[1],
        )
        antennas_right_deg = np.clip(
            action["antennas.right.pos"],
            self.config.antennas_pos_limits_deg[0],
            self.config.antennas_pos_limits_deg[1],
        )

        # Convert actions from degrees (lerobot convention) to radians (SDK convention)
        body_yaw_rad = np.deg2rad(body_yaw_deg)
        antennas_rad = [
            np.deg2rad(antennas_left_deg),
            np.deg2rad(antennas_right_deg),
        ]
        # Head position is in mm
        head_pose = create_head_pose(z=head_z_pos, mm=True)

        self.robot.set_target(
            head=head_pose,
            antennas=antennas_rad,
            body_yaw=body_yaw_rad,
        )
        return action

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
