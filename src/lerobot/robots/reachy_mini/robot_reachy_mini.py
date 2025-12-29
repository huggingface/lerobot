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
import contextlib
from typing import Any

import numpy as np

# Make sure to install the reachy-mini dependency from pyproject.toml
# pip install -e .[reachy_mini]
from reachy_mini import ReachyMini as ReachyMiniSDK

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
        self.robot: ReachyMiniSDK | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

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
            "body_rotation.pos": float,
            "stewart_1.pos": float,
            "stewart_2.pos": float,
            "stewart_3.pos": float,
            "stewart_4.pos": float,
            "stewart_5.pos": float,
            "stewart_6.pos": float,
            "right_antenna.pos": float,
            "left_antenna.pos": float,
            **{cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()},
        }
        return obs_features

    @property
    def action_features(self) -> dict:
        return {
            "body_rotation.pos": (self.config.body_yaw_limits_deg[0], self.config.body_yaw_limits_deg[1]),
            "stewart_1.pos": (self.config.stewart_pos_limits_deg[0], self.config.stewart_pos_limits_deg[1]),
            "stewart_2.pos": (self.config.stewart_pos_limits_deg[0], self.config.stewart_pos_limits_deg[1]),
            "stewart_3.pos": (self.config.stewart_pos_limits_deg[0], self.config.stewart_pos_limits_deg[1]),
            "stewart_4.pos": (self.config.stewart_pos_limits_deg[0], self.config.stewart_pos_limits_deg[1]),
            "stewart_5.pos": (self.config.stewart_pos_limits_deg[0], self.config.stewart_pos_limits_deg[1]),
            "stewart_6.pos": (self.config.stewart_pos_limits_deg[0], self.config.stewart_pos_limits_deg[1]),
            "right_antenna.pos": (
                self.config.antennas_pos_limits_deg[0],
                self.config.antennas_pos_limits_deg[1],
            ),
            "left_antenna.pos": (
                self.config.antennas_pos_limits_deg[0],
                self.config.antennas_pos_limits_deg[1],
            ),
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self.robot is None:
            raise ConnectionError("Reachy Mini is not connected.")

        obs = {}
        try:
            joint_positions = self.robot.get_current_joint_positions()
            # joint_positions returns (head_joints[7], antenna_joints[2])
            # head_joints: [body_yaw, stewart_1...6]
            # antenna_joints: [right, left]
            head_joints, antenna_joints = joint_positions

            obs.update(
                {
                    "body_rotation.pos": np.rad2deg(head_joints[0]),
                    "stewart_1.pos": np.rad2deg(head_joints[1]),
                    "stewart_2.pos": np.rad2deg(head_joints[2]),
                    "stewart_3.pos": np.rad2deg(head_joints[3]),
                    "stewart_4.pos": np.rad2deg(head_joints[4]),
                    "stewart_5.pos": np.rad2deg(head_joints[5]),
                    "stewart_6.pos": np.rad2deg(head_joints[6]),
                    "right_antenna.pos": np.rad2deg(antenna_joints[0]),
                    "left_antenna.pos": np.rad2deg(antenna_joints[1]),
                }
            )
        except (AttributeError, KeyError, IndexError) as e:
            # Fallback or error handling if SDK response is unexpected
            print(f"Error reading joints: {e}")
            # Try alternative attribute access if method fails (legacy SDK check)
            with contextlib.suppress(AttributeError):
                obs.update(
                    {
                        "body_rotation.pos": np.rad2deg(self.robot.body_yaw.present_position),
                        "right_antenna.pos": np.rad2deg(self.robot.antennas.right.present_position),
                        "left_antenna.pos": np.rad2deg(self.robot.antennas.left.present_position),
                        # Stewart motors might not be easily accessible via attributes in some SDK versions without proper mapping
                    }
                )

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self.robot is None:
            raise ConnectionError("Reachy Mini is not connected.")

        # Extract and clamp actions
        body_yaw_deg = np.clip(
            action["body_rotation.pos"],
            self.config.body_yaw_limits_deg[0],
            self.config.body_yaw_limits_deg[1],
        )

        stewart_joints_deg = []
        for i in range(1, 7):
            val = np.clip(
                action[f"stewart_{i}.pos"],
                self.config.stewart_pos_limits_deg[0],
                self.config.stewart_pos_limits_deg[1],
            )
            stewart_joints_deg.append(val)

        right_antenna_deg = np.clip(
            action["right_antenna.pos"],
            self.config.antennas_pos_limits_deg[0],
            self.config.antennas_pos_limits_deg[1],
        )
        left_antenna_deg = np.clip(
            action["left_antenna.pos"],
            self.config.antennas_pos_limits_deg[0],
            self.config.antennas_pos_limits_deg[1],
        )

        # Convert actions from degrees to radians
        body_yaw_rad = np.deg2rad(body_yaw_deg)
        stewart_joints_rad = [np.deg2rad(val) for val in stewart_joints_deg]
        right_antenna_rad = np.deg2rad(right_antenna_deg)
        left_antenna_rad = np.deg2rad(left_antenna_deg)

        # Construct lists for SDK
        head_joint_positions = [body_yaw_rad] + stewart_joints_rad
        antennas_joint_positions = [right_antenna_rad, left_antenna_rad]

        # Use the internal method to set joint positions directly
        self.robot._set_joint_positions(
            head_joint_positions=head_joint_positions,
            antennas_joint_positions=antennas_joint_positions,
        )

        return action

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
