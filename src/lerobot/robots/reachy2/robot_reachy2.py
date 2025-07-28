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

import time

import numpy as np
from reachy2_sdk import ReachySDK
from typing import Any

# from stretch_body.gamepad_teleop import GamePadTeleop
# from stretch_body.robot import Robot as StretchAPI
# from stretch_body.robot_params import RobotParams

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_IMAGES, OBS_STATE

from ..robot import Robot
from .configuration_reachy2 import Reachy2RobotConfig

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_MOTORS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
    # "mobile_base.vx": "mobile_base.vx",
    # "mobile_base.vy": "mobile_base.vy",
    # "mobile_base.vtheta": "mobile_base.vtheta",
}


class Reachy2Robot(Robot):
    """[Reachy 2](https://www.pollen-robotics.com/reachy/), by Pollen Robotics."""

    config_class = Reachy2RobotConfig
    name = "reachy2"

    def __init__(self, config: Reachy2RobotConfig):
        super().__init__(config)

        self.config = config
        self.robot_type = self.config.type

        self.reachy: None | ReachySDK = None
        self.cameras = make_cameras_from_configs(config.cameras)

        self.logs = {}

    @property
    def observation_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": len(REACHY2_MOTORS),
            "names": {"motors": list(REACHY2_MOTORS)},
        }

    @property
    def action_features(self) -> dict:
        return self.observation_features

    @property
    def camera_features(self) -> dict[str, dict]:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft[cam_key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def is_connected(self) -> bool:
        return self.reachy.is_connected() if self.reachy is not None else False

    def connect(self) -> None:
        self.reachy = ReachySDK(self.config.ip_address)
        if not self.is_connected:
            print("Error connecting to Reachy 2.")
            raise ConnectionError()

        # for cam in self.cameras.values():
        #     cam.connect()
        #     self.is_connected = self.is_connected and cam.is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.configure()

    def configure(self) -> None:
        self.reachy.turn_on()
        self.reachy.reset_default_limits()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def _get_state(self) -> dict:
        return {k: self.reachy.joints[v].present_position for k, v in REACHY2_MOTORS.items()}

    def get_observation(self) -> dict[str, np.ndarray]:
        obs_dict = {}

        # Read Reachy 2 state
        before_read_t = time.perf_counter()
        state = self._get_state()
        print(state)
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        state = np.asarray(list(state.values()))
        obs_dict[OBS_STATE] = state

        # Capture images from cameras
        # for cam_key, cam in self.cameras.items():
        #     before_camread_t = time.perf_counter()
        #     obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()
        #     self.logs[f"read_camera_{cam_key}_dt_s"] = cam.logs["delta_timestamp_s"]
        #     self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError()

        before_write_t = time.perf_counter()
        for key, val in action.items():
            if key not in REACHY2_MOTORS:
                raise KeyError(f"Key '{key}' is not a valid motor key in Reachy 2.")
            else:
                self.reachy.joints[REACHY2_MOTORS[key]].goal_position = val
        self.reachy.send_goal_positions()
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return action

    def disconnect(self) -> None:
        self.reachy.turn_off_smoothly()
        self.reachy.disconnect()
        # for cam in self.cameras.values():
        #     cam.disconnect()
