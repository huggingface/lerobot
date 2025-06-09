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
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as StretchAPI
from stretch_body.robot_params import RobotParams

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.datasets.utils import get_nested_item

from ..robot import Robot
from .configuration_stretch3 import Stretch3RobotConfig

# {lerobot_keys: stretch.api.keys}
STRETCH_MOTORS = {
    "head_pan.pos": "head.head_pan.pos",
    "head_tilt.pos": "head.head_tilt.pos",
    "lift.pos": "lift.pos",
    "arm.pos": "arm.pos",
    "wrist_pitch.pos": "end_of_arm.wrist_pitch.pos",
    "wrist_roll.pos": "end_of_arm.wrist_roll.pos",
    "wrist_yaw.pos": "end_of_arm.wrist_yaw.pos",
    "gripper.pos": "end_of_arm.stretch_gripper.pos",
    "base_x.vel": "base.x_vel",
    "base_y.vel": "base.y_vel",
    "base_theta.vel": "base.theta_vel",
}


class Stretch3Robot(Robot):
    """[Stretch 3](https://hello-robot.com/stretch-3-product), by Hello Robot."""

    config_class = Stretch3RobotConfig
    name = "stretch3"

    def __init__(self, config: Stretch3RobotConfig):
        raise NotImplementedError
        super().__init__(config)

        self.config = config
        self.robot_type = self.config.type

        self.api = StretchAPI()
        self.cameras = make_cameras_from_configs(config.cameras)

        self.is_connected = False
        self.logs = {}

        self.teleop = None  # TODO remove

        # TODO(aliberts): test this
        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")

        self.state_keys = None
        self.action_keys = None

    @property
    def observation_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(STRETCH_MOTORS),),
            "names": {"motors": list(STRETCH_MOTORS)},
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

    def connect(self) -> None:
        self.is_connected = self.api.startup()
        if not self.is_connected:
            print("Another process is already using Stretch. Try running 'stretch_free_robot_process.py'")
            raise ConnectionError()

        for cam in self.cameras.values():
            cam.connect()
            self.is_connected = self.is_connected and cam.is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.calibrate()

    def calibrate(self) -> None:
        if not self.api.is_homed():
            self.api.home()

    def _get_state(self) -> dict:
        status = self.api.get_status()
        return {k: get_nested_item(status, v, sep=".") for k, v in STRETCH_MOTORS.items()}

    def get_observation(self) -> dict[str, np.ndarray]:
        obs_dict = {}

        # Read Stretch state
        before_read_t = time.perf_counter()
        state = self._get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        state = np.asarray(list(state.values()))
        obs_dict[OBS_STATE] = state

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()
            self.logs[f"read_camera_{cam_key}_dt_s"] = cam.logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        if self.action_keys is None:
            dummy_action = self.teleop.gamepad_controller.get_state()
            self.action_keys = list(dummy_action.keys())

        action_dict = dict(zip(self.action_keys, action.tolist(), strict=True))

        before_write_t = time.perf_counter()
        self.teleop.do_motion(state=action_dict, robot=self)
        self.push_command()
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def teleop_safety_stop(self) -> None:
        if self.teleop is not None:
            self.teleop._safety_stop(robot=self)

    def disconnect(self) -> None:
        self.api.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

        for cam in self.cameras.values():
            cam.disconnect()

        self.is_connected = False
