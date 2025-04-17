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
from dataclasses import replace

import torch
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as StretchAPI
from stretch_body.robot_params import RobotParams

from lerobot.common.robot_devices.robots.configs import StretchRobotConfig


class StretchRobot(StretchAPI):
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, config: StretchRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = StretchRobotConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)

        self.robot_type = self.config.type
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = None
        self.logs = {}

        # TODO(aliberts): test this
        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")

        self.state_keys = None
        self.action_keys = None

    def connect(self) -> None:
        self.is_connected = self.startup()
        if not self.is_connected:
            print("Another process is already using Stretch. Try running 'stretch_free_robot_process.py'")
            raise ConnectionError()

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.run_calibration()

    def run_calibration(self) -> None:
        if not self.is_homed():
            self.home()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        before_read_t = time.perf_counter()
        state = self.get_state()
        action = self.teleop.gamepad_controller.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        self.teleop.do_motion(robot=self)
        self.push_command()
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if self.state_keys is None:
            self.state_keys = list(state)

        if not record_data:
            return

        state = torch.as_tensor(list(state.values()))
        action = torch.as_tensor(list(action.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def get_state(self) -> dict:
        status = self.get_status()
        return {
            "head_pan.pos": status["head"]["head_pan"]["pos"],
            "head_tilt.pos": status["head"]["head_tilt"]["pos"],
            "lift.pos": status["lift"]["pos"],
            "arm.pos": status["arm"]["pos"],
            "wrist_pitch.pos": status["end_of_arm"]["wrist_pitch"]["pos"],
            "wrist_roll.pos": status["end_of_arm"]["wrist_roll"]["pos"],
            "wrist_yaw.pos": status["end_of_arm"]["wrist_yaw"]["pos"],
            "gripper.pos": status["end_of_arm"]["stretch_gripper"]["pos"],
            "base_x.vel": status["base"]["x_vel"],
            "base_y.vel": status["base"]["y_vel"],
            "base_theta.vel": status["base"]["theta_vel"],
        }

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        state = self.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        state = torch.as_tensor(list(state.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
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
        self.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
