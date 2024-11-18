#!/usr/bin/env python

# Copyright 2024 The Pollen Robotics team and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from lerobot.common.robot_devices.cameras.utils import Camera
from reachy2_sdk import ReachySDK
from copy import copy



@dataclass
class ReachyRobotConfig:
    robot_type: str | None = "Reachy2"
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    ip_address: str | None = "localhost"

class ReachyRobot():
    """Wrapper of ReachySDK """

    def __init__(self, config: ReachyRobotConfig | None = None, **kwargs):
        if config is None:
            config = ReachyRobotConfig()



        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.cameras = self.config.cameras
        self.is_connected = False #at init Reachy2 is in fact connected...
        self.teleop = None
        self.logs = {}
        self.reachy=ReachySDK(host=config.ip_address)


        self.state_keys = None
        self.action_keys = None

    def connect(self) -> None:
        self.reachy.is_connected = self.reachy.connect()
        if not self.is_connected:
            print(f'Cannot connect to Reachy at address {self.config.ip_address}. Maybe a connection already exists.')
            raise ConnectionError()

        self.reachy.turn_on()
        print(self.cameras)
        if self.cameras is not None:
            for name in self.cameras:
                print(f'Connecting camera: {name}')
                self.cameras[name].connect()
                self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO

        return None

    def get_state(self) -> dict:

        return {
            "neck_yaw.pos": np.radians(self.reachy.head.neck.yaw.present_position),
            "neck_pitch.pos": np.radians(self.reachy.head.neck.pitch.present_position),
            "neck_roll.pos": np.radians(self.reachy.head.neck.roll.present_position),
            "r_shoulder_pitch.pos": np.radians(self.reachy.r_arm.shoulder.pitch.present_position),
            "r_shoulder_roll.pos": np.radians(self.reachy.r_arm.shoulder.roll.present_position),
            "r_elbow_yaw.pos": np.radians(self.reachy.r_arm.elbow.yaw.present_position),
            "r_elbow_pitch.pos": np.radians(self.reachy.r_arm.elbow.pitch.present_position),
            "r_wrist_roll.pos": np.radians(self.reachy.r_arm.wrist.roll.present_position),
            "r_wrist_pitch.pos": np.radians(self.reachy.r_arm.wrist.pitch.present_position),
            "r_wrist_yaw.pos": np.radians(self.reachy.r_arm.wrist.yaw.present_position),
            "r_gripper.pos": np.radians(self.reachy.r_arm.gripper.present_position),
            "l_shouldel_pitch.pos": np.radians(self.reachy.l_arm.shoulder.pitch.present_position),
            "l_shouldel_roll.pos": np.radians(self.reachy.l_arm.shoulder.roll.present_position),
            "l_elbow_yaw.pos": np.radians(self.reachy.l_arm.elbow.yaw.present_position),
            "l_elbow_pitch.pos": np.radians(self.reachy.l_arm.elbow.pitch.present_position),
            "l_wrist_roll.pos": np.radians(self.reachy.l_arm.wrist.roll.present_position),
            "l_wrist_pitch.pos": np.radians(self.reachy.l_arm.wrist.pitch.present_position),
            "l_wrist_yaw.pos": np.radians(self.reachy.l_arm.wrist.yaw.present_position),
            "l_gripper.pos": np.radians(self.reachy.l_arm.gripper.present_position),
            #TODO mobile base
        }

    def capture_observation(self) -> dict:

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
            images[name] = self.cameras[name].read() #Reachy cameras read() is not blocking?
            print(f'name: {name} img: {images[name]}')
            if images[name] is not None:
                images[name] = torch.from_numpy(copy(images[name][0])) #seems like I need to copy?
                self.logs[f"read_camera_{name}_dt_s"] = images[name][1] #full timestamp, TODO dt

        # Populate output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:

        if not self.is_connected:
            raise ConnectionError()

        #TODO
        return action

    def print_logs(self) -> None:
        pass


    def disconnect(self) -> None:
        self.is_connected = False
        self.reachy.turn_off_smoothly()
        # self.reachy.turn_off()
        time.sleep(6)

    def __del__(self):
        self.disconnect()
