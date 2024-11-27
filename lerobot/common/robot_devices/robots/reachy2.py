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
from copy import copy
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from reachy2_sdk import ReachySDK

from lerobot.common.robot_devices.cameras.reachy2 import ReachyCamera

REACHY_MOTORS = [
    "neck_yaw.pos",
    "neck_pitch.pos",
    "neck_roll.pos",
    "r_shoulder_pitch.pos",
    "r_shoulder_roll.pos",
    "r_elbow_yaw.pos",
    "r_elbow_pitch.pos",
    "r_wrist_roll.pos",
    "r_wrist_pitch.pos",
    "r_wrist_yaw.pos",
    "r_gripper.pos",
    "l_shoulder_pitch.pos",
    "l_shoulder_roll.pos",
    "l_elbow_yaw.pos",
    "l_elbow_pitch.pos",
    "l_wrist_roll.pos",
    "l_wrist_pitch.pos",
    "l_wrist_yaw.pos",
    "l_gripper.pos",
    "mobile_base.vx",
    "mobile_base.vy",
    "mobile_base.vtheta",
]


@dataclass
class ReachyRobotConfig:
    robot_type: str | None = "reachy2"
    cameras: dict[str, ReachyCamera] = field(default_factory=lambda: {})
    ip_address: str | None = "172.17.135.207"
    # ip_address: str | None = "192.168.0.197"
    # ip_address: str | None = "localhost"


class ReachyRobot:
    """Wrapper of ReachySDK"""

    def __init__(self, config: ReachyRobotConfig | None = None, **kwargs):
        if config is None:
            config = ReachyRobotConfig()

        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.cameras = self.config.cameras
        self.has_camera = True
        self.num_cameras = len(self.cameras)
        self.is_connected = False
        self.teleop = None
        self.logs = {}
        self.reachy = None
        self.mobile_base_available = False

        self.state_keys = None
        self.action_keys = None

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        motors = REACHY_MOTORS
        # if self.mobile_base_available:
        #     motors += REACHY_MOBILE_BASE
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": motors,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": motors,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    def connect(self) -> None:
        self.reachy = ReachySDK(host=self.config.ip_address)
        print("Connecting to Reachy")
        self.reachy.connect()
        self.is_connected = self.reachy.is_connected
        if not self.is_connected:
            print(
                f"Cannot connect to Reachy at address {self.config.ip_address}. Maybe a connection already exists."
            )
            raise ConnectionError()
        # self.reachy.turn_on()
        print(self.cameras)
        if self.cameras is not None:
            for name in self.cameras:
                print(f"Connecting camera: {name}")
                self.cameras[name].connect()
                self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.mobile_base_available = self.reachy.mobile_base is not None

    def run_calibration(self):
        pass

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not record_data:
            return
        action = {}
        action["neck_roll.pos"] = self.reachy.head.neck.roll.goal_position
        action["neck_pitch.pos"] = self.reachy.head.neck.pitch.goal_position
        action["neck_yaw.pos"] = self.reachy.head.neck.yaw.goal_position

        action["r_shoulder_pitch.pos"] = self.reachy.r_arm.shoulder.pitch.goal_position
        action["r_shoulder_roll.pos"] = self.reachy.r_arm.shoulder.roll.goal_position
        action["r_elbow_yaw.pos"] = self.reachy.r_arm.elbow.yaw.goal_position
        action["r_elbow_pitch.pos"] = self.reachy.r_arm.elbow.pitch.goal_position
        action["r_wrist_roll.pos"] = self.reachy.r_arm.wrist.roll.goal_position
        action["r_wrist_pitch.pos"] = self.reachy.r_arm.wrist.pitch.goal_position
        action["r_wrist_yaw.pos"] = self.reachy.r_arm.wrist.yaw.goal_position
        action["r_gripper.pos"] = self.reachy.r_arm.gripper.opening

        action["l_shoulder_pitch.pos"] = self.reachy.l_arm.shoulder.pitch.goal_position
        action["l_shoulder_roll.pos"] = self.reachy.l_arm.shoulder.roll.goal_position
        action["l_elbow_yaw.pos"] = self.reachy.l_arm.elbow.yaw.goal_position
        action["l_elbow_pitch.pos"] = self.reachy.l_arm.elbow.pitch.goal_position
        action["l_wrist_roll.pos"] = self.reachy.l_arm.wrist.roll.goal_position
        action["l_wrist_pitch.pos"] = self.reachy.l_arm.wrist.pitch.goal_position
        action["l_wrist_yaw.pos"] = self.reachy.l_arm.wrist.yaw.goal_position
        action["l_gripper.pos"] = self.reachy.l_arm.gripper.opening

        if self.mobile_base_available:
            last_cmd_vel = self.reachy.mobile_base.last_cmd_vel
            action["mobile_base_x.vel"] = last_cmd_vel["x"]
            action["mobile_base_y.vel"] = last_cmd_vel["y"]
            action["mobile_base_theta.vel"] = last_cmd_vel["theta"]
        else:
            action["mobile_base_x.vel"] = 0
            action["mobile_base_y.vel"] = 0
            action["mobile_base_theta.vel"] = 0

        dtype = self.motor_features["action"]["dtype"]
        action = np.array(list(action.values()), dtype=dtype)
        # action = torch.as_tensor(list(action.values()))

        obs_dict = self.capture_observation()
        action_dict = {}
        action_dict["action"] = action

        return obs_dict, action_dict

    def get_state(self) -> dict:
        # neck roll, pitch, yaw
        # r_shoulder_pitch, r_shoulder_roll, r_elbow_yaw, r_elbow_pitch, r_wrist_roll, r_wrist_pitch, r_wrist_yaw, r_gripper
        # l_shoulder_pitch, l_shoulder_roll, l_elbow_yaw, l_elbow_pitch, l_wrist_roll, l_wrist_pitch, l_wrist_yaw, l_gripper
        # mobile base x, y, theta
        if self.is_connected:
            if self.mobile_base_available:
                odometry = self.reachy.mobile_base.odometry
            else:
                odometry = {"x": 0, "y": 0, "theta": 0, "vx": 0, "vy": 0, "vtheta": 0}
            return {
                "neck_yaw.pos": self.reachy.head.neck.yaw.present_position,
                "neck_pitch.pos": self.reachy.head.neck.pitch.present_position,
                "neck_roll.pos": self.reachy.head.neck.roll.present_position,
                "r_shoulder_pitch.pos": self.reachy.r_arm.shoulder.pitch.present_position,
                "r_shoulder_roll.pos": self.reachy.r_arm.shoulder.roll.present_position,
                "r_elbow_yaw.pos": self.reachy.r_arm.elbow.yaw.present_position,
                "r_elbow_pitch.pos": self.reachy.r_arm.elbow.pitch.present_position,
                "r_wrist_roll.pos": self.reachy.r_arm.wrist.roll.present_position,
                "r_wrist_pitch.pos": self.reachy.r_arm.wrist.pitch.present_position,
                "r_wrist_yaw.pos": self.reachy.r_arm.wrist.yaw.present_position,
                "r_gripper.pos": self.reachy.r_arm.gripper.present_position,
                "l_shoulder_pitch.pos": self.reachy.l_arm.shoulder.pitch.present_position,
                "l_shoulder_roll.pos": self.reachy.l_arm.shoulder.roll.present_position,
                "l_elbow_yaw.pos": self.reachy.l_arm.elbow.yaw.present_position,
                "l_elbow_pitch.pos": self.reachy.l_arm.elbow.pitch.present_position,
                "l_wrist_roll.pos": self.reachy.l_arm.wrist.roll.present_position,
                "l_wrist_pitch.pos": self.reachy.l_arm.wrist.pitch.present_position,
                "l_wrist_yaw.pos": self.reachy.l_arm.wrist.yaw.present_position,
                "l_gripper.pos": self.reachy.l_arm.gripper.present_position,
                "mobile_base.vx": odometry["vx"],
                "mobile_base.vy": odometry["vy"],
                "mobile_base.vtheta": odometry["vtheta"],
            }
        else:
            return {}

    def capture_observation(self) -> dict:
        if self.is_connected:
            before_read_t = time.perf_counter()
            state = self.get_state()
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

            if self.state_keys is None:
                self.state_keys = list(state)

            dtype = self.motor_features["observation.state"]["dtype"]
            state = np.array(list(state.values()), dtype=dtype)
            # state = torch.as_tensor(list(state.values()))

            # Capture images from cameras
            images = {}
            for name in self.cameras:
                # before_camread_t = time.perf_counter()
                images[name] = self.cameras[name].read()  # Reachy cameras read() is not blocking?
                # print(f'name: {name} img: {images[name]}')
                if images[name] is not None:
                    # images[name] = copy(images[name][0])  # seems like I need to copy?
                    images[name] = torch.from_numpy(copy(images[name][0]))  # seems like I need to copy?
                    self.logs[f"read_camera_{name}_dt_s"] = images[name][1]  # full timestamp, TODO dt

            # Populate output dictionnaries
            obs_dict = {}
            obs_dict["observation.state"] = state
            for name in self.cameras:
                obs_dict[f"observation.images.{name}"] = images[name]

            return obs_dict
        else:
            return {}

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise ConnectionError()

        self.reachy.head.neck.yaw.goal_position = float(action[0])
        self.reachy.head.neck.pitch.goal_position = float(action[1])
        self.reachy.head.neck.roll.goal_position = float(action[2])

        self.reachy.r_arm.shoulder.pitch.goal_position = float(action[3])
        self.reachy.r_arm.shoulder.roll.goal_position = float(action[4])
        self.reachy.r_arm.elbow.yaw.goal_position = float(action[5])
        self.reachy.r_arm.elbow.pitch.goal_position = float(action[6])
        self.reachy.r_arm.wrist.roll.goal_position = float(action[7])
        self.reachy.r_arm.wrist.roll.goal_position = float(action[8])
        self.reachy.r_arm.wrist.yaw.goal_position = float(action[9])
        self.reachy.r_arm.gripper.set_opening(float(action[10]))

        self.reachy.l_arm.shoulder.pitch.goal_position = float(action[11])
        self.reachy.l_arm.shoulder.roll.goal_position = float(action[12])
        self.reachy.l_arm.elbow.yaw.goal_position = float(action[13])
        self.reachy.l_arm.elbow.pitch.goal_position = float(action[14])
        self.reachy.l_arm.wrist.roll.goal_position = float(action[15])
        self.reachy.l_arm.wrist.roll.goal_position = float(action[16])
        self.reachy.l_arm.wrist.yaw.goal_position = float(action[17])
        self.reachy.l_arm.gripper.set_opening(float(action[18]))

        s = time.time()
        self.reachy.send_goal_positions(check_positions=False)
        print("send_goal_positions", time.time() - s)

        if self.mobile_base_available:
            self.reachy.mobile_base.set_goal_speed(action[19], action[20], action[21])
            self.reachy.mobile_base.send_speed_command()

        # TODO: what shape is the action tensor?
        # 7 dofs per arm (x2)
        # 1 dof per gripper (x2)
        # 3 dofs for the neck
        # 3 dofs for the mobile base (x, y, theta)
        # 7+7+1+1+3+3 = 22
        return action

    def print_logs(self) -> None:
        pass

    def disconnect(self) -> None:
        print("Disconnecting")
        self.is_connected = False
        print("Turn off")
        # self.reachy.turn_off_smoothly()
        # self.reachy.turn_off()
        print("\t turn off done")
        self.reachy.disconnect()
