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

# To run teleoperate:
# python lerobot/scripts/control_robot.py teleoperate --robot-path lerobot/configs/robot/piper.yaml --fps 30

import time
from dataclasses import dataclass, field, replace

import torch
import numpy as np
from lerobot.common.robot_devices.robots.joystick_interface import JoystickIntervention, ControllerType
from piper_sdk import *

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


@dataclass
class PiperRobotConfig:
    robot_type: str | None = "piper"
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    leader_arms: dict = field(default_factory=lambda: {})
    # TODO(aliberts): add feature with max_relative target
    # TODO(aliberts): add comment on max_relative target
    max_relative_target: list[float] | float | None = None


class Rate:
    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.last_time = time.perf_counter()
    
    def sleep(self, elapsed: float):
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self.last_time = time.perf_counter()


class LowPassFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.last_rx = 0
        self.last_ry = 0
        self.last_rz = 0

    def filter(self, orientation):
        # Apply exponential moving average
        self.last_rx = self.alpha * orientation[0] + (1 - self.alpha) * self.last_rx
        self.last_ry = self.alpha * orientation[1] + (1 - self.alpha) * self.last_ry
        self.last_rz = self.alpha * orientation[2] + (1 - self.alpha) * self.last_rz
        return self.last_rx, self.last_ry, self.last_rz


class PiperRobot(ManipulatorRobot):
    """Wrapper of piper_sdk.robot.Robot"""

    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = PiperRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.leader_arms = self.config.leader_arms
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = JoystickIntervention(controller_type=ControllerType.XBOX, gripper_enabled=True)
        self.logs = {}

        self.state_keys = None
        self.action_keys = None
        self.rate = Rate(200)
        self.euler_filter = LowPassFilter()
        # init piper robot
        self.piper = C_PiperInterface("can0")
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0,1000,0x01, 0)
        self.state_scaling_factor = 1e6 

    def startup_robot(self, piper:C_PiperInterface):
        '''
        enable robot and check enable status, try 5s, if enable timeout, exit program
        '''
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("enable status:",enable_flag)
            piper.EnableArm(7)
            piper.GripperCtrl(0,1000,0x01, 0)
            print("--------------------")
            # check if timeout
            if elapsed_time > timeout:
                print("enable timeout....")
                elapsed_time_flag = True
                enable_flag = False
                break
            time.sleep(1)
            pass
        if not elapsed_time_flag:
            return enable_flag
        else:
            print("enable timeout, exit program")
            raise RuntimeError("Failed to enable robot motors within timeout period")

    def connect(self) -> None:
        self.is_connected = self.startup_robot(self.piper)

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.move_to_home()

    def move_to_home(self) -> None:
        # TODO(ke): add logic to move to home
        count = 0
        while True:
            if(count == 0):
                print("1-----------")
                action = [0.07,0,0.22,0,0.08,0,0]
            elif(count == 300):
                print("2-----------")
                action = [0.15,0.0,0.35,0.08,0.08,0.075,0.0] # 0.08 is maximum gripper position
            elif(count == 600):
                print("3-----------")
                action = [0.204381, -0.00177, 0.274648, -0.176753, 0.021866, 0.171871, 0.0]
            count += 1
            before_write_t = time.perf_counter()
            state = self.get_state()
            state = state["state"]
            state[3:6] = self.euler_filter.filter(state[3:6])
            self.send_action(action)
            self.rate.sleep(time.perf_counter() - before_write_t)
            if count > 800:
                break

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        before_read_t = time.perf_counter()
        state = self.get_state()
        state = state["state"]
        state[3:6] = self.euler_filter.filter(state[3:6])
        # get relative action from joystick
        action = self.teleop.action(state)
        action[:6] += state[:6]
        if self.teleop.home:
            self.move_to_home()

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        self.send_action(action)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        self.rate.sleep(time.perf_counter() - before_write_t)
        if self.state_keys is None:
            self.state_keys = list(state)

        if not record_data:
            return

        state = torch.as_tensor(state)
        action = torch.as_tensor(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def get_state(self) -> dict:
        end_effector_pose = self.piper.GetArmEndPoseMsgs()
        gripper_pose = self.piper.GetArmGripperMsgs()
        state = np.array([end_effector_pose.end_pose.X_axis,end_effector_pose.end_pose.Y_axis,end_effector_pose.end_pose.Z_axis
                    ,end_effector_pose.end_pose.RX_axis,end_effector_pose.end_pose.RY_axis,end_effector_pose.end_pose.RZ_axis
                    ,gripper_pose.gripper_state.grippers_angle])/self.state_scaling_factor
        return {
            "state": state,
        }

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        state = self.get_state()
        state["state"][3:6] = self.euler_filter.filter(state["state"][3:6])
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

        # Populate output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: list[float]) -> None:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()
        # check if action is tensor and if it is, convert it to list
        if isinstance(action, torch.Tensor):
            action = action.tolist()
        # clip rz value to be between -pi/2 and pi/2 for safety
        action[5] = max(-np.pi/2, min(np.pi/2, action[5]))
        X = round(action[0]*self.state_scaling_factor)
        Y = round(action[1]*self.state_scaling_factor)
        Z = round(action[2]*self.state_scaling_factor)
        RX = round(action[3]*self.state_scaling_factor)
        RY = round(action[4]*self.state_scaling_factor)
        RZ = round(action[5]*self.state_scaling_factor)
        Gripper = round(action[6]*self.state_scaling_factor)
        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.GripperCtrl(abs(Gripper), 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)

        # TODO(aliberts): return action_sent when motion is limited
        return torch.tensor(action)

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        if self.teleop is not None:
            self.teleop.close()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
