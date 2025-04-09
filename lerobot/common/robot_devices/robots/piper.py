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

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from piper_sdk import *
from lerobot.common.robot_devices.utils import busy_wait

from datetime import datetime
from copy import deepcopy

from lerobot.common.robot_devices.robots.configs import PiperRobotConfig
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

# @dataclass
# class PiperRobotConfig:
#     robot_type: str | None = "piper"
#     fps: int = 10
#     cameras: dict[str, Camera] = field(default_factory=lambda: {})
#     leader_arms: dict = field(default_factory=lambda: {})
#     # TODO(aliberts): add feature with max_relative target
#     # TODO(aliberts): add comment on max_relative target
#     max_relative_target: list[float] | float | None = None
#     joint_position_relative_bounds: dict[np.ndarray] | None = None


class Rate:
    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.last_time = time.perf_counter()
    
    def sleep(self, elapsed: float):
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self.last_time = time.perf_counter()


def rectify_signal(current, previous):
    """Rectify a single signal value using its previous value"""
    if abs(current - previous) >= .18:
        if current > previous:
            current -= .36
        else:
            current += .36
    return current


class EulerAngles:
    def __init__(self):
        self.prev_rx = 0
        self.prev_ry = 0
        self.prev_rz = 0

    def rectify(self, orientation):
        rx_rectified = rectify_signal(orientation[0], self.prev_rx)
        ry_rectified = rectify_signal(orientation[1], self.prev_ry)
        rz_rectified = rectify_signal(orientation[2], self.prev_rz)
        
        self.prev_rx = rx_rectified
        self.prev_ry = ry_rectified
        self.prev_rz = rz_rectified
        
        return [rx_rectified, ry_rectified, rz_rectified]


class PiperRobot(ManipulatorRobot):
    """Wrapper of piper_sdk.robot.Robot"""

    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):

        super().__init__(config)
        if config is None:
            config = PiperRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.fps = self.config.fps

        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.teleop = JoystickIntervention(controller_type=ControllerType.XBOX, gripper_enabled=True)
        self.logs = {}
        self.action_repeat = 2
        self.state_keys = None
        self.action_keys = None
        self.euler_filter = EulerAngles()
        # init piper robot
        self.piper = C_PiperInterface("can0")
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0,1000,0x01, 0)
        self.state_scaling_factor = 1e6 
        # self.default_pos = [0.200337, 0.020786, 0.289284, 0.179831, 0.010918, 0.173467, 0.0]
        self.default_pos = [0.171642, -0.028, 0.165, 0.179831, 0.010918, 0.173467, 0.0]
        self.joint_position_relative_bounds = self.config.joint_position_relative_bounds if self.config.joint_position_relative_bounds is not None else None
        self.previous_ee_position = np.array([0.0, 0.0, 0.0])
        self.current_state = np.array([0.0, 0.0, 0.0, 0.0])

        # self.data_rows = []
        # self.csv_filename = f"arm_poses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # Register the signal handler
        import signal
        signal.signal(signal.SIGINT, self.signal_handler)

    @property
    def motor_features(self) -> dict:
        # Or completely redefine the features
        return {
            "action": {
                "dtype": "float64",
                "shape": (4,),  # Change shape
                "names": ["x", "y", "z", "gripper"],  # New names
            },
            "observation.state": {
                "dtype": "float64",
                "shape": (7,),
                "names": ["x", "y", "z", "gripper", "rx", "ry", "rz"],
            },
        }
    # Signal handler for graceful shutdown
    def signal_handler(self, sig, frame):
        print('\nSaving data and exiting...')
        import sys
        # self.save_to_csv()
        self.disable_robot()
        sys.exit(0)

    def disable_robot(self):
        self.piper.DisableArm(7)
        self.piper.GripperCtrl(0,1000,0x02, 0)

    # Function to save data to CSV
    def save_to_csv(self):
        import csv
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Timestamp', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'grippers_angle', 'grippers_effort'])
            # Write all stored data
            writer.writerows(self.data_rows)
        print(f"Data saved to {self.csv_filename}")

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

    # Used when connecting to robot
    def move_to_home(self) -> None:
        count = 0
        while True:
            if(count == 0):
                print("1-----------")
                action = [0.07,0,0.22,0,0.08,0,0.08]
            # elif(count == 400):
            #     print("2-----------")
                # action = [0.15,0.0,0.35,0.08,0.08,0.025,0.0] # 0.08 is maximum gripper position
            elif(count == 300):
                print("2-----------")
                action = self.default_pos
            count += 1
            before_write_t = time.perf_counter()
            state = self.get_state()
            state = state["state"]
            # state[3:6] = self.euler_filter.rectify(state[3:6])
            self.send_action(action)
            # self.rate.sleep(time.perf_counter() - before_write_t)
            if count > 600:
                break
    
    # Used when returning to home after finishing a demo
    def move_to_home_2(self):
        count = 0
        while True:
            if count <= 100:
                action = self.default_pos
                action[6] = 0.08
            elif count > 100:
                action = self.default_pos
                action[6] = 0.0
            count += 1
            before_write_t = time.perf_counter()
            state = self.get_state()
            state = state["state"]
            # state[3:6] = self.euler_filter.rectify(state[3:6])
            self.send_action(action)
            # self.rate.sleep(time.perf_counter() - before_write_t)
            if count > 300:
                break


    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        before_read_t = time.perf_counter()
        state = self.get_state()
        self.get_intervention_start()
        state = state["state"]
        # state[3:6] = self.euler_filter.rectify(state[3:6])
        # get relative action from joystick
        action = self.teleop.action()
        # print(action)
        # Convert action to numpy array first
        action = np.array(action, dtype=np.float32)
        action_record = deepcopy(action[:3])
        # action_record = np.concatenate([action[:3], [action[-1]]])
        
        for i in range(self.action_repeat):
            action[:3] = state[:3] + action_record*(i+1)/self.action_repeat
            self.send_action(action)
            busy_wait(0.03)
 
        if self.teleop.home:
            self.move_to_home_2()
        if self.state_keys is None:
            self.state_keys = list(state)

        if not record_data:
            return
        # action_record[3:6] -= state[3:6] # just get delta orientation
        # it has to be done after send_action
        # state[:2] -= self.default_pos[:2]
        state = torch.as_tensor(state).to(torch.float32)
        action_record = torch.as_tensor(action_record).to(torch.float32)
        # print(action_record)

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
        # obs_dict.update({"gripper_effort": self.get_gripper_effort()})
        action_dict["action"] = action_record
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def get_state(self) -> dict:
        end_effector_pose = self.piper.GetArmEndPoseMsgs()
        gripper_pose = self.piper.GetArmGripperMsgs()
        self.previous_ee_position = self.current_state[:3]
        
        # Convert to float32 numpy array
        self.current_state = np.array([
            end_effector_pose.end_pose.X_axis,
            end_effector_pose.end_pose.Y_axis,
            end_effector_pose.end_pose.Z_axis,
            gripper_pose.gripper_state.grippers_angle
        ], dtype=np.float32) / self.state_scaling_factor
        # add velocity to state
        velocity = (self.current_state[:3] - self.previous_ee_position) * self.fps
        state = np.concatenate([self.current_state, 
                                velocity])
        
        # print(f"state: {state}")

        return {
            "state": state,
        }

    def get_gripper_state(self) -> float:
        gripper_pose = self.piper.GetArmGripperMsgs()
        return np.array([gripper_pose.gripper_state.grippers_angle, gripper_pose.gripper_state.grippers_effort])
    
    def get_intervention_start(self) -> bool:
        intervention_start, success, estop = self.teleop.get_intervention_start()
        if estop:
            self.disable_robot()
        return intervention_start, success
    
    # def get_ee_pos(self) -> list[float]:
    #     end_effector_pose = self.piper.GetArmEndPoseMsgs()
    #     gripper_pose = self.piper.GetArmGripperMsgs()
    #     return [end_effector_pose.end_pose.X_axis,end_effector_pose.end_pose.Y_axis,end_effector_pose.end_pose.Z_axis,gripper_pose.gripper_state.grippers_angle]
    
    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        state = self.get_state()
        self.get_intervention_start()
        # state = state["state"]
        # state["state"][3:6] = self.euler_filter.rectify(state["state"][3:6])
        # state["state"][:2] -= self.default_pos[:2]
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        state = torch.as_tensor(np.array(list(state.values())).astype(np.float32))
        state = state.squeeze(0)

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
        # obs_dict.update({"gripper_effort": self.get_gripper_effort()})
        return obs_dict

    def send_action(self, action: list[float]) -> None:
        if not self.is_connected:
            raise ConnectionError()
        # check if action is tensor and if it is, convert it to list
        if isinstance(action, torch.Tensor):
            action = action.tolist()
        # clip rz value to be between -pi/2 and pi/2 for safety
        # action[5] = max(-np.pi/2, min(np.pi/2, action[5]))
        X = round(action[0]*self.state_scaling_factor)
        Y = round(action[1]*self.state_scaling_factor)
        Z = round(action[2]*self.state_scaling_factor)
        RX = round(self.default_pos[3]*self.state_scaling_factor)
        RY = round(self.default_pos[4]*self.state_scaling_factor)
        RZ = round(self.default_pos[5]*self.state_scaling_factor)
        Gripper = round(action[-1]*self.state_scaling_factor)

        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.GripperCtrl(abs(Gripper), 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        # It is needed to give time for the CAN bus communication to complete
        busy_wait(0.01)

        # TODO(aliberts): return action_sent when motion is limited
        return torch.tensor(action)

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        if self.teleop is not None:
            self.teleop.close()

        # if len(self.cameras) > 0:
        #     for cam in self.cameras.values():
        #         cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()