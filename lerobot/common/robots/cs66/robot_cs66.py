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
import time
from functools import cached_property
from typing import Any , Dict

import numpy as np
from .robot import RobotController

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.constants import OBS_IMAGES, OBS_STATE

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .configuration_cs66 import EliteCS66Config

logger = logging.getLogger(__name__)

def normalize_angle_deg(angle):
        """将角度归一化到 [-180, 180]"""
        return ((angle + 180) % 360) - 180

class EliteCS66(Robot):
    """
    Cobot Elite CS66 developed by [Elite](https://www.elibot.com/).
    """

    config_class = EliteCS66Config
    name = "cs66"

    def __init__(self, config: EliteCS66Config):
        # raise NotImplementedError
        super().__init__(config)

        self.config = config
        # 初始化机器人
        self.robot = RobotController(robot_ip=self.config.ip)

        # 初始化相机参数
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self._is_connected = False
        self._calibrated = False
        self._last_suction_command = None


    # 机器人标定 cs66 机器人功能完成标定
    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        logger.info(f"{self} calibration completed.")
        self._calibrated = True


    # cs66和相机连接
    def connect(self) -> None:
        # 连接机器人并开启解释器模式
        self.robot.connect()
        self.robot.sendCMD("   interpreter_mode(clearQueueOnEnter = True, clearOnEnd = True)")
        # 初始化数字输DO-0为 False 吸盘为关闭状态
        # 以下命令分别使用了30001，30020，29999和40011端口功能，再次检验链接正常
        self.robot.interpreter("set_standard_digital_out(0, False)")
        time.sleep(0.01)
        # 29999端口功能测试报错，但是暂时没有用到。需要用时咨询艾利特技术支持
        # rtn = self.robot.dashboard_shell("help")
        data = self.robot.rt.get_output_data()
        sucker_status = data.actual_digital_output_bits
        joint_states = data.actual_joint_positions
        # print( "sucker status:", sucker_status, "joint states:", joint_states)
        if (sucker_status & 0b1) == 0 and joint_states is not None:
            print("sucker is shut down now")
            print(f"{self} joint states: {joint_states}")
        else:
            raise ConnectionError()
        
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True
        logger.info(f"{self} connected. ip address: {self.config.ip}")

        # TODO self.configure()
    #     self.move2start()

    # def move2start(self):
    #     """
    #     Move the robot to a predefined start position.
    #     This is a placeholder function and should be implemented with actual logic.
    #     """
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")
        
    #     # 这里假设有一个预定义的起始位置
    #     start_position = [90.0, -90.0, -90.0, -90.0, 90.0, -90.0]
        
    #     # 发送关节位置到机器人
    #     self.send_action({f"joint_{i+1}.pos": pos for i, pos in enumerate(start_position)})


    def configure(self):
        # Configure the cs66.
        raise NotImplementedError
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # 这里猜测需要使用伺服模式运行
        # 启动伺服模式
        err = self.robot.ServoMoveStart()
        if err != 0:
            raise RuntimeError(f"{self} failed to start servo mode. Error code: {err}")
        
    def get_observation(self) -> dict[str, Any]:
        """
        Get the current observation from the robot and cameras.

        Returns:
            dict[str, Any]: Observation dictionary with joint positions and camera images.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # 获取关节位置
        start = time.perf_counter()
        data = self.robot.rt.get_output_data()
        joint_states = data.actual_joint_positions
        # if joint_states is None:
        #     raise RuntimeError(f"{self} failed to get joint states.")

        obs_dict = {f"joint_{i+1}.pos": float(state) for i, state in enumerate(joint_states)}
        obs_dict["end_effector.pos"] = float(data.actual_digital_output_bits & 0b1)   # 0 或 1，表示吸盘状态
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def disconnect(self):
        """Disconnect the robot and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # 断开链接
        self.robot.disconnect()
        
        # 断开相机链接
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected.")

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        ee_pose = action.get("end_effector.pos", None)
        joint_keys = [f"joint_{i+1}.pos" for i in range(6)]
        # 发送给机器人的关节命令
        cmd_robot_joints = [
            np.deg2rad(
                normalize_angle_deg(float(action[key]) + self.config.joint_offsets.get(key.split('.')[0], 0.0))
            )
            for key in joint_keys if key in action
        ]

        # 需要重新将机器人关节状态重新构建成字典格式
        data = self.robot.rt.get_output_data()
        if data is None:
            raise RuntimeError(f"{self} failed to get robot state data.")
    
        # === 吸盘控制逻辑 ===
        suction_state = float(data.actual_digital_output_bits & 0b1)  # 0 或 1，转成 float
        if ee_pose is not None:
            target_suction = ee_pose < 50.0  # 小于 50 表示吸附
            # 如果目标状态和实际状态不一致，发出切换命令
            if suction_state != float(target_suction):
                self.robot.interpreter(f"set_standard_digital_out(0, {str(target_suction)})")
                self._last_suction_command = target_suction
                time.sleep(0.05)

        # robot_joints = [
        #     float(action[key]) + self.config.joint_offsets.get(key.split('.')[0], 0.0) for key in joint_keys if key in action
        # ]
        # 调试用，不send action给机器人
        # print(f"{self} got goal pose: {robot_joints}")
        # print(f"{self} got end effector pose: {ee_pose}")


        # # 获取关节状态
        # joint_states = data.actual_joint_positions
        # if joint_states is None:
        #     raise RuntimeError(f"{self} failed to get joint states.")

        # # 构建 joint_1.pos ~ joint_6.pos
        # command_dict = {
        #     f"joint_{i+1}.pos": float(pos)
        #     for i, pos in enumerate(joint_states[:6])
        # }

        # # 添加 end_effector.pos
        # command_dict["end_effector.pos"] = end_effector_val

        # 调试不启动 先看测试输出的值
        self.robot.interpreter("skipbuffer")
        self.robot.interpreter("servoj({}, t={}, lookahead_time={}, gain={})".format(
            cmd_robot_joints,
            self.config.dt,
            self.config.lookahead_time,
            self.config.gain,
        ))

        # 目前没有设置安全限值，执行的输入值和输出值可以看作一致
        return action
        # return command_dict
        
        
        ##########
        # 调试用
        # return cmd_robot_joints
        # return ee_pose


#########################################################

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """Set connection status."""
        self._is_connected = value

    @property
    def is_calibrated(self) -> bool:
        return getattr(self, "_calibrated", False)

    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @property
    def _motor_ft(self) -> dict[str, type]:
        data = self.robot.rt.get_output_data()
        joint_states = data.actual_joint_positions
        motors = {f"joint_{i+1}.pos": float for i in range(len(joint_states))}
        motors["end_effector.pos"] = float # 定义末端类型
        # if joint_states is None:
        #     raise RuntimeError(f"{self} failed to get joint states.")
        return motors
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Joint position types and camera observation shapes."""
        return {
            **self._motor_ft,
            **self._cameras_ft
        }
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motor_ft
    
    

#####################################################
####                testing                      ####
#####################################################

def main():
    config = EliteCS66Config(
        ip="192.168.101.11",  
        cameras={}        
     )
    elite = EliteCS66(config)

    try:
        elite.connect()
        print("✅ Connected")

        obs = elite.get_observation()
        print("✅ Observation obtained:")
        for k, v in obs.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        elite.disconnect()
        print("✅ Disconnected")


if __name__ == "__main__":
    main()