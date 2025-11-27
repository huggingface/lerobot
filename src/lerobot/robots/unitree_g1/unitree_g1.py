#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import struct
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as hg_LowCmd,
    LowState_ as hg_LowState,
)
from unitree_sdk2py.utils.crc import CRC

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)

kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20


class MotorState:
    def __init__(self):
        self.q = None  # position
        self.dq = None  # velocity
        self.tau_est = None  # estimated torque
        self.temperature = None  # motor temperature


class IMUState:
    def __init__(self):
        self.quaternion = None  # [w, x, y, z]
        self.gyroscope = None  # [x, y, z] angular velocity (rad/s)
        self.accelerometer = None  # [x, y, z] linear acceleration (m/s²)
        self.rpy = None  # [roll, pitch, yaw] (rad)
        self.temperature = None  # IMU temperature


# g1 observation class
class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]
        self.imu_state = IMUState()
        self.wireless_remote = None  # Raw wireless remote data


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    # unitree remote controller
    class RemoteController:
        def __init__(self):
            self.lx = 0
            self.ly = 0
            self.rx = 0
            self.ry = 0
            self.button = [0] * 16

        def set(self, data):
            # wireless_remote
            keys = struct.unpack("H", data[2:4])[0]
            for i in range(16):
                self.button[i] = (keys & (1 << i)) >> i
            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config

        self.control_dt = config.control_dt

        # connect robot
        self.connect()

        # initialize direct motor control interface
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread to read robot state
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.is_connected:
            time.sleep(0.1)
            logger.warning("[UnitreeG1] Waiting to connect to robot...")
        logger.warning("[UnitreeG1] Connected to robot.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.lowstate_subscriber.Read().mode_machine

        # initialize all motors with unified kp/kd from config
        lowstate = self.lowstate_buffer.GetData()

        self.kp = np.array(config.kp, dtype=np.float32)
        self.kd = np.array(config.kd, dtype=np.float32)

        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp[id.value]
            self.msg.motor_cmd[id].kd = self.kd[id.value]
            self.msg.motor_cmd[id].q = lowstate.motor_state[id.value].q

        # Initialize remote controller
        self.remote_controller = self.RemoteController()

    def _subscribe_motor_state(self):  # polls robot state @ 250Hz
        while True:
            start_time = time.time()
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # Capture motor states
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = msg.motor_state[id].tau_est
                    lowstate.motor_state[id].temperature = msg.motor_state[id].temperature

                # Capture IMU state
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # Capture wireless remote data
                lowstate.wireless_remote = msg.wireless_remote

                self.lowstate_buffer.SetData(lowstate)

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # maintina constant control dt
            time.sleep(sleep_time)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{G1_29_JointIndex(motor).name}.pos": float for motor in G1_29_JointIndex}

    def calibrate(self) -> None:  # robot is already calibrated
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        ChannelFactoryInitialize(0)

    def disconnect(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        return self.lowstate_buffer.GetData()

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self.lowstate_buffer.GetData() is None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{G1_29_JointIndex(motor).name}.pos": float for motor in G1_29_JointIndex}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.msg.crc = self.crc.Crc(action)
        self.lowcmd_publisher.Write(action)

    def get_gravity_orientation(self, quaternion):  # get gravity orientation from quaternion
        """Get gravity orientation from quaternion."""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def transform_imu_data(
        self, waist_yaw, waist_yaw_omega, imu_quat, imu_omega
    ):  # transform imu data from torso to pelvis frame
        """Transform IMU data from torso to pelvis frame."""
        rot_zWaist = R.from_euler("z", waist_yaw).as_matrix()
        rot_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
        rot_pelvis = np.dot(rot_torso, rot_zWaist.T)
        w = np.dot(rot_zWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
        return R.from_matrix(rot_pelvis).as_quat()[[3, 0, 1, 2]], w
