# !/usr/bin/env python

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
import os
import copy
from typing import Any

import numpy as np
import transforms3d as t3d
from teleop.utils.jacobi_robot import JacobiRobot
from controller import Supervisor
import cv2

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.robots.robot import Robot

from lerobot.common.robots.webots_xarm.webots_config_xarm import (
    WebotsXarmEndEffectorConfig,
)

logger = logging.getLogger(__name__)


class WebotsXarmEndEffector(Robot):
    config_class = WebotsXarmEndEffectorConfig
    name = "webots_x_armend_effector"

    def __init__(self, config: WebotsXarmEndEffectorConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self.jacobi = JacobiRobot(os.path.join(this_dir, "lite6.urdf"), ee_link="link6")

        self._arm = Supervisor()
        self.timestep_ = int(self._arm.getBasicTimeStep())
        self.joint_names_ = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.motors_ = [self._arm.getDevice(name) for name in self.joint_names_]

        self.position_sensors = []
        for motor in self.motors_:
            sensor = motor.getPositionSensor()
            sensor.enable(self.timestep_)
            self.position_sensors.append(sensor)
        self._arm.step(self.timestep_)

        self.left_motor_ = self._arm.getDevice("gripper_left_joint")
        self.right_motor_ = self._arm.getDevice("gripper_right_joint")

        self.is_gripper_open = True

        # open gripper on the beginning
        self.left_motor_.setPosition(0.0)
        self.right_motor_.setPosition(0.0)

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        joint_positions = [sensor.getValue() for sensor in self.position_sensors]
        for motor, position in zip(self.motors_, joint_positions):
            motor.setPosition(position)

        for joint in self.motors_:
            joint.setPosition(0)

        for i in range(1, 7):  # joints 1-6
            joint_name = f"joint{i}"
            self.jacobi.set_joint_position(joint_name, joint_positions[i - 1])

        self._camera = self._arm.getDevice("camera")

        logger.info(f"{self} connected.")
        if self._camera is None:
            print("Camera not found, disabling camera.")
        else:
            self._camera.enable(self.timestep_)

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (7),
            "names": {
                "joint1": 0,
                "joint2": 1,
                "joint3": 2,
                "joint4": 3,
                "joint5": 4,
                "joint6": 5,
                "gripper": 6,
            },
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'pose', 'gripper', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'.

        Returns:
            The joint-space action that was sent to the motors
        """

        gripper = action.get("gripper", None)
        action = copy.deepcopy(action)
        action["gripper.pos"] = gripper if gripper is not None else 0.0

        if (
            "delta_x" in action
            and "delta_y" in action
            and "delta_z" in action
            and "delta_roll" in action
            and "delta_pitch" in action
            and "delta_yaw" in action
        ):
            pose = self.jacobi.get_ee_pose()
            delta_pose = np.eye(4)
            delta_pose[:3, 3] = [
                action["delta_x"],
                action["delta_y"],
                action["delta_z"],
            ]
            roll = action["delta_roll"]
            pitch = action["delta_pitch"]
            yaw = action["delta_yaw"]
            delta_rotation = t3d.euler.euler2mat(roll, pitch, yaw)
            delta_pose[:3, :3] = delta_rotation

            action["pose"] = np.eye(4)
            action["pose"][:3, :3] = delta_rotation @ pose[:3, :3]
            action["pose"][:3, 3] = pose[:3, 3] + delta_pose[:3, 3]

        if "pose" in action:
            # Convert pose to joint positions using Jacobi
            pose = action["pose"]
            self.jacobi.servo_to_pose(pose)
            # Get joint positions from Jacobi
            joint_positions = []
            for i in range(1, 7):  # joints 1-6
                joint_pos = self.jacobi.get_joint_position(f"joint{i}")
                joint_positions.append(joint_pos)
                action[f"joint{i}.pos"] = joint_pos

            for motor, position in zip(self.motors_, joint_positions):
                motor.setPosition(position)

        # Send gripper command
        if gripper is not None:
            gripper_pos = -0.01 if gripper < 1.0 else 0.0
            print("Gripper position:", gripper_pos)
            self.open_gripper(gripper_pos)

        self.step()

        return super().send_action(action)

    def open_gripper(self, pos=0.0):
        self.left_motor_.setPosition(pos)
        self.right_motor_.setPosition(pos)

    def get_observation(self) -> dict[str, Any]:
        # Read joint positions from xarm
        joint_angles = [sensor.getValue() for sensor in self.position_sensors]

        ret = 0
        if joint_angles is None:
            ret = -1

        obs_dict = {}
        if ret == 0:  # Success
            # Convert joint angles to observation dict
            for i, angle in enumerate(joint_angles[:6]):  # First 6 angles are joints
                obs_dict[f"joint{i+1}.pos"] = angle

        obs_dict["gripper.pos"] = 0 if self.is_gripper_open else 2
        return obs_dict

    def reset(self):
        pass

    def disconnect(self) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        pass

    def step(self) -> bool:
        return self._arm.step(self.timestep_) != -1

    @property
    def is_connected(self) -> bool:
        pass

    @property
    def observation_features(self) -> dict[str, Any]:
        """Define observation features."""
        features = {
            "dtype": "float32",
            "shape": {},
            "names": {},
        }

        # Joint positions
        for i in range(1, 7):
            joint_name = f"joint{i}.pos"
            features["shape"][joint_name] = (1,)
            features["names"][joint_name] = joint_name

        # Gripper position
        features["shape"]["gripper.pos"] = (1,)
        features["names"]["gripper.pos"] = "gripper.pos"

        # Camera features
        for cam_key, cam in self.cameras.items():
            features["shape"][cam_key] = cam.shape
            features["names"][cam_key] = cam_key

        return features

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
