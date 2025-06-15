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
import time
import os
import copy
from typing import Any

import numpy as np
import transforms3d as t3d
from teleop.utils.jacobi_robot import JacobiRobot
from controller import Supervisor
from PIL import Image
import cv2
import time

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.common.robots.robot import Robot

from lerobot.common.robots.webots_xarm.webots_config_xarm import WebotsXarmEndEffectorConfig

logger = logging.getLogger(__name__)

class WebotsXarmEndEffector(Robot):
    config_class = WebotsXarmEndEffectorConfig
    name = "webots_xarm_end_effector"

    def __init__(self, config: WebotsXarmEndEffectorConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self.jacobi = JacobiRobot(os.path.join(this_dir, "lite6.urdf"), ee_link="link6")
        
        self.arm_ = Supervisor()
        self.timestep_ = int(self.arm_.getBasicTimeStep())
        self.joint_names_ = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.motors_ = [self.arm_.getDevice(name) for name in self.joint_names_]
        
        self.position_sensors = []
        for motor in self.motors_:
            sensor = motor.getPositionSensor()
            sensor.enable(self.timestep_)
            self.position_sensors.append(sensor)
        self.arm_.step(self.timestep_)
        
        self.left_motor_ = self.arm_.getDevice("gripper_left_joint")
        self.right_motor_ = self.arm_.getDevice("gripper_right_joint")
        
        self.is_gripper_open = True
        
        # open gripper on the begining
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
        print(joint_positions)
        print("00000000000")
        
        for joint in self.motors_:
            joint.setPosition(0)
        
        for i in range(1, 7):  # joints 1-6
            joint_name = f"joint{i}"
            self.jacobi.set_joint_position(joint_name, joint_positions[i - 1])

        self.camera_ = self.arm_.getDevice("camera")

        logger.info(f"{self} connected.")
        if self.camera_ is None:
            print("❌ Kamera nije pronađena. Proveri tačno ime u Webotsu.")
        else:
            self.camera_.enable(self.timestep_)

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

        gripper = action["gripper"] if "gripper" in action else None
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
                # print(position)
            

        # Send gripper command
        if gripper is not None:
            if gripper < 1.0:
                self.arm.close_lite6_gripper()
            else:
                self.arm.open_lite6_gripper()

        return super().send_action(action)
    
    def close_gripper(self, pos=-0.01):
        
        if self.is_gripper_open: 
            self.left_motor_.setPosition(pos)
            self.right_motor_.setPosition(pos)
            self.is_gripper_open = False
    
    def open_gripper(self, pos=0.0):
        
        if not self.is_gripper_open: 
            self.left_motor_.setPosition(pos)
            self.right_motor_.setPosition(pos)
            self.is_gripper_open = True

    def get_observation(self) -> dict[str, Any]:
        # Read arm position
        start = time.perf_counter()

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

        obs_dict["gripper.pos"] = 1 if self.is_gripper_open else 0

        # Capture images from cameras
        image = self.camera_.getImage()
        width = self.camera_.getWidth()
        height = self.camera_.getHeight()
        img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        rgb_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_array, 'RGB')

        # timestamp = int(time.time() * 1000)  # vreme u milisekundama
        # filename = f"/home/marija/Documents/spes_lerobot/lerobot/lerobot/common/robots/webots_xarm/images/img_{timestamp}.png"
        # pil_image.save(filename)
        
        # for cam_key, cam in self.cameras.items():
        #     start = time.perf_counter()
        #     obs_dict[cam_key] = cam.async_read()
        #     dt_ms = (time.perf_counter() - start) * 1e3
        #     logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

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
        return self.arm_.step(self.timestep_) != -1


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

    @property
    def arm(self):
        return self.arm_

    @arm.setter
    def arm(self, value):
        self.arm_ = value

    @property
    def jacobi(self):
        return self._jacobi

    @jacobi.setter
    def jacobi(self, value):
        self._jacobi = value

if __name__ == "__main__":
    import transforms3d as t3d

    # Example usage
    config = WebotsXarmEndEffectorConfig()
    robot = WebotsXarmEndEffector(config)
    robot.connect()
    
    # export WEBOTS_HOME=/usr/local/webots
    # $WEBOTS_HOME/webots-controller lerobot/common/robots/webots_xarm/webots_xarm_end_effector.py 


    # Example action
    action = {
        "pose": t3d.affines.compose(
            [0.25, 0.0, 0.4],  # Translation
            t3d.euler.euler2mat(3.14, 0, 0.0),  # Rotation (no rotation)
            [1.0, 1.0, 1.0],  # Scale
        ),
        # "gripper": 2.0,
    }

    action["delta_x"] = 0.5
    action["delta_y"] = 0.0
    action["delta_z"] = 0.01
    action["delta_roll"] = 0.0
    action["delta_pitch"] = 0.0
    action["delta_yaw"] = 0.0
    
    robot.close_gripper(-0.01)
    
    while robot.step():
        robot.send_action(action)
        obs = robot.get_observation()
        print(obs)
        time.sleep(0.05)
        robot.open_gripper(0.0)
