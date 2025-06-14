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
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI
from teleop.utils.jacobi_robot import JacobiRobot

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.common.robots.robot import Robot

from lerobot.common.robots.xarm.config_xarm import XarmEndEffectorConfig

logger = logging.getLogger(__name__)


# pip install xarm-python-sdk
class XarmEndEffector(Robot):
    config_class = XarmEndEffectorConfig
    name = "xarm_end_effector"

    def __init__(self, config: XarmEndEffectorConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self._is_connected = False
        self.arm = None
        self.jacobi = JacobiRobot(os.path.join(this_dir, "lite6.urdf"), ee_link="link6")

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to the robot and set it to a servo mode
        self.arm = XArmAPI("192.168.1.184")
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(state=0)  # Sport state

        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True
        self.configure()
        logger.info(f"{self} connected.")

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

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        gripper = action["gripper"]

        full_action = action.copy()
        full_action["gripper.pos"] = gripper

        if "pose" in action:
            # Convert pose to joint positions using Jacobi
            pose = action["pose"]
            success = self.jacobi.servo_to_pose(pose)
            if success:
                # Get joint positions from Jacobi
                joint_positions = []
                for i in range(1, 7):  # joints 1-6
                    joint_pos = self.jacobi.get_joint_position(f"joint{i}")
                    joint_positions.append(joint_pos)
                    full_action[f"joint{i}.pos"] = joint_pos

                # Send joint positions to xarm
                self.arm.set_servo_angle(angle=joint_positions, wait=False)
            else:
                logger.warning("Failed to solve inverse kinematics for pose")

        # Extract joint actions for motors
        joint_action = {"gripper.pos": gripper}
        for i in range(1, 7):
            if f"joint{i}" in action:
                joint_action[f"joint{i}.pos"] = action[f"joint{i}"]
                # Send individual joint command if direct joint control
                if "pose" not in action:
                    current_angles = self.arm.get_servo_angle()[
                        1
                    ]  # [1] contains the angles
                    current_angles[i - 1] = action[f"joint{i}"]
                    self.arm.set_servo_angle(angle=current_angles, wait=False)

        # Send gripper command
        if gripper is not None:
            if gripper < 0.5:
                self.arm.set_gripper_enable(enable=False)
            else:
                self.arm.set_gripper_enable(enable=True)

        return super().send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        # Read joint positions from xarm
        ret, joint_angles = self.arm.get_servo_angle()
        ret_gripper, gripper_pos = self.arm.get_gripper_position()

        obs_dict = {}
        if ret == 0:  # Success
            # Convert joint angles to observation dict
            for i, angle in enumerate(joint_angles):
                obs_dict[f"joint{i+1}.pos"] = angle

        if ret_gripper == 0:  # Success
            # Normalize gripper position to 0-1 range
            obs_dict["gripper.pos"] = gripper_pos / 850.0

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        pass

    def disconnect(self) -> None:
        """Disconnect from the robot and cameras."""
        if not self.is_connected:
            return
        
        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None
        
        for cam in self.cameras.values():
            cam.disconnect()
        
        self.is_connected = False
        logger.info(f"{self} disconnected.")

    def calibrate(self) -> None:
        """Calibrate the robot (optional for xarm)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # XArm doesn't typically require calibration
        # This could be used for homing or setting reference positions
        logger.info(f"{self} calibration completed.")

    def configure(self) -> None:
        """Configure robot settings."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Set motion parameters
        self.arm.set_tcp_maxacc(1000)  # Max acceleration
        logger.info(f"{self} configured.")

    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        # XArm is considered always calibrated when connected
        return self.is_connected

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """Set connection status."""
        self._is_connected = value

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
        return self._arm

    @arm.setter
    def arm(self, value):
        self._arm = value

    @property
    def jacobi(self):
        return self._jacobi

    @jacobi.setter
    def jacobi(self, value):
        self._jacobi = value



if __name__ == "__main__":
    import transforms3d as t3d

    # Example usage
    config = XarmEndEffectorConfig()
    robot = XarmEndEffector(config)
    robot.connect()
    
    # Example action
    action = {
        "pose": t3d.affines.compose(
            [0.5, 0.0, 0.2],  # Translation
            t3d.euler.euler2mat(0.0, 3.14, 0.0),  # Rotation (no rotation)
            [1.0, 1.0, 1.0]  # Scale
        ),
        "gripper": 1.0,
    }
    
    robot.send_action(action)
    obs = robot.get_observation()
    print(obs)
    
    robot.disconnect()
