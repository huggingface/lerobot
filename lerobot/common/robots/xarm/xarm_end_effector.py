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

from lerobot.common.robots.xarm.config_xarm import XarmEndEffector

logger = logging.getLogger(__name__)


# pip install xarm-python-sdk
class XarmEndEffector(Robot):
    config_class = XarmEndEffector
    name = "xarm_end_effector"

    def __init__(self, config: XarmEndEffector):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self.is_connected = False
        self.arm = None
        self.jacobi = JacobiRobot(this_dir, "lite6.urdf", ee_link="link6")

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



if __name__ == "__main__":
    # Example usage
    config = XarmEndEffector(
        cameras=[],
        max_relative_target=0.1,
        disable_torque_on_disconnect=True,
    )
    robot = XarmEndEffector(config)
    robot.connect()
    
    # Example action
    action = {
        "pose": np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0]),
        "gripper": 1.0,
    }
    
    robot.send_action(action)
    obs = robot.get_observation()
    print(obs)
    
    robot.disconnect()
