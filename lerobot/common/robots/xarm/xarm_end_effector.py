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
import math
from typing import Any

import numpy as np
import transforms3d as t3d
from xarm.wrapper import XArmAPI
from teleop.utils.jacobi_robot import JacobiRobot

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.common.robots.robot import Robot

from lerobot.common.robots.xarm.config_xarm import XarmEndEffectorConfig

logger = logging.getLogger(__name__)


class Lite6Gripper:
    def __init__(self, arm: XArmAPI):
        self._arm = arm
        self._prev_gripper_state = None
        self._gripper_state = 0.0
        self._gripper_open_time = 0.0
        self._gripper_stopped = False

    def open(self):
        self._arm.open_lite6_gripper()

    def close(self):
        self._arm.close_lite6_gripper()

    def stop(self):
        self._arm.stop_lite6_gripper()

    def set_gripper_state(self, gripper_state: float) -> None:
        """Set gripper state and handle opening/closing logic."""
        self._gripper_state = gripper_state

        if self._gripper_state is not None:
            if (
                self._prev_gripper_state is None
                or self._prev_gripper_state != self._gripper_state
            ):
                if self._gripper_state < 1.0:
                    self._gripper_stopped = False
                    self.close()
                else:
                    self._gripper_open_time = time.time()
                    self._gripper_stopped = False
                    self.open()

            if (
                not self._gripper_stopped
                and self._gripper_state >= 1.0
                and time.time() - self._gripper_open_time > 1.0
            ):
                # If gripper was closed and now is open, stop the gripper
                self._gripper_stopped = True
                self.stop()

        self._prev_gripper_state = self._gripper_state

    def get_gripper_state(self) -> float:
        """Get current gripper state."""
        return self._gripper_state

    def reset_gripper(self) -> None:
        """Reset gripper state."""
        self._prev_gripper_state = None
        self._gripper_state = 0.0
        self._gripper_open_time = 0.0
        self._gripper_stopped = False


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
        self._arm = None
        self._gripper = None
        self._jacobi = JacobiRobot(
            os.path.join(this_dir, "lite6.urdf"), ee_link="link6"
        )
        self._initial_pose = None

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to the robot and set it to a servo mode
        self._arm = XArmAPI("192.168.1.184", is_radian=True)
        self._arm.connect()
        self._arm.motion_enable(enable=True)
        self._arm.set_mode(1)  # Position mode
        self._arm.set_state(state=0)  # Sport state

        # Initialize gripper
        self._gripper = Lite6Gripper(self._arm)

        # set joint positions to jacobi, read from arm
        code, joint_positions = self._arm.get_servo_angle()
        if code != 0:
            raise DeviceNotConnectedError(f"Failed to get joint angles from {self}")
        for i in range(1, 7):  # joints 1-6
            joint_name = f"joint{i}"
            self._jacobi.set_joint_position(joint_name, joint_positions[i - 1])

        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Return mapping of motor feature names to their Python types (float)."""
        motors = {f"joint{i}.pos": float for i in range(1, 7)}
        motors["gripper.pos"] = float
        return motors

    @property
    def action_features(self) -> dict[str, type]:
        """Joint commands expected by the robot (all scalar floats)."""
        return self._motors_ft

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

        action = copy.deepcopy(action)

        if (
            "delta_x" in action
            and "delta_y" in action
            and "delta_z" in action
            and "delta_roll" in action
            and "delta_pitch" in action
            and "delta_yaw" in action
        ):
            pose = self._jacobi.get_ee_pose()
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

        if "pose_from_initial" in action:
            delta_pose = action["pose_from_initial"]
            action["pose"] = np.eye(4)
            action["pose"][:3, :3] = delta_pose[:3, :3] @ self._initial_pose[:3, :3]
            action["pose"][:3, 3] = self._initial_pose[:3, 3] + delta_pose[:3, 3]

        if "pose" in action:
            # Convert pose to joint positions using Jacobi
            pose = action["pose"]
            self._jacobi.servo_to_pose(pose)
            # Get joint positions from Jacobi
            for i in range(1, 7):
                joint_pos = self._jacobi.get_joint_position(f"joint{i}")
                action[f"joint{i}.pos"] = joint_pos

        if "home" in action:
            joint_targets = [
                0,
                math.radians(10),
                math.radians(32),
                0,
                math.radians(22),
                0,
            ]
            for i in range(1, 7):
                joint_name = f"joint{i}"
                joint_pos = self._jacobi.get_joint_position(joint_name)
                error = joint_targets[i - 1] - joint_pos
                error = np.clip(error, -0.1, 0.1)
                target_pos = joint_pos + error * 0.07
                self._jacobi.set_joint_position(joint_name, target_pos)
                action[f"joint{i}.pos"] = target_pos

        # Execute joint positions
        joint_positions = []
        for i in range(1, 7):
            joint_pos = action[f"joint{i}.pos"]
            joint_positions.append(joint_pos)
        self._arm.set_servo_angle_j(joint_positions)

        # Send gripper command
        if "gripper" in action:
            self._gripper.set_gripper_state(action["gripper"])
            action["gripper.pos"] = self._gripper.get_gripper_state()

        # Return the joint-space command dictionary so that the recorder can
        # store every value in the dataset.
        return action

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        # Read joint positions from xarm
        code, (joint_angles, joint_velocities, joint_currents) = (
            self._arm.get_joint_states()
        )

        obs_dict = {}
        for i, angle in enumerate(joint_angles[:6]):  # First 6 angles are joints
            obs_dict[f"joint{i+1}.pos"] = angle
        obs_dict["gripper.pos"] = self._gripper.get_gripper_state()

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        if self._gripper:
            self._gripper.reset_gripper()

    def disconnect(self) -> None:
        """Disconnect from the robot and cameras."""
        if not self.is_connected:
            return

        if self._gripper is not None:
            self._gripper.stop()
            self._gripper = None

        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None

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

        self._initial_pose = self._jacobi.get_ee_pose()
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
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        """Joint positions and camera image shapes returned by the robot."""
        return {**self._motors_ft, **self._cameras_ft}

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
            [0.25, 0.0, 0.2],  # Translation
            t3d.euler.euler2mat(3.14, 0, 0.0),  # Rotation (no rotation)
            [1.0, 1.0, 1.0],  # Scale
        ),
        # "gripper": 2.0,
    }

    action["delta_x"] = 0.0
    action["delta_y"] = 0.0
    action["delta_z"] = 0.01
    action["delta_roll"] = 0.0
    action["delta_pitch"] = 0.0
    action["delta_yaw"] = 0.0

    for _ in range(30):
        robot.send_action(action)
        time.sleep(0.05)

    obs = robot.get_observation()

    robot.disconnect()
