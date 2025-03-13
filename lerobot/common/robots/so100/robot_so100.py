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

import json
import logging
import time

import numpy as np

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    TorqueMode,
    run_full_arm_calibration,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .configuration_so100 import SO100RobotConfig


class SO100Robot(Robot):
    """
    [SO-100 Follower Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = SO100RobotConfig
    name = "so100"

    def __init__(self, config: SO100RobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.id = config.id

        self.arm = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": config.shoulder_pan,
                "shoulder_lift": config.shoulder_lift,
                "elbow_flex": config.elbow_flex,
                "wrist_flex": config.wrist_flex,
                "wrist_roll": config.wrist_roll,
                "gripper": config.gripper,
            },
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        self.is_connected = False
        self.logs = {}

    @property
    def state_feature(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def action_feature(self) -> dict:
        return self.state_feature

    @property
    def camera_features(self) -> dict[str, dict]:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft[cam_key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        logging.info("Connecting arm.")
        self.arm.connect()

        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.arm.write("Torque_Enable", TorqueMode.DISABLED.value)
        self.calibrate()

        # Mode=0 for Position Control
        self.arm.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        self.arm.write("P_Coefficient", 16)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.arm.write("I_Coefficient", 0)
        self.arm.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.arm.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.arm.write("Maximum_Acceleration", 254)
        self.arm.write("Acceleration", 254)

        logging.info("Activating torque.")
        self.arm.write("Torque_Enable", TorqueMode.ENABLED.value)

        # Check arm can be read
        self.arm.read("Present_Position")

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True

    def calibrate(self) -> None:
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        arm_calib_path = self.calibration_dir / f"{self.config.id}.json"

        if arm_calib_path.exists():
            with open(arm_calib_path) as f:
                calibration = json.load(f)
        else:
            # TODO(rcadene): display a warning in __init__ if calibration file not available
            logging.info(f"Missing calibration file '{arm_calib_path}'")
            calibration = run_full_arm_calibration(self.arm, self.robot_type, self.name, "follower")

            logging.info(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
            arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arm_calib_path, "w") as f:
                json.dump(calibration, f)

        self.arm.set_calibration(calibration)

    def get_observation(self) -> dict[str, np.ndarray]:
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        obs_dict = {}

        # Read arm position
        before_read_t = time.perf_counter()
        obs_dict[OBS_STATE] = self.arm.read("Present_Position")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()
            self.logs[f"read_camera_{cam_key}_dt_s"] = cam.logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (np.ndarray): array containing the goal positions for the motors.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        goal_pos = action

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.arm.read("Present_Position")
            goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.arm.write("Goal_Position", goal_pos.astype(np.int32))

        return goal_pos

    def print_logs(self):
        # TODO(aliberts): move robot-specific logs logic here
        pass

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        self.is_connected = False
