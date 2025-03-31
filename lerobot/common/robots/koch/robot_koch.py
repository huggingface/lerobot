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
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
    TorqueMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .configuration_koch import KochRobotConfig


class KochRobot(Robot):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochRobotConfig
    name = "koch"

    def __init__(self, config: KochRobotConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.arm = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl430-w250", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
            },
        )
        self.cameras = make_cameras_from_configs(config.cameras)

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

    @property
    def is_connected(self) -> bool:
        # TODO(aliberts): add cam.is_connected for cam in self.cameras
        return self.arm.is_connected

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        logging.info("Connecting arm.")
        self.arm.connect()
        if not self.is_calibrated:
            self.calibrate()

        self.configure()

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

    def configure(self) -> None:
        for name in self.arm.names:
            # Torque must be deactivated to change values in the motors' EEPROM area
            # We assume that at configuration time, arm is in a rest position,
            # and torque can be safely disabled to run calibration.
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)
            if name != "gripper":
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.arm.write("Operating_Mode", name, OperatingMode.EXTENDED_POSITION.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.arm.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

        # Set better PID values to close the gap between recorded states and actions
        # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
        self.arm.write("Position_P_Gain", "elbow_flex", 1500)
        self.arm.write("Position_I_Gain", "elbow_flex", 0)
        self.arm.write("Position_D_Gain", "elbow_flex", 600)

        for name in self.arm.names:
            self.arm.write("Torque_Enable", name, TorqueMode.ENABLED.value)

        logging.info("Torque enabled.")

    @property
    def is_calibrated(self) -> bool:
        motors_calibration = self.arm.get_calibration()
        return motors_calibration == self.calibration

    def calibrate(self) -> None:
        print(f"\nRunning calibration of {self.id} Koch robot")
        for name in self.arm.names:
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)
            self.arm.write("Operating_Mode", name, OperatingMode.EXTENDED_POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.arm.set_half_turn_homings()

        fixed_range = ["shoulder_pan", "wrist_roll"]
        auto_range_motors = [name for name in self.arm.names if name not in fixed_range]
        print(
            "Move all joints except 'wrist_roll' (id=5) sequentially through their entire ranges of motion."
        )
        print("Recording positions. Press ENTER to stop...")
        ranges = self.arm.register_ranges_of_motion(auto_range_motors)
        for name in fixed_range:
            ranges[name] = {"min": 0, "max": 4095}

        self.calibration = {}
        for name, motor in self.arm.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=ranges[name]["min"],
                range_max=ranges[name]["max"],
            )

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Read arm position
        before_read_t = time.perf_counter()
        obs_dict[OBS_STATE] = self.arm.sync_read("Present_Position")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()
            self.logs[f"read_camera_{cam_key}_dt_s"] = cam.logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        goal_pos = action

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.arm.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.arm.sync_write("Goal_Position", goal_pos)

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
