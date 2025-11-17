# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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

from functools import cached_property
import time
from typing import Any
from venv import logger

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.robots.robot import Robot
from lerobot.robots.sourccey.sourccey.sourccey_follower.sourccey_follower_calibrator import SourcceyFollowerCalibrator
from lerobot.robots.sourccey.sourccey.sourccey_follower.config_sourccey_follower import SourcceyFollowerConfig
from lerobot.robots.utils import ensure_safe_goal_position

class SourcceyFollower(Robot):
    config_class = SourcceyFollowerConfig
    name = "sourccey_follower"

    def __init__(self, config: SourcceyFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        motor_ids = [1, 2, 3, 4, 5, 6]
        if self.config.orientation == "right":
            motor_ids = [7, 8, 9, 10, 11, 12]

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(motor_ids[0], config.motor_models["shoulder_pan"], norm_mode_body),
                "shoulder_lift": Motor(motor_ids[1], config.motor_models["shoulder_lift"], norm_mode_body),
                "elbow_flex": Motor(motor_ids[2], config.motor_models["elbow_flex"], norm_mode_body),
                "wrist_flex": Motor(motor_ids[3], config.motor_models["wrist_flex"], norm_mode_body),
                "wrist_roll": Motor(motor_ids[4], config.motor_models["wrist_roll"], norm_mode_body),
                "gripper": Motor(motor_ids[5], config.motor_models["gripper"], MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # Initialize calibrator
        self.calibrator = SourcceyFollowerCalibrator(
            robot=self
        )

    def __del__(self):
        if (self.is_connected):
            self.disconnect()

    ###################################################################
    # Properties and Attributes
    ###################################################################
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    ###################################################################
    # Connection Management
    ###################################################################
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        print(f"Disconnecting Sourccey {self.config.orientation} Follower")
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    ###################################################################
    # Calibration and Configuration Management
    ###################################################################
    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Perform manual calibration."""
        self.calibration = self.calibrator.manual_calibrate()

    def auto_calibrate(self, reversed: bool = False, full_reset: bool = False) -> None:
        """Perform automatic calibration."""
        if full_reset:
            self.calibration = self.calibrator.auto_calibrate(reversed=reversed)
        else:
            self.calibration = self.calibrator.default_calibrate(reversed=reversed)

    def configure(self) -> None:
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

            if motor == "gripper":
                self.bus.write("P_Coefficient", motor, 170)
                self.bus.write("I_Coefficient", motor, 2)
                self.bus.write("D_Coefficient", motor, 55)
                self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                self.bus.write("Protection_Current", motor, 400)  # 50% of max current to avoid burnout
                self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded
            elif motor == "shoulder_lift":
                self.bus.write("P_Coefficient", motor, 12)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 48)  # Optimal damping (64 was too high)
                self.bus.write("Max_Torque_Limit", motor, 2000)
                self.bus.write("Protection_Current", motor, 4200)  # 4.2A for STS3250
                self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded
                self.bus.write("Minimum_Startup_Force", motor, 10)
                self.bus.write("CW_Dead_Zone", motor, 2)
                self.bus.write("CCW_Dead_Zone", motor, 2)
                self.bus.write("Acceleration", motor, 180)
            elif motor == "elbow_flex":
                self.bus.write("P_Coefficient", motor, 12)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 48)  # Optimal damping (64 was too high)
                self.bus.write("Max_Torque_Limit", motor, 2000)
                self.bus.write("Protection_Current", motor, 4200)  # 4.2A for STS3250
                self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded
                self.bus.write("Minimum_Startup_Force", motor, 10)
                self.bus.write("CW_Dead_Zone", motor, 2)
                self.bus.write("CCW_Dead_Zone", motor, 2)
                self.bus.write("Acceleration", motor, 180)
            else:
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 2)
                self.bus.write("D_Coefficient", motor, 32)
                self.bus.write("Max_Torque_Limit", motor, 1500)  # 80% of max torque
                self.bus.write("Protection_Current", motor, 1500)  # 80% of max current
                self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    ###################################################################
    # Data Management
    ###################################################################
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Check for NaN values and skip sending actions if any are found
        present_pos = self.bus.sync_read("Present_Position")
        if any(np.isnan(v) for v in goal_pos.values()) or any(np.isnan(v) for v in present_pos.values()):
            logger.warning("NaN values detected in goal positions. Skipping action execution.")
            return {f"{motor}.pos": val for motor, val in present_pos.items()}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

        # Check safety after sending goals
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    ###################################################################
    # Safety Functions (Must be used after extremely extensive testing
    # to ensure these functions don't make joints anything unsafe
    # or break the arm.)
    ###################################################################
    def _check_current_safety(self) -> list[str]:
        """
        Check if any motor is over current limit and return safety status.

        Returns:
            tuple: (is_safe, overcurrent_motors)
            - is_safe: True if all motors are under current limit
            - overcurrent_motors: List of motor names that are over current
        """
        # Read current from all motors
        currents = self.bus.sync_read("Present_Current")
        overcurrent_motors = []
        for motor, current in currents.items():
            if current > self.config.max_current_safety_threshold:
                overcurrent_motors.append(motor)
                logger.warning(f"Safety triggered: {motor} current {current}mA > {self.config.max_current_safety_threshold}mA")
        return overcurrent_motors

    def _handle_overcurrent_motors(
        self,
        overcurrent_motors: list[str],
        goal_pos: dict[str, float],
        present_pos: dict[str, float],
    ) -> dict[str, float]:
        """
        Handle overcurrent motors by replacing their goal positions with present positions.

        Args:
            goal_pos: Dictionary of goal positions with keys like "shoulder_pan.pos"
            present_pos: Dictionary of present positions with keys like "shoulder_pan.pos"
            overcurrent_motors: List of motor names that are over current (e.g., ["shoulder_pan", "elbow_flex"])

        Returns:
            Dictionary of goal positions with overcurrent motors replaced by present positions
        """
        if not overcurrent_motors or len(overcurrent_motors) == 0:
            return goal_pos

        # Create copies of the goal positions to modify
        modified_goal_pos = goal_pos.copy()
        for motor_name in overcurrent_motors:
            goal_key = f"{motor_name}.pos"
            if goal_key in modified_goal_pos:
                modified_goal_pos[goal_key] = present_pos[motor_name]
                logger.warning(f"Replaced goal position for {motor_name} with present position: {present_pos[motor_name]}")

        # Sync write the modified goal positions
        goal_pos_raw = {k.replace(".pos", ""): v for k, v in modified_goal_pos.items()}
        self.bus.sync_write("Goal_Position", goal_pos_raw)
        return modified_goal_pos
