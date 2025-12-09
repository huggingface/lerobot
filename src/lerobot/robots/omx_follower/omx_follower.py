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
from importlib.resources import files
from typing import Any

import draccus

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_omx_follower import OmxFollowerConfig

logger = logging.getLogger(__name__)


class OmxFollower(Robot):
    """
    OMX Follower robot with Dynamixel motors.
    Motors 1,2,3: xl430-w250
    Motors 4,5,6: xl330-m288
    """

    config_class = OmxFollowerConfig
    name = "omx_follower"

    def __init__(self, config: OmxFollowerConfig):
        super().__init__(config)
        self.config = config
        
        # Load default calibration from package if user calibration file doesn't exist
        # Also check if cached calibration has wrong range values and reload from default
        should_reload_default = False
        if self.calibration:
            # Check if shoulder_lift or elbow_flex have wrong range values (not 0~4095)
            for motor_name in ["shoulder_lift", "elbow_flex"]:
                if motor_name in self.calibration:
                    cal = self.calibration[motor_name]
                    if cal.range_min != 0 or cal.range_max != 4095:
                        logger.warning(
                            f"Found incorrect calibration range for {motor_name}: "
                            f"range_min={cal.range_min}, range_max={cal.range_max}. "
                            "Will reload from default calibration file."
                        )
                        should_reload_default = True
                        break
        
        if not self.calibration or should_reload_default:
            try:
                default_calibration_path = files("lerobot.robots.omx_follower") / "omx_follower_default.json"
                if default_calibration_path.is_file():
                    with default_calibration_path.open() as f, draccus.config_type("json"):
                        self.calibration = draccus.load(dict[str, MotorCalibration], f)
                    logger.info(f"Loaded default calibration from package for {self.id}")
                    if should_reload_default:
                        # Save corrected calibration to cache
                        self._save_calibration()
                        logger.info(f"Saved corrected calibration to {self.calibration_fpath}")
            except Exception as e:
                logger.debug(f"Could not load default calibration from package: {e}")
        
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # Get motor IDs from calibration if available, otherwise use defaults
        motor_ids = {}
        if self.calibration:
            motor_ids = {motor: cal.id for motor, cal in self.calibration.items()}
        else:
            # Default motor IDs (will be overridden by calibration file)
            motor_ids = {
                "shoulder_pan": 11,
                "shoulder_lift": 12,
                "elbow_flex": 13,
                "wrist_flex": 14,
                "wrist_roll": 15,
                "gripper": 16,
            }
        
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(motor_ids.get("shoulder_pan", 11), "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(motor_ids.get("shoulder_lift", 12), "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(motor_ids.get("elbow_flex", 13), "xl430-w250", norm_mode_body),
                "wrist_flex": Motor(motor_ids.get("wrist_flex", 14), "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(motor_ids.get("wrist_roll", 15), "xl330-m288", norm_mode_body),
                "gripper": Motor(motor_ids.get("gripper", 16), "xl330-m288", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

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

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        
        For OMX robots that come pre-calibrated:
        - If default calibration from package doesn't match motors, read from motors and save
        - This allows using pre-calibrated robots without manual calibration
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        
        # OMX robots don't require calibration - use calibration file values directly
        # Don't write to motors, just use the calibration file's range_min/max values
        if self.calibration:
            # Use calibration file values directly (don't write to motors)
            self.bus.calibration = self.calibration

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # OMX robots don't require calibration - always return True
        return True

    def calibrate(self) -> None:
        """
        OMX robots don't require calibration - use factory values from calibration file.
        This method is overwritten to prevent accidental calibration that could overwrite
        pre-calibrated motor values.
        """
        if self.calibration:
            # Use calibration file values (factory values) - don't write to motors
            logger.info(f"OMX robot {self.id} uses pre-calibrated values from calibration file. No calibration needed.")
            # Ensure bus calibration is set to match the file
            self.bus.calibration = self.calibration
        else:
            logger.warning(f"No calibration file found for {self.id}. Calibration file is required for OMX robots.")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            
            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
            # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling
            # the arm, you could end up with a servo with a position 0 or 4095 at a crucial point
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # Use 'position control current based' for gripper to be limited by the limit of the current. For
            # the follower gripper, it means it can grasp an object without forcing too much even tho, its
            # goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with
            # our finger to make it move, and it will move back to its original target position when we
            # release the force.
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
            
            # Set gripper drive mode to INVERTED to reverse gripper direction
            self.bus.write("Drive_Mode", "gripper", DriveMode.INVERTED.value)
            
            # Set gripper's goal pos in current position mode for spring effect (if configured)
            if self.config.gripper_open_pos is not None:
                self.bus.enable_torque("gripper")
                if self.is_calibrated:
                    self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        # Read motors that need special handling (shoulder_pan, shoulder_lift, elbow_flex)
        special_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
        other_motors = [m for m in self.bus.motors.keys() if m not in special_motors]
        obs_dict = self.bus.sync_read("Present_Position", other_motors) if other_motors else {}
        
        # Read special motors without normalization to use calibration range
        for motor_name in special_motors:
            if motor_name not in self.bus.motors:
                continue
            raw_pos = self.bus.read("Present_Position", motor_name, normalize=False)
            drive_mode = self.bus.apply_drive_mode and self.bus.calibration[motor_name].drive_mode
            norm_mode = self.bus.motors[motor_name].norm_mode
            
            if motor_name == "shoulder_pan":
                # Use extended range for EXTENDED_POSITION mode
                extended_range = 2097152  # 512 turns * 4096 steps per turn
                min_ = -extended_range // 2
                max_ = extended_range // 2 - 1
            else:
                # For shoulder_lift and elbow_flex, use calibration range_min/max
                # This ensures leader and follower use the same normalization range
                min_ = self.bus.calibration[motor_name].range_min
                max_ = self.bus.calibration[motor_name].range_max
            
            if norm_mode == MotorNormMode.RANGE_M100_100:
                norm = (((raw_pos - min_) / (max_ - min_)) * 200) - 100 if max_ != min_ else 0
                obs_dict[motor_name] = -norm if drive_mode else norm
            elif norm_mode == MotorNormMode.RANGE_0_100:
                norm = ((raw_pos - min_) / (max_ - min_)) * 100 if max_ != min_ else 0
                obs_dict[motor_name] = 100 - norm if drive_mode else norm
            elif norm_mode == MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.bus.model_resolution_table[self.bus.motors[motor_name].model] - 1
                obs_dict[motor_name] = (raw_pos - mid) * 360 / max_res
            else:
                obs_dict[motor_name] = raw_pos
        
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
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Handle special motors (shoulder_pan, shoulder_lift, elbow_flex) separately
        special_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
        special_goals = {}
        for motor_name in special_motors:
            if motor_name in goal_pos:
                special_goals[motor_name] = goal_pos.pop(motor_name)
        
        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", list(goal_pos.keys()) if goal_pos else None)
            for motor_name in special_goals:
                present_pos[motor_name] = self.bus.read("Present_Position", motor_name, normalize=False)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            for motor_name, goal_val in special_goals.items():
                goal_present_pos[motor_name] = (goal_val, present_pos[motor_name])
            goal_pos_updated = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            # Extract special motors back
            for motor_name in special_motors:
                if motor_name in goal_pos_updated:
                    special_goals[motor_name] = goal_pos_updated.pop(motor_name)

        # Send goal position to the arm (other motors)
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)
        
        # Send special motors' goal positions separately using calibration range
        for motor_name, goal_val in special_goals.items():
            drive_mode = self.bus.apply_drive_mode and self.bus.calibration[motor_name].drive_mode
            norm_mode = self.bus.motors[motor_name].norm_mode
            
            if motor_name == "shoulder_pan":
                # Use extended range for EXTENDED_POSITION mode
                extended_range = 2097152  # 512 turns * 4096 steps per turn
                min_ = -extended_range // 2
                max_ = extended_range // 2 - 1
            else:
                # For shoulder_lift and elbow_flex, use calibration range_min/max
                # This ensures leader and follower use the same normalization range
                min_ = self.bus.calibration[motor_name].range_min
                max_ = self.bus.calibration[motor_name].range_max
            
            if norm_mode == MotorNormMode.RANGE_M100_100:
                val = -goal_val if drive_mode else goal_val
                raw_goal = int(((val + 100) / 200) * (max_ - min_) + min_) if max_ != min_ else int((min_ + max_) / 2)
            elif norm_mode == MotorNormMode.RANGE_0_100:
                val = 100 - goal_val if drive_mode else goal_val
                raw_goal = int((val / 100) * (max_ - min_) + min_) if max_ != min_ else int((min_ + max_) / 2)
            elif norm_mode == MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.bus.model_resolution_table[self.bus.motors[motor_name].model] - 1
                raw_goal = int((goal_val * max_res / 360) + mid)
            else:
                raw_goal = int(goal_val)
            self.bus.write("Goal_Position", motor_name, raw_goal, normalize=False)
            goal_pos[motor_name] = goal_val
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

