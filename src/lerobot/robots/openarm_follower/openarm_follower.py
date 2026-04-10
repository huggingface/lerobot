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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)

logger = logging.getLogger(__name__)


class OpenArmFollower(Robot):
    """
    OpenArms Follower Robot: 7 DOF Damiao arm (CAN) + UMI-style Feetech gripper (serial).
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        # Arm motors (Damiao on CAN bus)
        arm_motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(send_id, motor_type_str, MotorNormMode.DEGREES)
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            arm_motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=arm_motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        # Gripper motors (Feetech STS3215 on serial bus)
        gripper_motors: dict[str, Motor] = {
            name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_0_100)
            for name, motor_id in config.gripper_motor_ids.items()
        }
        self.gripper_bus = FeetechMotorsBus(
            port=config.gripper_port,
            motors=gripper_motors,
            calibration=self.calibration,
        )

        if config.side is not None:
            if config.side == "left":
                config.joint_limits = LEFT_DEFAULT_JOINTS_LIMITS
            elif config.side == "right":
                config.joint_limits = RIGHT_DEFAULT_JOINTS_LIMITS
            else:
                raise ValueError(
                    "config.side must be either 'left', 'right' (for default values) or 'None' (for CLI values)"
                )
        else:
            logger.info(
                "Set config.side to either 'left' or 'right' to use pre-configured values for joint limits."
            )
        logger.info(f"Values used for joint limits: {config.joint_limits}.")

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float
            features[f"{motor}.torque"] = float
        for motor in self.gripper_bus.motors:
            features[f"{motor}.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.bus.is_connected
            and self.gripper_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arms are in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        logger.info(f"Connecting gripper on {self.config.gripper_port}...")
        self.gripper_bus.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        if self.bus.is_calibrated:
            self.bus.set_zero_position()

        self.bus.enable_torque()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated and self.gripper_bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration for both the Damiao arm and Feetech gripper.

        Arm calibration: set zero position with arm hanging, ±90° default range.
        Gripper calibration: SO100-style half-turn homing + range recording.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                self.gripper_bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")

        # --- Arm calibration (Damiao) ---
        self.bus.disable_torque()
        input(
            "\nCalibration: Set Zero Position\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )
        self.bus.write_calibration(self.calibration)

        # --- Gripper calibration (Feetech) ---
        self.gripper_bus.disable_torque()
        for motor in self.gripper_bus.motors:
            self.gripper_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move gripper to the middle of its range of motion and press ENTER....")
        homing_offsets = self.gripper_bus.set_half_turn_homings()

        gripper_motor_names = list(self.gripper_bus.motors.keys())
        print(
            f"Move gripper joints ({', '.join(gripper_motor_names)}) through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.gripper_bus.record_ranges_of_motion(gripper_motor_names)

        for motor_name, m in self.gripper_bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )
        self.gripper_bus.write_calibration(self.calibration)

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure both arm (Damiao) and gripper (Feetech) motors."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()

        with self.gripper_bus.torque_disabled():
            self.gripper_bus.configure_motors()
            for motor in self.gripper_bus.motors:
                self.gripper_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.gripper_bus.write("P_Coefficient", motor, 16)
                self.gripper_bus.write("I_Coefficient", motor, 0)
                self.gripper_bus.write("D_Coefficient", motor, 32)
                self.gripper_bus.write("Max_Torque_Limit", motor, 500)
                self.gripper_bus.write("Protection_Current", motor, 250)
                self.gripper_bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read all motor states from arm (CAN) and gripper (serial), plus cameras."""
        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}

        # Arm motors (Damiao) — pos/vel/torque in one CAN refresh cycle
        states = self.bus.sync_read_all_states()
        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
            obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

        # Gripper motors (Feetech) — position only
        gripper_positions = self.gripper_bus.sync_read("Present_Position")
        for motor, val in gripper_positions.items():
            obs_dict[f"{motor}.pos"] = val

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """
        Send action command to robot. Arm joints go to Damiao CAN bus,
        gripper joints go to Feetech serial bus.

        Args:
            action: Dictionary with motor positions (e.g., "joint_1.pos", "proximal.pos")
            custom_kp: Optional custom kp gains per arm motor
            custom_kd: Optional custom kd gains per arm motor

        Returns:
            The action actually sent (potentially clipped)
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Apply joint limit clipping
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f} to {clipped_position:.2f}")
                goal_pos[motor_name] = clipped_position

        # Split into arm and gripper actions
        arm_motors = set(self.bus.motors.keys())
        gripper_motors = set(self.gripper_bus.motors.keys())
        arm_goal = {k: v for k, v in goal_pos.items() if k in arm_motors}
        gripper_goal = {k: v for k, v in goal_pos.items() if k in gripper_motors}

        # Cap arm goal position when too far away from present position
        if self.config.max_relative_target is not None and arm_goal:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal.items()}
            arm_goal = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Arm: batch MIT control (Damiao)
        if arm_goal:
            arm_motor_names = list(self.bus.motors.keys())
            commands = {}
            for motor_name, position_degrees in arm_goal.items():
                idx = arm_motor_names.index(motor_name) if motor_name in arm_motor_names else 0
                if custom_kp is not None and motor_name in custom_kp:
                    kp = custom_kp[motor_name]
                else:
                    kp = (
                        self.config.position_kp[idx]
                        if isinstance(self.config.position_kp, list)
                        else self.config.position_kp
                    )
                if custom_kd is not None and motor_name in custom_kd:
                    kd = custom_kd[motor_name]
                else:
                    kd = (
                        self.config.position_kd[idx]
                        if isinstance(self.config.position_kd, list)
                        else self.config.position_kd
                    )
                commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)
            self.bus._mit_control_batch(commands)

        # Gripper: position control (Feetech)
        if gripper_goal:
            self.gripper_bus.sync_write("Goal_Position", gripper_goal)

        goal_pos.update(arm_goal)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        self.gripper_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
