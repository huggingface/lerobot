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

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.robots.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig

from ..robot import Robot
from .config_xlerobot import XLeRobotConfig

logger = logging.getLogger(__name__)


class XLeRobot(Robot):
    """
    XLeRobot: A bimanual robot with SO-100 Follower Arms plus additional head joints and base motors.

    Left arm: SO100Follower + 2 positional control motors as head joints
    Right arm: SO100Follower + 3 velocity control motors as base (similar to lekiwi)
    """

    config_class = XLeRobotConfig
    name = "xlerobot"

    def __init__(self, config: XLeRobotConfig):
        super().__init__(config)
        self.config = config

        # Left arm configuration (SO100Follower)
        left_arm_config = SO100FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        # Right arm configuration (SO100Follower)
        right_arm_config = SO100FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        # Head joints configuration (positional control)
        norm_mode_head = MotorNormMode.DEGREES if config.head_use_degrees else MotorNormMode.RANGE_M100_100
        self.head_bus = FeetechMotorsBus(
            port=self.config.head_port,
            motors={
                "head_pan": Motor(1, "sts3215", norm_mode_head),
                "head_tilt": Motor(2, "sts3215", norm_mode_head),
            },
            calibration=self.calibration,
        )

        # Base motors configuration (velocity control)
        norm_mode_base = MotorNormMode.DEGREES if config.base_use_degrees else MotorNormMode.RANGE_M100_100
        self.base_bus = FeetechMotorsBus(
            port=self.config.base_port,
            motors={
                "base_left_wheel": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )

        self.left_arm = SO100Follower(left_arm_config)
        self.right_arm = SO100Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Left arm motors (positional control)
        left_motors = {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors}

        # Right arm motors (positional control)
        right_motors = {f"right_{motor}.pos": float for motor in self.right_arm.bus.motors}

        # Head motors (positional control)
        head_motors = {f"head_{motor}.pos": float for motor in self.head_bus.motors}

        # Base motors (velocity control)
        base_motors = {
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }

        return left_motors | right_motors | head_motors | base_motors

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
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and self.head_bus.is_connected
            and self.base_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # Connect head bus
        if not self.head_bus.is_connected:
            self.head_bus.connect()
            logger.info(f"{self} head bus connected")

        # Connect base bus
        if not self.base_bus.is_connected:
            self.base_bus.connect()
            logger.info(f"{self} base bus connected")

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return (
            self.left_arm.is_calibrated
            and self.right_arm.is_calibrated
            and self.head_bus.is_calibrated
            and self.base_bus.is_calibrated
        )

    def calibrate(self) -> None:
        print("开始校准XLeRobot...")
        print("=" * 50)

        print("1. 校准左臂 (左手 + 头部设备)")
        print("   请确保左臂和头部设备已正确连接")
        input("按Enter键开始校准左臂...")
        try:
            self.left_arm.calibrate()
            print("✓ 左臂校准成功")
        except Exception as e:
            print(f"✗ 左臂校准失败: {e}")
            print("请检查左臂连接，或跳过左臂校准")
            retry = input("是否重试左臂校准? (y/n): ").strip().lower()
            if retry in ["y", "yes", "是"]:
                self.left_arm.calibrate()
                print("✓ 左臂校准成功")

        print("\n2. 校准右臂 (右手 + 底盘设备)")
        print("   请确保右臂和底盘设备已正确连接")
        input("按Enter键开始校准右臂...")
        try:
            self.right_arm.calibrate()
            print("✓ 右臂校准成功")
        except Exception as e:
            print(f"✗ 右臂校准失败: {e}")
            print("请检查右臂连接，或跳过右臂校准")
            retry = input("是否重试右臂校准? (y/n): ").strip().lower()
            if retry in ["y", "yes", "是"]:
                self.right_arm.calibrate()
                print("✓ 右臂校准成功")

        print("\n3. 校准底盘电机")
        print("   底盘电机与右臂共用端口")
        print("   注意: 底盘电机使用简单校准，设置默认范围")
        input("按Enter键开始校准底盘电机...")

        try:
            # 参考LeKiwi的做法，为底盘电机设置简单校准
            if self.base_bus.calibration:
                # 如果已有校准文件，询问是否使用
                user_input = input("按Enter键使用现有校准文件，或输入'c'重新校准: ")
                if user_input.strip().lower() != "c":
                    self.base_bus.write_calibration(self.base_bus.calibration)
                    print("使用现有底盘电机校准文件")
                    return

            print("为底盘电机设置默认校准...")

            # 为底盘电机设置默认校准（参考LeKiwi）
            base_calibration = {}
            for motor_name, motor in self.base_bus.motors.items():
                base_calibration[motor_name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=0,  # 底盘电机不需要homing offset
                    range_min=0,  # 最小范围
                    range_max=4095,  # 最大范围
                )

            self.base_bus.write_calibration(base_calibration)
            print("✓ 底盘电机校准完成（使用默认设置）")

        except Exception as e:
            print(f"✗ 底盘电机校准失败: {e}")
            print("请检查底盘电机连接，或跳过底盘校准")
            retry = input("是否重试底盘校准? (y/n): ").strip().lower()
            if retry in ["y", "yes", "是"]:
                # 重试底盘校准
                base_calibration = {}
                for motor_name, motor in self.base_bus.motors.items():
                    base_calibration[motor_name] = MotorCalibration(
                        id=motor.id,
                        drive_mode=0,
                        homing_offset=0,
                        range_min=0,
                        range_max=4095,
                    )
                self.base_bus.write_calibration(base_calibration)
                print("✓ 底盘电机校准完成（使用默认设置）")

        print("\nXLeRobot校准完成！")
        print("注意: 头部电机不需要校准，使用默认设置")
        print("=" * 50)

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

        # Configure head motors (position mode)
        with self.head_bus.torque_disabled():
            self.head_bus.configure_motors()
            for motor in self.head_bus.motors:
                self.head_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.head_bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.head_bus.write("I_Coefficient", motor, 0)
                self.head_bus.write("D_Coefficient", motor, 32)

        # Configure base motors (velocity mode)
        with self.base_bus.torque_disabled():
            self.base_bus.configure_motors()
            for motor in self.base_bus.motors:
                self.base_bus.write("Operating_Mode", motor, OperatingMode.VELOCITY.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.base_bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.base_bus.write("I_Coefficient", motor, 0)
                self.base_bus.write("D_Coefficient", motor, 32)

    def setup_motors(self) -> None:
        print("开始设置XLeRobot电机...")
        print("=" * 50)

        print("1. 设置左臂电机 (左手 + 头部设备)")
        print("   请确保左臂和头部设备已正确连接")
        input("按Enter键开始设置左臂电机...")
        self.left_arm.setup_motors()

        print("\n2. 设置右臂电机 (右手 + 底盘设备)")
        print("   请确保右臂和底盘设备已正确连接")
        input("按Enter键开始设置右臂电机...")
        self.right_arm.setup_motors()

        print("\n3. 设置底盘电机")
        print("   底盘电机与右臂共用端口")
        print("   注意: 底盘电机使用默认设置，无需复杂设置")
        for motor in reversed(self.base_bus.motors):
            input(f"请连接控制器板到底盘电机 '{motor}' 并按Enter键...")
            self.base_bus.setup_motor(motor)
            print(f"底盘电机 '{motor}' ID设置为 {self.base_bus.motors[motor].id}")

        print("\nXLeRobot电机设置完成！")
        print("注意: 头部电机使用默认设置，无需单独设置")
        print("=" * 50)

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        """Convert degrees per second to raw speed value."""
        # This is a simplified conversion, adjust based on your motor specifications
        return int(degps * 10)  # Scale factor may need adjustment

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        """Convert raw speed value to degrees per second."""
        return raw_speed / 10.0  # Scale factor may need adjustment

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert body frame velocity commands to wheel speed commands.

        Args:
            x: Forward velocity (m/s)
            y: Lateral velocity (m/s)
            theta: Angular velocity (rad/s)
            wheel_radius: Wheel radius in meters
            base_radius: Distance from center to wheel in meters
            max_raw: Maximum raw speed value

        Returns:
            Dictionary with wheel speed commands
        """
        # Convert to wheel speeds using differential drive kinematics
        left_wheel_speed = (x - theta * base_radius) / wheel_radius
        back_wheel_speed = (y + theta * base_radius) / wheel_radius
        right_wheel_speed = (x + theta * base_radius) / wheel_radius

        # Convert to raw values and clamp
        left_raw = np.clip(self._degps_to_raw(np.degrees(left_wheel_speed)), -max_raw, max_raw)
        back_raw = np.clip(self._degps_to_raw(np.degrees(back_wheel_speed)), -max_raw, max_raw)
        right_raw = np.clip(self._degps_to_raw(np.degrees(right_wheel_speed)), -max_raw, max_raw)

        return {
            "base_left_wheel": left_raw,
            "base_back_wheel": back_raw,
            "base_right_wheel": right_raw,
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        Convert wheel speed commands to body frame velocity.

        Args:
            left_wheel_speed: Left wheel speed (raw)
            back_wheel_speed: Back wheel speed (raw)
            right_wheel_speed: Right wheel speed (raw)
            wheel_radius: Wheel radius in meters
            base_radius: Distance from center to wheel in meters

        Returns:
            Dictionary with body frame velocities
        """
        # Convert raw speeds to rad/s
        left_radps = np.radians(self._raw_to_degps(left_wheel_speed))
        back_radps = np.radians(self._raw_to_degps(back_wheel_speed))
        right_radps = np.radians(self._raw_to_degps(right_wheel_speed))

        # Convert to body frame velocities
        x_vel = (left_radps + right_radps) * wheel_radius / 2
        y_vel = back_radps * wheel_radius
        theta_vel = (right_radps - left_radps) * wheel_radius / (2 * base_radius)

        return {
            "x.vel": x_vel,
            "y.vel": y_vel,
            "theta.vel": theta_vel,
        }

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix for left arm
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix for right arm
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # Add head joint positions (without calibration)
        start = time.perf_counter()
        head_obs = self.head_bus.sync_read("Present_Position", normalize=False)
        obs_dict.update({f"head_{motor}.pos": val for motor, val in head_obs.items()})
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read head state: {dt_ms:.1f}ms")

        # Add base wheel velocities (without calibration)
        start = time.perf_counter()
        base_obs = self.base_bus.sync_read("Present_Velocity", normalize=False)
        base_velocities = self._wheel_raw_to_body(
            base_obs["base_left_wheel"],
            base_obs["base_back_wheel"],
            base_obs["base_right_wheel"],
        )
        obs_dict.update(base_velocities)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read base state: {dt_ms:.1f}ms")

        # Add camera observations
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Separate actions for different components
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        head_action = {
            key.removeprefix("head_"): value for key, value in action.items() if key.startswith("head_")
        }

        # Handle base velocity commands
        base_velocities = {}
        if "x.vel" in action and "y.vel" in action and "theta.vel" in action:
            base_velocities = self._body_to_wheel_raw(
                action["x.vel"],
                action["y.vel"],
                action["theta.vel"],
            )

        # Send actions to respective components
        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Send head actions (without calibration)
        if head_action:
            self.head_bus.sync_write("Goal_Position", head_action, normalize=False)
            send_action_head = {f"head_{key}.pos": value for key, value in head_action.items()}
        else:
            send_action_head = {}

        # Send base actions (without calibration)
        if base_velocities:
            self.base_bus.sync_write("Goal_Velocity", base_velocities, normalize=False)
            send_action_base = {
                "x.vel": action["x.vel"],
                "y.vel": action["y.vel"],
                "theta.vel": action["theta.vel"],
            }
        else:
            send_action_base = {}

        # Combine all actions with prefixes
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {
            **prefixed_send_action_left,
            **prefixed_send_action_right,
            **send_action_head,
            **send_action_base,
        }

    def stop_base(self):
        """Stop all base motors."""
        stop_commands = {
            "base_left_wheel": 0,
            "base_back_wheel": 0,
            "base_right_wheel": 0,
        }
        self.base_bus.sync_write("Goal_Velocity", stop_commands, normalize=False)

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        self.head_bus.disconnect()
        self.base_bus.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
