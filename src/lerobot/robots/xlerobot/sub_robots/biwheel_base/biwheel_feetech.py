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
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .biwheel_base import BiwheelBase
from .config_biwheel_base import BiwheelFeetechConfig

logger = logging.getLogger(__name__)


class BiwheelFeetech(BiwheelBase):
    """Biwheel robot class driven by Feetech servos."""

    config_class = BiwheelFeetechConfig
    name = "biwheel_feetech"
    supports_shared_bus = True

    def __init__(self, config: BiwheelFeetechConfig):
        super().__init__(config)
        self.config = config

        left_id, right_id = self.config.base_motor_ids
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "base_left_wheel": Motor(left_id, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(right_id, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.base_motors = list(self.bus.motors.keys())

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True, handshake: bool | None = None) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        handshake_flag = self.config.handshake_on_connect if handshake is None else handshake
        self.bus.connect(handshake=handshake_flag)

        if calibrate and not self.is_calibrated:
            self.calibrate()

        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info("Calibrating %s base motors", self)

        homing_offsets = dict.fromkeys(self.base_motors, 0)
        range_mins = dict.fromkeys(self.base_motors, 0)
        range_maxes = dict.fromkeys(self.base_motors, 4095)

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("Calibration saved at %s", self.calibration_fpath)

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.bus.enable_torque()

    def setup_motors(self) -> None:
        for motor in self.base_motors:
            input(f"Connect the controller board to the '{motor}' motor only and press ENTER.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        return raw_speed / steps_per_deg

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_int = int(round(degps * steps_per_deg))
        return max(min(speed_int, 0x7FFF), -0x8000)

    def _body_to_wheel_raw(
        self,
        x: float,
        theta: float,
        wheel_radius: float = 0.05,
        wheel_base: float = 0.25,
        max_raw: int = 3000,
    ) -> dict[str, int]:
        """
        Convert desired body-frame velocities into wheel raw commands for differential drive.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          wheel_base  : Distance between left and right wheels (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_right_wheel": value}.

        Notes:
          - Differential drive kinematics: only x and theta are controllable
          - y velocity is ignored (differential drive cannot move sideways)
        """
        wheel_radius = wheel_radius or self.config.wheel_radius
        wheel_base = wheel_base or self.config.wheel_base
        max_raw = max_raw or self.config.max_wheel_raw

        theta_rad = np.deg2rad(theta)

        half_wheelbase = wheel_base / 2
        left_wheel_speed = (x - theta_rad * half_wheelbase) / wheel_radius
        right_wheel_speed = (x + theta_rad * half_wheelbase) / wheel_radius

        # Convert to deg/s
        wheel_speeds_degps = np.rad2deg(np.array([left_wheel_speed, right_wheel_speed]))
        # Apply scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = np.abs(wheel_speeds_degps) * steps_per_deg
        max_raw_computed = np.max(raw_floats)

        if max_raw_computed > max_raw:
            wheel_speeds_degps *= max_raw / max_raw_computed

        # Convert to raw integers
        left_wheel_raw = self._degps_to_raw(wheel_speeds_degps[0])
        right_wheel_raw = self._degps_to_raw(wheel_speeds_degps[1])

        # Apply motor direction inversion if configured
        if self.config.invert_left_motor:
            left_wheel_raw = -left_wheel_raw
        if self.config.invert_right_motor:
            right_wheel_raw = -right_wheel_raw

        return {
            "base_left_wheel": left_wheel_raw,
            "base_right_wheel": right_wheel_raw,
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed: int,
        right_wheel_speed: int,
        wheel_radius: float = 0.05,
        wheel_base: float = 0.25,
    ) -> dict[str, float]:
        """
        Convert wheel raw command feedback back into body-frame velocities for differential drive.

        Parameters:
            left_wheel_speed  : Raw command for left wheel.
            right_wheel_speed : Raw command for right wheel.
            wheel_radius      : Radius of each wheel (meters). Defaults to self.config.wheel_radius.
            wheel_base        : Distance between left and right wheels (meters). Defaults to self.config.wheelbase.

        Returns:
            Dictionary containing:
            - "x.vel": Linear velocity in m/s
            - "theta.vel": Angular velocity in deg/s
        """
        # Use default values from config if not provided
        wheel_radius = wheel_radius or self.config.wheel_radius
        wheel_base = wheel_base or self.config.wheel_base

        # Convert raw commands to angular speeds (deg/s)
        left_degps = self._raw_to_degps(left_wheel_speed)
        right_degps = self._raw_to_degps(right_wheel_speed)

        # Convert angular speeds from deg/s to rad/s
        left_radps = np.deg2rad(left_degps)
        right_radps = np.deg2rad(right_degps)

        # Calculate linear speed (m/s) for each wheel
        left_linear_speed = left_radps * wheel_radius
        right_linear_speed = right_radps * wheel_radius

        # Apply differential drive inverse kinematics
        # Linear velocity: v = (v_left + v_right) / 2
        x_vel = (left_linear_speed + right_linear_speed) / 2.0

        # Angular velocity: ω = (v_right - v_left) / L
        theta_rad = (right_linear_speed - left_linear_speed) / wheel_base
        theta_vel = np.rad2deg(theta_rad)

        return {
            "x.vel": x_vel,  # Linear velocity (m/s)
            "theta.vel": theta_vel,  # Angular velocity (deg/s)
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)
            base_vel = self._wheel_raw_to_body(
                base_wheel_vel["base_left_wheel"],
                base_wheel_vel["base_right_wheel"],
            )
        except ConnectionError as e:
            logger.warning(f"Failed to read observation: {e}.")
            raise

        return base_vel

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        base_goal_vel = {
            "x.vel": float(action.get("x.vel", 0.0)),
            "theta.vel": float(action.get("theta.vel", 0.0)),
        }

        wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"],
            base_goal_vel["theta.vel"],
        )

        # Debug logging
        logger.debug(
            f"Action: x.vel={base_goal_vel['x.vel']:.3f} m/s, theta.vel={base_goal_vel['theta.vel']:.1f} deg/s "
            f"-> Wheels: L={wheel_goal_vel['base_left_wheel']}, R={wheel_goal_vel['base_right_wheel']}"
        )

        self.bus.sync_write("Goal_Velocity", wheel_goal_vel)
        return base_goal_vel

    def stop_base(self) -> None:
        try:
            self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
            logger.info("Base motors stopped")
        except ConnectionError as e:
            logger.warning(f"Failed to stop base motors: {e}")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        logger.info("%s disconnected.", self)
