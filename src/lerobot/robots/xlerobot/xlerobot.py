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
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .xlerobot_config import XLerobotConfig
from .xlerobot_base_keyboard import BaseKeyboardController
from .xlerobot_form_factor import ROBOT_PARTS, BUS_MAPPINGS, DEFAULT_IDS_BY_LAYOUT

logger = logging.getLogger(__name__)


class XLerobot(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = XLerobotConfig
    name = "xlerobot"

    def __init__(self, config: XLerobotConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        self.base_keyboard = BaseKeyboardController(self.teleop_keys)

        # Get bus mapping for the selected layout
        bus_mapping = BUS_MAPPINGS[config.motor_layout]

        # Build buses dynamically from robot parts + bus mapping
        self.buses: dict[str, FeetechMotorsBus] = {}
        self.motor_to_bus: dict[str, str] = {}  # motor_name -> bus_name
        self.role_to_motors: dict[str, list[str]] = {}  # role -> list of motor names

        motor_ids = dict(self.config.motor_ids)
        layout_defaults = DEFAULT_IDS_BY_LAYOUT.get(self.config.motor_layout, {})
        calibration_ids = {name: calib.id for name, calib in self.calibration.items()}

        for bus_name, bus_config in bus_mapping.items():
            port = config.ports.get(bus_config.port_key)
            if not port:
                continue

            # Find motors that match this bus's patterns
            bus_motors = {}
            calibration_subset = {}

            for motor_name, motor_def in ROBOT_PARTS.items():
                # Check if this motor matches any pattern for this bus
                matches = False
                for pattern in bus_config.motor_patterns:
                    if pattern.endswith("*"):
                        prefix = pattern[:-1]
                        if motor_name.startswith(prefix):
                            matches = True
                            break
                    elif motor_name == pattern:
                        matches = True
                        break

                if not matches:
                    continue

                # Create Motor object
                norm_mode = motor_def.norm_mode if config.use_degrees else MotorNormMode.RANGE_M100_100
                configured_id = motor_ids.get(motor_name)
                if configured_id is not None:
                    motor_id = configured_id
                else:
                    motor_id = calibration_ids.get(
                        motor_name, layout_defaults.get(motor_name, motor_def.id)
                    )
                bus_motors[motor_name] = Motor(motor_id, motor_def.model, norm_mode)

                # Track mappings
                self.motor_to_bus[motor_name] = bus_name
                if motor_def.role not in self.role_to_motors:
                    self.role_to_motors[motor_def.role] = []
                self.role_to_motors[motor_def.role].append(motor_name)

                # Extract calibration
                if self.calibration and motor_name in self.calibration:
                    calibration_subset[motor_name] = self.calibration[motor_name]

            # Create bus
            self.buses[bus_name] = FeetechMotorsBus(
                port=port,
                motors=bus_motors,
                calibration=calibration_subset,
            )

        self.cameras = make_cameras_from_configs(self.config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        keys = []

        # Add position keys for arms and head
        for role in ["left_arm", "right_arm", "head"]:
            for motor_name in self.role_to_motors.get(role, []):
                keys.append(f"{motor_name}.pos")

        # Add velocity keys for base
        if self.role_to_motors.get("base"):
            keys.extend(["x.vel", "y.vel", "theta.vel"])

        return dict.fromkeys(tuple(keys), float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return (all(bus.is_connected for bus in self.buses.values()) and all(cam.is_connected for cam in self.cameras.values()))

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        handshake = self.config.verify_motors_on_connect
        for bus in self.buses.values():
            bus.connect(handshake=handshake)

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return all(bus.is_calibrated for bus in self.buses.values())

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")

        all_calibrations = {}

        # Calibrate each role separately
        for role in ["left_arm", "right_arm", "head", "base"]:
            role_motors = self.role_to_motors.get(role, [])
            if not role_motors:
                continue

            # Get the bus for the first motor of this role
            first_motor = role_motors[0]
            bus_name = self.motor_to_bus[first_motor]
            bus = self.buses[bus_name]

            if role == "base":
                # Base wheels: simple calibration (full rotation)
                calibration_role = {}
                for motor_name in role_motors:
                    motor = bus.motors[motor_name]
                    calibration_role[motor_name] = MotorCalibration(
                        id=motor.id,
                        drive_mode=0,
                        homing_offset=0,
                        range_min=0,
                        range_max=4095,
                    )
                bus.write_calibration(calibration_role)
                all_calibrations.update(calibration_role)
            else:
                # Arms and head: full calibration process
                bus.disable_torque()
                for motor_name in role_motors:
                    bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)

                input(f"Move {role} motors to the middle of their range of motion and press ENTER....")
                homing_offsets = bus.set_half_turn_homings(role_motors)

                print(f"Move all {role} joints sequentially through their "
                      "entire ranges of motion.\nRecording positions. Press ENTER to stop...")
                range_mins, range_maxes = bus.record_ranges_of_motion(role_motors)

                calibration_role = {}
                for motor_name in role_motors:
                    motor = bus.motors[motor_name]
                    calibration_role[motor_name] = MotorCalibration(
                        id=motor.id,
                        drive_mode=0,
                        homing_offset=homing_offsets[motor_name],
                        range_min=range_mins[motor_name],
                        range_max=range_maxes[motor_name],
                    )
                bus.write_calibration(calibration_role)
                all_calibrations.update(calibration_role)

        # Combine all calibrations and save
        self.calibration = all_calibrations
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)


    def configure(self):
        # Set-up all actuators
        # We assume that at connection time, robot is in a rest position,
        # and torque can be safely disabled to run configuration
        for bus in self.buses.values():
            bus.disable_torque()
            bus.configure_motors()

        # Configure each motor based on its role and definition
        for motor_name, motor_def in ROBOT_PARTS.items():
            if motor_name not in self.motor_to_bus:
                continue

            bus_name = self.motor_to_bus[motor_name]
            bus = self.buses[bus_name]

            # Set operating mode based on motor definition
            if motor_def.operating_mode == "position":
                bus.write("Operating_Mode", motor_name, OperatingMode.POSITION.value)
                # Set PID for position mode
                bus.write("P_Coefficient", motor_name, motor_def.pid_p)
                bus.write("I_Coefficient", motor_name, motor_def.pid_i)
                bus.write("D_Coefficient", motor_name, motor_def.pid_d)
            elif motor_def.operating_mode == "velocity":
                bus.write("Operating_Mode", motor_name, OperatingMode.VELOCITY.value)

        # Enable torque on all buses
        for bus in self.buses.values():
            bus.enable_torque()


    def setup_motors(self) -> None:
        # Setup motors grouped by bus
        for bus_name, bus in self.buses.items():
            print(f"\n=== Setting up {bus_name} ===")
            bus_motors = list(bus.motors.keys())
            for motor in reversed(bus_motors):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                bus.setup_motor(motor)
                print(f"'{motor}' motor id set to {bus.motors[motor].id}")


    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

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
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x, y, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
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
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        return self.base_keyboard.compute_action(pressed_keys)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read actuators position for arms and head, velocity for base
        start = time.perf_counter()

        # Group motors by bus for efficient reading
        bus_reads = {}
        for role in ["left_arm", "right_arm", "head", "base"]:
            role_motors = self.role_to_motors.get(role, [])
            if not role_motors:
                continue

            # Group motors by bus
            for motor_name in role_motors:
                bus_name = self.motor_to_bus[motor_name]
                if bus_name not in bus_reads:
                    bus_reads[bus_name] = {"position": [], "velocity": []}

                if role == "base":
                    bus_reads[bus_name]["velocity"].append(motor_name)
                else:
                    bus_reads[bus_name]["position"].append(motor_name)

        # Read from each bus
        all_positions = {}
        all_velocities = {}
        for bus_name, motor_lists in bus_reads.items():
            bus = self.buses[bus_name]
            if motor_lists["position"]:
                positions = bus.sync_read("Present_Position", motor_lists["position"])
                all_positions.update(positions)
            if motor_lists["velocity"]:
                velocities = bus.sync_read("Present_Velocity", motor_lists["velocity"])
                all_velocities.update(velocities)

        # Convert base wheel velocities to body frame
        base_vel = {}
        base_motors = self.role_to_motors.get("base", [])
        if base_motors and all_velocities:
            base_vel = self._wheel_raw_to_body(
                all_velocities.get("base_left_wheel", 0),
                all_velocities.get("base_back_wheel", 0),
                all_velocities.get("base_right_wheel", 0),
            )

        # Build observation dict
        obs_dict = {}
        for motor_name, pos in all_positions.items():
            obs_dict[f"{motor_name}.pos"] = pos
        obs_dict.update(base_vel)

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
        """Command xlerobot to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = self.base_keyboard.augment_action(action)

        # Parse action by role
        role_actions = {"left_arm": {}, "right_arm": {}, "head": {}, "base": {}}
        base_goal_vel = {}

        for key, value in action.items():
            if key.endswith(".pos"):
                motor_name = key.replace(".pos", "")
                for role in ["left_arm", "right_arm", "head"]:
                    if motor_name in self.role_to_motors.get(role, []):
                        role_actions[role][key] = value
                        break
            elif key.endswith(".vel"):
                base_goal_vel[key] = value

        # Convert base velocities to wheel commands
        base_wheel_goal_vel = {}
        if base_goal_vel:
            base_wheel_goal_vel = self._body_to_wheel_raw(
                base_goal_vel.get("x.vel", 0.0),
                base_goal_vel.get("y.vel", 0.0),
                base_goal_vel.get("theta.vel", 0.0),
            )

        # Safety clamping if configured
        if self.config.max_relative_target is not None:
            # Read present positions for all position-controlled motors
            present_pos = {}
            for role in ["left_arm", "right_arm", "head"]:
                role_motors = self.role_to_motors.get(role, [])
                if not role_motors:
                    continue

                # Get bus for first motor of this role
                first_motor = role_motors[0]
                bus_name = self.motor_to_bus[first_motor]
                bus = self.buses[bus_name]

                # Read positions for all motors of this role
                positions = bus.sync_read("Present_Position", role_motors)
                present_pos.update(positions)

            # Apply safety clamping
            goal_present_pos = {}
            for role in ["left_arm", "right_arm", "head"]:
                for key, value in role_actions[role].items():
                    motor_name = key.replace(".pos", "")
                    if motor_name in present_pos:
                        goal_present_pos[key] = (value, present_pos[motor_name])

            if goal_present_pos:
                safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
                # Update role actions with safe positions
                for key, safe_value in safe_goal_pos.items():
                    motor_name = key.replace(".pos", "")
                    for role in ["left_arm", "right_arm", "head"]:
                        if motor_name in self.role_to_motors.get(role, []):
                            role_actions[role][key] = safe_value
                            break

        # Group actions by bus for efficient writing
        bus_writes = {}
        for role in ["left_arm", "right_arm", "head"]:
            role_motors = self.role_to_motors.get(role, [])
            if not role_motors:
                continue

            # Group by bus
            for key, value in role_actions[role].items():
                motor_name = key.replace(".pos", "")
                bus_name = self.motor_to_bus[motor_name]
                if bus_name not in bus_writes:
                    bus_writes[bus_name] = {}
                bus_writes[bus_name][motor_name] = value

        # Add base wheel commands
        if base_wheel_goal_vel:
            base_motors = self.role_to_motors.get("base", [])
            for motor_name in base_motors:
                bus_name = self.motor_to_bus[motor_name]
                if bus_name not in bus_writes:
                    bus_writes[bus_name] = {}
                bus_writes[bus_name][motor_name] = base_wheel_goal_vel.get(motor_name, 0)

        # Write to buses
        base_motors = self.role_to_motors.get("base", [])
        for bus_name, commands in bus_writes.items():
            bus = self.buses[bus_name]
            if any(motor in base_motors for motor in commands.keys()):
                # Base motors use velocity commands
                bus.sync_write("Goal_Velocity", commands)
            else:
                # Other motors use position commands
                bus.sync_write("Goal_Position", commands)

        # Return the action that was sent
        result = {}
        for role_actions_dict in role_actions.values():
            result.update(role_actions_dict)
        result.update(base_goal_vel)
        return result

    def stop_base(self):
        base_motors = self.role_to_motors.get("base", [])
        if base_motors:
            # Find the bus for base motors
            first_base_motor = base_motors[0]
            bus_name = self.motor_to_bus[first_base_motor]
            bus = self.buses[bus_name]
            bus.sync_write("Goal_Velocity", dict.fromkeys(base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        for bus in self.buses.values():
            bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
