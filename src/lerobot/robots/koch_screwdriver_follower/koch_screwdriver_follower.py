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
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_koch_screwdriver_follower import KochScrewdriverFollowerConfig

logger = logging.getLogger(__name__)


class KochScrewdriverFollower(Robot):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochScrewdriverFollowerConfig
    name = "koch_screwdriver_follower"

    def __init__(self, config: KochScrewdriverFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(3, "xl330-m288", norm_mode_body),
                "wrist_flex": Motor(4, "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(5, "xl330-m288", norm_mode_body),
                # Using MotorNormMode.RANGE_M100_100. Using lekiwi's wheel servos as a reference lerobot/common/robots/lekiwi/lekiwi.py
                "screwdriver": Motor(6, "xl330-m288", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    # called by observation_features method
    @property
    def _motors_ft(self) -> dict[str, type]:
        # Set the screwdriver to .vel. Using lekiwi's wheel servos as a reference lerobot/common/robots/lekiwi/lekiwi.py
        return {
            f"{motor}.vel" if motor == "screwdriver" else f"{motor}.pos": float for motor in self.bus.motors
        }

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
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            # Set the screwdriver to velocity mode
            if motor == "screwdriver":
                print(f"Operating_Mode: {motor} {OperatingMode.VELOCITY.value}")
                self.bus.write("Operating_Mode", motor, OperatingMode.VELOCITY.value)
            else:
                print(f"Operating_Mode: {motor} {OperatingMode.EXTENDED_POSITION.value}")
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll", "screwdriver"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            # Use 'extended position mode' for all motors except screwdriver, because in joint mode the servos
            # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling
            # the arm, you could end up with a servo with a position 0 or 4095 at a crucial point
            for motor in self.bus.motors:
                if motor != "screwdriver":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # Screwdriver needs to be in velocity mode. Using lekiwi's base_motors wheel servos config as a reference lerobot/common/robots/lekiwi/lekiwi.py
            self.bus.write("Operating_Mode", "screwdriver", OperatingMode.VELOCITY.value)

            # Apply current & velocity limits for the screwdriver motor to avoid
            # over-current shutdowns.  Current_Limit expects raw units.
            self._screw_limit = int(self.config.screwdriver_current_limit)
            self.bus.write("Current_Limit", "screwdriver", self._screw_limit)

            # Optional: limit maximum velocity (raw units) for safety.
            self.bus.write("Velocity_Limit", "screwdriver", 400)

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)

        # State variable used by the software clutch
        self._clutch_engaged: bool = False
        self._clutch_release_time: float = 0.0

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read positions only for joints that are in position mode (exclude screwdriver)
        pos_motors = [m for m in self.bus.motors if m != "screwdriver"]

        # Set num_retry=3 to help prevent:
        # ConnectionError: Failed to sync read 'Present_Velocity' on ids=[n] after 1 tries. [TxRxResult] There is no status packet!
        # FATAL: exception not rethrown
        pos_dict = self.bus.sync_read("Present_Position", pos_motors, num_retry=3)
        obs_dict = {}
        for motor, val in pos_dict.items():
            obs_dict[f"{motor}.pos"] = val

        # Set num_retry=3 to help prevent:
        # ConnectionError: Failed to sync read 'Present_Velocity' on ids=[n] after 1 tries. [TxRxResult] There is no status packet!
        # FATAL: exception not rethrown
        screwdriver_vel_raw = self.bus.sync_read("Present_Velocity", ["screwdriver"], num_retry=3)[
            "screwdriver"
        ]
        obs_dict["screwdriver.vel"] = screwdriver_vel_raw

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

        # Split positional and velocity commands
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        goal_vel = {key.removesuffix(".vel"): int(val) for key, val in action.items() if key.endswith(".vel")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None and goal_pos:
            present_pos = self.bus.sync_read(
                "Present_Position", [m for m in self.bus.motors if m != "screwdriver"]
            )
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send commands to the arm
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)
        if goal_vel:
            # Apply software clutch for the screwdriver motor
            if "screwdriver" in goal_vel:
                goal_vel["screwdriver"] = self._apply_clutch(goal_vel["screwdriver"])

            self.bus.sync_write("Goal_Velocity", goal_vel)

        # Merge and return the actually sent commands
        sent_action = {f"{motor}.pos": val for motor, val in goal_pos.items()}
        sent_action.update({f"{motor}.vel": val for motor, val in goal_vel.items()})
        return sent_action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Software-clutch helpers
    # ------------------------------------------------------------------

    def _read_screwdriver_current(self) -> int:
        """Return present current (raw units) for the screwdriver motor."""

        return self.bus.sync_read("Present_Current", ["screwdriver"], num_retry=1)["screwdriver"]

    def _apply_clutch(self, vel_cmd: int) -> int:
        """Cut velocity to 0 if current close to limit and update clutch flag."""

        present = abs(self._read_screwdriver_current())
        threshold_on = self._screw_limit * self.config.clutch_ratio  # engage clutch
        threshold_off = self._screw_limit * (self.config.clutch_ratio * 0.6)  # release clutch (hysteresis)

        now = time.perf_counter()

        # If still in cooldown window → force velocity to 0
        if self._clutch_engaged and now < self._clutch_release_time:
            return 0

        if self._clutch_engaged and now >= self._clutch_release_time:
            # Cool-down ended, try re-enable torque and resume normal control
            try:
                self.bus.enable_torque("screwdriver")
            except Exception as e:
                logger.debug(f"Could not re-enable torque: {e}")
            self._clutch_engaged = False

        if not self._clutch_engaged and present >= threshold_on:
            # Engage clutch: cut velocity and (best-effort) disable torque to drop current fast
            try:
                self.bus.disable_torque("screwdriver")
            except Exception as e:
                logger.debug(f"Torque disable failed: {e}")
            self._clutch_engaged = True
            # Start cool-down timer
            self._clutch_release_time = now + self.config.clutch_cooldown_s
            print(f"Clutch engaged: {present} >= {threshold_on}")
            return 0

        return vel_cmd
