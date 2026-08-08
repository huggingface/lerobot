#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


class SOFollower(Robot):
    """The SO-family follower arm: a 5-DOF arm plus gripper on a Feetech bus.

    `SO100Follower` and `SO101Follower` are aliases of this class. The two arms differ in calibration and
    gearing, not control code, so both are driven through the same implementation with a different
    `config_class` and `name`.

    Actions and observations are keyed `"<motor>.pos"`; cameras named in the config appear in observations
    under their own keys. See [`~robots.Robot`] for the contract every method here implements.

    Args:
        config (`SOFollowerRobotConfig`):
            The robot's configuration. Its `port` and `cameras` determine what is connected.

    Example:
        ```python
        >>> from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        >>> robot = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM0"))  # doctest: +SKIP
        >>> with robot:  # doctest: +SKIP
        ...     observation = robot.get_observation()
        ...     robot.send_action({"shoulder_pan.pos": 0.0})
        ```
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        # choose normalization mode depending on config if available
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            if getattr(self.cameras[cam], "use_rgb", True):
                features[cam] = (self.cameras[cam].height, self.cameras[cam].width, 3)
            if getattr(self.cameras[cam], "use_depth", False):
                features[f"{cam}_depth"] = (self.cameras[cam].height, self.cameras[cam].width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """The arm's joint positions plus one entry per configured camera.

        Returns:
            `dict[str, type | tuple]`: `"<motor>.pos"` keys mapped to `float`, and one key per camera
            mapped to its `(height, width, channels)` shape.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """The arm's goal joint positions.

        Returns:
            `dict[str, type]`: `"<motor>.pos"` keys mapped to `float`.
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Whether the motor bus and every configured camera are connected.

        Returns:
            `bool`: `True` only when all of them are.
        """
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect the motor bus and cameras, calibrating and configuring the arm.

        > [!WARNING]
        > The arm is assumed to be at rest when this is called, because torque is disabled to run
        > calibration. Do not call it with the arm holding a load.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to run calibration when the motors disagree with the calibration file, or no file
                exists yet. Calibration is interactive and prompts on stdin.

        Raises:
            DeviceAlreadyConnectedError: If the robot is already connected.
        """
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

    @property
    def is_calibrated(self) -> bool:
        """Whether the motors' stored calibration matches the calibration file.

        Returns:
            `bool`: `True` when the arm needs no recalibration.
        """
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Calibrate the arm, writing the result to the motors and the calibration file.

        This is interactive: it prompts on stdin to reuse an existing calibration file, and otherwise asks
        you to move the arm to its middle position and then through each joint's full range.
        """
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

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
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        """Write the position-mode operating mode and the configured PID gains to every motor.

        The gripper additionally gets reduced torque, current and overload limits so that gripping a rigid
        object does not burn out its motor.
        """
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, self.config.position_p_coefficient)
                self.bus.write("I_Coefficient", motor, self.config.position_i_coefficient)
                self.bus.write("D_Coefficient", motor, self.config.position_d_coefficient)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        """Assign each motor its bus ID, one at a time.

        Run this once when building an arm. It is interactive: it prompts you to connect the controller
        board to a single motor at a time, working from the gripper back to the base.
        """
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read the arm's joint positions and one frame from each camera.

        Returns:
            `dict[str, Any]`: Keys matching [`~robots.Robot.observation_features`].

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """Disconnect the motor bus and every camera.

        Torque is released first unless `disable_torque_on_disconnect` is `False`.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower = SOFollower
SO101Follower = SOFollower
