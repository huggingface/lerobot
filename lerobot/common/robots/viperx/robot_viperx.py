"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time

import numpy as np

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import TorqueMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    run_arm_calibration,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .configuration_viperx import ViperXRobotConfig


class ViperXRobot(Robot):
    """
    [ViperX](https://www.trossenrobotics.com/viperx-300) developed by Trossen Robotics
    """

    config_class = ViperXRobotConfig
    name = "viperx"

    def __init__(
        self,
        config: ViperXRobotConfig,
    ):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.arm = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": config.waist,
                "shoulder": config.shoulder,
                "shoulder_shadow": config.shoulder_shadow,
                "elbow": config.elbow,
                "elbow_shadow": config.elbow_shadow,
                "forearm_roll": config.forearm_roll,
                "wrist_angle": config.wrist_angle,
                "wrist_rotate": config.wrist_rotate,
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
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    def _set_shadow_motors(self):
        """
        Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
        As a result, if only one of them is required to move to a certain position,
        the other will follow. This is to avoid breaking the motors.
        """
        shoulder_idx = self.config.shoulder[0]
        self.arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

        elbow_idx = self.config.elbow[0]
        self.arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

    def connect(self):
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

        self._set_shadow_motors()

        # Set a velocity limit of 131 as advised by Trossen Robotics
        self.arm.write("Velocity_Limit", 131)

        # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
        # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
        # you could end up with a servo with a position 0 or 4095 at a crucial point See [
        # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
        all_motors_except_gripper = [name for name in self.arm.motor_names if name != "gripper"]
        if len(all_motors_except_gripper) > 0:
            # 4 corresponds to Extended Position on Aloha motors
            self.arm.write("Operating_Mode", 4, all_motors_except_gripper)

        # Use 'position control current based' for follower gripper to be limited by the limit of the current.
        # It can grasp an object without forcing too much even tho,
        # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
        self.arm.write("Operating_Mode", 5, "gripper")

        # Note: We can't enable torque on the leader gripper since "xc430-w150" doesn't have
        # a Current Controlled Position mode.

        logging.info("Activating torque.")
        self.arm.write("Torque_Enable", TorqueMode.ENABLED.value)

        # Check arm can be read
        self.arm.read("Present_Position")

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True

    def calibrate(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        if self.calibration_fpath.exists():
            with open(self.calibration_fpath) as f:
                calibration = json.load(f)
        else:
            # TODO(rcadene): display a warning in __init__ if calibration file not available
            logging.info(f"Missing calibration file '{self.calibration_fpath}'")
            calibration = run_arm_calibration(self.arm, self.robot_type, self.name, "follower")

            logging.info(f"Calibration is done! Saving calibration file '{self.calibration_fpath}'")
            self.calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calibration_fpath, "w") as f:
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
