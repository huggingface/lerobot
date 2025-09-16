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
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_bi_piper import BiPiperConfig

logger = logging.getLogger(__name__)


class BiPiper(Robot):
    """
    Bimanual [Piper Robot](https://global.agilex.ai/products/piper) using piper_sdk for CAN communication.
    This bimanual implementation works when two piper arms are controlled using two piper arms as masters. As piper arms pairs support direct teleoperation
    by connecting leader and follower arms to the same CAN port, LeRobot will only be used for data recording.
    """

    config_class = BiPiperConfig
    name = "bi_piper"

    def __init__(self, config: BiPiperConfig):
        super().__init__(config)
        self.config = config

        try:
            from piper_sdk import C_PiperInterface_V2

            self.C_PiperInterface_V2 = C_PiperInterface_V2
        except ImportError as e:
            raise ImportError(
                "piper_sdk is not installed. Please install it with: pip install piper_sdk"
            ) from e

        self.left_arm = None
        self.right_arm = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        motor_features = {}
        for i in range(1, 7):
            motor_features[f"left_joint_{i}.pos"] = float
        motor_features["left_gripper.pos"] = float

        for i in range(1, 7):
            motor_features[f"right_joint_{i}.pos"] = float
        motor_features["right_gripper.pos"] = float

        return motor_features

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
        left_connected = self.left_arm is not None
        right_connected = self.right_arm is not None
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return left_connected and right_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        try:
            logger.info(f"Connecting to left arm on CAN port: {self.config.left_arm_can_port}")
            self.left_arm = self.C_PiperInterface_V2(self.config.left_arm_can_port)
            self.left_arm.ConnectPort(True)

            logger.info(f"Connecting to right arm on CAN port: {self.config.right_arm_can_port}")
            self.right_arm = self.C_PiperInterface_V2(self.config.right_arm_can_port)
            self.right_arm.ConnectPort(True)

            for cam in self.cameras.values():
                cam.connect()

            logger.info("BiPiper robot connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect to BiPiper robot: {e}")
            raise

    @property
    def is_calibrated(self) -> bool:
        """BiPiper robots are assumed to be always calibrated"""
        return True

    def calibrate(self) -> None:
        """BiPiper robots don't require manual calibration"""
        pass

    def configure(self) -> None:
        """Configure the BiPiper robot - no specific configuration needed"""
        pass

    def disconnect(self) -> None:
        try:
            if self.left_arm is not None:
                try:
                    self.left_arm.DisconnectPort()
                    logger.info("Left arm disconnected from CAN port")
                except Exception as e:
                    logger.warning(f"Error disconnecting left arm: {e}")
                self.left_arm = None

            if self.right_arm is not None:
                try:
                    self.right_arm.DisconnectPort()
                    logger.info("Right arm disconnected from CAN port")
                except Exception as e:
                    logger.warning(f"Error disconnecting right arm: {e}")
                self.right_arm = None

            for cam in self.cameras.values():
                cam.disconnect()

            logger.info("BiPiper robot disconnected successfully")

        except Exception as e:
            logger.error(f"Error during BiPiper robot disconnect: {e}")

    def get_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("BiPiper robot is not connected")

        observation = {}

        try:
            left_joint_msgs = self.left_arm.GetArmJointMsgs()
            left_gripper_msgs = self.left_arm.GetArmGripperMsgs()

            self._parse_joint_messages(left_joint_msgs, "left", observation)
            self._parse_gripper_messages(left_gripper_msgs, "left", observation)

            right_joint_msgs = self.right_arm.GetArmJointMsgs()
            right_gripper_msgs = self.right_arm.GetArmGripperMsgs()

            self._parse_joint_messages(right_joint_msgs, "right", observation)

            self._parse_gripper_messages(right_gripper_msgs, "right", observation)

            for cam_name, cam in self.cameras.items():
                observation[cam_name] = cam.async_read()

        except Exception as e:
            logger.error(f"Error capturing observation: {e}")
            raise

        return observation

    def _parse_joint_messages(self, joint_msgs, arm_prefix: str, observation: dict) -> None:
        """
        Parse joint messages from piper SDK format.
        Expected format: joint_msgs.joint_state.joint_1, joint_msgs.joint_state.joint_2, etc.
        """
        try:
            # Extract joint values using direct attribute access
            for i in range(1, 7):
                joint_key = f"{arm_prefix}_joint_{i}.pos"
                try:
                    # Access joint value using joint_msgs.joint_state.joint_{i}
                    joint_attr = f"joint_{i}"
                    joint_value = getattr(joint_msgs.joint_state, joint_attr)
                    observation[joint_key] = float(joint_value)
                except AttributeError:
                    logger.warning(f"Joint {i} attribute not found in joint_state")
                    observation[joint_key] = 0.0
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse joint {i} value")
                    observation[joint_key] = 0.0

        except Exception as e:
            logger.error(f"Error parsing joint messages: {e}")
            # Fallback: set all joints to 0
            for i in range(1, 7):
                observation[f"{arm_prefix}_joint_{i}.pos"] = 0.0

    def _parse_gripper_messages(self, gripper_msgs, arm_prefix: str, observation: dict) -> None:
        """
        Parse gripper messages from piper SDK format.
        https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD#getarmgrippermsgs
        """
        try:
            angle_raw = gripper_msgs.gripper_state.grippers_angle
            angle_mm = float(angle_raw) / 1000.0
            observation[f"{arm_prefix}_gripper.pos"] = angle_mm
            logger.debug(f"{arm_prefix} gripper position: {angle_mm}mm (raw: {angle_raw})")

        except Exception as e:
            logger.error(f"Error parsing {arm_prefix} gripper messages: {e}")
            logger.error(f"Gripper message type: {type(gripper_msgs)}")
            logger.error(f"Gripper message content: {gripper_msgs}")
            observation[f"{arm_prefix}_gripper.pos"] = 0.0

    def send_action(self, action: dict) -> Any:
        pass
