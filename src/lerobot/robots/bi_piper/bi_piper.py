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

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_bi_piper import BiPiperConfig

logger = logging.getLogger(__name__)


class BiPiper(Robot):
    """
    Bimanual Piper Robot using piper_sdk for CAN communication
    """

    config_class = BiPiperConfig
    name = "bi_piper"

    def __init__(self, config: BiPiperConfig):
        super().__init__(config)
        self.config = config

        # Import piper_sdk here to avoid import errors if not installed
        try:
            from piper_sdk import C_PiperInterface_V2

            self.C_PiperInterface_V2 = C_PiperInterface_V2
        except ImportError as e:
            raise ImportError(
                "piper_sdk is not installed. Please install it with: pip install piper_sdk"
            ) from e

        # Initialize left and right arm interfaces
        self.left_arm = None
        self.right_arm = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Define the motor features for both arms"""
        motor_features = {}
        # Left arm: joint_1 to joint_6 and gripper
        for i in range(1, 7):
            motor_features[f"left_joint_{i}.pos"] = float
        motor_features["left_gripper.pos"] = float

        # Right arm: joint_1 to joint_6 and gripper
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
        """
        Connect to both Piper arms via CAN ports
        """
        try:
            # Connect left arm
            logger.info(f"Connecting to left arm on CAN port: {self.config.left_arm_can_port}")
            self.left_arm = self.C_PiperInterface_V2(self.config.left_arm_can_port)
            self.left_arm.ConnectPort(True)

            # Connect right arm
            logger.info(f"Connecting to right arm on CAN port: {self.config.right_arm_can_port}")
            self.right_arm = self.C_PiperInterface_V2(self.config.right_arm_can_port)
            self.right_arm.ConnectPort(True)

            # Connect cameras
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
        logger.info("BiPiper robot calibration - no action needed, assumed calibrated")
        pass

    def configure(self) -> None:
        """Configure the BiPiper robot - no specific configuration needed"""
        logger.info("BiPiper robot configuration - no action needed")
        pass

    def disconnect(self) -> None:
        """
        Disconnect from both Piper arms and cameras
        """
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
        """
        Capture current joint positions and camera images
        """
        if not self.is_connected:
            raise RuntimeError("BiPiper robot is not connected")

        observation = {}

        try:
            # Get left arm joint positions
            left_joint_msgs = self.left_arm.GetArmJointMsgs()
            left_gripper_msgs = self.left_arm.GetArmGripperMsgs()

            # Parse joint positions for left arm
            # Based on the format: ArmMsgFeedBackJointStates with Joint 1-6 values
            self._parse_joint_messages(left_joint_msgs, "left", observation)

            # Parse gripper position for left arm
            # Based on the format: ArmMsgFeedBackGripper with grippers_angle
            self._parse_gripper_messages(left_gripper_msgs, "left", observation)

            # Get right arm joint positions
            right_joint_msgs = self.right_arm.GetArmJointMsgs()
            right_gripper_msgs = self.right_arm.GetArmGripperMsgs()

            # Parse joint positions for right arm
            self._parse_joint_messages(right_joint_msgs, "right", observation)

            # Parse gripper position for right arm
            self._parse_gripper_messages(right_gripper_msgs, "right", observation)

            # Capture camera images
            for cam_name, cam in self.cameras.items():
                observation[cam_name] = cam.capture()

        except Exception as e:
            logger.error(f"Error capturing observation: {e}")
            raise

        return observation

    def _parse_joint_messages(self, joint_msgs, arm_prefix: str, observation: dict) -> None:
        """
        Parse joint messages from piper SDK format.
        Expected format includes Joint 1-6 values in the message.
        """
        try:
            # Convert message to string to parse it
            msg_str = str(joint_msgs)

            # Extract joint values using string parsing
            # Looking for patterns like "Joint 1:value", "Joint 2:value", etc.
            for i in range(1, 7):
                joint_key = f"{arm_prefix}_joint_{i}.pos"

                # Look for "Joint {i}:" pattern in the message
                pattern = f"Joint {i}:"
                start_idx = msg_str.find(pattern)

                if start_idx != -1:
                    # Find the value after "Joint {i}:"
                    value_start = start_idx + len(pattern)
                    # Find the end of the value (next newline or end of string)
                    value_end = msg_str.find("\n", value_start)
                    if value_end == -1:
                        value_end = len(msg_str)

                    value_str = msg_str[value_start:value_end].strip()
                    try:
                        observation[joint_key] = float(value_str)
                    except ValueError:
                        logger.warning(f"Could not parse joint {i} value: {value_str}")
                        observation[joint_key] = 0.0
                else:
                    logger.warning(f"Joint {i} not found in message")
                    observation[joint_key] = 0.0

        except Exception as e:
            logger.error(f"Error parsing joint messages: {e}")
            # Fallback: set all joints to 0
            for i in range(1, 7):
                observation[f"{arm_prefix}_joint_{i}.pos"] = 0.0

    def _parse_gripper_messages(self, gripper_msgs, arm_prefix: str, observation: dict) -> None:
        """
        Parse gripper messages from piper SDK format.
        Expected format: gripper_state object with grippers_angle attribute
        grippers_angle is in 0.001° units, needs conversion to degrees.
        """
        try:
            # Direct object access - convert from 0.001° to degrees
            angle_raw = gripper_msgs.grippers_angle
            angle_degrees = float(angle_raw) / 1000.0
            observation[f"{arm_prefix}_gripper.pos"] = angle_degrees
            logger.debug(f"{arm_prefix} gripper angle: {angle_degrees}° (raw: {angle_raw})")

        except Exception as e:
            logger.error(f"Error parsing {arm_prefix} gripper messages: {e}")
            logger.error(f"Gripper message type: {type(gripper_msgs)}")
            logger.error(f"Gripper message content: {gripper_msgs}")
            observation[f"{arm_prefix}_gripper.pos"] = 0.0

    def send_action(self, action: dict) -> dict:
        """
        Send action to both Piper arms
        Note: This is a placeholder since you mentioned the robot movement is handled separately
        """
        if not self.is_connected:
            raise RuntimeError("BiPiper robot is not connected")

        # Since you mentioned that robot movement is handled separately,
        # we don't need to implement actual movement commands here
        # This method is required by the Robot interface but can be a no-op
        logger.debug("send_action called - movement handled separately")

        # Return the action as-is since no modifications are made
        return action
