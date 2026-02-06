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

import woangripper_api_py as woangripper

from lerobot.utils.errors import DeviceNotConnectedError

from ...robots.onerobotics_follower.woan_arm import WoanAdapter

logger = logging.getLogger("woan_arm")


class WoanTeleopLeaderAdapter(WoanAdapter):
    """
    Specialized Adapter for Teleoperation Leader scenarios.

    This adapter can have specialized methods if needed.
    Starting gravity compensation when connected, etc.
    """

    def connect(self) -> None:
        """
        Connect to the Woan Arm robot and start gravity compensation.
        """
        super().connect()
        # time.sleep(0.5)  # Wait a bit to ensure connection is stable
        if self.is_connected:
            self._arm.ArmGravityCompensation()
            logger.info(f"{self} gravity compensation started.")
            if self.config.enable_gripper_joystick:
                self._gripper_joystick = woangripper.GripperControl()
                if not self._gripper_joystick.initialize_joystick(
                    self.config.device_path, self.config.slcan_type
                ):
                    logger.warning("Failed to connect to gripper joystick control.")

    def send_action(self, action):
        """
        Only handle reset action specifically.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if action.get("reset", False):
            # Leader stops gravity compensation and then resets
            logger.debug("Leader adapter executing reset action.")
            self._arm.StopGravityCompensation()
            self._execute_reset_position()
            return action
        # For other actions, skip and just return
        logger.debug("Leader adapter ignoring non-reset action.")
        return action

    def get_observation(self):
        """
        Get observation from the robot, including joint states and gripper joystick status if enabled.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = super().get_observation()
        if self.config.enable_gripper_joystick:
            obs_dict["gripper.position"] = self._gripper_joystick.get_joystick_pos()

        return obs_dict
