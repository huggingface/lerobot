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

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode


from ..teleoperator import Teleoperator
from .config_panda_leader import PandaTeleoperatorConfig

from lerobot.common.motors.franka_api.API import *

logger = logging.getLogger(__name__)


class PandaTeleoperator(Teleoperator):
    """
    [WidowX](https://www.trossenrobotics.com/widowx-250) developed by Trossen Robotics
    """

    config_class = PandaTeleoperatorConfig
    name = "panda_leader"

    def __init__(self, config: PandaTeleoperatorConfig):
        #raise NotImplementedError
        super().__init__(config)
        self.config = config
        self.robot_type = self.config.type
        self.api = API(server_address=config.ip)

        # Joint names in order
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"panda_joint{idx}.pos": float for idx in range(1,8)}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return True # TODO: check if API is correctly connected

    def connect(self, calibrate: bool = True):
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        raise NotImplementedError  # TODO(aliberts): adapt code below (copied from koch)
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        joint_angles = self.api.get_JointAngles()
        #action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        action = {f"{joint}.pos": float(joint_angles[i]) for i, joint in enumerate(self.joint_names)}
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        #self.bus.disconnect()
        logger.info(f"{self} disconnected.")
