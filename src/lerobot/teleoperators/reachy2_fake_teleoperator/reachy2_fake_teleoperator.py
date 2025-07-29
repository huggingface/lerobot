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

# from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
# from lerobot.motors import Motor, MotorCalibration, MotorNormMode
# from lerobot.motors.feetech import (
#     FeetechMotorsBus,
#     OperatingMode,
# )
from reachy2_sdk import ReachySDK


from ..teleoperator import Teleoperator
from .config_reachy2_fake_teleoperator import Reachy2FakeTeleoperatorConfig

logger = logging.getLogger(__name__)

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_MOTORS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
    # "mobile_base.vx": "mobile_base.vx",
    # "mobile_base.vy": "mobile_base.vy",
    # "mobile_base.vtheta": "mobile_base.vtheta",
}


class Reachy2FakeTeleoperator(Teleoperator):
    """
    [Reachy 2](https://www.pollen-robotics.com/reachy/), by Pollen Robotics.
    """

    config_class = Reachy2FakeTeleoperatorConfig
    name = "reachy2_specific"

    def __init__(self, config: Reachy2FakeTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.reachy: None | ReachySDK = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(
            REACHY2_MOTORS.keys(),
            float,
        )

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.reachy.is_connected() if self.reachy is not None else False

    def connect(self, calibrate: bool = True) -> None:
        self.reachy = ReachySDK(self.config.ip_address)
        if not self.is_connected:
            print("Error connecting to Reachy 2.")
            raise ConnectionError()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = {k: self.reachy.joints[v].goal_position for k, v in REACHY2_MOTORS.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        self.reachy.disconnect()
