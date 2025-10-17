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

from reachy2_sdk import ReachySDK

from ..teleoperator import Teleoperator
from .config_reachy2_teleoperator import Reachy2TeleoperatorConfig

logger = logging.getLogger(__name__)

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_NECK_JOINTS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
}

REACHY2_ANTENNAS_JOINTS = {
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
}

REACHY2_R_ARM_JOINTS = {
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
}

REACHY2_L_ARM_JOINTS = {
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
}

REACHY2_VEL = {
    "mobile_base.vx": "vx",
    "mobile_base.vy": "vy",
    "mobile_base.vtheta": "vtheta",
}


class Reachy2Teleoperator(Teleoperator):
    """
    [Reachy 2](https://www.pollen-robotics.com/reachy/), by Pollen Robotics.
    """

    config_class = Reachy2TeleoperatorConfig
    name = "reachy2_specific"

    def __init__(self, config: Reachy2TeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.reachy: None | ReachySDK = None

        self.joints_dict: dict[str, str] = self._generate_joints_dict()

    def _generate_joints_dict(self) -> dict[str, str]:
        joints = {}
        if self.config.with_neck:
            joints.update(REACHY2_NECK_JOINTS)
        if self.config.with_l_arm:
            joints.update(REACHY2_L_ARM_JOINTS)
        if self.config.with_r_arm:
            joints.update(REACHY2_R_ARM_JOINTS)
        if self.config.with_antennas:
            joints.update(REACHY2_ANTENNAS_JOINTS)
        return joints

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.with_mobile_base:
            return {
                **dict.fromkeys(
                    self.joints_dict.keys(),
                    float,
                ),
                **dict.fromkeys(
                    REACHY2_VEL.keys(),
                    float,
                ),
            }
        else:
            return dict.fromkeys(self.joints_dict.keys(), float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.reachy.is_connected() if self.reachy is not None else False

    def connect(self, calibrate: bool = True) -> None:
        self.reachy = ReachySDK(self.config.ip_address)
        if not self.is_connected:
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

        if self.reachy and self.is_connected:
            if self.config.use_present_position:
                joint_action = {
                    k: self.reachy.joints[v].present_position for k, v in self.joints_dict.items()
                }
            else:
                joint_action = {k: self.reachy.joints[v].goal_position for k, v in self.joints_dict.items()}

            if not self.config.with_mobile_base:
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read action: {dt_ms:.1f}ms")
                return joint_action

            if self.config.use_present_position:
                vel_action = {k: self.reachy.mobile_base.odometry[v] for k, v in REACHY2_VEL.items()}
            else:
                vel_action = {k: self.reachy.mobile_base.last_cmd_vel[v] for k, v in REACHY2_VEL.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return {**joint_action, **vel_action}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if self.reachy and self.is_connected:
            self.reachy.disconnect()
