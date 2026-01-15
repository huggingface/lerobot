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

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex

from ..teleoperator import Teleoperator
from ..homunculus.config_homunculus import HomunculusArmConfig
from ..homunculus.homunculus_arm import HomunculusArm
from .config_unitree_g1_bimanual import UnitreeG1BimanualConfig

logger = logging.getLogger(__name__)

_HOMUNCULUS_TO_G1_ARM = {
    "shoulder_pitch": "ShoulderPitch",
    "shoulder_roll": "ShoulderRoll",
    "shoulder_yaw": "ShoulderYaw",
    "elbow_flex": "Elbow",
    "wrist_roll": "WristRoll",
    "wrist_pitch": "WristPitch",
    "wrist_yaw": "WristYaw",
}


class UnitreeG1Bimanual(Teleoperator):
    """
    Bimanual Homunculus arms teleoperator for Unitree G1 arms.
    """

    config_class = UnitreeG1BimanualConfig
    name = "unitree_g1_bimanual"

    def __init__(self, config: UnitreeG1BimanualConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = HomunculusArmConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            baud_rate=config.left_arm_config.baud_rate,
        )
        right_arm_config = HomunculusArmConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            baud_rate=config.right_arm_config.baud_rate,
        )

        self.left_arm = HomunculusArm(left_arm_config)
        self.right_arm = HomunculusArm(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.q": float for name in self._g1_arm_joint_names()}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, float]:
        left_action = self.left_arm.get_action()
        right_action = self.right_arm.get_action()

        action_dict: dict[str, float] = {}
        action_dict.update(self._map_homunculus_to_g1(left_action, "left"))
        action_dict.update(self._map_homunculus_to_g1(right_action, "right"))
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def _map_homunculus_to_g1(self, action: dict[str, float], side: str) -> dict[str, float]:
        mapped: dict[str, float] = {}
        side_prefix = "kLeft" if side == "left" else "kRight"
        for key, value in action.items():
            if not key.endswith(".pos"):
                continue
            joint = key[: -len(".pos")]
            g1_joint = _HOMUNCULUS_TO_G1_ARM.get(joint)
            if g1_joint is None:
                continue
            g1_name = f"{side_prefix}{g1_joint}"
            if g1_name not in self._g1_arm_joint_names_set:
                continue
            mapped[f"{g1_name}.q"] = float(value)
        return mapped

    @cached_property
    def _g1_arm_joint_names(self) -> list[str]:
        return [joint.name for joint in G1_29_JointArmIndex]

    @cached_property
    def _g1_arm_joint_names_set(self) -> set[str]:
        return set(self._g1_arm_joint_names)

