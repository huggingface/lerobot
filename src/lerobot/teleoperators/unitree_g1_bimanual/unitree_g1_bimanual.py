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
import os
import re
import xml.etree.ElementTree as ET
from functools import cached_property

from huggingface_hub import snapshot_download

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex

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

        self._g1_joint_limits = self._load_g1_joint_limits()

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
        return {f"{name}.q": float for name in self._g1_joint_names()}

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

        # Default all joints to 0.0 so dataset logging has full action keys.
        action_dict: dict[str, float] = {f"{name}.q": 0.0 for name in self._g1_joint_names}
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
            mapped[f"{g1_name}.q"] = float(self._map_to_g1_range(g1_name, value))
        return mapped

    @cached_property
    def _g1_arm_joint_names(self) -> list[str]:
        return [joint.name for joint in G1_29_JointArmIndex]

    @cached_property
    def _g1_arm_joint_names_set(self) -> set[str]:
        return set(self._g1_arm_joint_names)

    @cached_property
    def _g1_joint_names(self) -> list[str]:
        return [joint.name for joint in G1_29_JointIndex]

    def _load_g1_joint_limits(self) -> dict[str, tuple[float, float]]:
        repo_path = snapshot_download(self.config.g1_model_repo_id)
        model_path = os.path.join(repo_path, self.config.g1_model_filename)
        if not os.path.exists(model_path):
            logger.warning(f"{self}: G1 model file not found at {model_path}")
            return {}

        limits: dict[str, tuple[float, float]] = {}
        try:
            tree = ET.parse(model_path)
        except Exception as exc:
            logger.warning(f"{self}: Failed to parse G1 model file: {exc}")
            return {}

        root = tree.getroot()
        for joint in root.iter("joint"):
            name = joint.get("name")
            range_attr = joint.get("range")
            if not name or not range_attr:
                continue
            try:
                min_val, max_val = (float(x) for x in range_attr.split())
            except ValueError:
                continue
            g1_name = self._xml_joint_to_g1_name(name)
            if g1_name and g1_name in self._g1_arm_joint_names_set:
                limits[g1_name] = (min_val, max_val)

        if not limits:
            logger.warning(f"{self}: No G1 joint limits found in {model_path}")
        return limits

    def _map_to_g1_range(self, g1_joint: str, value: float) -> float:
        """Map Homunculus normalized value (-100..100) to G1 joint limits."""
        min_max = self._g1_joint_limits.get(g1_joint)
        if not min_max:
            return float(value)

        min_val, max_val = min_max
        clamped = max(-100.0, min(100.0, float(value)))
        return min_val + (clamped + 100.0) * (max_val - min_val) / 200.0

    def _xml_joint_to_g1_name(self, xml_joint_name: str) -> str | None:
        match = re.match(r"^(left|right)_(.+)_joint$", xml_joint_name)
        if not match:
            return None

        side, snake = match.groups()
        camel = "".join(part.capitalize() for part in snake.split("_"))
        side_prefix = "kLeft" if side == "left" else "kRight"
        return f"{side_prefix}{camel}"

