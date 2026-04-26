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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lerobot.model import RobotKinematics
from lerobot.processor import (
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.types import TransitionKey
from lerobot.utils.rotation import Rotation

from .config_nero_follower import NEOKeyboardEEConfig, NEOFollowerRobotConfig

logger = logging.getLogger(__name__)


class JointSpaceKinematics(Protocol):
    def forward_kinematics(self, joint_pos_rad: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, current_joint_pos_rad: np.ndarray, desired_ee_pose: np.ndarray) -> np.ndarray: ...


@dataclass
class NERORadianKinematicsAdapter:
    """Adapter that exposes RobotKinematics in radians for NERO runtime."""

    kinematics: RobotKinematics

    def forward_kinematics(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        return self.kinematics.forward_kinematics(np.rad2deg(joint_pos_rad))

    def inverse_kinematics(self, current_joint_pos_rad: np.ndarray, desired_ee_pose: np.ndarray) -> np.ndarray:
        result_deg = self.kinematics.inverse_kinematics(np.rad2deg(current_joint_pos_rad), desired_ee_pose)
        return np.deg2rad(result_deg)


@dataclass
class NEROKeyboardEEToJoints(RobotActionProcessorStep):
    """Convert keyboard task-space commands into safe NERO joint targets via IK."""

    kinematics: JointSpaceKinematics
    joint_names: list[str]
    config: NEOKeyboardEEConfig

    _target_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)

    def _get_joint_vector(self, observation: RobotObservation) -> np.ndarray:
        values = []
        for name in self.joint_names:
            key = f"{name}.pos"
            if key not in observation:
                raise ValueError(f"Missing required observation key: {key}")
            values.append(float(observation[key]))
        return np.array(values, dtype=float)

    def _clip_pose(self, pose: np.ndarray) -> np.ndarray:
        if self.config.position_bounds_min is None or self.config.position_bounds_max is None:
            return pose

        pos = pose[:3, 3]
        clipped = np.clip(
            pos,
            np.array(self.config.position_bounds_min, dtype=float),
            np.array(self.config.position_bounds_max, dtype=float),
        )
        pose[:3, 3] = clipped
        return pose

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for keyboard EE IK processing")

        q_curr = self._get_joint_vector(observation.copy())
        if self._target_ee_pose is None:
            self._target_ee_pose = self.kinematics.forward_kinematics(q_curr)

        enabled = bool(float(action.get("enabled", 0.0)))
        gripper_vel = float(action.get("gripper_vel", 0.0))

        if not enabled:
            return {}

        dx = float(np.clip(action.get("target_x", 0.0), -self.config.max_linear_step_m, self.config.max_linear_step_m))
        dy = float(np.clip(action.get("target_y", 0.0), -self.config.max_linear_step_m, self.config.max_linear_step_m))
        dz = float(np.clip(action.get("target_z", 0.0), -self.config.max_linear_step_m, self.config.max_linear_step_m))
        dwx = float(
            np.clip(action.get("target_wx", 0.0), -self.config.max_angular_step_rad, self.config.max_angular_step_rad)
        )
        dwy = float(
            np.clip(action.get("target_wy", 0.0), -self.config.max_angular_step_rad, self.config.max_angular_step_rad)
        )
        dwz = float(
            np.clip(action.get("target_wz", 0.0), -self.config.max_angular_step_rad, self.config.max_angular_step_rad)
        )

        desired = self._target_ee_pose.copy()
        desired[:3, 3] = desired[:3, 3] + np.array([dx, dy, dz], dtype=float)
        desired[:3, :3] = desired[:3, :3] @ Rotation.from_rotvec([dwx, dwy, dwz]).as_matrix()
        desired = self._clip_pose(desired)

        try:
            q_target = self.kinematics.inverse_kinematics(q_curr, desired)
        except Exception as exc:  # nosec B110
            logger.warning("NERO keyboard IK failed, dropping arm command: %s", exc)
            return {}

        self._target_ee_pose = desired
        result = {f"{name}.pos": float(q_target[i]) for i, name in enumerate(self.joint_names)}

        if "gripper.pos" in observation:
            gripper_now = float(observation["gripper.pos"])
            gripper_target = np.clip(
                gripper_now + gripper_vel * self.config.gripper_delta_per_step,
                self.config.gripper_min,
                self.config.gripper_max,
            )
            result["gripper.pos"] = float(gripper_target)

        return result

    def reset(self):
        self._target_ee_pose = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for key in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(key, None)

        for name in self.joint_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(1,),
            )
        features[PipelineFeatureType.ACTION]["gripper.pos"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(1,),
        )
        return features


def make_nero_keyboard_ee_robot_action_processor(
    config: NEOFollowerRobotConfig,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    if config.keyboard_ee.urdf_path is None:
        raise ValueError(
            "NERO keyboard end-effector teleop requires --robot.keyboard_ee.urdf_path=<path/to/nero.urdf>"
        )

    kinematics = NERORadianKinematicsAdapter(
        RobotKinematics(
            urdf_path=config.keyboard_ee.urdf_path,
            target_frame_name=config.keyboard_ee.target_frame_name,
            joint_names=config.keyboard_ee.joint_names,
        )
    )

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            NEROKeyboardEEToJoints(
                kinematics=kinematics,
                joint_names=config.keyboard_ee.joint_names,
                config=config.keyboard_ee,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
