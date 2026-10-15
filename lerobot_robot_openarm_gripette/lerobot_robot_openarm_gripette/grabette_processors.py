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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from gripette.grasp_projection import GraspProjection

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, RobotObservation, TransitionKey
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep

ROBOT_JOINTS = [f"joint_{i}" for i in range(1, 8)]
POSITION_NAMES = ["x", "y", "z"]
ROTATION_NAMES = [f"r6d_{i}" for i in range(6)]
GRIPPER_NAMES = ["strategy", "closure"]

logger = logging.getLogger(__name__)


@ProcessorStepRegistry.register("gripette_policy_observation")
@dataclass
class GripettePolicyObservation(ObservationProcessorStep):
    projection: GraspProjection = field(default_factory=GraspProjection)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        strategy, closure = self.projection.encode(
            float(observation["proximal.pos"]), float(observation["distal.pos"])
        )
        return {
            **{key: value for key, value in observation.items() if not key.endswith(".pos")},
            "strategy": strategy,
            "closure": closure,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for name in [*ROBOT_JOINTS, "proximal", "distal"]:
            observation_features.pop(f"{name}.pos", None)
        for name in GRIPPER_NAMES:
            observation_features[name] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        return features


@ProcessorStepRegistry.register("grabette_relative_pose_to_joints")
@dataclass
class GrabetteRelativePoseToJoints(RobotActionProcessorStep):
    kinematics: RobotKinematics
    joint_signs: tuple[float, ...] = (1.0,) * 7
    max_joint_step: float = 5.0
    position_weight: float = 100.0
    orientation_weight: float = 10.0
    max_iters: int = 30
    resync_on_chunk_start: bool = False
    projection: GraspProjection = field(default_factory=GraspProjection)
    _q_cmd: np.ndarray | None = field(default=None, init=False, repr=False)
    _anchor: np.ndarray | None = field(default=None, init=False, repr=False)
    _count_queued_actions: Callable[[], int] | None = field(default=None, init=False, repr=False)
    _n_action_steps: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        signs = np.asarray(self.joint_signs, dtype=np.float64)
        if signs.shape != (7,) or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("joint_signs must contain seven values, each +1 or -1")
        if self.max_joint_step <= 0:
            raise ValueError("max_joint_step must be positive")

    def bind_action_queue(self, count_queued_actions: Callable[[], int], n_action_steps: int) -> None:
        self._count_queued_actions = count_queued_actions
        self._n_action_steps = n_action_steps

    def action(self, action: RobotAction) -> RobotAction:
        missing = [name for name in [*POSITION_NAMES, *ROTATION_NAMES, *GRIPPER_NAMES] if name not in action]
        if missing:
            raise ValueError(f"Missing Grabette policy action fields: {missing}")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Robot observation is required for Grabette inverse kinematics")

        signs = np.asarray(self.joint_signs, dtype=np.float64)
        queue_depth = self._count_queued_actions() if self._count_queued_actions is not None else 0
        chunk_start = self._anchor is None or queue_depth == max(0, self._n_action_steps - 1)
        if self._q_cmd is None or (chunk_start and self.resync_on_chunk_start):
            self._q_cmd = signs * np.array(
                [float(observation[f"{joint}.pos"]) for joint in ROBOT_JOINTS], dtype=np.float64
            )

        rows = np.array([action[name] for name in ROTATION_NAMES], dtype=np.float64).reshape(2, 3)
        row1 = rows[0] / np.linalg.norm(rows[0])
        row2 = rows[1] - np.dot(row1, rows[1]) * row1
        row2 = row2 / np.linalg.norm(row2)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.stack([row1, row2, np.cross(row1, row2)])
        pose[:3, 3] = [float(action[name]) for name in POSITION_NAMES]

        ry = np.arcsin(np.clip(-pose[2, 0], -1.0, 1.0))
        if abs(np.cos(ry)) > 1e-6:
            rx = np.arctan2(pose[2, 1], pose[2, 2])
            rz = np.arctan2(pose[1, 0], pose[0, 0])
        else:
            rx = np.arctan2(-pose[1, 2], pose[1, 1])
            rz = 0.0
        rotation_deg = np.degrees([rx, ry, rz])
        rotation_deg[np.abs(rotation_deg) < 5e-4] = 0.0
        logger.info(
            "Predicted relative action: X=%+.4f m Y=%+.4f m Z=%+.4f m | "
            "RX=%+.1f deg RY=%+.1f deg RZ=%+.1f deg",
            *pose[:3, 3],
            *rotation_deg,
        )

        if chunk_start:
            self._anchor = self.kinematics.forward_kinematics(self._q_cmd) @ np.linalg.inv(pose)

        q_target = self.kinematics.inverse_kinematics(
            self._q_cmd,
            self._anchor @ pose,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
            max_iters=self.max_iters,
        )
        jump = np.abs(q_target - self._q_cmd)
        if jump.max() > self.max_joint_step:
            raise RuntimeError(
                f"IK moved {ROBOT_JOINTS[jump.argmax()]} by {jump.max():.1f} deg in one tick, aborting."
            )
        self._q_cmd = q_target
        proximal, distal = self.projection.decode(float(action["strategy"]), float(action["closure"]))
        return {
            **{
                f"{joint}.pos": float(sign * target)
                for joint, sign, target in zip(ROBOT_JOINTS, signs, q_target, strict=True)
            },
            "proximal.pos": proximal,
            "distal.pos": distal,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        action_features.clear()
        for name in [*ROBOT_JOINTS, "proximal", "distal"]:
            action_features[f"{name}.pos"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features

    def reset(self) -> None:
        self._q_cmd = None
        self._anchor = None
