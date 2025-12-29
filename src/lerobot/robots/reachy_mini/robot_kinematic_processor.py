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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)


class ReachyMiniKinematics:
    """
    Reachy Mini Kinematics adapter using AnalyticalKinematics from reachy_mini SDK.
    Adapts the API to match lerobot.model.kinematics.RobotKinematics protocol.
    """

    def __init__(self, automatic_body_yaw: bool = True):
        self.kin = AnalyticalKinematics(automatic_body_yaw=automatic_body_yaw)
        # Joints order expected by AnalyticalKinematics: body_yaw, stewart_1...6
        self.joint_names = [
            "body_rotation",
            "stewart_1",
            "stewart_2",
            "stewart_3",
            "stewart_4",
            "stewart_5",
            "stewart_6",
        ]

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.
        Args:
            joint_pos_deg: Joint positions in degrees (length 7).
        Returns:
            4x4 transformation matrix of the end-effector pose.
        """
        if len(joint_pos_deg) < 7:
            raise ValueError(f"Expected at least 7 joints, got {len(joint_pos_deg)}")

        # Convert to radians
        # Note: only take the first 7 joints corresponding to head
        joint_pos_rad = np.deg2rad(joint_pos_deg[:7])

        # AnalyticalKinematics.fk returns 4x4 matrix
        # Check collision and no_iterations are optional/defaults in SDK
        return self.kin.fk(joint_pos_rad)

    def inverse_kinematics(
        self,
        current_joint_pos_deg: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,  # Unused by Analytical IK
        orientation_weight: float = 0.01,  # Unused by Analytical IK
    ) -> np.ndarray:
        """
        Compute inverse kinematics.
        Args:
            current_joint_pos_deg: Current joint positions in degrees. Used for body_yaw hint.
            desired_ee_pose: Target end-effector pose as 4x4 matrix.
        Returns:
            Joint positions in degrees (length 7).
        """
        # Extract body yaw from current position (first element)
        body_yaw_rad = np.deg2rad(current_joint_pos_deg[0]) if len(current_joint_pos_deg) > 0 else 0.0

        # Compute IK
        # AnalyticalKinematics.ik returns 7 joints in radians
        joints_rad = self.kin.ik(desired_ee_pose, body_yaw=body_yaw_rad)

        # Convert to degrees
        joints_deg = np.rad2deg(joints_rad)

        return joints_deg


@ProcessorStepRegistry.register("reachy_inverse_kinematics_ee_to_joints")
@dataclass
class ReachyInverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using Reachy Mini IK.
    """

    kinematics: ReachyMiniKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        # Reachy doesn't have a gripper in the head chain usually, but we keep generic structure
        # If antennas are treated as 'gripper' or separate, they are not part of head IK.

        if None in (x, y, z, wx, wy, wz):
             raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Extract joints corresponding to motor_names
        q_raw_list = []
        for name in self.motor_names:
            key = f"{name}.pos"
            if key in observation:
                q_raw_list.append(float(observation[key]))
            else:
                 # Fallback if motor not in observation? Should raise error.
                 raise ValueError(f"Missing observation for {key}")

        q_raw = np.array(q_raw_list, dtype=float)

        if self.initial_guess_current_joints:
            self.q_curr = q_raw
        else:
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute IK
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        for i, name in enumerate(self.motor_names):
             action[f"{name}.pos"] = float(q_target[i])

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        self.q_curr = None


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: ReachyMiniKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]
    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()

    # We don't remove joint positions from dictionary as we might want them?
    # But usage in SO-100 removes them.
    for n in motor_names:
        joints.pop(f"{n}.pos")

    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    return joints


@ProcessorStepRegistry.register("reachy_forward_kinematics_joints_to_ee_observation")
@dataclass
class ReachyForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using Reachy Mini FK.
    """

    kinematics: ReachyMiniKinematics
    motor_names: list[str]

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        for k in ["x", "y", "z", "wx", "wy", "wz"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features

