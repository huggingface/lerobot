#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Processor step that maps XR controller actions to robot EE targets.

Analogous to ``MapPhoneActionToRobotAction``, this bridges the clutch-rebased EE pose to
the IK pipeline's input contract (``EEBoundsAndSafety`` -> ``InverseKinematicsEEToJoints``).
Pure (no ``isaacteleop``), so it is unit-testable without the XR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.utils.rotation import Rotation

from .base import _GRIPPER_MOTOR_SCALE


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps an absolute base-frame EE pose + gripper closedness to the IK input contract.

    Pure, stateless rename (the device's in-pipeline clutch retargeter already produced the
    absolute base-frame target). Each frame it writes:

    - ``ee.x/y/z`` = ``ee_pose[:3]`` (position [m]);
    - ``ee.wx/wy/wz`` = rotvec of ``ee_pose[3:7]`` (orientation; the IK tracks it softly at a
      small ``orientation_weight`` on the 5-DOF SO-101);
    - ``ee.gripper_pos`` = ``gripper_open + closedness * (gripper_close - gripper_open)``.

    The gripper endpoints are in the follower's own action units, so a robot whose jaw is
    not the SO-101's RANGE_0_100 (the reBot B601-RS drives its gripper in degrees) only has
    to state its two endpoints. Defaults reproduce the SO-101 mapping: 100 = fully open.

    Input keys: ``ee_pose`` ``(7,)`` ``[x,y,z,qx,qy,qz,qw]``, ``closedness`` float in [0, 1].
    """

    gripper_open: float = _GRIPPER_MOTOR_SCALE
    gripper_close: float = 0.0

    def action(self, action: RobotAction) -> RobotAction:
        ee_pose = action.pop("ee_pose")
        closedness = float(action.pop("closedness"))

        action["ee.x"] = float(ee_pose[0])
        action["ee.y"] = float(ee_pose[1])
        action["ee.z"] = float(ee_pose[2])
        # Orientation target as a rotvec (quat [qx,qy,qz,qw] -> axis-angle); the IK
        # consumes ee.w* as a rotvec and tracks it with orientation_weight.
        rotvec = Rotation.from_quat(ee_pose[3:7]).as_rotvec()
        action["ee.wx"] = float(rotvec[0])
        action["ee.wy"] = float(rotvec[1])
        action["ee.wz"] = float(rotvec[2])
        action["ee.gripper_pos"] = self.gripper_open + closedness * (self.gripper_close - self.gripper_open)
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["ee_pose", "closedness"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "ee.gripper_pos",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features
