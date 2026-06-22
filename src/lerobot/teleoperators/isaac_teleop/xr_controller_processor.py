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

Analogous to ``MapPhoneActionToRobotAction`` in
``lerobot/teleoperators/phone/phone_processor.py``, this step bridges
:meth:`XRController.get_action` output to the input contract of the downstream
closed-loop IK pipeline (``EEBoundsAndSafety`` -> ``InverseKinematicsEEToJoints``).

This module does **not** import ``isaacteleop`` (only the local ``_GRIPPER_MOTOR_SCALE``
from ``base``), so it can be unit-tested without the XR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction
from lerobot.utils.rotation import Rotation

from .base import _GRIPPER_MOTOR_SCALE


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps an absolute base-frame EE pose + gripper closedness to the IK input contract.

    A pure, stateless per-frame rename with no clutch logic of its own: the owning
    loop owns the clutch (latches the engage origin and rebases the controller
    delta onto the EE), so the ``ee_pose`` reaching this step is the already-rebased
    *absolute* base-frame target. Each frame it:

    - writes ``ee.x/y/z = ee_pose[:3]`` (the absolute base-frame position target);
    - writes ``ee.wx/wy/wz`` = the rotvec of the absolute base-frame orientation
      target ``ee_pose[3:7]`` (the controller grip quaternion, already clutch-rebased
      upstream). ``InverseKinematicsEEToJoints`` reads these as a rotvec orientation
      target; with a small ``orientation_weight`` the 5-DOF SO-101 tracks it softly
      (position dominates). All six ``ee.*`` components must be present for the IK
      step to accept the action;
    - writes ``ee.gripper_pos = (1 - closedness) * _GRIPPER_MOTOR_SCALE`` (absolute jaw
      target in motor units ``[0, 100]``, RANGE_0_100; the SO-101 calibrates 100=open,
      0=closed, so closedness is inverted here), passed straight through to
      ``gripper.pos`` by the IK step.

    Input keys (from the owning loop's clutch):
        - ``ee_pose``: ``np.ndarray`` shape ``(7,)`` — ``[x,y,z,qx,qy,qz,qw]`` (base frame),
          the clutch-rebased absolute position + orientation target.
        - ``closedness``: ``float`` — jaw closedness in ``[0, 1]`` (0=open, 1=closed).

    Output keys:
        - ``ee.x``, ``ee.y``, ``ee.z``: ``float`` — absolute base-frame position target [m].
        - ``ee.wx``, ``ee.wy``, ``ee.wz``: ``float`` — absolute base-frame orientation
          target as a rotvec [rad].
        - ``ee.gripper_pos``: ``float`` — absolute jaw target in motor units ``[0, 100]``.
    """

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
        # Inverted: closedness c=1 (closed) -> 0, c=0 (open) -> 100 (SO-101 calibration).
        action["ee.gripper_pos"] = (1.0 - closedness) * _GRIPPER_MOTOR_SCALE
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
