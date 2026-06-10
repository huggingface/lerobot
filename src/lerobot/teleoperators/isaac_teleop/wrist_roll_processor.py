#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Post-IK processor step that overwrites the wrist-roll joint from an angle.

The SO-101 is a 5-DOF arm, so ``InverseKinematicsEEToJoints`` cannot track a
full 6-DOF pose: the terminal roll axis does not move the end-effector
*position*, so the IK solver leaves it under-determined. The XR pipeline instead
carries an explicit absolute ``wrist_roll`` angle [rad] (from
``SO101RollRetargeter``) and this step writes it directly onto the ``wrist_roll``
joint *after* IK, overriding whatever the solver produced.

This module is pure dict math (``numpy`` only) and does **not** import
``isaacteleop``, so it can be unit-tested without the XR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction


@ProcessorStepRegistry.register("overwrite_wrist_roll_from_angle")
@dataclass
class OverwriteWristRollFromAngle(RobotActionProcessorStep):
    """Overwrites the ``wrist_roll`` joint target from an absolute angle [rad].

    Runs AFTER ``InverseKinematicsEEToJoints`` (which must keep
    ``initial_guess_current_joints=True`` so the IK seed re-tracks the measured
    joints each frame and the overwritten roll never drifts the seed). Consumes
    the ``wrist_roll`` angle [rad] carried by
    :class:`MapXRControllerActionToRobotAction` and converts it to the
    bus/``RobotKinematics`` unit (degrees):

        ``action["wrist_roll.pos"] = rad2deg(wrist_roll)``

    This intentionally overrides the IK-produced ``wrist_roll`` joint: on a 5-DOF
    arm the terminal roll axis does not affect the EE position, so the position-only
    IK leaves it under-determined and the operator's explicit roll is authoritative.

    No-ops when no roll command is present: :class:`MapXRControllerActionToRobotAction`
    omits the ``wrist_roll`` key before the first engage (and the clutch otherwise
    HOLDs the last roll while disengaged). When the key is absent, this step leaves
    the IK-produced ``wrist_roll.pos`` untouched rather than snapping the wrist.

    Input keys:
        - ``wrist_roll``: ``float`` — absolute wrist-roll angle [rad]. Optional: when
          absent (pre-engage / disengaged-before-first-engage) the step is a no-op.

    Output keys:
        - ``wrist_roll.pos``: ``float`` — wrist-roll joint target [deg]. Only written
          when a ``wrist_roll`` command is present.
    """

    def action(self, action: RobotAction) -> RobotAction:
        wrist_roll = action.pop("wrist_roll", None)
        if wrist_roll is None:
            # No roll command (pre-engage / disengaged before first engage): leave
            # the IK-produced wrist_roll.pos untouched so the wrist is not snapped.
            return action
        action["wrist_roll.pos"] = float(np.rad2deg(float(wrist_roll)))
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("wrist_roll", None)
        features[PipelineFeatureType.ACTION]["wrist_roll.pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )
        return features
