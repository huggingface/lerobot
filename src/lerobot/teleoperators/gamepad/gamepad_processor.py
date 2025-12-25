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

from dataclasses import dataclass
import numpy as np
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_gamepad_action_to_robot_action")
@dataclass
class MapGamepadActionToRobotAction(RobotActionProcessorStep):
    """
    Maps gamepad deltas to standardized robot action inputs expected downstream.

    Inputs (from `GamepadTeleop.get_action()`):
        - "delta_x", "delta_y", "delta_z" (float) in controller space

    Outputs (for downstream steps like `EEReferenceAndDelta`):
        - "enabled" (bool): True when any delta is non-zero
        - "target_x", "target_y", "target_z" (float): passed-through deltas
        - "target_wx", "target_wy", "target_wz" (float): zero (no rotation control initially)
        - "gripper_vel" (float): zero (gripper handled later; held by velocity=0)
    """

    def action(self, action: RobotAction) -> RobotAction:
        dx = float(action.pop("delta_x", 0.0))
        dy = float(action.pop("delta_y", 0.0))
        dz = float(action.pop("delta_z", 0.0))

        # Clip dx, dy, dz to be between -0.5 and 0.5 to achieve smoother movement
        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)
        dz = np.clip(dz, -0.5, 0.5)

        print(f"dx: {dx}, dy: {dy}, dz: {dz}")

        enabled = abs(dx) > 1e-8 or abs(dy) > 1e-8 or abs(dz) > 1e-8

        action["enabled"] = enabled
        action["target_x"] = dx if enabled else 0.0
        action["target_y"] = dy if enabled else 0.0
        action["target_z"] = dz if enabled else 0.0
        action["target_wx"] = 0.0
        action["target_wy"] = 0.0
        action["target_wz"] = 0.0
        action["gripper_vel"] = 0.0
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["delta_x", "delta_y", "delta_z"]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


