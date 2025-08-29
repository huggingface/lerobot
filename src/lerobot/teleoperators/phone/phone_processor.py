# !/usr/bin/env python

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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import ActionProcessor, ProcessorStepRegistry
from lerobot.teleoperators.phone.config_phone import PhoneOS


@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapPhoneActionToRobotAction(ActionProcessor):
    """
    Map calibrated phone pose (actions) to the inputs for robot actions

    Expected input ACTION keys:
    {
        "action.phone.enabled": bool,
        "action.phone.pos": np.ndarray,
        "action.phone.rot": Rotation,
        "action.phone.raw_inputs": dict,
    }

    Output ACTION keys:
    {
        "action.enabled": bool,
        "action.ee.{x,y,z,wx,wy,wz}" : float
        "action.gripper": float,
    }
    """

    platform: PhoneOS
    _enabled_prev: bool = field(default=False, init=False, repr=False)

    def action(self, act: dict) -> dict:
        # Pop them from the action
        enabled = bool(act.pop("action.phone.enabled", 0))
        pos = act.pop("action.phone.pos", None)
        rot = act.pop("action.phone.rot", None)
        inputs = act.pop("action.phone.raw_inputs", {})

        if pos is None or rot is None:
            return act

        rotvec = rot.as_rotvec()  # Absolute orientation as rotvec

        # Map certain inputs to certain actions
        if self.platform == PhoneOS.IOS:
            gripper = float(inputs.get("a3", 0.0))
        else:
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper = (
                a - b
            )  # Positive if a is pressed, negative if b is pressed, 0 if both or neither are pressed

        # For some actions we need to invert the axis
        act["action.enabled"] = enabled
        act["action.target_x"] = -pos[1] if enabled else 0.0
        act["action.target_y"] = pos[0] if enabled else 0.0
        act["action.target_z"] = pos[2] if enabled else 0.0
        act["action.target_wx"] = rotvec[1] if enabled else 0.0
        act["action.target_wy"] = rotvec[0] if enabled else 0.0
        act["action.target_wz"] = -rotvec[2] if enabled else 0.0
        act["action.gripper"] = gripper  # Still send gripper action when disabled
        return act

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features.pop("action.phone.enabled", None)
        features.pop("action.phone.pos", None)
        features.pop("action.phone.rot", None)
        features.pop("action.phone.raw_inputs", None)

        features["action.enabled"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.target_x"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.target_y"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.target_z"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.target_wx"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.target_wy"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.target_wz"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        features["action.gripper"] = (PolicyFeature(type=FeatureType.ACTION, shape=(1,)),)
        return features
