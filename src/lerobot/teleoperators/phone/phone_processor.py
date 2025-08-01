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

from dataclasses import dataclass

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey
from lerobot.teleoperators.phone.config_phone import PhoneOS


@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapPhoneActionToRobotAction:
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
        "action.target_x": float,
        "action.target_y": float,
        "action.target_z": float,
        "action.target_qx": float,
        "action.target_qy": float,
        "action.target_qz": float,
        "action.target_qw": float,
        "action.gripper": float,
        "action.x": float,
        "action.y": float,
        "action.theta": float,
    }
    """

    platform: PhoneOS

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}

        if act is None or not isinstance(act, dict):
            return transition

        # Pop them from the action
        enabled = act.pop("action.phone.enabled", 0)
        pos = act.pop("action.phone.pos", None)
        rot = act.pop("action.phone.rot", None)
        inputs = act.pop("action.phone.raw_inputs", {})

        if pos is None or rot is None:
            return transition

        quat = rot.as_quat()
        if quat[3] < 0:  # ensure qw >= 0
            quat = -quat

        # Map certain inputs to certain actions
        if self.platform == PhoneOS.IOS:
            gripper = float(inputs.get("a3", 0.0))
            x = float(inputs.get("a1", 0.0))
            y = float(inputs.get("a2", 0.0))
            theta = float(inputs.get("a7", 0.0))
        else:
            sc = float(inputs.get("scale", 1.0))
            gripper = max(min(sc - 1.0, 1.0), -1.0)
            x = y = theta = 0.0

        # For some actions we need to invert the axis
        act.update(
            {
                "action.enabled": enabled,
                "action.target_x": -pos[1] if enabled else 0.0,
                "action.target_y": pos[0] if enabled else 0.0,
                "action.target_z": pos[2] if enabled else 0.0,
                "action.target_qx": quat[1] if enabled else 0.0,
                "action.target_qy": quat[0] if enabled else 0.0,
                "action.target_qz": -quat[2] if enabled else 0.0,
                "action.target_qw": quat[3] if enabled else 0.0,
                "action.gripper": gripper,  # Still send gripper action when disabled
                "action.x": x if enabled else 0.0,
                "action.y": y if enabled else 0.0,
                "action.theta": theta if enabled else 0.0,
            }
        )

        transition[TransitionKey.ACTION] = act
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Accept both scoped/unscoped inputs, always emit scoped outputs
        for k in ("phone.enabled", "phone.pos", "phone.rot", "phone.raw_inputs"):
            features.pop(f"action.{k}", None)
            features.pop(k, None)
        features.update(
            {
                "action.enabled": bool,
                "action.target_x": float,
                "action.target_y": float,
                "action.target_z": float,
                "action.target_qx": float,
                "action.target_qy": float,
                "action.target_qz": float,
                "action.target_qw": float,
                "action.gripper": float,
                "action.x": float,
                "action.y": float,
                "action.theta": float,
            }
        )
        return features
