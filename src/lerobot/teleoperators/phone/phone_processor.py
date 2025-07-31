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


@ProcessorStepRegistry.register("phone_axis_remap_to_action")
@dataclass
class PhoneAxisRemapToAction:
    """Map calibrated phone pose to the inputs for robot action"""

    platform: PhoneOS

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION)

        # pop these from observation because only needed for this step
        enabled = obs.pop("phone.enabled", 0)
        pos = obs.pop("phone.pos", None)
        rot = obs.pop("phone.rot", None)
        inputs = obs.pop("phone.raw_inputs", {})

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
        act = {
            "enabled": enabled,
            "target_x": -pos[1] if enabled else 0.0,
            "target_y": pos[0] if enabled else 0.0,
            "target_z": pos[2] if enabled else 0.0,
            "target_qx": quat[1] if enabled else 0.0,
            "target_qy": quat[0] if enabled else 0.0,
            "target_qz": -quat[2] if enabled else 0.0,
            "target_qw": quat[3] if enabled else 0.0,
            "gripper": gripper,  # Still send gripper action when disabled
            "x": x if enabled else 0.0,
            "y": y if enabled else 0.0,
            "theta": theta if enabled else 0.0,
        }

        transition[TransitionKey.ACTION] = act
        return transition

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # Add new features and remove enabled from features
        return {
            "enabled": bool,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_qx": float,
            "target_qy": float,
            "target_qz": float,
            "target_qw": float,
            "gripper": float,
            "x": float,
            "y": float,
            "theta": float,
        }
