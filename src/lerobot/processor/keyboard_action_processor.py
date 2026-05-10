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

from dataclasses import dataclass
from types import MappingProxyType

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.types import RobotAction, TransitionKey

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep

SO_KEYBOARD_JOINT_DELTAS = MappingProxyType(
    {
        "a": {"shoulder_pan": -1.0},
        "d": {"shoulder_pan": 1.0},
        "w": {"shoulder_lift": 1.0},
        "s": {"shoulder_lift": -1.0},
        "i": {"elbow_flex": 1.0},
        "k": {"elbow_flex": -1.0},
        "j": {"wrist_flex": -1.0},
        "l": {"wrist_flex": 1.0},
        "u": {"wrist_roll": -1.0},
        "o": {"wrist_roll": 1.0},
        "r": {"gripper": 1.0},
        "f": {"gripper": -1.0},
    }
)


@ProcessorStepRegistry.register("map_keyboard_to_so_joint_positions")
@dataclass
class MapKeyboardToSOJointPositionsStep(RobotActionProcessorStep):
    """Map raw keyboard keys to SO follower joint position targets.

    ``KeyboardTeleop`` emits active character keys (for example ``{"w": None}``).
    SO follower robots expect absolute joint targets with ``*.pos`` keys. This
    step keeps the current observation as the hold position, then applies small
    per-frame deltas for any active keyboard keys.
    """

    motor_names: tuple[str, ...]
    joint_step_size: float = 1.0
    gripper_step_size: float = 2.0

    def _get_current_positions(self) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError(
                "MapKeyboardToSOJointPositionsStep requires a robot observation dict "
                f"but got {type(observation)}."
            )

        missing_keys = [f"{motor}.pos" for motor in self.motor_names if f"{motor}.pos" not in observation]
        if missing_keys:
            raise ValueError(
                "MapKeyboardToSOJointPositionsStep requires current SO follower joint positions "
                f"in the observation. Missing keys: {missing_keys}."
            )

        return {f"{motor}.pos": float(observation[f"{motor}.pos"]) for motor in self.motor_names}

    def action(self, action: RobotAction) -> RobotAction:
        target_positions = self._get_current_positions()
        active_keys = {str(key).lower() for key in action}
        motor_deltas = dict.fromkeys(self.motor_names, 0.0)

        for key in active_keys:
            for motor, direction in SO_KEYBOARD_JOINT_DELTAS.get(key, {}).items():
                if motor not in motor_deltas:
                    continue
                step_size = self.gripper_step_size if motor == "gripper" else self.joint_step_size
                motor_deltas[motor] += direction * step_size

        for motor, delta in motor_deltas.items():
            pos_key = f"{motor}.pos"
            target_positions[pos_key] += delta
            if motor == "gripper":
                target_positions[pos_key] = min(100.0, max(0.0, target_positions[pos_key]))

        return target_positions

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {
            feature_type: feature_values.copy() for feature_type, feature_values in features.items()
        }
        transformed[PipelineFeatureType.ACTION] = {
            f"{motor}.pos": PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for motor in self.motor_names
        }
        transformed.setdefault(PipelineFeatureType.OBSERVATION, {})
        return transformed
