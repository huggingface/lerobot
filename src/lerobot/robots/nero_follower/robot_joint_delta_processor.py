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

from __future__ import annotations

from dataclasses import dataclass

from lerobot.processor import (
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.types import TransitionKey


@dataclass
class NEROKeyboardJointDeltasToAbsolute(RobotActionProcessorStep):
    """将键盘关节增量指令转换为绝对关节角度。

    从 observation 读取当前关节位置，加上 delta 得到绝对目标位置。
    """

    joint_names: list[str]
    gripper_min: float = 0.0
    gripper_max: float = 100.0

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return {}

        enabled = bool(float(action.get("enabled", 0.0)))
        if not enabled:
            return {}

        result = {}
        for i, name in enumerate(self.joint_names):
            delta_key = f"{name}.delta"
            pos_key = f"{name}.pos"
            delta = float(action.get(delta_key, 0.0))
            if delta == 0.0:
                continue
            current = float(observation.get(pos_key, 0.0))
            result[pos_key] = current + delta

        gripper_delta = float(action.get("gripper.delta", 0.0))
        if gripper_delta != 0.0:
            gripper_now = float(observation.get("gripper.pos", 0.0))
            gripper_target = max(self.gripper_min, min(self.gripper_max, gripper_now + gripper_delta))
            result["gripper.pos"] = gripper_target

        return result

    def reset(self):
        pass

    def transform_features(self, features):
        return features


def make_nero_keyboard_joint_robot_action_processor(
    joint_names: list[str],
    gripper_min: float = 0.0,
    gripper_max: float = 100.0,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            NEROKeyboardJointDeltasToAbsolute(
                joint_names=joint_names,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
