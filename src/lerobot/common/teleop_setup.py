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

from dataclasses import replace
from typing import TYPE_CHECKING

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.processor.base_action_processor import BaseActionProcessorStep
from lerobot.processor.hold_joints_processor import HoldUncommandedJointsProcessorStep
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.composite import CompositeTeleoperator

if TYPE_CHECKING:
    from lerobot.robots import Robot


def make_mobile_teleop(
    robot: "Robot",
    base: Teleoperator,
    arm: Teleoperator | None = None,
    *,
    base_mapping: BaseActionProcessorStep | None = None,
    action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction] | None = None,
) -> tuple[Teleoperator, RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]]:
    """Combine a base input and an optional arm leader, mapping base inputs and holding other joints.

    Reuse the returned pipeline across recording/reset phases; reset it for a new session.
    """
    teleop = (
        base
        if arm is None
        else CompositeTeleoperator([(base, ""), (arm, getattr(robot, "arm_action_prefix", ""))])
    )
    pipeline = action_processor if action_processor is not None else make_default_processors()[0]
    steps = list(pipeline.steps)
    mapping_type = BaseActionProcessorStep.get_known_choices().get(base.name)
    if base_mapping is None:
        base_mapping = mapping_type() if mapping_type is not None else None
    elif mapping_type is None or not isinstance(base_mapping, mapping_type):
        raise ValueError(f"The base mapping must match the {base.name!r} base input.")
    if base_mapping is not None:
        # Each session owns its mapping state (e.g. keyboard speed and pressed keys).
        steps.insert(0, replace(base_mapping))
    steps.append(
        HoldUncommandedJointsProcessorStep(
            tuple(key for key in robot.action_features if key.endswith(".pos"))
        )
    )
    return teleop, replace(pipeline, steps=steps)
