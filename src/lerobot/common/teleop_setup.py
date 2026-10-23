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

"""Assemble mobile control devices and their action pipeline at session setup."""

from dataclasses import replace
from typing import TYPE_CHECKING

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.processor.hold_joints_processor import HoldUncommandedJointsProcessorStep
from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep
from lerobot.processor.pipeline import ProcessorStep
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.composite import CompositeTeleoperator
from lerobot.teleoperators.keyboard import KeyboardTeleop

if TYPE_CHECKING:
    from lerobot.robots import Robot


def make_mobile_teleop(
    robot: "Robot",
    base: Teleoperator,
    arm: Teleoperator | None = None,
    *,
    keyboard_mapping: KeyboardToBaseActionProcessorStep | None = None,
    hold_uncommanded: bool = True,
    action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction] | None = None,
) -> tuple[Teleoperator, RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]]:
    """Return ordinary device(s) plus a pipeline that maps their combined raw action.

    A single base is returned unchanged. With an arm, a generic composite prefixes its keys
    using the robot's optional ``arm_action_prefix``. The pipeline owns keyboard mapping and
    joint targets; reuse it for recording and reset phases, and reset it on a new session.
    No concrete robot type or legacy robot-owned keyboard mapping is needed.
    """
    teleop = (
        base
        if arm is None
        else CompositeTeleoperator([(base, ""), (arm, getattr(robot, "arm_action_prefix", ""))])
    )
    pipeline = action_processor if action_processor is not None else make_default_processors()[0]
    steps: list[ProcessorStep] = []
    if isinstance(base, KeyboardTeleop) and base.name == "keyboard":
        # A new session gets fresh speed/key state even when a config object is reused.
        steps.append(
            replace(keyboard_mapping) if keyboard_mapping is not None else KeyboardToBaseActionProcessorStep()
        )
    steps.extend(pipeline.steps)
    if hold_uncommanded:
        steps.append(
            HoldUncommandedJointsProcessorStep(
                tuple(key for key in robot.action_features if key.endswith(".pos"))
            )
        )
    return teleop, replace(pipeline, steps=steps)
