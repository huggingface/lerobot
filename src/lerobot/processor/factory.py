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
from typing import Any

from .converters import (
    identity_transition,
    observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from .core import EnvTransition, RobotAction
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline


def make_default_processors():
    teleop_action_processor: RobotProcessorPipeline[RobotAction, EnvTransition] = RobotProcessorPipeline[
        RobotAction, EnvTransition
    ](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_to_transition,
        to_output=identity_transition,
    )
    robot_action_processor: RobotProcessorPipeline[EnvTransition, RobotAction] = RobotProcessorPipeline[
        EnvTransition, RobotAction
    ](
        steps=[IdentityProcessorStep()],
        to_transition=identity_transition,
        to_output=transition_to_robot_action,
    )
    robot_observation_processor: RobotProcessorPipeline[dict[str, Any], EnvTransition] = (
        RobotProcessorPipeline[dict[str, Any], EnvTransition](
            steps=[IdentityProcessorStep()],
            to_transition=observation_to_transition,
            to_output=identity_transition,
        )
    )
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)
