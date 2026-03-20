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

from __future__ import annotations

import logging

from lerobot.types import RobotAction, RobotObservation

from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline

logger = logging.getLogger(__name__)

# Teleoperator types that output delta EE actions (delta_x, delta_y, delta_z, gripper)
DELTA_TELEOP_TYPES = {"gamepad", "keyboard_ee"}

# Robot types that accept joint-space actions and have IK support
JOINT_SPACE_ROBOT_TYPES = {"so100_follower", "so101_follower"}

# Motor names for SO100/SO101 arms (order matters for joint control)
SO_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors(
    teleop_config: object | None = None,
    robot_config: object | None = None,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    """Build the processor pipelines for the given teleop + robot combination.

    When a delta teleop (gamepad, keyboard_ee) is paired with a joint-space robot
    (so100/so101_follower), the teleop action processor is set up to map gamepad
    axes directly to joint positions. Otherwise identity processors are returned.

    Args:
        teleop_config: The teleoperator configuration (must have a ``.type`` property).
            When *None*, identity processors are returned.
        robot_config: The robot configuration (must have a ``.type`` property).
            When *None*, identity processors are returned.

    Returns:
        A 3-tuple of (teleop_action_processor, robot_action_processor,
        robot_observation_processor).
    """
    teleop_action_processor = make_default_teleop_action_processor()

    if teleop_config is not None and robot_config is not None:
        teleop_type = teleop_config.type
        robot_type = robot_config.type

        if teleop_type in DELTA_TELEOP_TYPES and robot_type in JOINT_SPACE_ROBOT_TYPES:
            from lerobot.processor.delta_action_processor import MapGamepadToJointPositionsStep

            logger.info("Building direct joint control pipeline (gamepad axes -> joint positions)")
            teleop_action_processor = RobotProcessorPipeline[
                tuple[RobotAction, RobotObservation], RobotAction
            ](
                steps=[MapGamepadToJointPositionsStep(motor_names=SO_MOTOR_NAMES)],
                to_transition=robot_action_observation_to_transition,
                to_output=transition_to_robot_action,
            )

    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)
