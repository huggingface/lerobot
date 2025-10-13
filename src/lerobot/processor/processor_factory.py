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

"""Factory functions for creating robot and teleoperator action processors.

This module provides centralized factory functions that dispatch to the appropriate
processor pipelines based on robot/teleoperator type. This allows scripts like
teleoperate.py and record.py to support multiple robot types without hardcoding
specific processor implementations.
"""

from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.robots import Robot, RobotConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig


def make_teleop_action_processor(
    teleop_config: TeleoperatorConfig,
    teleop: Teleoperator,
    display_data: bool,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Create teleoperator action processor based on teleoperator type.

    This factory function selects the appropriate processor pipeline for converting
    raw teleoperator actions (typically joint angles) to the desired action space
    (typically end-effector poses for FK/IK-based systems).

    Args:
        teleop_config: Configuration for the teleoperator
        teleop: Teleoperator instance
        display_data: Whether to enable visualization in rerun

    Returns:
        Processor pipeline for teleoperator actions, or IdentityProcessor if no
        specific processor is needed for this teleoperator type
    """
    if teleop_config.type == "koch_leader":
        from lerobot.teleoperators.koch_leader.config_koch_leader import make_koch_teleop_processors

        return make_koch_teleop_processors(teleop, display_data)
    elif teleop_config.type == "bi_koch_leader":
        from lerobot.teleoperators.bi_koch_leader.config_bi_koch_leader import (
            make_bimanual_koch_teleop_processors,
        )

        return make_bimanual_koch_teleop_processors(teleop, display_data)
    else:
        # For teleoperators without specific processors, return identity processor
        from lerobot.processor import IdentityProcessor

        return RobotProcessorPipeline(
            steps=[IdentityProcessor()],
        )


def make_robot_action_processor(
    robot_config: RobotConfig,
    robot: Robot,
    display_data: bool,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Create robot action processor based on robot type.

    This factory function selects the appropriate processor pipeline for converting
    actions from the desired action space (typically end-effector poses for FK/IK-based
    systems) to robot commands (typically joint angles).

    Args:
        robot_config: Configuration for the robot
        robot: Robot instance
        display_data: Whether to enable visualization in rerun

    Returns:
        Processor pipeline for robot actions, or IdentityProcessor if no
        specific processor is needed for this robot type
    """
    if robot_config.type == "koch_follower":
        from lerobot.robots.koch_follower.config_koch_follower import make_koch_robot_processors

        return make_koch_robot_processors(robot, display_data)
    elif robot_config.type == "bi_koch_follower":
        from lerobot.robots.bi_koch_follower.config_bi_koch_follower import (
            make_bimanual_koch_robot_processors,
        )

        return make_bimanual_koch_robot_processors(robot, display_data)
    else:
        # For robots without specific processors, return identity processor
        from lerobot.processor import IdentityProcessor

        return RobotProcessorPipeline(
            steps=[IdentityProcessor()],
        )


def make_fk_processor(
    robot_config: RobotConfig,
    robot: Robot,
    display_data: bool,
) -> RobotProcessorPipeline[RobotAction, RobotAction]:
    """Create forward kinematics processor for computing EE pose from joint angles.

    This factory function selects the appropriate FK processor for converting
    joint angles to end-effector poses. This is useful for computing the current
    EE state from robot observations.

    Args:
        robot_config: Configuration for the robot
        robot: Robot instance
        display_data: Whether to enable visualization in rerun

    Returns:
        FK processor pipeline, or IdentityProcessor if FK is not needed for this robot
    """
    if robot_config.type == "koch_follower":
        from lerobot.teleoperators.koch_leader.config_koch_leader import make_koch_teleop_processors

        return make_koch_teleop_processors(robot, display_data)
    elif robot_config.type == "bi_koch_follower":
        from lerobot.teleoperators.bi_koch_leader.config_bi_koch_leader import (
            make_bimanual_koch_teleop_processors,
        )

        return make_bimanual_koch_teleop_processors(robot, display_data)
    else:
        # For robots without FK processors, return identity processor
        from lerobot.processor import IdentityProcessor

        return RobotProcessorPipeline(
            steps=[IdentityProcessor()],
        )
