# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
from pprint import pformat

from lerobot.robots import RobotConfig

from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower, KochFollowerConfig

        return KochFollower(cast(KochFollowerConfig,config))
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower, SO100FollowerConfig

        return SO100Follower(cast(SO100FollowerConfig,config))
    elif config.type == "so100_follower_end_effector":
        from .so100_follower import SO100FollowerEndEffector, SO100FollowerEndEffectorConfig

        return SO100FollowerEndEffector(cast(SO100FollowerEndEffectorConfig, config))
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower, SO101FollowerConfig

        return SO101Follower(cast(SO101FollowerConfig, config))
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi, LeKiwiConfig

        return LeKiwi(cast(LeKiwiConfig, config))
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot, Stretch3RobotConfig

        return Stretch3Robot(cast(Stretch3RobotConfig, config))
    elif config.type == "viperx":
        from .viperx import ViperX, ViperXConfig

        return ViperX(cast(ViperXConfig, config))
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand, HopeJrHandConfig

        return HopeJrHand(cast(HopeJrHandConfig, config))
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm, HopeJrArmConfig

        return HopeJrArm(cast(HopeJrArmConfig, config))
    elif config.type == "bi_so100_follower":
        from .bi_so100_follower import BiSO100Follower, BiSO100FollowerConfig

        return BiSO100Follower(cast(BiSO100FollowerConfig, config))
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot, MockRobotConfig

        return MockRobot(cast(MockRobotConfig, config))
    else:
        raise ValueError(config.type)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[str, float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions
