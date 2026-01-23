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
from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .config import RobotConfig
from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    # TODO(Steven): Consider just using the make_device_from_device_class for all types
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "omx_follower":
        from .omx_follower import OmxFollower

        return OmxFollower(config)
    elif config.type == "so100_follower":
        from .so_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so101_follower":
        from .so_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand

        return HopeJrHand(config)
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm

        return HopeJrArm(config)
    elif config.type == "bi_so_follower":
        from .bi_so_follower import BiSOFollower

        return BiSOFollower(config)
    elif config.type == "reachy2":
        from .reachy2 import Reachy2Robot

        return Reachy2Robot(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    else:
        try:
            return cast(Robot, make_device_from_device_class(config))
        except Exception as e:
            raise ValueError(f"Error creating robot with config {config}: {e}") from e


# TODO(pepijn): Move to pipeline step to make sure we don't have to do this in the robot code and send action to robot is clean for use in dataset
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
