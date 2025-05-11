import logging
from pprint import pformat

from lerobot.common.robots import RobotConfig

from .robot import Robot


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "aloha":
        raise NotImplementedError  # TODO

    elif robot_type == "koch_follower":
        from .koch_follower.config_koch_follower import KochFollowerConfig

        return KochFollowerConfig(**kwargs)
    # elif robot_type == "koch_bimanual":
    #     return KochBimanualRobotConfig(**kwargs)
    elif robot_type == "moss_follower":
        from .moss_follower.configuration_moss import MossRobotConfig

        return MossRobotConfig(**kwargs)
    elif robot_type == "so100_follower":
        from .so100_follower.config_so100_follower import SO100FollowerConfig

        return SO100FollowerConfig(**kwargs)
    elif robot_type == "stretch":
        from .stretch3.configuration_stretch3 import Stretch3RobotConfig

        return Stretch3RobotConfig(**kwargs)
    elif robot_type == "lekiwi":
        from .lekiwi.config_lekiwi import LeKiwiConfig

        return LeKiwiConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwiClient

        return LeKiwiClient(config)
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot

        return Stretch3Robot(config)
    elif config.type == "viperx":
        from .viperx import ViperX

        return ViperX(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    else:
        raise ValueError(config.type)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[float]
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


# TODO(aliberts): Remove
def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"
