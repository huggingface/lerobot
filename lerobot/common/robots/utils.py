import logging
from pprint import pformat
from typing import Protocol

from lerobot.common.robots import RobotConfig


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


# TODO(aliberts): Remove and point to lerobot.common.robots.Robot
class Robot(Protocol):
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "aloha":
        from .aloha.configuration_aloha import AlohaRobotConfig

        return AlohaRobotConfig(**kwargs)
    elif robot_type == "koch":
        from .koch.configuration_koch import KochRobotConfig

        return KochRobotConfig(**kwargs)
    # elif robot_type == "koch_bimanual":
    #     return KochBimanualRobotConfig(**kwargs)
    elif robot_type == "moss":
        from .moss.configuration_moss import MossRobotConfig

        return MossRobotConfig(**kwargs)
    elif robot_type == "so100":
        from .so100.configuration_so100 import SO100RobotConfig

        return SO100RobotConfig(**kwargs)
    elif robot_type == "stretch":
        from .stretch3.configuration_stretch3 import Stretch3RobotConfig

        return Stretch3RobotConfig(**kwargs)
    elif robot_type == "lekiwi":
        from .lekiwi.configuration_lekiwi import LeKiwiRobotConfig

        return LeKiwiRobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    from .lekiwi.configuration_lekiwi import LeKiwiRobotConfig
    from .manipulator import ManipulatorRobotConfig

    if isinstance(config, ManipulatorRobotConfig):
        from lerobot.common.robots.manipulator import ManipulatorRobot

        return ManipulatorRobot(config)
    elif isinstance(config, LeKiwiRobotConfig):
        from lerobot.common.robots.mobile_manipulator import MobileManipulator

        return MobileManipulator(config)
    else:
        from lerobot.common.robots.stretch3.robot_stretch3 import Stretch3Robot

        return Stretch3Robot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = {key: max_relative_target for key in goal_present_pos}
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
