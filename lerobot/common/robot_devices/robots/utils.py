from typing import Protocol

from lerobot.common.robot_devices.robots.configs import (
    AlohaRobotConfig,
    KochBimanualRobotConfig,
    KochRobotConfig,
    LeKiwiRobotConfig,
    ManipulatorRobotConfig,
    MossRobotConfig,
    RobotConfig,
    So100RobotConfig,
    StretchRobotConfig,
    NoOpRobotConfig,
    ARX5RobotConfig,
    ARX5SingleArmRobotConfig,
    ARX5BimanualRobotConfig,
    ARX5SingleArmFollowOnlyConfig,
    ARX5BimanualFollowOnlyConfig,
)


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
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
        return AlohaRobotConfig(**kwargs)
    elif robot_type == "koch":
        return KochRobotConfig(**kwargs)
    elif robot_type == "koch_bimanual":
        return KochBimanualRobotConfig(**kwargs)
    elif robot_type == "moss":
        return MossRobotConfig(**kwargs)
    elif robot_type == "so100":
        return So100RobotConfig(**kwargs)
    elif robot_type == "stretch":
        return StretchRobotConfig(**kwargs)
    elif robot_type == "arx5":
        return ARX5SingleArmRobotConfig(**kwargs)
    elif robot_type == "arx5_bimanual":
        return ARX5BimanualRobotConfig(**kwargs)
    elif robot_type == "arx5_follow":
        return ARX5SingleArmFollowOnlyConfig(**kwargs)
    elif robot_type == "arx5_bimanual_follow":
        return ARX5BimanualFollowOnlyConfig(**kwargs)
    elif robot_type == "lekiwi":
        return LeKiwiRobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    if isinstance(config, ManipulatorRobotConfig):
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

        return ManipulatorRobot(config)
    elif isinstance(config, ARX5SingleArmRobotConfig):
        from lerobot.common.robot_devices.robots.arx5 import ARX5Robot
        common_config = ARX5RobotConfig(
            leader_arms=config.leader_arms,
            follower_arms=config.follower_arms,
            cameras=config.cameras
        )
        return ARX5Robot(common_config)
    elif isinstance(config, ARX5BimanualRobotConfig):
        from lerobot.common.robot_devices.robots.arx5 import ARX5Robot
        common_config = ARX5RobotConfig(
            leader_arms=config.leader_arms,
            follower_arms=config.follower_arms,
            cameras=config.cameras
        )
        return ARX5Robot(common_config)
    elif isinstance(config, ARX5SingleArmFollowOnlyConfig):
        from lerobot.common.robot_devices.robots.arx5 import ARX5Robot
        common_config = ARX5RobotConfig(
            leader_arms=config.leader_arms,
            follower_arms=config.follower_arms,
            cameras=config.cameras
        )
        return ARX5Robot(common_config)
    elif isinstance(config, ARX5BimanualFollowOnlyConfig):
        from lerobot.common.robot_devices.robots.arx5 import ARX5Robot
        common_config = ARX5RobotConfig(
            leader_arms=config.leader_arms,
            follower_arms=config.follower_arms,
            cameras=config.cameras
        )
        return ARX5Robot(common_config)
    elif isinstance(config, LeKiwiRobotConfig):
        from lerobot.common.robot_devices.robots.mobile_manipulator import MobileManipulator

        return MobileManipulator(config)
    else:
        from lerobot.common.robot_devices.robots.stretch import StretchRobot

        return StretchRobot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)
