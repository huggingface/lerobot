from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

robot_config = SO101FollowerConfig(
    port="COM6",
    id="DI_VLA_FOLLOWER",
)

teleop_config = SO101LeaderConfig(
    port="COM5",
    id="DI_VLA_LEADER",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action: dict[str, float] = teleop_device.get_action()
    robot.send_action(action)
