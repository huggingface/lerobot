from lerobot.common.robot_devices.motors.feetech import *
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


so_arm100_configs=So100RobotConfig()

config = FeetechMotorsBusConfig(
    port=so_arm100_configs.follower_arms["main"].port,
    motors=so_arm100_configs.follower_arms["main"].motors
)

robot = ManipulatorRobot(so_arm100_configs)

robot.connect()

follower_pos = robot.follower_arms["main"].read("Present_Position")

print(follower_pos)

follower_pos -=[-39.63867,147.30469,157.06055,21.269531,-113.90625,6.626506]
robot.follower_arms["main"].write("Goal_Position", follower_pos)

# 开启力矩自锁
# robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
# robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)


robot.disconnect()
