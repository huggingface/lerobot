import lerobot
from lerobot.common.robot_devices.motors.ufactory import xArmWrapper
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

# Defines how to communicate with the motors of the leader and follower arms
leader_arms = {
    "main": lerobot.common.robot_devices.motors.ufactory.xArmWrapper(
        port="192.168.1.236",
        motors={
            # name: (index, model)
            "joint1": (1, "ufactory-850"),
            "joint2": (2, "ufactory-850"),
            "joint3": (3, "ufactory-850"),
            "joint4": (4, "ufactory-850"),
            "joint5": (5, "ufactory-850"),
            "joint6": (6, "ufactory-850"),
            "gripper": (7, "ufactory-850"),
        },
    ),
}
follower_arms = {
    "main": lerobot.common.robot_devices.motors.ufactory.xArmWrapper(
        port="192.168.1.218",
        motors={
            # name: (index, model)
            "joint1": (1, "ufactory-850"),
            "joint2": (2, "ufactory-850"),
            "joint3": (3, "ufactory-850"),
            "joint4": (4, "ufactory-850"),
            "joint5": (5, "ufactory-850"),
            "joint6": (6, "ufactory-850"),
            "gripper": (7, "ufactory-850"),
        },
    ),
}
robot = ManipulatorRobot(
    robot_type="u850",
    calibration_dir=".cache/calibration/u850",
    leader_arms=leader_arms,
    follower_arms=follower_arms,
)

# Connect motors buses and cameras if any (Required)
robot.connect()

try:
    while True:
        robot.teleop_step()
except KeyboardInterrupt:
    print('Operation interrupted by user.')

robot.disconnect()
