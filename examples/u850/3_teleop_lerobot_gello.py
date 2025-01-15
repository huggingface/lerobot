import time
import numpy as np

from wasabi import color

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.motors.ufactory import XArmWrapper
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.gello import GelloDynamixelWrapper
# from lerobot.common.robot_devices.motors.gello_aux import GelloDynamixelWrapper


# Defines how to communicate with the motors of the leader and follower arms
leader_arms = {
    "left": GelloDynamixelWrapper(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9BTDLW-if00-port0",
        motors={
            # name: (index, model)
            "joint1": (1, "xl330-m288"),
            "joint2": (2, "xl330-m288"), 
            "joint3": (3, "xl330-m288"),
            "joint4": (4, "xl330-m288"),
            "joint5": (5, "xl330-m288"),
            "joint6": (6, "xl330-m288"),
            "gripper": (7, "xl330-m077"),
        },
    ),
    "right": GelloDynamixelWrapper(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9BTEGF-if00-port0",
        motors={
            # name: (index, model)
            "joint1": (1, "xl330-m288"),
            "joint2": (2, "xl330-m288"), 
            "joint3": (3, "xl330-m288"),
            "joint4": (4, "xl330-m288"),
            "joint5": (5, "xl330-m288"),
            "joint6": (6, "xl330-m288"),
            "gripper": (7, "xl330-m077"),
        },
    ),    
}
follower_arms = {
    "left": XArmWrapper(
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
    "right": XArmWrapper(
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
    # cameras={
    #     "top": OpenCVCamera(4, fps=30, width=640, height=480),
    #     "wrist": OpenCVCamera(10, fps=30, width=640, height=480),
    # },
)

# Connect motors buses and cameras if any (Required)
robot.connect()

# print(robot.leader_arms["main"].get_position())
# print("---")
# print(robot.follower_arms["main"].get_position())

try:
    import time

    start_time = time.time()
    iteration_count = 0

    while True:
        robot.teleop_step()
        # time.sleep(0.033)  # 30 Hz -> barely smooth
        # time.sleep(0.004)  # 250 Hz -> very smooth

        iteration_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:  # Print frequency every second
            frequency = iteration_count / elapsed_time
            print(f"Current teleoperation frequency: {frequency:.2f} Hz")
            start_time = time.time()
            iteration_count = 0

        # # Recording data, only joints
        # leader_pos = robot.leader_arms["main"].get_position()
        # follower_pos = robot.follower_arms["main"].get_position()
        # observation, action = robot.teleop_step(record_data=True)

        # print(follower_pos)
        # print(observation)
        # print(leader_pos)
        # print(action)
        # print("---")

        # Recording data with cameras
        # observation, action = robot.teleop_step(record_data=True)
        # print(observation["observation.images.top"].shape)
        # print(observation["observation.images.wrist"].shape)
        # print(observation["observation.images.top"].min().item())
        # print(observation["observation.images.top"].max().item())
        # print("---")


except KeyboardInterrupt:
    print("Operation interrupted by user.")


print(color("Disconnecting...", "red"))
robot.disconnect()
