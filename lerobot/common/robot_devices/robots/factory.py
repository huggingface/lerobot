def make_robot(name):
    if name == "koch":
        # TODO(rcadene): Add configurable robot from command line and yaml config
        # TODO(rcadene): Add example with and without cameras
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
        from lerobot.common.robot_devices.robots.koch import KochRobot

        robot = KochRobot(
            leader_arms={
                "main": DynamixelMotorsBus(
                    port="/dev/tty.usbmodem575E0031751",
                    motors={
                        # name: (index, model)
                        "shoulder_pan": (1, "xl330-m077"),
                        "shoulder_lift": (2, "xl330-m077"),
                        "elbow_flex": (3, "xl330-m077"),
                        "wrist_flex": (4, "xl330-m077"),
                        "wrist_roll": (5, "xl330-m077"),
                        "gripper": (6, "xl330-m077"),
                    },
                ),
            },
            follower_arms={
                "main": DynamixelMotorsBus(
                    port="/dev/tty.usbmodem575E0032081",
                    motors={
                        # name: (index, model)
                        "shoulder_pan": (1, "xl430-w250"),
                        "shoulder_lift": (2, "xl430-w250"),
                        "elbow_flex": (3, "xl330-m288"),
                        "wrist_flex": (4, "xl330-m288"),
                        "wrist_roll": (5, "xl330-m288"),
                        "gripper": (6, "xl330-m288"),
                    },
                ),
            },
            cameras={
                "laptop": OpenCVCamera(0, fps=30, width=640, height=480),
                "phone": OpenCVCamera(1, fps=30, width=640, height=480),
            },
        )
    else:
        raise ValueError(f"Robot '{name}' not found.")

    return robot
