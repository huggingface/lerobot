import logging

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robots.config import RobotMode
from lerobot.common.robots.lekiwi.configuration_daemon_lekiwi import DaemonLeKiwiRobotConfig
from lerobot.common.robots.lekiwi.daemon_lekiwi import DaemonLeKiwiRobot
from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100 import SO100Teleop, SO100TeleopConfig

DUMMY_FEATURES = {
    "observation.state": {
        "dtype": "float64",
        "shape": (9,),
        "names": {
            "motors": [
                "arm_shoulder_pan",
                "arm_shoulder_lift",
                "arm_elbow_flex",
                "arm_wrist_flex",
                "arm_wrist_roll",
                "arm_gripper",
                "base_left_wheel",
                "base_right_wheel",
                "base_back_wheel",
            ]
        },
    },
    "action": {
        "dtype": "float64",
        "shape": (9,),
        "names": {
            "motors": [
                "arm_shoulder_pan",
                "arm_shoulder_lift",
                "arm_elbow_flex",
                "arm_wrist_flex",
                "arm_wrist_roll",
                "arm_gripper",
                "base_left_wheel",
                "base_right_wheel",
                "base_back_wheel",
            ]
        },
    },
    "observation.images.front": {
        "dtype": "image",
        "shape": (640, 480, 3),
        "names": [
            "width",
            "height",
            "channels",
        ],
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": [
            "width",
            "height",
            "channels",
        ],
    },
}


def main():
    logging.info("Configuring Teleop Devices")
    leader_arm_config = SO100TeleopConfig(port="/dev/tty.usbmodem58760429271")
    leader_arm = SO100Teleop(leader_arm_config)

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    logging.info("Configuring LeKiwiRobot Daemon")
    robot_config = DaemonLeKiwiRobotConfig(
        id="daemonlekiwi", calibration_dir=".cache/calibration/lekiwi", robot_mode=RobotMode.TELEOP
    )
    robot = DaemonLeKiwiRobot(robot_config)

    logging.info("Creating LeRobot Dataset")

    # TODO(Steven): Check this creation
    dataset = LeRobotDataset.create(
        repo_id="user/lekiwi",
        fps=10,
        features=DUMMY_FEATURES,
    )

    logging.info("Connecting Teleop Devices")
    leader_arm.connect()
    keyboard.connect()

    logging.info("Connecting remote LeKiwiRobot")
    robot.connect()

    logging.info("Starting LeKiwiRobot teleoperation")
    i = 0
    while i < 1000:
        arm_action = leader_arm.get_action()
        base_action = keyboard.get_action()
        action = np.append(arm_action, base_action) if base_action.size > 0 else arm_action

        # TODO(Steven): Deal with policy action space
        # robot.set_mode(RobotMode.AUTO)
        # policy_action = policy.get_action() # This might be in body frame, key space or smt else
        # robot.send_action(policy_action)

        action_sent = robot.send_action(action)
        observation = robot.get_observation()

        frame = {"action": action_sent}
        frame.update(observation)
        frame.update({"task": "Dummy Task Dataset"})

        logging.info("Saved a frame into the dataset")
        dataset.add_frame(frame)
        i += 1

    dataset.save_episode()
    # dataset.push_to_hub()

    logging.info("Disconnecting Teleop Devices and LeKiwiRobot Daemon")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    logging.info("Finished LeKiwiRobot cleanly")


if __name__ == "__main__":
    main()
