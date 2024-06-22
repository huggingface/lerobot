

import time
from lerobot.common.robot_devices.robots.aloha import AlohaRobot


def teleoperate():
    robot = AlohaRobot(use_cameras=False)
    robot.init_teleop()

    while True:
        now = time.time()
        robot.teleop_step(record_data=False)

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


if __name__ == "__main__":
    teleoperate()