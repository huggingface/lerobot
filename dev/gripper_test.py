import sys
import time
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower


def main() -> None:
    cfg = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="follower0",
        max_relative_target=None,
    )

    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)

    robot.bus.enable_torque("gripper")

    present = robot.bus.read("Present_Position", "gripper", normalize=False)
    model = robot.bus.motors["gripper"].model
    max_res = robot.bus.model_resolution_table[model] - 1

    delta = 200
    target = present + delta
    if target < 0:
        target = 0
    if target > max_res:
        target = max_res

    robot.bus.write("Goal_Position", "gripper", target, normalize=False)

    time.sleep(1.0)

    robot.disconnect()


if __name__ == "__main__":
    main()


