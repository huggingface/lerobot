import time

# from safetensors.torch import load_file, save_file
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging

if __name__ == "__main__":
    init_logging()

    control_mode = "test"
    robot_path = "lerobot/configs/robot/reachy2.yaml"
    robot_overrides = None

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    print(robot.is_connected)
    # print(robot.get_state())
    print(robot.capture_observation())
    time.sleep(5)
    robot.disconnect()
