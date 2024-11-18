
import argparse
import logging
import time
from pathlib import Path
from typing import List

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import (create_lerobot_dataset,
                                                      delete_current_episode,
                                                      init_dataset,
                                                      save_current_episode)
from lerobot.common.robot_devices.control_utils import (
    control_loop, has_method, init_keyboard_listener, init_policy,
    log_control_info, record_episode, reset_environment,
    sanity_check_dataset_name, stop_recording, warmup_record)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import (init_hydra_config, init_logging,
                                        log_say, none_or_int)


import cv2
if __name__ == '__main__':
    init_logging()

    control_mode = "test"
    robot_path = "lerobot/configs/robot/reachy2.yaml"
    robot_overrides = None

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    print(robot.get_state())
    print(robot.capture_observation())
