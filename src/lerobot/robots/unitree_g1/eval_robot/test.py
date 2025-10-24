"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import numpy as np
import cv2
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from eval_robot.utils.utils import cleanup_resources, EvalRealConfig

from eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from eval_robot.utils.utils import to_list, to_scalar
from eval_robot.robot_control.robot_arm_test import (
    G1_29_ArmController, G1_29_JointIndex
)
import logging_mp
from eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def replay_main():

    #damp needs to be on? do i start the robot as well 

    arm_ik = G1_29_ArmIK()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=False)#motors move here upon init. there's a bug where when closing python the motors die

if __name__ == "__main__":
    replay_main()
