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
from lerobot.robots.unitree_g1.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from lerobot.robots.unitree_g1.eval_robot.utils.utils import cleanup_resources, EvalRealConfig

from lerobot.robots.unitree_g1.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from lerobot.robots.unitree_g1.eval_robot.utils.utils import to_list, to_scalar
from lerobot.robots.unitree_g1.eval_robot.robot_control.robot_arm_test import (
    G1_29_ArmController, G1_29_JointIndex
)
import logging_mp
from lerobot.robots.unitree_g1.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def replay_main():

    #damp needs to be on? do i start the robot as well 

    arm_ik = G1_29_ArmIK()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=False)#motors move here upon init. there's a bug where when closing python the motors die
    #arm_ctrl.ctrl_dual_arm_go_home()
    dataset = LeRobotDataset(repo_id='nepyope/unitree_box_push_30fps_actions_fix', root="", episodes=[0])
    actions = dataset.hf_dataset.select_columns("action")
    print(actions)
    episode_idx = 8
    episode_info = dataset.meta.episodes[episode_idx]

    from_idx = episode_info["dataset_from_index"]
    to_idx = episode_info["dataset_to_index"]
    step = dataset[from_idx]
    init_left_arm_pose = step["observation.state"][:14].cpu().numpy()
    tau = arm_ik.solve_tau(init_left_arm_pose)
    #arm_ctrl.ctrl_dual_arm(init_left_arm_pose, tau)
    print('ok')

    # Create config object for image client
    import argparse
    cfg = argparse.Namespace(
        sim=False,      # Real robot (not simulation)
        arm="G1_29",
        ee="dex3",
        motion=False
    )
    
    #replay actions from the dataset
    for idx in range(dataset.num_frames):

            arm_joint_indices = set(range(15, 29))  # 15–28 are arms
            for jid in G1_29_JointIndex:
                if jid.value not in arm_joint_indices:
                    arm_ctrl.msg.motor_cmd[jid].mode = 1
                    arm_ctrl.msg.motor_cmd[jid].q = 0.0
                    arm_ctrl.msg.motor_cmd[jid].dq = 0.0
                    arm_ctrl.msg.motor_cmd[jid].tau = 0.0
            loop_start_time = time.perf_counter()

            action_np = actions[idx]["action"].numpy()

            # exec action
            arm_action = action_np[:14]
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)
            logger_mp.info(f"arm_action {arm_action}, tau {tau}")

            # Maintain frequency
            time.sleep(max(0, (1.0 / 30) - (time.perf_counter() - loop_start_time)))
    
    #some thread issue idk motion_mode true gets rid of the shakes motion mode 

if __name__ == "__main__":
    replay_main()
