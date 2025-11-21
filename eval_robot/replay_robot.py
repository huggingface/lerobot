"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import numpy as np

from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import cleanup_resources, EvalRealConfig

from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import to_list, to_scalar

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


@parser.wrap()
def replay_main(cfg: EvalRealConfig):
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    image_info = setup_image_client(cfg)
    robot_interface = setup_robot_interface(cfg)

    """The main control and evaluation loop."""
    # Unpack interfaces for convenience
    arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
        robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
    )
    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
        image_info[key]
        for key in [
            "tv_img_array",
            "wrist_img_array",
            "tv_img_shape",
            "wrist_img_shape",
            "is_binocular",
            "has_wrist_cam",
        ]
    )

    logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root, episodes=[cfg.episodes])
    actions = dataset.hf_dataset.select_columns("action")

    # init pose
    from_idx = dataset.episode_data_index["from"][0].item()
    step = dataset[from_idx]
    init_left_arm_pose = step["observation.state"][:14].cpu().numpy()

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == "s":
        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        logger_mp.info("Initializing robot to starting pose...")
        tau = arm_ik.solve_tau(init_left_arm_pose)
        arm_ctrl.ctrl_dual_arm(init_left_arm_pose, tau)
        time.sleep(1)
        for idx in range(dataset.num_frames):
            loop_start_time = time.perf_counter()

            left_ee_state = right_ee_state = np.array([])
            action_np = actions[idx]["action"].numpy()

            # exec action
            arm_action = action_np[:arm_dof]
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)
            logger_mp.info(f"arm_action {arm_action}, tau {tau}")

            if cfg.ee:
                ee_action_start_idx = arm_dof
                left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]

                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:] = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                state = np.concatenate((current_arm_q, left_ee_state, right_ee_state))

                visualization_data(idx, observation, state, action_np, rerun_logger)

            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    cleanup_resources(image_info)


if __name__ == "__main__":
    replay_main()
