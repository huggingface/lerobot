"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext

from lerobot.policies.factory import make_policy
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from multiprocessing.sharedctypes import SynchronizedArray

from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def eval_policy(
    cfg: EvalRealConfig,
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    policy.reset()  # Set policy to evaluation mode

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

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

        # Get initial pose from the first step of the dataset
        episode_idx = 0
        episode_info = dataset.meta.episodes[episode_idx]

        from_idx = episode_info["dataset_from_index"]
        to_idx = episode_info["dataset_to_index"]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move
            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                #convert wrist obs to tensors 
                observation["observation.images.cam_left_wrist"] = torch.from_numpy(observation["observation.images.cam_left_wrist"])
                observation["observation.images.cam_right_wrist"] = torch.from_numpy(observation["observation.images.cam_right_wrist"])
                left_ee_state = right_ee_state = np.array([])
                #if cfg.ee:
                #    with ee_shared_mem["lock"]:
                #        full_state = np.array(ee_shared_mem["state"][:])
                #        left_ee_state = full_state[:ee_dof]
                #        right_ee_state = full_state[ee_dof:]
                #pad with zeros 
                #left_ee_state = np.zeros(ee_dof)
                #right_ee_state = np.zeros(ee_dof)
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                )
                action_np = action.cpu().numpy()
                # 3. Execute Action
                arm_action = action_np[:arm_dof]*0.1
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                # if cfg.ee:
                #     ee_action_start_idx = arm_dof
                #     left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                #     right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                #     # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                #     if isinstance(ee_shared_mem["left"], SynchronizedArray):
                #         ee_shared_mem["left"][:] = to_list(left_ee_action)
                #         ee_shared_mem["right"][:] = to_list(right_ee_action)
                #     elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                #         ee_shared_mem["left"].value = to_scalar(left_ee_action)
                #         ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg=cfg, policy=policy, dataset=dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
