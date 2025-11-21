"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging
import sys
import os

import numpy as np

# Suppress DDS debug output
class SuppressDDSOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
)
from unitree_lerobot.eval_robot.utils.sim_savedata_utils import (
    EvalRealConfig,
    process_data_add,
    is_success,
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

    #logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    policy.reset()  # Set policy to evaluation mode

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        
        # Suppress DDS debug output during robot setup
        robot_interface = setup_robot_interface(cfg)
        # Unpack interfaces for convenience
        (
            arm_ctrl,
            arm_ik,
            ee_shared_mem,
            arm_dof,
            ee_dof,
            sim_state_subscriber,
            sim_reward_subscriber,
            episode_writer,
            reset_pose_publisher,
        ) = (
            robot_interface[key]
            for key in [
                "arm_ctrl",
                "arm_ik",
                "ee_shared_mem",
                "arm_dof",
                "ee_dof",
                "sim_state_subscriber",
                "sim_reward_subscriber",
                "episode_writer",
                "reset_pose_publisher",
            ]
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
        #is_binocular = True#add flag properly

        # Get initial pose from the first step of the dataset
        from_idx = dataset.episode_data_index["from"][0].item()
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        idx = 0
        full_state = None

        reward_stats = {
            "reward_sum": 0.0,
            "episode_num": 0.0,
        }

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        #logger_mp.info("Initializing robot to starting pose...")
        tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
        robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)#hits init pose just fine
        time.sleep(1.0)  # Give time for the robot to move

        # --- Run Main Loop ---
        #logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
        while True:
            if cfg.save_data:
                if reward_stats["episode_num"] == 0:
                    episode_writer.create_episode()
            loop_start_time = time.perf_counter()

            # 1. Get Observations
            observation, current_arm_q = process_images_and_observations(
                tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
            )
    

            left_ee_state = right_ee_state = np.array([])
            if cfg.ee:
               with ee_shared_mem["lock"]:
                  full_state = np.array(ee_shared_mem["state"][:])
                  left_ee_state = full_state[:ee_dof]
                  right_ee_state = full_state[ee_dof:]
            state_tensor = torch.from_numpy(
                np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
            ).float()
            observation["observation.state"] = state_tensor
            # 2. Get Action from Policy
            #action_np = np.random.random(arm_dof)
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
            arm_action = action_np[:arm_dof]
            #arm action is random vector

            tau = arm_ik.solve_tau(arm_action)

            # Suppress DDS debug output during control commands
            arm_ctrl.ctrl_dual_arm(arm_action, tau)

            if cfg.ee:
                ee_action_start_idx = arm_dof
                left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:] = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)
            # save data
            #if cfg.save_data:
                # process_data_add(episode_writer, observation, current_arm_q, full_state, action, arm_dof, ee_dof)

                # is_success(
                #     sim_reward_subscriber,
                #     episode_writer,
                #     reset_pose_publisher,
                #     policy,
                #     cfg,
                #     reward_stats,
                #     init_arm_pose,
                #     robot_interface,
                # )

            if cfg.visualization:
                visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
            idx += 1
            reward_stats["episode_num"] = reward_stats["episode_num"] + 1
            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
        pass
    finally:
        if image_info:
            cleanup_resources(image_info)
        # Clean up sim state subscriber if it exists
    if sim_state_subscriber and not getattr(sim_state_subscriber, "stopped", False):
        sim_state_subscriber.stop_subscribe()
    if sim_reward_subscriber and not getattr(sim_reward_subscriber, "stopped", False):
        sim_reward_subscriber.stop_subscribe()


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
