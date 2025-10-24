"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
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

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.make_robot import setup_robot_interface
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

    # init pose
    from_idx = dataset.episode_data_index["from"][0].item()
    step = dataset[from_idx]
    to_idx = dataset.episode_data_index["to"][0].item()

    ground_truth_actions = []
    predicted_actions = []

    if cfg.send_real_robot:
        robot_interface = setup_robot_interface(cfg)
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

    # ===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == "s":
        if cfg.send_real_robot:
            # Initialize robot to starting pose
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)

            time.sleep(1)

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            loop_start_time = time.perf_counter()

            step = dataset[step_idx]
            observation = extract_observation(step)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
            )
            action_np = action.cpu().numpy()

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action_np)

            if cfg.send_real_robot:
                # Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                # logger_mp.info(f"Arm Action: {arm_action}")

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                visualization_data(step_idx, observation, observation["observation.state"], action_np, rerun_logger)

            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)

        # Get the number of timesteps and action dimensions
        n_timesteps, n_dims = ground_truth_actions.shape

        # Create a figure with subplots for each action dimension
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
        fig.suptitle("Ground Truth vs Predicted Actions")

        # Plot each dimension
        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes

            ax.plot(ground_truth_actions[:, i], label="Ground Truth", color="blue")
            ax.plot(predicted_actions[:, i], label="Predicted", color="red", linestyle="--")
            ax.set_ylabel(f"Dim {i + 1}")
            ax.legend()

        # Set common x-label
        axes[-1].set_xlabel("Timestep")

        plt.tight_layout()
        # plt.show()

        time.sleep(1)
        plt.savefig("figure.png")


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
        eval_policy(cfg, policy, dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
