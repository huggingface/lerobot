import logging
import time

import numpy as np
import rerun as rr
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device


def lekiwi_record_loop(
    robot: LeKiwiClient,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop_arm: Teleoperator | None = None,
    teleop_keyboard: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    log_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation = robot.get_observation()

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        if policy is not None:
            arm_action, base_action = {}, {}
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None and teleop_arm and teleop_keyboard is not None:
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)

            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)

        if log_data:
            display_data(observation, arm_action, base_action)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


def display_data(observation, arm_action, base_action):
    """Display all data in Rerun."""

    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation_{obs}", rr.Scalars(val))
        elif isinstance(val, (np.ndarray, torch.Tensor)):
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            if len(val.shape) == 1:  # 1D array - log as individual scalars
                for i, v in enumerate(val):
                    rr.log(f"observation_{obs}_{i}", rr.Scalars(v))
            else:  # 2D or 3D array - log as image
                rr.log(f"observation_{obs}", rr.Image(val), static=True)

    # Log arm actions
    for act, val in arm_action.items():
        if isinstance(val, float):
            rr.log(f"action_{act}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action_{act}_{i}", rr.Scalars(v))

    # Log base actions
    for act, val in base_action.items():
        if isinstance(val, float):
            rr.log(f"base_action_{act}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"base_action_{act}_{i}", rr.Scalars(v))
