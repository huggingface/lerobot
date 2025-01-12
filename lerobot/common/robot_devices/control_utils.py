########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any, Dict, List, Optional

import torch
import tqdm
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path


@dataclass
class LogItem:
    name: str
    value: float
    unit: str
    color: str = "white"

    def to_dict(self):
        return asdict(self)

def stringify_and_log(log_items: List[LogItem]):
    parts = []
    for item in log_items:
        if item.unit:
            info_str = f"{item.name}:{item.value:.2f} {item.unit}"
        else:
            info_str = f"{item.name}:{int(item.value)}"
        
        if item.color != "white":
            info_str = colored(info_str, item.color)
        
        parts.append(info_str)
    
    info_str = " ".join(parts)
    logging.info(info_str)

def serialize_log_items(log_items: List[LogItem]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in log_items]

def log_control_info(robot: Robot, dt_s: float, fps: Optional[float] = None,
                    episode_index: Optional[int] = None,
                    frame_index: Optional[int] = None) -> List[LogItem]:
    log_items: List[LogItem] = []

    # Add episode and frame information if provided
    if episode_index is not None:
        log_items.append(LogItem(name="ep", value=float(episode_index), unit=""))
    if frame_index is not None:
        log_items.append(LogItem(name="frame", value=float(frame_index), unit=""))

    # Helper function to create LogItem instances
    def create_log_item(shortname: str, dt_val_s: float, base_fps: Optional[float]) -> LogItem:
        value_ms = dt_val_s * 1000
        frequency = 1 / dt_val_s if dt_val_s > 0 else 0.0
        unit = f"ms ({frequency:.1f}Hz)"
        color = "white"
        if base_fps is not None and frequency < (base_fps - 1):
            color = "yellow"
        return LogItem(name=shortname, value=value_ms, unit=unit, color=color)

    # Log total step time
    log_items.append(create_log_item("dt", dt_s, fps))

    # Robot-specific logs
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_items.append(create_log_item("dtRlead", robot.logs[key], fps))

        for name in robot.follower_arms:
            key_write = f"write_follower_{name}_goal_pos_dt_s"
            if key_write in robot.logs:
                log_items.append(create_log_item("dtWfoll", robot.logs[key_write], fps))

            key_read = f"read_follower_{name}_pos_dt_s"
            if key_read in robot.logs:
                log_items.append(create_log_item("dtRfoll", robot.logs[key_read], fps))

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_items.append(create_log_item(f"dtR{name}", robot.logs[key], fps))

    stringify_and_log(log_items)
    return log_items



@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def has_method(_object: object, method_name: str):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def init_policy(pretrained_policy_name_or_path, policy_overrides):
    """Instantiate the policy and load fps, device and use_amp from config yaml"""
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)
    use_amp = hydra_cfg.use_amp
    policy_fps = hydra_cfg.env.fps

    policy.eval()
    policy.to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)
    return policy, policy_fps, device, use_amp


def warmup_record(
    robot,
    enable_teleoperation,
    warmup_time_s,
    fps,
    control_context
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        fps=fps,
        teleoperate=enable_teleoperation,
        control_context=control_context,
    )


def record_episode(
    robot,
    dataset,
    episode_time_s,
    policy,
    device,
    use_amp,
    fps,
    control_context
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        dataset=dataset,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
        control_context=control_context,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    dataset: LeRobotDataset | None = None,
    policy=None,
    device=None,
    use_amp=None,
    fps=None,
    control_context=None,
):
    events = control_context.get_events() if control_context is not None else None

    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    total_time = 0
    try:
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if teleoperate:
                observation, action = robot.teleop_step(record_data=True)
            else:
                observation = robot.capture_observation()

                if policy is not None:
                    pred_action = predict_action(observation, policy, device, use_amp)
                    # Action can eventually be clipped using `max_relative_target`,
                    # so action actually sent is saved in the dataset.
                    action = robot.send_action(pred_action)
                    action = {"action": action}

            if dataset is not None:
                frame = {**observation, **action}
                dataset.add_frame(frame)

            timestamp = time.perf_counter() - start_episode_t
            total_time += timestamp
            countdown_time = max(0, control_time_s - timestamp)

            control_context.update_with_observations(observation, start_loop_t, countdown_time)

            if events["exit_early"]:
                events["exit_early"] = False
                break

    except Exception as e:
        print(f"Error in control loop: {e}")


def reset_environment(robot, control_context, reset_time_s):
    # TODO(rcadene): refactor warmup_record and reset_environment
    # TODO(alibets): allow for teleop during reset
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    events = control_context.get_events()

    timestamp = 0
    start_vencod_t = time.perf_counter()

    # Wait if necessary
    with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
        while timestamp < reset_time_s:
            time.sleep(1)
            timestamp = time.perf_counter() - start_vencod_t
            countdown_time = max(0, reset_time_s - timestamp)
            control_context.update_with_observations(None, 0, countdown_time)
            pbar.update(1)

            if events["exit_early"]:
                events["exit_early"] = False
                break

def sanity_check_dataset_name(repo_id, policy):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
