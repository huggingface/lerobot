########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import cv2
import torch
import tqdm
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path

import pygame
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Create an enum for ControlPhase
class ControlPhase:
    TELEOPERATE = "Teleoperate"
    WARMUP = "Warmup"
    RECORD = "Record"
    RESET = "reset"

@dataclass
class ControlContextConfig:
    display_cameras: bool = False
    play_sounds: bool = False
    assign_rewards: bool = False
    debug_mode: bool = False
    control_phase: ControlPhase = ControlPhase.TELEOPERATE
    num_episodes: int = 0

class ControlContext:
    def __init__(self, config: Optional[ControlContextConfig] = None):
        self.config = config or ControlContextConfig()
        pygame.init()
        if not self.config.display_cameras:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)

        self.screen = None
        self.image_positions = {}
        self.padding = 20
        self.title_height = 30
        self.controls_width = 300  # Width of the controls panel
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "next_reward": 0
        }

        if config.assign_rewards:
            self.events["next_reward"] = 0

        self.pressed_keys = []
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)  # Smaller font for controls list
        self.current_episode_index = 0
        
        # Color theme
        self.text_bg_color = (0, 0, 0)  # Black
        self.text_color = (0, 255, 0)  # Green
        
        # Define the control instructions
        self.controls = [
            ("Right Arrow", "Exit current loop"),
            ("Left Arrow", "Rerecord last episode"),
            ("Escape", "Stop recording"),
            ("Space", "Toggle reward (if enabled)"),
        ]

    def calculate_window_size(self, images: Dict[str, np.ndarray]):
        """Calculate required window size based on images"""
        max_width = 0
        max_height = 0
        n_images = len(images)
        
        # Calculate grid dimensions
        grid_cols = min(2, n_images)
        grid_rows = (n_images + 1) // 2
        
        for image in images.values():
            max_width = max(max_width, image.shape[1])
            max_height = max(max_height, image.shape[0])
            
        # Adjust total width and height calculations to remove extra padding
        total_width = max_width * grid_cols + self.controls_width
        total_height = max_height * grid_rows + self.title_height
        
        return total_width, total_height, grid_cols

    def render_controls_panel(self, window_width: int, window_height: int):
        """Render the controls panel on the right side"""
        # Draw controls background
        controls_rect = pygame.Rect(
            window_width - self.controls_width, 
            self.title_height,
            self.controls_width, 
            window_height - self.title_height
        )
        pygame.draw.rect(self.screen, self.text_bg_color, controls_rect)
        pygame.draw.line(self.screen, self.text_color, 
                        (controls_rect.left, self.title_height),
                        (controls_rect.left, window_height), 2)

        # Draw "Controls" header
        header = self.font.render("Controls", True, self.text_color)
        header_rect = header.get_rect(
            centerx=window_width - self.controls_width/2,
            top=self.title_height + 10
        )
        self.screen.blit(header, header_rect)

        # Draw control instructions
        y_pos = header_rect.bottom + 20
        for key, action in self.controls:
            # Draw key
            key_surface = self.small_font.render(key, True, self.text_color)
            key_rect = key_surface.get_rect(
                left=window_width - self.controls_width + 20,
                top=y_pos
            )
            self.screen.blit(key_surface, key_rect)

            # Draw action
            action_surface = self.small_font.render(action, True, self.text_color)
            action_rect = action_surface.get_rect(
                left=key_rect.right + 10,
                top=y_pos
            )
            self.screen.blit(action_surface, action_rect)

            y_pos += 30

        # Add status information below controls
        y_pos += 20  # Add some spacing
        
        # Control phase
        phase_text = f"Control Phase: {self.config.control_phase}"
        phase_surface = self.small_font.render(phase_text, True, self.text_color)
        phase_rect = phase_surface.get_rect(
            left=window_width - self.controls_width + 20,
            top=y_pos
        )
        self.screen.blit(phase_surface, phase_rect)
        
        # Pressed keys
        y_pos += 30
        keys_text = f"Pressed: {', '.join(self.pressed_keys)}" if self.pressed_keys else "Pressed: None"
        keys_surface = self.small_font.render(keys_text, True, self.text_color)
        keys_rect = keys_surface.get_rect(
            left=window_width - self.controls_width + 20,
            top=y_pos
        )
        self.screen.blit(keys_surface, keys_rect)

    def handle_events(self):
        """Handle pygame events and update internal state"""
        for event in pygame.event.get():
            if self.config.debug_mode:
                print(event)
            if event.type == pygame.QUIT:
                self.events["stop_recording"] = True
                self.events["exit_early"] = True
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                self.pressed_keys.append(key_name)
                
                if event.key == pygame.K_RIGHT:
                    print("Right arrow key pressed. Exiting loop...")
                    self.events["exit_early"] = True
                elif event.key == pygame.K_LEFT:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    self.events["rerecord_episode"] = True
                    self.events["exit_early"] = True
                elif event.key == pygame.K_ESCAPE:
                    print("Escape key pressed. Stopping data recording...")
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
                elif self.config.assign_rewards and event.key == pygame.K_SPACE:
                    self.events["next_reward"] = 1 if self.events["next_reward"] == 0 else 0
                    print(f"Space key pressed. New reward: {self.events['next_reward']}")
            
            elif event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                if key_name in self.pressed_keys:
                    self.pressed_keys.remove(key_name)
        
        return self.events
        
    def render_camera_frames(self, observation: Dict[str, np.ndarray]):
        """Update display with new images from observation dict"""
        image_keys = [key for key in observation if "image" in key]
        images = {k: observation[k].numpy() for k in image_keys}
        if not images:
            return

        # Initialize or resize window if needed
        window_width, window_height, grid_cols = self.calculate_window_size(images)
        if self.screen is None or self.screen.get_size() != (window_width, window_height):
            self.screen = pygame.display.set_mode((window_width, window_height))

            # @TODO - label this window with the camera name
            pygame.display.set_caption("Camera 0")

        self.screen.fill(self.text_bg_color)

        # Update image positions and draw images
        for idx, (key, image) in enumerate(images.items()):
            # Calculate grid position
            col = idx % grid_cols
            row = idx // grid_cols
            
            # Calculate pixel position - adjust for controls panel
            x = col * (image.shape[1] + self.padding)
            y = row * (image.shape[0] + self.title_height + self.padding)
            
            # Convert numpy array to pygame surface
            image_surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
            self.screen.blit(image_surface, (x, y + self.title_height))

        pygame.draw.rect(self.screen, self.text_bg_color, (0, 0, window_width, self.title_height))
        
        # Prepare top bar text
        top_text_str = ""
        if self.config.control_phase == ControlPhase.RECORD:
            top_text_str = f"Episode: {self.current_episode_index}/{self.config.num_episodes}"
            if self.config.assign_rewards:
                next_reward = self.events["next_reward"]
                top_text_str += f" | Reward: {next_reward}"

        top_text = self.font.render(top_text_str, True, self.text_color)            
        text_rect = top_text.get_rect(center=(window_width//2, self.title_height//2))
        self.screen.blit(top_text, text_rect)

        # Draw controls panel
        self.render_controls_panel(window_width, window_height)

        pygame.display.flip()
        
    def update(self, observation: Dict[str, np.ndarray]):
        if self.config.display_cameras:
            self.render_camera_frames(observation)
        self.handle_events()
        return self
    
    def update_current_episode(self, episode_index):
        self.current_episode_index = episode_index
        return self
        
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()

    def get_events(self):
        """Return current events state"""
        return self.events.copy()

# @TODO(jackvial): Move this to the control context and make configurable
def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


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


def init_keyboard_listener(assign_rewards=False):
    """
    Initializes a keyboard listener to enable early termination of an episode 
    or environment reset by pressing the right arrow key ('->'). This may require 
    sudo permissions to allow the terminal to monitor keyboard events.

    Args:
        assign_rewards (bool): If True, allows annotating the collected trajectory 
        with a binary reward at the end of the episode to indicate success.
    """
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    if assign_rewards:
        events["next.reward"] = 0

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
            elif assign_rewards and key == keyboard.Key.space:
                events["next.reward"] = 1 if events["next.reward"] == 0 else 0
                print(
                    "Space key pressed. Assigning new reward to the subsequent frames. New reward:",
                    events["next.reward"],
                )

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
    robot=None,
    enable_teleoperation=False,
    warmup_time_s=10,
    fps=30,
    control_context: ControlContext = None,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        control_context=control_context,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    episode_time_s,
    control_context,
    policy,
    device,
    use_amp,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        control_context=control_context,
        dataset=dataset,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    control_context: ControlContext = None,
    dataset: LeRobotDataset | None = None,
    policy=None,
    device=None,
    use_amp=None,
    fps=None,
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

            control_context.update(observation)

            if fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            # log_control_info(robot, dt_s, fps=fps)

            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"]:
                events["exit_early"] = False
                break
    except Exception as e:
        print(f"Error in control loop: {e}")
    finally:
        # Clean up display window
        if control_context is not None:
            control_context.close()
        


def reset_environment(robot, events, reset_time_s):
    # TODO(rcadene): refactor warmup_record and reset_environment
    # TODO(alibets): allow for teleop during reset
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    timestamp = 0
    start_vencod_t = time.perf_counter()
    if "next.reward" in events:
        events["next.reward"] = 0

    # Wait if necessary
    with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
        while timestamp < reset_time_s:
            time.sleep(1)
            timestamp = time.perf_counter() - start_vencod_t
            pbar.update(1)
            if events["exit_early"]:
                events["exit_early"] = False
                break


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


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
