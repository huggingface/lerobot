# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
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


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)

    # --- DEBUG: Log raw observation ---
    if not hasattr(predict_action, "_step_count"):
        predict_action._step_count = 0
    predict_action._step_count += 1
    _step = predict_action._step_count
    _debug = (_step <= 3) or (_step % 50 == 0)  # Log first 3 steps, then every 50

    if _debug:
        print(f"\n{'='*80}")
        print(f"[INFERENCE DEBUG] Step {_step}")
        print(f"{'='*80}")
        print(f"[STEP 1] Raw observation keys: {list(observation.keys())}")
        for k, v in observation.items():
            if isinstance(v, np.ndarray):
                if "image" in k:
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.2f}, {v.max():.2f}]")
                elif "state" in k:
                    print(f"  {k}: shape={v.shape}, values={v.flatten()[:6]}")
                else:
                    print(f"  {k}: shape={v.shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

        # Save images for first step
        if _step == 1:
            import os
            debug_dir = "/tmp/lerobot_inference_debug"
            os.makedirs(debug_dir, exist_ok=True)
            for k, v in observation.items():
                if isinstance(v, np.ndarray) and "image" in k:
                    try:
                        from PIL import Image
                        img = Image.fromarray(v)
                        save_path = os.path.join(debug_dir, f"step{_step}_{k.replace('.', '_')}.jpg")
                        img.save(save_path)
                        print(f"  -> Saved image to {save_path}")
                    except Exception as e:
                        print(f"  -> Failed to save image: {e}")

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)

        if _debug:
            print(f"\n[STEP 2] After prepare_observation_for_inference:")
            for k, v in observation.items():
                if isinstance(v, torch.Tensor):
                    if "image" in k:
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]")
                    elif "state" in k:
                        print(f"  {k}: shape={v.shape}, values={v.flatten()[:6].cpu().numpy()}")
                    else:
                        print(f"  {k}: shape={v.shape}")
                else:
                    print(f"  {k}: {type(v).__name__} = {v}")

        observation = preprocessor(observation)

        if _debug:
            print(f"\n[STEP 3] After preprocessor:")
            for k, v in observation.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() < 100:
                        print(f"  {k}: shape={v.shape}, values={v.flatten()[:10].cpu().numpy()}")
                    else:
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"  {k}: list of {len(v)} items")
                    for i, item in enumerate(v):
                        if isinstance(item, torch.Tensor):
                            print(f"    [{i}]: shape={item.shape}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with keys {list(v.keys())}")
                else:
                    print(f"  {k}: {type(v).__name__}")

            # Check if raw_state made it through
            if "raw_state" in observation:
                rs = observation["raw_state"]
                print(f"\n  raw_state present in preprocessor output!")
                if isinstance(rs, dict):
                    for rk, rv in rs.items():
                        print(f"    {rk}: shape={rv.shape}, values={rv.flatten()[:6]}")

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        if _debug:
            print(f"\n[STEP 4] After policy.select_action:")
            print(f"  action shape: {action.shape}")
            print(f"  action values (normalized): {action.flatten()[:6].cpu().numpy()}")
            print(f"  action range: [{action.min():.4f}, {action.max():.4f}]")

        action = postprocessor(action)

        if _debug:
            print(f"\n[STEP 6] After postprocessor (FINAL action):")
            print(f"  action shape: {action.shape}")
            print(f"  action values (absolute): {action.flatten()[:6].cpu().numpy()}")
            print(f"  action range: [{action.min():.4f}, {action.max():.4f}]")
            print(f"{'='*80}\n")

    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
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


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
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
