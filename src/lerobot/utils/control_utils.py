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
import time
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
from collections import deque


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

# 追加: RTC用の状態 (policyインスタンスごと)
_RTC_STATE: dict[int, dict[str, Any]] = {}


def _get_rtc_state(policy) -> dict[str, Any]:
    sid = id(policy)
    st = _RTC_STATE.get(sid)
    if st is None:
        st = {
            "queue": deque(),
            "prev_leftover": None,
            "last_infer_s": None,
        }
        _RTC_STATE[sid] = st
    return st


def _to_1step_action(a):
    """
    RTCチャンクから取り出した要素を「1ステップのaction (A,)」に正規化する。
    flattenは絶対にしない (flattenすると (T*A,) になって壊れる).
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)

    if not isinstance(a, torch.Tensor):
        return a

    # (1, A) -> (A,)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]

    # (T, A) みたいなのが紛れたら先頭ステップだけ取る
    if a.ndim == 2 and a.shape[0] > 1:
        a = a[0]

    # (B, T, A) とかの事故も先頭を辿って潰す
    while a.ndim > 1:
        a = a[0]

    return a


def _enqueue_chunk(st: dict[str, Any], chunk):
    q = st["queue"]

    if isinstance(chunk, torch.Tensor):
        # expect (T, A) or (1, T, A)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]  # (T, A)

        if chunk.ndim != 2:
            raise RuntimeError(f"RTC chunk tensor has unexpected shape: {tuple(chunk.shape)}")

        T = int(chunk.shape[0])
        for i in range(T):
            q.append(_to_1step_action(chunk[i]))

        # log once
        if not getattr(_enqueue_chunk, "_logged", False):
            logging.info("[RTC] enqueued T=%d, queue_len=%d", T, len(q))
            _enqueue_chunk._logged = True
        return

    if isinstance(chunk, (list, tuple)):
        for x in chunk:
            q.append(_to_1step_action(x))
        if not getattr(_enqueue_chunk, "_logged", False):
            logging.info("[RTC] enqueued list len=%d, queue_len=%d", len(chunk), len(q))
            _enqueue_chunk._logged = True
        return

    raise RuntimeError(f"RTC chunk has unsupported type: {type(chunk)}")


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
    fps: int = 30,
):
    observation = copy(observation)

    amp_ctx = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )

    with torch.inference_mode(), amp_ctx:
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        rtc_cfg = getattr(policy.config, "rtc_config", None)
        rtc_enabled = bool(getattr(rtc_cfg, "enabled", False))

        if not rtc_enabled:
            action = policy.select_action(observation)
            action = _to_1step_action(action)
            action = postprocessor(action)
            action = _to_1step_action(action)
            return action

        st = _get_rtc_state(policy)
        st["frame_ctr"] = st.get("frame_ctr", 0) + 1

        if not getattr(predict_action, "_rtc_logged", False):
            logging.info(
                "[RTC] ENTER enabled=%s exec_horizon=%s max_guidance=%s schedule=%s",
                rtc_enabled,
                getattr(rtc_cfg, "execution_horizon", None),
                getattr(rtc_cfg, "max_guidance_weight", None),
                getattr(rtc_cfg, "prefix_attention_schedule", None),
            )
            predict_action._rtc_logged = True

        if len(st["queue"]) > 0:
            action = st["queue"].popleft()
            action = _to_1step_action(action)
            action = postprocessor(action)
            action = _to_1step_action(action)
            return action

        # measure actual compute latency of chunk generation
        t0 = time.perf_counter()
        out = policy.predict_action_chunk(
            observation,
            inference_delay=0,  # placeholder, will be overridden below if leftover exists
            prev_chunk_left_over=st["prev_leftover"],
        )
        t1 = time.perf_counter()

        step_s = 1.0 / float(fps)
        inference_delay_steps = max(0, int(round((t1 - t0) / step_s)))

        # unpack out:
        # - (chunk, leftover)
        # - {"chunk": ..., "prev_chunk_left_over": ...} etc.
        chunk = None
        prev_leftover = None

        if isinstance(out, tuple) and len(out) == 2:
            chunk, prev_leftover = out
        elif isinstance(out, dict):
            # best-effort keys
            chunk = out.get("chunk", None) or out.get("actions", None) or out.get("action_chunk", None)
            prev_leftover = out.get("prev_chunk_left_over", None) or out.get("prev_leftover", None)
        else:
            chunk = out

        # update leftover if we got one
        if prev_leftover is not None:
            st["prev_leftover"] = prev_leftover

        # log every chunk generation (now delay is compute-latency based)
        logging.info(
            "[RTC] CHUNK_GEN frame=%d compute_delay_steps=%d prev_leftover_none=%s",
            st["frame_ctr"],
            inference_delay_steps,
            st["prev_leftover"] is None,
        )

        if not getattr(predict_action, "_chunk_logged", False):
            if isinstance(chunk, torch.Tensor):
                logging.info("[RTC] chunk tensor shape=%s dtype=%s", tuple(chunk.shape), chunk.dtype)
            else:
                logging.info(
                    "[RTC] chunk type=%s len=%s first_type=%s",
                    type(chunk),
                    len(chunk) if hasattr(chunk, "__len__") else None,
                    type(chunk[0]) if isinstance(chunk, (list, tuple)) and len(chunk) > 0 else None,
                )
            if not getattr(predict_action, "_out_logged", False):
                logging.info("[RTC] out type=%s", type(out))
                predict_action._out_logged = True
            predict_action._chunk_logged = True

        _enqueue_chunk(st, chunk)

        action = st["queue"].popleft()
        action = _to_1step_action(action)
        action = postprocessor(action)
        action = _to_1step_action(action)
        return action

# def predict_action(
#     observation: dict[str, np.ndarray],
#     policy: PreTrainedPolicy,
#     device: torch.device,
#     preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
#     postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
#     use_amp: bool,
#     task: str | None = None,
#     robot_type: str | None = None,
# ):
#     """
#     Performs a single-step inference to predict a robot action from an observation.

#     This function encapsulates the full inference pipeline:
#     1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
#     2. Runs the preprocessor pipeline on the observation.
#     3. Feeds the processed observation to the policy to get a raw action.
#     4. Runs the postprocessor pipeline on the raw action.
#     5. Formats the final action by removing the batch dimension and moving it to the CPU.

#     Args:
#         observation: A dictionary of NumPy arrays representing the robot's current observation.
#         policy: The `PreTrainedPolicy` model to use for action prediction.
#         device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
#         preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
#         postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
#         use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
#         task: An optional string identifier for the task.
#         robot_type: An optional string identifier for the robot type.

#     Returns:
#         A `torch.Tensor` containing the predicted action, ready for the robot.
#     """
#     observation = copy(observation)
#     with (
#         torch.inference_mode(),
#         torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
#     ):
#         # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
#         observation = prepare_observation_for_inference(observation, device, task, robot_type)
#         observation = preprocessor(observation)

#         # Compute the next action with the policy
#         # based on the current observation
#         action = policy.select_action(observation)

#         action = postprocessor(action)

#     return action


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
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
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
