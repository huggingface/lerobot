#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
OpenArms Policy Evaluation with UMI-style Relative Actions

Evaluates a policy trained with relative actions (use_relative_actions=True).
During inference, the policy outputs relative deltas which are added to the
current robot position to get absolute targets.

This follows the UMI paper's "relative trajectory" action representation:
    action_absolute[t] = action_relative[t] + current_position

Example usage:
    python examples/openarms/evaluate_relative.py
"""

import time
from pathlib import Path

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import predict_action
from lerobot.processor import make_default_processors
from lerobot.processor.core import RobotAction
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.relative_actions import convert_from_relative_actions_dict, convert_state_to_relative
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# Configuration - Update these for your setup
HF_MODEL_ID = "your-org/your-relative-policy"  # Policy trained with use_relative_actions=True
HF_EVAL_DATASET_ID = "your-org/your-eval-dataset"
TASK_DESCRIPTION = "your task description"

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 300

# Robot CAN interfaces
FOLLOWER_LEFT_PORT = "can0"
FOLLOWER_RIGHT_PORT = "can1"

# Camera configuration
CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=FPS),
}


def make_robot_action(action_values: dict, features: dict) -> RobotAction:
    """Convert action values to robot action dict, filtering by features."""
    robot_action = {}
    for key in features:
        if key.startswith(ACTION + "."):
            action_key = key.removeprefix(ACTION + ".")
            if action_key in action_values:
                robot_action[action_key] = action_values[action_key]
    return robot_action


def inference_loop_relative(
    robot,
    policy,
    preprocessor,
    postprocessor,
    dataset,
    events,
    fps: int,
    control_time_s: float,
    single_task: str,
    display_data: bool = True,
    state_key: str = "observation.state",
):
    """
    Inference loop for policies trained with UMI-style relative actions and state.
    
    Key differences from standard inference:
    - Observation state is converted to relative (provides velocity info)
    - Policy outputs relative deltas (action_relative)
    - We add current robot position to get absolute targets:
      action_absolute = action_relative + current_position
    """
    device = get_safe_torch_device(policy.config.device)
    
    timestamp = 0
    start_t = time.perf_counter()
    
    while timestamp < control_time_s:
        loop_start = time.perf_counter()
        
        if events["exit_early"] or events["stop_recording"]:
            break
        
        # Get current robot observation
        obs = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        
        # Get current joint positions (reference for relative conversion)
        current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
        
        # Convert observation state to relative (UMI-style)
        # This gives velocity-like information to the policy
        if state_key in observation_frame:
            state_tensor = observation_frame[state_key]
            if isinstance(state_tensor, torch.Tensor):
                observation_frame[state_key] = convert_state_to_relative(state_tensor)
        
        # Run policy inference - outputs relative actions
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        
        # Convert relative actions to absolute
        # action_values contains relative deltas, current_pos has absolute positions
        relative_action = make_robot_action(action_values, dataset.features)
        absolute_action = convert_from_relative_actions_dict(relative_action, current_pos)
        
        # Send absolute action to robot
        robot.send_action(absolute_action)
        
        # Record to dataset (store the absolute action that was sent)
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, absolute_action, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        
        if display_data:
            log_rerun_data(observation=obs, action=absolute_action)
        
        dt = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt)
        timestamp = time.perf_counter() - start_t


def main():
    """Main evaluation function for relative action policies."""
    print("=" * 65)
    print("  OpenArms Evaluation - UMI-style Relative Actions")
    print("=" * 65)
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Evaluation Dataset: {HF_EVAL_DATASET_ID}")
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Episode Duration: {EPISODE_TIME_SEC}s")
    print("\nNote: Policy outputs are relative deltas, converted to absolute at inference time")
    
    # Setup robot
    follower_config = OpenArmsFollowerConfig(
        port_left=FOLLOWER_LEFT_PORT,
        port_right=FOLLOWER_RIGHT_PORT,
        can_interface="socketcan",
        id="openarms_follower",
        disable_torque_on_disconnect=True,
        max_relative_target=10.0,
        cameras=CAMERA_CONFIG,
    )
    
    follower = OpenArmsFollower(follower_config)
    follower.connect(calibrate=False)
    
    if not follower.is_connected:
        raise RuntimeError("Follower robot failed to connect!")

    # Build processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Build dataset features
    action_features_hw = {k: v for k, v in follower.action_features.items() if k.endswith(".pos")}
    
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
    )
    
    # Check existing dataset
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / HF_EVAL_DATASET_ID
    if dataset_path.exists():
        print(f"\nDataset already exists at: {dataset_path}")
        choice = input("Continue and append? (y/n): ").strip().lower()
        if choice != 'y':
            follower.disconnect()
            return
    
    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_EVAL_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=12, 
    )
    
    # Load policy
    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID
    policy = make_policy(policy_config, ds_meta=dataset.meta)
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )

    # Initialize controls
    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarms_eval_relative")
    episode_idx = 0
    
    print("\nControls:")
    print("  ESC    - Stop recording and save")
    print("  →      - End current episode")
    print("  ←      - Re-record episode")
    
    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Evaluating episode {episode_idx + 1} of {NUM_EPISODES}")
            print(f"\nRunning relative action inference for episode {episode_idx + 1}...")
            
            # Run inference with relative action conversion
            inference_loop_relative(
                robot=follower,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                events=events,
                fps=FPS,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            
            # Handle re-recording
            if events.get("rerecord_episode", False):
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # Save episode
            if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                print(f"Saving episode {episode_idx + 1} ({dataset.episode_buffer['size']} frames)...")
                dataset.save_episode()
                episode_idx += 1
            
            events["exit_early"] = False
            
            # Wait for manual reset between episodes
            if not events["stop_recording"] and episode_idx < NUM_EPISODES:
                log_say("Waiting for manual reset")
                input("Press ENTER when ready for next episode...")
        
        print(f"\nEvaluation complete! {episode_idx} episodes recorded")
        log_say("Evaluation complete", blocking=True)
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    finally:
        follower.disconnect()
        
        if listener is not None:
            listener.stop()
        
        dataset.finalize()
        print("\nUploading to Hugging Face Hub...")
        dataset.push_to_hub(private=True)


if __name__ == "__main__":
    main()

