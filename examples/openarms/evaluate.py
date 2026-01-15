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
OpenArms Policy Evaluation

Evaluates a trained policy on the OpenArms robot by running inference and recording
the evaluation episodes to a dataset. Supports optional leader arm for manual resets.

Example usage:
    python examples/openarms/evaluate.py
"""

import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


HF_MODEL_ID = "lerobot-data-collection/level1_rac2_100k"  # TODO: Replace with your trained model
HF_EVAL_DATASET_ID = "lerobot-data-collection/three-folds-pi0_eval_raccc3"  # TODO: Replace with your eval dataset name
TASK_DESCRIPTION = "Fold the T-shirt properly" # TODO: Replace with your task, this should match!!

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 1000
RESET_TIME_SEC = 60

# Robot CAN interfaces
FOLLOWER_LEFT_PORT = "can0"
FOLLOWER_RIGHT_PORT = "can1"

# If enabled, you can manually reset the environment between evaluation episodes
USE_LEADER_FOR_RESETS = False  # Set to False if you don't want to use leader
LEADER_LEFT_PORT = "can2"
LEADER_RIGHT_PORT = "can3"

# Camera configuration
CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video0", width=1280, height=720, fps=FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=1280, height=720, fps=FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=FPS),
}

def main():
    """Main evaluation function."""
    print("OpenArms Policy Evaluation")
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Evaluation Dataset: {HF_EVAL_DATASET_ID}")
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Episode Duration: {EPISODE_TIME_SEC}s")
    print(f"Reset Duration: {RESET_TIME_SEC}s")
    print(f"Use Leader for Resets: {USE_LEADER_FOR_RESETS}")
    
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

    
    leader = None
    if USE_LEADER_FOR_RESETS:
        leader_config = OpenArmsLeaderConfig(
            port_left=LEADER_LEFT_PORT,
            port_right=LEADER_RIGHT_PORT,
            can_interface="socketcan",
            id="openarms_leader",
            manual_control=False,  # Enable torque control for gravity compensation
        )
        
        leader = OpenArmsLeader(leader_config)
        leader.connect(calibrate=False)
        
        if not leader.is_connected:
            raise RuntimeError("Leader robot failed to connect!")
        
        # Enable gravity compensation
        if leader.pin_robot is not None:
            leader.bus_right.enable_torque()
            leader.bus_left.enable_torque()
            time.sleep(0.1)
            print(f"Leader connected with gravity compensation ({LEADER_LEFT_PORT}, {LEADER_RIGHT_PORT})")
        else:
            print(f"Leader connected but gravity compensation unavailable (no URDF)")

    # Build default processors for action and observation
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # Build dataset features from robot features and processors
    # For actions, only include positions (no velocity or torque)
    action_features_hw = {}
    for key, value in follower.action_features.items():
        if key.endswith(".pos"):
            action_features_hw[key] = value
    
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
    
    # Check if dataset already exists
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / HF_EVAL_DATASET_ID
    if dataset_path.exists():
        print(f"Evaluation dataset already exists at: {dataset_path}")
        print("This will append new episodes to the existing dataset.")
        choice = input("  Continue? (y/n): ").strip().lower()
        if choice != 'y':
            print("  Aborting evaluation.")
            follower.disconnect()
            if leader:
                leader.disconnect()
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
    
    # Load policy config from pretrained model and create policy using factory
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

    print(f"\nRunning evaluation...")
    # Initialize keyboard listener and visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarms_evaluation")
    episode_idx = 0
    
    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Evaluating episode {episode_idx + 1} of {NUM_EPISODES}")
            print(f"\nRunning inference for episode {episode_idx + 1}...")
            
            # Run inference with policy
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            
            # Handle re-recording
            if events["rerecord_episode"]:
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
            
            # Reset environment between episodes (if not last episode)
            if not events["stop_recording"] and episode_idx < NUM_EPISODES:
                if USE_LEADER_FOR_RESETS and leader:
                    log_say("Reset the environment using leader arms")
                    print(f"\nManual reset period ({RESET_TIME_SEC}s)...")
                    
                    # Use leader for manual reset with gravity compensation
                    import numpy as np
                    
                    dt = 1 / FPS
                    reset_start_time = time.perf_counter()
                    
                    while time.perf_counter() - reset_start_time < RESET_TIME_SEC:
                        if events["exit_early"] or events["stop_recording"]:
                            break
                        
                        loop_start = time.perf_counter()
                        
                        # Get leader state
                        leader_action = leader.get_action()
                        
                        # Extract positions and velocities
                        leader_positions_deg = {}
                        leader_velocities_deg_per_sec = {}
                        
                        for motor in leader.bus_right.motors:
                            pos_key = f"right_{motor}.pos"
                            vel_key = f"right_{motor}.vel"
                            if pos_key in leader_action:
                                leader_positions_deg[f"right_{motor}"] = leader_action[pos_key]
                            if vel_key in leader_action:
                                leader_velocities_deg_per_sec[f"right_{motor}"] = leader_action[vel_key]
                        
                        for motor in leader.bus_left.motors:
                            pos_key = f"left_{motor}.pos"
                            vel_key = f"left_{motor}.vel"
                            if pos_key in leader_action:
                                leader_positions_deg[f"left_{motor}"] = leader_action[pos_key]
                            if vel_key in leader_action:
                                leader_velocities_deg_per_sec[f"left_{motor}"] = leader_action[vel_key]
                        
                        # Calculate gravity and friction torques
                        leader_positions_rad = {k: np.deg2rad(v) for k, v in leader_positions_deg.items()}
                        leader_gravity_torques_nm = leader._gravity_from_q(leader_positions_rad)
                        
                        leader_velocities_rad_per_sec = {k: np.deg2rad(v) for k, v in leader_velocities_deg_per_sec.items()}
                        leader_friction_torques_nm = leader._friction_from_velocity(
                            leader_velocities_rad_per_sec,
                            friction_scale=1.0
                        )
                        
                        # Combine torques
                        leader_total_torques_nm = {}
                        for motor_name in leader_gravity_torques_nm:
                            gravity = leader_gravity_torques_nm.get(motor_name, 0.0)
                            friction = leader_friction_torques_nm.get(motor_name, 0.0)
                            leader_total_torques_nm[motor_name] = gravity + friction
                        
                        # Apply compensation
                        for motor in leader.bus_right.motors:
                            full_name = f"right_{motor}"
                            position = leader_positions_deg.get(full_name, 0.0)
                            torque = leader_total_torques_nm.get(full_name, 0.0)
                            kd = leader.get_damping_kd(motor)
                            
                            leader.bus_right._mit_control(
                                motor=motor, kp=0.0, kd=kd,
                                position_degrees=position,
                                velocity_deg_per_sec=0.0,
                                torque=torque,
                            )
                        
                        for motor in leader.bus_left.motors:
                            full_name = f"left_{motor}"
                            position = leader_positions_deg.get(full_name, 0.0)
                            torque = leader_total_torques_nm.get(full_name, 0.0)
                            kd = leader.get_damping_kd(motor)
                            
                            leader.bus_left._mit_control(
                                motor=motor, kp=0.0, kd=kd,
                                position_degrees=position,
                                velocity_deg_per_sec=0.0,
                                torque=torque,
                            )
                        
                        # Send leader positions to follower
                        follower_action = {}
                        for joint in leader_positions_deg.keys():
                            pos_key = f"{joint}.pos"
                            if pos_key in leader_action:
                                follower_action[pos_key] = leader_action[pos_key]
                        
                        if follower_action:
                            follower.send_action(follower_action)
                        
                        # Maintain loop rate
                        loop_duration = time.perf_counter() - loop_start
                        sleep_time = dt - loop_duration
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
                    print("Reset complete")
                else:
                    log_say("Waiting for manual reset")
                    print(f"Manually reset the environment and press ENTER to continue")
                    input("Press ENTER when ready...")
        
        print(f"Evaluation complete! {episode_idx} episodes recorded")
        log_say("Evaluation complete", blocking=True)
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    finally:
        if leader:
            leader.bus_right.disable_torque()
            leader.bus_left.disable_torque()
            time.sleep(0.1)
            leader.disconnect()

        follower.disconnect()
        
        if listener is not None:
            listener.stop()
        
        dataset.finalize()
        print("\nUploading to Hugging Face Hub...")
        dataset.push_to_hub(private=True)


if __name__ == "__main__":
    main()

