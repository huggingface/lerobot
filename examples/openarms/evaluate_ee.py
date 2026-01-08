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
OpenArms End-Effector Policy Evaluation

Evaluates a policy trained on end-effector (EE) space by:
1. Converting robot joint observations to EE poses (FK)
2. Running policy inference with EE state
3. Converting EE action output back to joint positions (IK)
4. Sending joint commands to robot

Example usage:
    python examples/openarms/evaluate_ee.py
    python examples/openarms/evaluate_ee.py --model lerobot/my-ee-policy
"""

import time
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.relative_actions import (
    convert_state_to_relative,
    convert_from_relative_actions,
    PerTimestepNormalizer,
)
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.openarms.config_openarms_follower import OpenArmsFollowerConfig
from lerobot.robots.openarms.openarms_follower import OpenArmsFollower
from lerobot.robots.openarms.robot_kinematic_processor import (
    BimanualEEBoundsAndSafety,
    BimanualForwardKinematicsJointsToEE,
    BimanualInverseKinematicsEEToJoints,
)
from lerobot.teleoperators.openarms.config_openarms_leader import OpenArmsLeaderConfig
from lerobot.teleoperators.openarms.openarms_leader import OpenArmsLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# Configuration
HF_MODEL_ID = "lerobot-data-collection/pi0_ee"  # TODO: Replace with your EE-trained model
HF_EVAL_DATASET_ID = "your-org/your-ee-eval-dataset"  # TODO: Replace with your eval dataset
TASK_DESCRIPTION = "ee-policy-task"  # TODO: Replace with your task

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 1000
RESET_TIME_SEC = 60

# Robot CAN interfaces
FOLLOWER_LEFT_PORT = "can0"
FOLLOWER_RIGHT_PORT = "can1"

# Leader for manual resets (disabled by default)
USE_LEADER_FOR_RESETS = False
LEADER_LEFT_PORT = "can2"
LEADER_RIGHT_PORT = "can3"

# Camera configuration
CAMERA_CONFIG = {
    "left_wrist": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=FPS),
    "right_wrist": OpenCVCameraConfig(index_or_path="/dev/video1", width=640, height=480, fps=FPS),
    "base": OpenCVCameraConfig(index_or_path="/dev/video3", width=640, height=480, fps=FPS),
}

# Kinematics configuration
DEFAULT_URDF = "src/lerobot/robots/openarms/urdf/openarm_bimanual_pybullet.urdf"
DEFAULT_LEFT_EE_FRAME = "openarm_left_hand_tcp"
DEFAULT_RIGHT_EE_FRAME = "openarm_right_hand_tcp"

MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]
LEFT_URDF_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_URDF_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]


def load_relative_config(model_path: Path | str) -> tuple[PerTimestepNormalizer | None, bool, bool]:
    """Auto-detect relative action/state settings and load normalizer from checkpoint."""
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    normalizer = None
    use_relative_actions = False
    use_relative_state = False
    
    # Try local path first
    if model_path.exists():
        stats_path = model_path / "relative_stats.pt"
        if stats_path.exists():
            normalizer = PerTimestepNormalizer.load(stats_path)
            use_relative_actions = True
            print(f"  Loaded per-timestep stats from: {stats_path}")
        
        config_path = model_path / "train_config.json"
        if config_path.exists():
            cfg = TrainPipelineConfig.from_pretrained(model_path)
            use_relative_actions = getattr(cfg, "use_relative_actions", use_relative_actions)
            use_relative_state = getattr(cfg, "use_relative_state", False)
    else:
        # Try hub
        try:
            from huggingface_hub import hf_hub_download
            try:
                stats_file = hf_hub_download(repo_id=str(model_path), filename="relative_stats.pt")
                normalizer = PerTimestepNormalizer.load(stats_file)
                use_relative_actions = True
                print("  Loaded per-timestep stats from hub")
            except Exception:
                pass  # No stats file means no relative actions
            
            try:
                config_file = hf_hub_download(repo_id=str(model_path), filename="train_config.json")
                cfg = TrainPipelineConfig.from_pretrained(Path(config_file).parent)
                use_relative_actions = getattr(cfg, "use_relative_actions", use_relative_actions)
                use_relative_state = getattr(cfg, "use_relative_state", False)
            except Exception:
                pass
        except Exception as e:
            print(f"  Warning: Could not load relative config: {e}")
    
    return normalizer, use_relative_actions, use_relative_state


def build_kinematics_pipelines(urdf_path: str, left_ee_frame: str, right_ee_frame: str):
    """Build FK and IK pipelines for bimanual robot."""
    left_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=left_ee_frame,
        joint_names=LEFT_URDF_JOINTS,
    )
    right_kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=right_ee_frame,
        joint_names=RIGHT_URDF_JOINTS,
    )

    # Joints -> EE (Forward Kinematics)
    joints_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            BimanualForwardKinematicsJointsToEE(
                left_kinematics=left_kinematics,
                right_kinematics=right_kinematics,
                motor_names=MOTOR_NAMES,
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # EE -> Joints (Inverse Kinematics)
    ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            BimanualEEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            BimanualInverseKinematicsEEToJoints(
                left_kinematics=left_kinematics,
                right_kinematics=right_kinematics,
                motor_names=MOTOR_NAMES,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    return joints_to_ee, ee_to_joints


def convert_obs_joints_to_ee(obs: dict, joints_to_ee_pipeline) -> dict:
    """Convert joint observations to EE space."""
    # Extract joint positions from observation
    joint_positions = {}
    for key, value in obs.items():
        if key.startswith("observation.state.") and key.endswith(".pos"):
            # e.g., observation.state.left_joint_1.pos -> left_joint_1.pos
            motor_key = key.replace("observation.state.", "")
            joint_positions[motor_key] = value
    
    if not joint_positions:
        return obs
    
    # Apply FK to get EE poses
    ee_poses = joints_to_ee_pipeline(joint_positions)
    
    # Build new observation with EE state
    new_obs = {}
    for key, value in obs.items():
        if not (key.startswith("observation.state.") and key.endswith(".pos")):
            new_obs[key] = value
    
    # Add EE poses as state
    for key, value in ee_poses.items():
        new_obs[f"observation.state.{key}"] = value
    
    return new_obs


def convert_action_ee_to_joints(
    ee_action: dict,
    current_obs: dict,
    ee_to_joints_pipeline,
) -> dict:
    """Convert EE action to joint positions using IK."""
    # Extract EE components from action
    ee_action_dict = {}
    for key, value in ee_action.items():
        if "ee." in key:
            # e.g., action.left_ee.x -> left_ee.x
            ee_key = key.replace("action.", "")
            ee_action_dict[ee_key] = value
    
    if not ee_action_dict:
        return ee_action
    
    # Build current observation for IK (joint positions)
    current_joints = {}
    for key, value in current_obs.items():
        if key.startswith("observation.state.") and "joint" in key and key.endswith(".pos"):
            motor_key = key.replace("observation.state.", "")
            current_joints[motor_key] = value
    
    # Apply IK
    joint_action = ee_to_joints_pipeline((ee_action_dict, current_joints))
    
    # Format as action dict
    result = {}
    for key, value in joint_action.items():
        result[f"action.{key}"] = value
    
    return result


def run_ee_inference_loop(
    robot: OpenArmsFollower,
    policy,
    preprocessor,
    postprocessor,
    joints_to_ee,
    ee_to_joints,
    dataset: LeRobotDataset,
    fps: int,
    duration_s: float,
    events: dict,
    task: str,
    use_relative_actions: bool = False,
    use_relative_state: bool = False,
    relative_normalizer: PerTimestepNormalizer | None = None,
    display_data: bool = True,
):
    """Run inference loop with EE conversion and optional UMI-style relative actions."""
    device = get_safe_torch_device(policy.config.device)
    
    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    
    dt = 1.0 / fps
    timestamp = 0
    start_time = time.perf_counter()
    step = 0
    
    mode_str = ""
    if use_relative_actions:
        mode_str += " [relative actions]"
    if use_relative_state:
        mode_str += " [relative state]"
    print(f"\nRunning EE inference for {duration_s}s...{mode_str}")
    
    while timestamp < duration_s:
        loop_start = time.perf_counter()
        
        if events.get("exit_early"):
            events["exit_early"] = False
            break
        
        # 1. Get robot observation (joint positions)
        robot_obs = robot.get_observation()
        
        # 2. Convert joint observation to EE space using FK
        joint_state = {}
        for key, value in robot_obs.items():
            if key.endswith(".pos"):
                joint_state[key] = value
        
        ee_state = joints_to_ee(joint_state.copy())
        
        # 3. Build observation frame with EE state for policy input
        # Get expected state dimension from policy's input features
        state_feature = policy.config.input_features.get("observation.state")
        expected_dim = state_feature.shape[0] if state_feature else None
        
        # Build state array from EE values (sorted keys)
        ee_keys = sorted(ee_state.keys())
        ee_values = [ee_state[k] for k in ee_keys]
        
        # Truncate to match expected dimension (FK may output more than policy expects)
        if expected_dim and len(ee_values) > expected_dim:
            ee_values = ee_values[:expected_dim]
            ee_keys = ee_keys[:expected_dim]
        
        # Store current EE position for relative action conversion (using same order)
        current_ee_pos = torch.tensor(ee_values)
        
        # Convert to relative state if enabled (UMI-style)
        if use_relative_state:
            ee_state_tensor = torch.tensor(ee_values)
            relative_state = convert_state_to_relative(ee_state_tensor.unsqueeze(0))
            ee_values = [float(relative_state[0, i]) for i in range(len(ee_values))]
        
        # Build observation dict for policy (images + state as numpy arrays)
        observation_frame = {}
        
        # Add images - robot.cameras contains camera names as keys
        for cam_name in robot.cameras:
            if cam_name in robot_obs:
                observation_frame[f"observation.images.{cam_name}"] = robot_obs[cam_name]
        
        # Add state as numpy array
        observation_frame["observation.state"] = np.array(ee_values, dtype=np.float32)
        
        # 4. Run policy inference using predict_action
        action_tensor = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        
        # 5. Convert action tensor to dict
        ee_action = make_robot_action(action_tensor, dataset.features)
        
        # 6. Convert relative action back to absolute if needed
        if use_relative_actions:
            action_keys = sorted(ee_action.keys())
            action_vals = torch.tensor([ee_action[k] for k in action_keys])
            
            # Unnormalize if we have a normalizer
            if relative_normalizer is not None:
                action_vals = relative_normalizer.unnormalize(action_vals.unsqueeze(0).unsqueeze(0))
                action_vals = action_vals.squeeze(0).squeeze(0)
            
            # Convert from relative to absolute
            absolute_action = convert_from_relative_actions(action_vals.unsqueeze(0), current_ee_pos)
            
            # Convert back to dict
            ee_action = {k: float(absolute_action[0, i]) for i, k in enumerate(action_keys)}
        
        # 7. Convert EE action to joint positions using IK
        joint_action = ee_to_joints((ee_action.copy(), joint_state.copy()))
        
        # 8. Send joint commands to robot
        robot.send_action(joint_action)
        
        # 9. Save frame to dataset (save original robot obs + joint action)
        if dataset is not None:
            obs_frame = build_dataset_frame(dataset.features, robot_obs, prefix=OBS_STR)
            act_frame = build_dataset_frame(dataset.features, joint_action, prefix=ACTION)
            frame = {**obs_frame, **act_frame, "task": task}
            dataset.add_frame(frame)
        
        # 10. Visualization
        if display_data:
            log_rerun_data(observation=robot_obs, action=joint_action)
        
        # Progress logging
        step += 1
        if step % (fps * 5) == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Step {step}, elapsed: {elapsed:.1f}s")
        
        # Maintain loop rate
        loop_duration = time.perf_counter() - loop_start
        sleep_time = dt - loop_duration
        if sleep_time > 0:
            precise_sleep(sleep_time)
        
        timestamp = time.perf_counter() - start_time
    
    print(f"  Completed {step} steps")


def main():
    """Main evaluation function for EE policies."""
    print("=" * 70)
    print("OpenArms End-Effector Policy Evaluation")
    print("=" * 70)
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Dataset: {HF_EVAL_DATASET_ID}")
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Episode Duration: {EPISODE_TIME_SEC}s")
    print("=" * 70)
    
    # Resolve URDF path
    urdf_path = Path(__file__).parent.parent.parent / DEFAULT_URDF
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    urdf_path = str(urdf_path)
    
    # Build kinematics pipelines
    print("\n[1/5] Building kinematics pipelines...")
    joints_to_ee, ee_to_joints = build_kinematics_pipelines(
        urdf_path, DEFAULT_LEFT_EE_FRAME, DEFAULT_RIGHT_EE_FRAME
    )
    print("  FK and IK pipelines ready")
    
    # Initialize robot
    print("\n[2/5] Connecting to robot...")
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
        raise RuntimeError("Robot failed to connect!")
    print("  Robot connected")
    
    # Initialize leader for resets
    leader = None
    if USE_LEADER_FOR_RESETS:
        print("\n  Connecting leader for resets...")
        leader_config = OpenArmsLeaderConfig(
            port_left=LEADER_LEFT_PORT,
            port_right=LEADER_RIGHT_PORT,
            can_interface="socketcan",
            id="openarms_leader",
            manual_control=False,
        )
        leader = OpenArmsLeader(leader_config)
        leader.connect(calibrate=False)
        
        if leader.is_connected and leader.pin_robot is not None:
            leader.bus_right.enable_torque()
            leader.bus_left.enable_torque()
            print("  Leader connected with gravity compensation")
    
    # Create dataset for saving evaluation data
    print(f"\n[3/5] Creating evaluation dataset...")
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
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
    
    dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / HF_EVAL_DATASET_ID
    if dataset_path.exists():
        print(f"  Dataset exists at: {dataset_path}")
        if input("  Continue and overwrite? (y/n): ").strip().lower() != 'y':
            follower.disconnect()
            return
    
    dataset = LeRobotDataset.create(
        repo_id=HF_EVAL_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=12,
    )
    print("  Dataset created")
    
    # Load policy
    print(f"\n[4/5] Loading policy from {HF_MODEL_ID}...")
    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID
    
    # Pass dataset meta for policy creation (required by make_policy)
    policy = make_policy(policy_config, ds_meta=dataset.meta)
    
    # Load preprocessor/postprocessor from pretrained model
    # DO NOT pass dataset_stats - let it load from pretrained model
    # (evaluation dataset has different features than training dataset)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=HF_MODEL_ID,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )
    print("  Policy loaded")
    
    # Auto-detect relative action/state settings from checkpoint
    relative_normalizer, use_relative_actions, use_relative_state = load_relative_config(HF_MODEL_ID)
    
    mode = "absolute"
    if use_relative_actions:
        mode = "relative actions + state" if use_relative_state else "relative actions only"
    print(f"  Mode: {mode}")
    
    # Initialize keyboard listener and visualization
    print("\n[5/5] Starting evaluation...")
    listener, events = init_keyboard_listener()
    init_rerun(session_name="openarms_eval_ee")
    
    print("\nControls: ESC=stop, →=next episode, ←=rerecord")
    episode_idx = 0
    
    try:
        while episode_idx < NUM_EPISODES and not events.get("stop_recording"):
            log_say(f"Episode {episode_idx + 1} of {NUM_EPISODES}")
            print(f"\n{'='*50}")
            print(f"Episode {episode_idx + 1}/{NUM_EPISODES}")
            print(f"{'='*50}")
            
            input("\nPress ENTER to start episode...")
            events["exit_early"] = False
            
            # Run inference with EE conversion
            run_ee_inference_loop(
                robot=follower,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                joints_to_ee=joints_to_ee,
                ee_to_joints=ee_to_joints,
                dataset=dataset,
                fps=FPS,
                duration_s=EPISODE_TIME_SEC,
                events=events,
                task=TASK_DESCRIPTION,
                use_relative_actions=use_relative_actions,
                use_relative_state=use_relative_state,
                relative_normalizer=relative_normalizer,
            )
            
            # Handle re-recording
            if events.get("rerecord_episode", False):
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # Save episode if we have data
            if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
                print(f"  Saving episode {episode_idx + 1}...")
                dataset.save_episode()
                episode_idx += 1
            
            events["exit_early"] = False
            
            # Reset between episodes
            if episode_idx < NUM_EPISODES and not events.get("stop_recording"):
                if USE_LEADER_FOR_RESETS and leader and leader.is_connected:
                    log_say("Reset environment using leader arms")
                    print(f"\nManual reset ({RESET_TIME_SEC}s) - use leader arms...")
                    
                    reset_start = time.perf_counter()
                    while time.perf_counter() - reset_start < RESET_TIME_SEC:
                        if events.get("exit_early") or events.get("stop_recording"):
                            break
                        
                        leader_action = leader.get_action()
                        follower_action = {k: v for k, v in leader_action.items() if k.endswith(".pos")}
                        if follower_action:
                            follower.send_action(follower_action)
                        time.sleep(1/FPS)
                else:
                    input("\nReset environment and press ENTER...")
        
        print(f"\n✓ Evaluation complete! {episode_idx} episodes recorded")
        log_say("Evaluation complete", blocking=True)
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted")
    
    finally:
        if leader:
            if hasattr(leader, 'bus_right'):
                leader.bus_right.disable_torque()
            if hasattr(leader, 'bus_left'):
                leader.bus_left.disable_torque()
            leader.disconnect()
        
        follower.disconnect()
        
        if listener is not None:
            listener.stop()
        
        # Finalize and push dataset
        dataset.finalize()
        print("Uploading to Hub...")
        dataset.push_to_hub(private=True)
        
        print("✓ Done!")


if __name__ == "__main__":
    main()

