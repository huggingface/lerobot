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
from lerobot.utils.relative_actions import (
    convert_state_to_relative,
    convert_from_relative_actions,
    PerTimestepNormalizer,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
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
TASK_DESCRIPTION = "ee-policy-task"  # TODO: Replace with your task

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 1000
RESET_TIME_SEC = 60

# UMI-style relative action/state (must match training config)
USE_RELATIVE_ACTIONS = False  # If True, policy outputs relative EE actions
USE_RELATIVE_STATE = False    # If True, convert state to relative before policy input
RELATIVE_STATS_PATH = None    # Path to relative_stats.pt (for per-timestep normalization)

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
    fps: int,
    duration_s: float,
    events: dict,
    use_relative_actions: bool = False,
    use_relative_state: bool = False,
    relative_normalizer: PerTimestepNormalizer | None = None,
):
    """Run inference loop with EE conversion and optional UMI-style relative actions."""
    dt = 1.0 / fps
    start_time = time.perf_counter()
    step = 0
    
    mode_str = ""
    if use_relative_actions:
        mode_str += " [relative actions]"
    if use_relative_state:
        mode_str += " [relative state]"
    print(f"\nRunning EE inference for {duration_s}s...{mode_str}")
    
    while time.perf_counter() - start_time < duration_s:
        if events.get("exit_early") or events.get("stop_recording"):
            break
        
        loop_start = time.perf_counter()
        
        # 1. Get robot observation (joint positions)
        robot_obs = robot.get_observation()
        
        # 2. Convert joint observation to EE space using FK
        joint_state = {}
        for key, value in robot_obs.items():
            if key.endswith(".pos"):
                joint_state[key] = value
        
        ee_state = joints_to_ee(joint_state.copy())
        
        # Store current EE position for relative action conversion
        current_ee_pos = torch.tensor([ee_state.get(k, 0.0) for k in sorted(ee_state.keys())])
        
        # 3. Build policy input with EE state
        # Convert to relative state if enabled (UMI-style)
        if use_relative_state:
            ee_state_tensor = torch.tensor([ee_state[k] for k in sorted(ee_state.keys())])
            relative_state = convert_state_to_relative(ee_state_tensor.unsqueeze(0))
            ee_state = {k: float(relative_state[0, i]) for i, k in enumerate(sorted(ee_state.keys()))}
        
        policy_obs = {"observation.state": ee_state}
        
        # Add images
        for cam_name in robot.cameras:
            img = robot_obs.get(f"{cam_name}.image")
            if img is not None:
                policy_obs[f"observation.images.{cam_name}"] = img
        
        # 4. Preprocess and run policy
        batch = preprocessor(policy_obs)
        
        # Add batch dimension if needed
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dim() == 1:
                batch[key] = batch[key].unsqueeze(0)
            elif isinstance(batch[key], torch.Tensor) and batch[key].dim() == 3:
                batch[key] = batch[key].unsqueeze(0)
        
        with torch.inference_mode():
            action = policy.select_action(batch)
        
        # 5. Postprocess to get EE action
        ee_action = postprocessor(action)
        
        # 6. Convert relative action back to absolute if needed
        if use_relative_actions:
            # Convert dict to tensor for relative->absolute conversion
            action_keys = sorted([k for k in ee_action.keys() if "ee." in k or k.endswith(".pos")])
            action_tensor = torch.tensor([ee_action.get(k, 0.0) for k in action_keys])
            
            # Unnormalize if we have a normalizer
            if relative_normalizer is not None:
                action_tensor = relative_normalizer.unnormalize(action_tensor.unsqueeze(0).unsqueeze(0))
                action_tensor = action_tensor.squeeze(0).squeeze(0)
            
            # Convert from relative to absolute
            absolute_action = convert_from_relative_actions(action_tensor.unsqueeze(0), current_ee_pos)
            
            # Convert back to dict
            ee_action = {k: float(absolute_action[0, i]) for i, k in enumerate(action_keys)}
        
        # 7. Convert EE action to joint positions using IK
        joint_action = ee_to_joints((ee_action.copy(), joint_state.copy()))
        
        # 8. Send joint commands to robot
        robot.send_action(joint_action)
        
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
    
    print(f"  Completed {step} steps")


def main():
    """Main evaluation function for EE policies."""
    print("=" * 70)
    print("OpenArms End-Effector Policy Evaluation")
    print("=" * 70)
    print(f"\nModel: {HF_MODEL_ID}")
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Episode Duration: {EPISODE_TIME_SEC}s")
    print(f"\nUMI-style relative mode:")
    print(f"  Relative actions: {USE_RELATIVE_ACTIONS}")
    print(f"  Relative state: {USE_RELATIVE_STATE}")
    if RELATIVE_STATS_PATH:
        print(f"  Stats path: {RELATIVE_STATS_PATH}")
    print("=" * 70)
    
    # Resolve URDF path
    urdf_path = Path(__file__).parent.parent.parent / DEFAULT_URDF
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    urdf_path = str(urdf_path)
    
    # Build kinematics pipelines
    print("\n[1/4] Building kinematics pipelines...")
    joints_to_ee, ee_to_joints = build_kinematics_pipelines(
        urdf_path, DEFAULT_LEFT_EE_FRAME, DEFAULT_RIGHT_EE_FRAME
    )
    print("  FK and IK pipelines ready")
    
    # Initialize robot
    print("\n[2/4] Connecting to robot...")
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
    
    # Load policy
    print(f"\n[3/4] Loading policy from {HF_MODEL_ID}...")
    policy_config = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
    policy_config.pretrained_path = HF_MODEL_ID
    
    # Create policy without dataset meta (use config defaults)
    policy = make_policy(policy_config, ds_meta=None)
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=HF_MODEL_ID,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)}
        },
    )
    print("  Policy loaded")
    
    # Load relative action normalizer if using relative actions
    relative_normalizer = None
    if USE_RELATIVE_ACTIONS and RELATIVE_STATS_PATH:
        stats_path = Path(RELATIVE_STATS_PATH)
        if stats_path.exists():
            print(f"  Loading relative stats from: {stats_path}")
            relative_normalizer = PerTimestepNormalizer.load(stats_path)
        else:
            print(f"  WARNING: Relative stats not found at {stats_path}")
    
    # Initialize keyboard listener
    print("\n[4/4] Starting evaluation...")
    listener, events = init_keyboard_listener()
    
    try:
        for episode_idx in range(NUM_EPISODES):
            if events.get("stop_recording"):
                break
            
            log_say(f"Starting episode {episode_idx + 1} of {NUM_EPISODES}")
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
                fps=FPS,
                duration_s=EPISODE_TIME_SEC,
                events=events,
                use_relative_actions=USE_RELATIVE_ACTIONS,
                use_relative_state=USE_RELATIVE_STATE,
                relative_normalizer=relative_normalizer,
            )
            
            # Reset between episodes
            if episode_idx < NUM_EPISODES - 1 and not events.get("stop_recording"):
                if USE_LEADER_FOR_RESETS and leader and leader.is_connected:
                    log_say("Reset environment using leader arms")
                    print(f"\nManual reset ({RESET_TIME_SEC}s) - use leader arms...")
                    
                    # Simple teleop reset loop
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
                    log_say("Manual reset required")
                    input("Reset environment and press ENTER...")
        
        print(f"\n✓ Evaluation complete! {NUM_EPISODES} episodes")
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
        
        print("✓ Done!")


if __name__ == "__main__":
    main()

